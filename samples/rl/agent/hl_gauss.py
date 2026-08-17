"""HL-Gauss categorical value support.

The critic predicts a distribution over a fixed return support and decodes its
expectation, rather than regressing a scalar. Screeps returns are open-ended and
differ in scale between PPO and joint pretraining, so the production geometry is
uniform in ``symlog(return)`` rather than in raw-return space.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def symlog(value: Tensor) -> Tensor:
    """Signed log transform with unit-scale linear behavior around zero."""
    value = value.float()
    return value.sign() * torch.log1p(value.abs())


def symexp(value: Tensor) -> Tensor:
    """Exact inverse of :func:`symlog`."""
    value = value.float()
    return value.sign() * torch.expm1(value.abs())


class HLGaussSupport(nn.Module):
    """Gaussian-smoothed categorical support with explicit target bounds.

    The ordinary constructor retains a linear edge support for focused tests
    and experiments. Production critics use :meth:`symmetric_symlog`, which
    places exact raw-return anchors at ``-max_abs_return``, zero, and
    ``+max_abs_return`` while adding Gaussian tail margin outside the declared
    target range.

    Targets are never clamped. Non-finite or out-of-contract targets raise a
    descriptive error, and :meth:`target_diagnostics` exposes proximity to and
    overflow beyond the declared range before projection.
    """

    def __init__(
        self,
        num_bins: int,
        v_min: float,
        v_max: float,
        sigma_ratio: float,
        eps: float = 1e-10,
    ):
        super().__init__()
        self._configure(
            num_bins=int(num_bins),
            support_min=float(v_min),
            support_max=float(v_max),
            target_min=float(v_min),
            target_max=float(v_max),
            sigma_ratio=float(sigma_ratio),
            eps=float(eps),
            transform="identity",
        )

    @classmethod
    def symmetric_symlog(
        cls,
        *,
        max_abs_return: float,
        interior_bins: int,
        margin_bins: int,
        sigma_ratio: float,
        eps: float = 1e-10,
    ) -> "HLGaussSupport":
        """Build a zero-centered signed-log support with protected endpoints.

        ``interior_bins`` is the number of uniform divisions from the negative
        to positive raw target anchor and must be even, so zero and both target
        extrema are bin centers. ``margin_bins`` adds full bins beyond each
        anchor; the additional half-bin comes from centering the anchors.
        """
        if not math.isfinite(max_abs_return) or max_abs_return <= 0:
            raise ValueError("max_abs_return must be finite and positive")
        if interior_bins < 2 or interior_bins % 2:
            raise ValueError("interior_bins must be a positive even integer >= 2")
        if margin_bins < 0:
            raise ValueError("margin_bins must be non-negative")
        if not math.isfinite(sigma_ratio) or sigma_ratio <= 0:
            raise ValueError("sigma_ratio must be finite and positive")

        anchor = math.log1p(float(max_abs_return))
        width = 2.0 * anchor / interior_bins
        support_abs = anchor + (margin_bins + 0.5) * width
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.interior_bins = int(interior_bins)
        obj.margin_bins = int(margin_bins)
        obj.max_abs_return = float(max_abs_return)
        obj._configure(
            num_bins=interior_bins + 1 + 2 * margin_bins,
            support_min=-support_abs,
            support_max=support_abs,
            # Predictions are convex combinations of centers and can therefore
            # enter the margin outside the expected-return anchors. Validate
            # targets against the inverse-transformed *edges*, not the anchors,
            # so a high bootstrap produced by the categorical head remains a
            # valid next target. Such targets are reported as saturation.
            target_min=-math.expm1(support_abs),
            target_max=math.expm1(support_abs),
            sigma_ratio=float(sigma_ratio),
            eps=float(eps),
            transform="symlog",
        )
        return obj

    def _configure(
        self,
        *,
        num_bins: int,
        support_min: float,
        support_max: float,
        target_min: float,
        target_max: float,
        sigma_ratio: float,
        eps: float,
        transform: str,
    ) -> None:
        if num_bins < 2:
            raise ValueError("HL-Gauss needs at least two bins")
        if not support_min < support_max:
            raise ValueError("HL-Gauss support_min must be below support_max")
        if not target_min < target_max:
            raise ValueError("HL-Gauss target_min must be below target_max")
        if not math.isfinite(sigma_ratio) or sigma_ratio <= 0:
            raise ValueError("sigma_ratio must be finite and positive")
        self.num_bins = int(num_bins)
        self.v_min = float(target_min)
        self.v_max = float(target_max)
        self.transform = transform
        self.support_min = float(support_min)
        self.support_max = float(support_max)
        self.bin_width = (support_max - support_min) / self.num_bins
        self.sigma = float(sigma_ratio) * self.bin_width
        self.sigma_ratio = float(sigma_ratio)
        self.eps = float(eps)
        edges = torch.linspace(support_min, support_max, self.num_bins + 1)
        self.register_buffer("edges", edges)
        self.register_buffer("centers", (edges[:-1] + edges[1:]) / 2.0)

    def _transform(self, targets: Tensor) -> Tensor:
        return symlog(targets) if self.transform == "symlog" else targets.float()

    def _inverse(self, values: Tensor) -> Tensor:
        return symexp(values) if self.transform == "symlog" else values.float()

    def target_diagnostics(self, targets: Tensor) -> dict[str, Tensor]:
        """Return device-local diagnostics; callers decide when to synchronize."""
        raw = targets.detach().float()
        finite = torch.isfinite(raw)
        overflow = (~finite) | (raw < self.v_min) | (raw > self.v_max)
        transformed = self._transform(torch.where(finite, raw, torch.zeros_like(raw)))
        # Saturation means a target is within three label sigmas of the actual
        # support edge. Correct production geometry has enough margin that even
        # target anchors do not saturate.
        near_edge = (
            (transformed - self.support_min < 3.0 * self.sigma)
            | (self.support_max - transformed < 3.0 * self.sigma)
        ) & finite
        denom = max(1, raw.numel())
        return {
            "overflow_count": overflow.sum(),
            "overflow_fraction": overflow.float().sum() / denom,
            "saturation_fraction": near_edge.float().sum() / denom,
            "target_min": raw.masked_fill(~finite, float("inf")).amin(),
            "target_max": raw.masked_fill(~finite, float("-inf")).amax(),
        }

    def validate_targets(self, targets: Tensor) -> None:
        """Raise if any target is non-finite or beyond the declared range."""
        diagnostics = self.target_diagnostics(targets)
        if int(diagnostics["overflow_count"].item()) == 0:
            return
        raise ValueError(
            "HL-Gauss target outside declared raw-return support "
            f"[{self.v_min:g}, {self.v_max:g}]: observed "
            f"[{float(diagnostics['target_min'].item()):g}, "
            f"{float(diagnostics['target_max'].item()):g}], "
            f"overflow_count={int(diagnostics['overflow_count'].item())}"
        )

    def project(self, targets: Tensor, *, validate: bool = True) -> Tensor:
        """Map scalar targets to Gaussian-smoothed categorical labels."""
        if validate:
            self.validate_targets(targets)
        transformed = self._transform(targets).unsqueeze(-1)
        cdf = torch.erf(
            (self.edges - transformed) / (self.sigma * math.sqrt(2.0))
        )
        mass = cdf[..., 1:] - cdf[..., :-1]
        return mass / mass.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def project_to_logprobs(self, targets: Tensor, eps: float = 1e-20) -> Tensor:
        return self.project(targets).clamp_min(eps).log()

    def to_expected_scalar(self, logits: Tensor) -> Tensor:
        """Decode by inverse-transforming the expected support coordinate."""
        transformed = (logits.float().softmax(-1) * self.centers).sum(-1)
        return self._inverse(transformed)

    def cross_entropy(
        self, logits: Tensor, targets: Tensor, *, validate: bool = True
    ) -> Tensor:
        """Per-position CE between projected targets and predicted logits."""
        return -(
            self.project(targets, validate=validate)
            * logits.float().log_softmax(-1)
        ).sum(-1)
