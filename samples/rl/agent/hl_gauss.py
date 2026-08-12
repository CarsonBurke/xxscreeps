"""HL-Gauss categorical value support (parameter-golf / cleanrl v215 recipe).

Ported for Screeps RL under CleanRL-normalized returns (support typically ±10).
Train with cross-entropy; decode with expected bin center for GAE.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class HLGaussSupport(nn.Module):
    """Uniform-edge discretized support with HL-Gauss target projection."""

    def __init__(
        self,
        num_bins: int,
        v_min: float,
        v_max: float,
        sigma_ratio: float,
        eps: float = 1e-10,
    ):
        super().__init__()
        self.num_bins = int(num_bins)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.bin_width = (self.v_max - self.v_min) / self.num_bins
        self.sigma = float(sigma_ratio) * self.bin_width
        self.eps = float(eps)
        edges = torch.linspace(self.v_min, self.v_max, self.num_bins + 1)
        self.register_buffer("edges", edges)
        self.register_buffer("centers", (edges[:-1] + edges[1:]) / 2.0)

    def project(self, targets: Tensor) -> Tensor:
        """(...,) scalar targets -> (..., num_bins) categorical distributions."""
        targets = targets.float().clamp(self.v_min, self.v_max).unsqueeze(-1)
        cdf = torch.erf((self.edges - targets) / (self.sigma * math.sqrt(2.0)))
        total = cdf[..., -1:] - cdf[..., :1]
        return (cdf[..., 1:] - cdf[..., :-1]) / total.clamp_min(self.eps)

    def project_to_logprobs(self, targets: Tensor, eps: float = 1e-20) -> Tensor:
        return self.project(targets).clamp_min(eps).log()

    def to_expected_scalar(self, logits: Tensor) -> Tensor:
        """Decode logits as E[bin center] under softmax(logits)."""
        return (logits.float().softmax(-1) * self.centers).sum(-1)

    def cross_entropy(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Per-position CE between projected targets and the head's softmax."""
        return -(self.project(targets) * logits.float().log_softmax(-1)).sum(-1)
