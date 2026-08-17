"""Optimized Muon for the RL actor and critic, with fused AdamW auxiliaries.

Ported from `../modded-nanogpt` (`train_gpt.py`, `NorMuonAndAdam`) and reduced
to this stack's shape: one device, float32 parameters, no learning-rate
schedule.

Kept from the reference implementation:

- **Polar Express** orthogonalization instead of fixed-coefficient
  Newton-Schulz. Five matmul rounds either way, but the per-round coefficient
  triples are tuned, so the update lands much closer to the true polar factor.
- **NorMuon** low-rank second-moment variance reduction, renormalized so the
  update keeps Muon's spectral scale. `muon_lr` therefore keeps its meaning.
- **Cautious weight decay** fused into the same kernel as the parameter update,
  instead of a separate pass over a snapshot of every matrix.
- Nesterov momentum accumulated in float32, orthogonalization in bfloat16 on
  CUDA.
- Hyperparameters passed as 0-D CPU tensors, so momentum warmup changes a value
  the graph reads rather than a constant it specialized on.

Dropped deliberately:

- Distributed banks, reduce-scatter and all-gather: this trainer is single
  device.
- bfloat16 mantissa tracking: these parameters are float32, so the update is
  already applied at full precision.
- Every schedule. The learning rate is constant. Momentum warmup is the one
  ramp retained, because Muon's first steps are unstable at 0.95.

Matrices of equal shape are stacked and stepped as one batch. The 30 hidden
matrices per network form three groups (20x128x128, 5x512x128, 5x128x512), so a
step is three batched kernel chains rather than thirty small ones - the small
matmuls here are launch-bound, not compute-bound.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


OPTIMIZER_FORMAT = 2
OPTIMIZER_KIND = "hybrid_normuon_adamw"

# Polar Express coefficients for num_iters=5, safety_factor=2e-2, cushion=2.
# https://arxiv.org/pdf/2505.16932
POLAR_EXPRESS_COEFFS: tuple[tuple[float, float, float], ...] = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)
SPECTRAL_SAFETY = 2e-2

# Pretraining keeps the learning rate it was tuned and checkpointed with.
PRETRAIN_MUON_LR = 1e-2
# PPO's default is RMS-matched to the AdamW trunk step it replaces, so adopting
# Muon changes update geometry rather than step size. An orthogonalized update
# to an RxC matrix has Frobenius norm sqrt(min(R, C)) before the aspect-ratio
# adjustment sqrt(max(1, R/C)), so the mean coordinate moves by roughly
# 0.06-0.09 x muon_lr for this model's 128x128, 512x128 and 128x512 matrices.
# At muon_lr=1.2e-3 that is about 1e-4, which is PPO's AdamW actor lr.
PPO_MUON_LR = 1.2e-3
MUON_WEIGHT_DECAY = 2.5e-2
MUON_MOMENTUM_MIN = 0.85
MUON_MOMENTUM_MAX = 0.95
MUON_MOMENTUM_WARMUP_STEPS = 300
MUON_BETA2 = 0.9


def split_hidden_matrices(
    model: nn.Module,
) -> tuple[list[Tensor], list[Tensor], tuple[str, ...], tuple[str, ...]]:
    """Partition a policy/value model into Muon hidden matrices and Adam auxiliaries.

    Muon is restricted to the hidden linear projections of the room encoder and
    entity trunk. Embeddings, normalization parameters, biases, policy output
    heads and the complete HL-Gauss head stay on AdamW, following Muon's
    intended domain.
    """
    modules = dict(model.named_modules())
    muon: list[Tensor] = []
    adam: list[Tensor] = []
    muon_names: list[str] = []
    adam_names: list[str] = []
    for name, parameter in model.named_parameters():
        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        module = modules.get(module_name)
        hidden_linear = (
            parameter.ndim == 2
            and isinstance(module, nn.Linear)
            and (
                name.startswith("trunk.room_enc.blocks.")
                or name.startswith("trunk.entity_blocks.")
            )
        )
        if hidden_linear:
            muon.append(parameter)
            muon_names.append(name)
        else:
            adam.append(parameter)
            adam_names.append(name)

    all_ids = [id(parameter) for parameter in (*muon, *adam)]
    expected_ids = [id(parameter) for parameter in model.parameters()]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(expected_ids):
        raise RuntimeError("optimizer parameter partition is not exhaustive")
    if not muon or not adam:
        raise RuntimeError("optimizer requires both hidden and auxiliary parameters")
    return muon, adam, tuple(muon_names), tuple(adam_names)


def polar_express(matrices: Tensor) -> Tensor:
    """Nearest-orthogonal factor of every matrix in a `[G, R, C]` batch.

    Iterates `X <- a*X + (b*A + c*A@A) @ X` with `A = X @ X.T` on the wide
    orientation, so the Gram matrix is the smaller of the two products. Tall
    batches are transposed in and back out.
    """
    if matrices.ndim != 3:
        raise ValueError("polar_express expects a [G, R, C] batch of matrices")
    transposed = matrices.size(-2) > matrices.size(-1)
    x = matrices.mT.contiguous() if transposed else matrices.contiguous()
    # Ensure the spectral norm is at most one before iterating.
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * (1.0 + SPECTRAL_SAFETY) + 1e-6)
    for a, b, c in POLAR_EXPRESS_COEFFS:
        gram = x @ x.mT
        poly = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, poly, x, beta=a)
    return x.mT if transposed else x


def normuon_group_step(
    params: Tensor,
    grads: Tensor,
    momentum_buffer: Tensor,
    second_moment: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    decay_t: Tensor,
    beta2: float,
    ortho_dtype: torch.dtype,
) -> None:
    """One NorMuon step over a stacked same-shape group, in place on `params`.

    `params` and `grads` are private stacked copies; the caller scatters
    `params` back to the live tensors. `momentum_t`, `lr_t` and `decay_t` are
    0-D CPU tensors so warmup does not invalidate a compiled graph.
    """
    momentum = momentum_t.to(grads.dtype)
    momentum_buffer.lerp_(grads, 1.0 - momentum)
    update = grads.lerp_(momentum_buffer, momentum)

    ortho = polar_express(update.to(ortho_dtype)).float()

    # NorMuon: low-rank second moment along the longer axis, then rescale so the
    # group's update norm is unchanged by the reweighting.
    reduce_dim = -1 if ortho.size(-2) >= ortho.size(-1) else -2
    axis = ortho.size(reduce_dim)
    mean_square = ortho.square().mean(dim=reduce_dim, keepdim=True)
    norm_before = mean_square.sum(dim=(-2, -1), keepdim=True).mul(axis).sqrt()
    second_moment.lerp_(mean_square, 1.0 - beta2)
    scale = second_moment.clamp_min(1e-10).rsqrt()
    norm_after = (mean_square * axis * scale.square()).sum(
        dim=(-2, -1), keepdim=True,
    ).sqrt()
    ortho = ortho * (scale * (norm_before / norm_after.clamp_min(1e-10)))

    # Cautious weight decay: decay only the coordinates whose learned step
    # already shrinks the weight. Both terms read the pre-update parameter.
    step = ortho * lr_t
    shrinking = (step * params) > 0
    decay = params * shrinking * decay_t
    params.sub_(step).sub_(decay)


class _MuonGroup:
    """Stacked buffers for one set of identically shaped hidden matrices."""

    __slots__ = (
        "params", "names", "shape", "lr_scale",
        "stacked_params", "stacked_grads", "param_views", "grad_views",
        "momentum_buffer", "second_moment",
        "lr_t", "decay_t", "momentum_t",
    )

    def __init__(self, params: list[Tensor], names: list[str]) -> None:
        reference = params[0]
        self.params = params
        self.names = names
        self.shape = tuple(reference.shape)
        rows, cols = self.shape
        # Muon's original aspect-ratio adjustment, identical to
        # `torch.optim.Muon(adjust_lr_fn="original")`.
        self.lr_scale = float(max(1.0, rows / cols) ** 0.5)
        device = reference.device
        count = len(params)
        self.stacked_params = torch.zeros(
            (count, rows, cols), dtype=torch.float32, device=device,
        )
        self.stacked_grads = torch.zeros_like(self.stacked_params)
        self.param_views = list(self.stacked_params.unbind(0))
        self.grad_views = list(self.stacked_grads.unbind(0))
        self.momentum_buffer = torch.zeros_like(self.stacked_params)
        moment_shape = (count, rows, 1) if rows >= cols else (count, 1, cols)
        self.second_moment = torch.zeros(
            moment_shape, dtype=torch.float32, device=device,
        )
        self.lr_t = torch.zeros((), dtype=torch.float32, device="cpu")
        self.decay_t = torch.zeros((), dtype=torch.float32, device="cpu")
        self.momentum_t = torch.zeros((), dtype=torch.float32, device="cpu")


class HybridMuonAdamW:
    """Muon hidden matrices plus fused-AdamW auxiliary parameters.

    Public surface matches a `torch.optim.Optimizer` closely enough for this
    trainer: `param_groups`, `zero_grad`, `step`, `state_dict`,
    `load_state_dict`. There is no learning-rate schedule; `set_learning_rates`
    exists for explicit operator overrides on resume.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        adam_lr: float,
        muon_lr: float = PRETRAIN_MUON_LR,
        muon_weight_decay: float = MUON_WEIGHT_DECAY,
        muon_momentum_min: float = MUON_MOMENTUM_MIN,
        muon_momentum_max: float = MUON_MOMENTUM_MAX,
        muon_momentum_warmup_steps: int = MUON_MOMENTUM_WARMUP_STEPS,
        muon_beta2: float = MUON_BETA2,
        compile_kernels: bool | None = None,
    ) -> None:
        if muon_lr <= 0 or adam_lr <= 0:
            raise ValueError("optimizer learning rates must be positive")
        if muon_weight_decay < 0:
            raise ValueError("Muon cautious weight decay must be non-negative")
        if not 0 <= muon_momentum_min <= muon_momentum_max < 1:
            raise ValueError("expected 0 <= minimum momentum <= maximum momentum < 1")
        if muon_momentum_warmup_steps < 0:
            raise ValueError("Muon momentum warmup must be non-negative")
        if not 0 < muon_beta2 < 1:
            raise ValueError("Muon second-moment decay must be in (0, 1)")

        muon, adam, muon_names, adam_names = split_hidden_matrices(model)
        device = next(model.parameters()).device
        self.adam = torch.optim.AdamW(
            adam,
            lr=adam_lr,
            eps=1e-5,
            weight_decay=0.0,
            fused=device.type == "cuda",
        )
        self.muon_names = muon_names
        self.adam_names = adam_names
        self.muon_lr = float(muon_lr)
        self.muon_weight_decay = float(muon_weight_decay)
        self.momentum_min = float(muon_momentum_min)
        self.momentum_max = float(muon_momentum_max)
        self.momentum_warmup_steps = int(muon_momentum_warmup_steps)
        self.beta2 = float(muon_beta2)
        self.step_count = 0

        grouped: dict[tuple[int, ...], tuple[list[Tensor], list[str]]] = {}
        for name, parameter in zip(muon_names, muon):
            if parameter.ndim != 2:
                raise ValueError(f"Muon parameter {name} must be a matrix")
            key = tuple(parameter.shape)
            params, names = grouped.setdefault(key, ([], []))
            params.append(parameter)
            names.append(name)
        self.groups = [
            _MuonGroup(params, names)
            for _, (params, names) in sorted(grouped.items())
        ]

        # bfloat16 orthogonalization is a CUDA choice: it is the dtype the
        # iteration was tuned in and halves bandwidth. CPU keeps float32
        # because CPU bfloat16 matmul is slower, not faster.
        self.ortho_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        if compile_kernels is None:
            # Inductor's compile cost exceeds anything it saves on this CPU
            # step, and CPU is only used by contract tests.
            compile_kernels = device.type == "cuda"
        self.compile_kernels = bool(compile_kernels)
        self._step_impl = (
            torch.compile(normuon_group_step, dynamic=False, fullgraph=True)
            if self.compile_kernels
            else normuon_group_step
        )

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        """Muon groups first, then AdamW's. Reported, never scheduled."""
        muon_groups = [
            {"params": group.params, "lr": self.muon_lr, "kind": "muon"}
            for group in self.groups
        ]
        return [*muon_groups, *self.adam.param_groups]

    @property
    def adam_lr(self) -> float:
        return float(self.adam.param_groups[0]["lr"])

    @property
    def momentum(self) -> float:
        """Current Nesterov momentum, ramped over the warmup window."""
        return self._momentum()

    def set_learning_rates(
        self, *, adam_lr: float | None = None, muon_lr: float | None = None,
    ) -> None:
        if adam_lr is not None:
            if adam_lr <= 0:
                raise ValueError("AdamW learning rate must be positive")
            for group in self.adam.param_groups:
                group["lr"] = float(adam_lr)
        if muon_lr is not None:
            if muon_lr <= 0:
                raise ValueError("Muon learning rate must be positive")
            self.muon_lr = float(muon_lr)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.adam.zero_grad(set_to_none=set_to_none)
        for group in self.groups:
            for parameter in group.params:
                if set_to_none:
                    parameter.grad = None
                elif parameter.grad is not None:
                    parameter.grad.zero_()

    def _momentum(self) -> float:
        if self.momentum_warmup_steps == 0:
            return self.momentum_max
        progress = min(1.0, self.step_count / self.momentum_warmup_steps)
        return self.momentum_min + progress * (self.momentum_max - self.momentum_min)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        if closure is not None:
            raise ValueError("HybridMuonAdamW does not support closures")
        momentum = self._momentum()
        for group in self.groups:
            grads = [parameter.grad for parameter in group.params]
            present = [grad is not None for grad in grads]
            if not any(present):
                continue
            if not all(present):
                missing = [
                    name for name, has in zip(group.names, present) if not has
                ]
                raise RuntimeError(
                    f"Muon group {group.shape} has partial gradients: {missing}"
                )
            group.lr_t.fill_(self.muon_lr * group.lr_scale)
            group.decay_t.fill_(self.muon_lr * self.muon_weight_decay)
            group.momentum_t.fill_(momentum)
            torch._foreach_copy_(group.param_views, group.params)
            torch._foreach_copy_(group.grad_views, grads)
            self._step_impl(
                group.stacked_params,
                group.stacked_grads,
                group.momentum_buffer,
                group.second_moment,
                group.momentum_t,
                group.lr_t,
                group.decay_t,
                self.beta2,
                self.ortho_dtype,
            )
            torch._foreach_copy_(group.params, group.param_views)
        result = self.adam.step()
        self.step_count += 1
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": OPTIMIZER_FORMAT,
            "kind": OPTIMIZER_KIND,
            "adam": self.adam.state_dict(),
            "muon": [
                {
                    "shape": list(group.shape),
                    "names": list(group.names),
                    "momentum_buffer": group.momentum_buffer,
                    "second_moment": group.second_moment,
                }
                for group in self.groups
            ],
            "step_count": self.step_count,
            "muon_names": self.muon_names,
            "adam_names": self.adam_names,
            "muon_lr": self.muon_lr,
            "muon_weight_decay": self.muon_weight_decay,
            "momentum_min": self.momentum_min,
            "momentum_max": self.momentum_max,
            "momentum_warmup_steps": self.momentum_warmup_steps,
            "beta2": self.beta2,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        expected = {
            "format": OPTIMIZER_FORMAT,
            "kind": OPTIMIZER_KIND,
            "muon_names": self.muon_names,
            "adam_names": self.adam_names,
            "muon_weight_decay": self.muon_weight_decay,
            "momentum_min": self.momentum_min,
            "momentum_max": self.momentum_max,
            "momentum_warmup_steps": self.momentum_warmup_steps,
            "beta2": self.beta2,
        }
        mismatched = {
            key: (state_dict.get(key), value)
            for key, value in expected.items()
            if state_dict.get(key) != value
        }
        if mismatched:
            raise ValueError(f"hybrid optimizer state is incompatible: {mismatched}")
        records = state_dict["muon"]
        if len(records) != len(self.groups):
            raise ValueError("hybrid optimizer state has a different Muon group layout")
        for group, record in zip(self.groups, records):
            if list(group.shape) != list(record["shape"]) or list(group.names) != list(
                record["names"],
            ):
                raise ValueError(
                    f"Muon group {group.shape} does not match stored group "
                    f"{tuple(record['shape'])}"
                )
            group.momentum_buffer.copy_(record["momentum_buffer"])
            group.second_moment.copy_(record["second_moment"])
        self.adam.load_state_dict(state_dict["adam"])
        # The stored learning rate is provenance; an explicit override wins.
        self.muon_lr = float(state_dict["muon_lr"])
        self.step_count = int(state_dict["step_count"])


def optimizer_parameter_counts(optimizer: HybridMuonAdamW) -> dict[str, int]:
    """Return auditable parameter counts for logs and artifact metadata."""
    return {
        "muon": sum(
            parameter.numel()
            for group in optimizer.groups
            for parameter in group.params
        ),
        "adamw": sum(
            parameter.numel()
            for group in optimizer.adam.param_groups
            for parameter in group["params"]
        ),
    }
