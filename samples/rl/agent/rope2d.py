"""2D rotary position embeddings for a patch grid (ViT-style)."""
from __future__ import annotations

import torch
from torch import Tensor, nn


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPE2D(nn.Module):
    """Apply independent 1D RoPE on x and y halves of the head dimension."""

    def __init__(self, head_dim: int, max_h: int = 16, max_w: int = 16, base: float = 10000.0):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError("head_dim must be divisible by 4 for 2D RoPE")
        self.head_dim = head_dim
        self.dim_h = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim_h, 2).float() / self.dim_h))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # precompute cos/sin tables for grid
        gy = torch.arange(max_h).float()
        gx = torch.arange(max_w).float()
        freq_y = torch.einsum("i,j->ij", gy, inv_freq)  # H, dim_h/2
        freq_x = torch.einsum("i,j->ij", gx, inv_freq)
        # interleave to dim_h
        sin_y = torch.stack((freq_y.sin(), freq_y.sin()), dim=-1).flatten(-2)
        cos_y = torch.stack((freq_y.cos(), freq_y.cos()), dim=-1).flatten(-2)
        sin_x = torch.stack((freq_x.sin(), freq_x.sin()), dim=-1).flatten(-2)
        cos_x = torch.stack((freq_x.cos(), freq_x.cos()), dim=-1).flatten(-2)
        self.register_buffer("sin_y", sin_y, persistent=False)  # H, dim_h
        self.register_buffer("cos_y", cos_y, persistent=False)
        self.register_buffer("sin_x", sin_x, persistent=False)
        self.register_buffer("cos_x", cos_x, persistent=False)

    def forward(self, q: Tensor, k: Tensor, h: int, w: int) -> tuple[Tensor, Tensor]:
        """
        q,k: [B, heads, L, head_dim] with L = h*w (row-major).
        """
        # positions
        # y coords: 0,0,... w times, then 1,...
        device = q.device
        ys = torch.arange(h, device=device).repeat_interleave(w)  # L
        xs = torch.arange(w, device=device).repeat(h)

        cos_y = self.cos_y[ys]  # L, dim_h
        sin_y = self.sin_y[ys]
        cos_x = self.cos_x[xs]
        sin_x = self.sin_x[xs]

        # split head into y-half and x-half
        qy, qx = q[..., : self.dim_h], q[..., self.dim_h :]
        ky, kx = k[..., : self.dim_h], k[..., self.dim_h :]

        def apply(t: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
            # t: B,H,L,D  cos/sin: L,D
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
            return t * cos + _rotate_half(t) * sin

        qy = apply(qy, cos_y, sin_y)
        ky = apply(ky, cos_y, sin_y)
        qx = apply(qx, cos_x, sin_x)
        kx = apply(kx, cos_x, sin_x)
        return torch.cat((qy, qx), dim=-1), torch.cat((ky, kx), dim=-1)
