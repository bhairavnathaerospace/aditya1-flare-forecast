"""Building blocks: strictly causal temporal convolutions and attention pooling.

Causality is not a stylistic choice here.  A forecaster that convolves
symmetrically over its input window sees a few steps into the future at every
layer; stack nine of those and the model is quietly reading the answer.  Every
convolution below pads on the left only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    """Conv1d that pads only on the left, so output t depends on inputs <= t."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int = 1, **kw):
        super().__init__(in_ch, out_ch, kernel_size, dilation=dilation,
                         padding=0, **kw)
        self._pad = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        return super().forward(F.pad(x, (self._pad, 0)))


class ChannelNorm1d(nn.Module):
    """Normalise across channels at each time step independently.

    The obvious choice here is GroupNorm, and it is wrong: over a (B, C, T)
    tensor GroupNorm pools statistics across the **time** axis, so step t is
    rescaled using information from steps after t.  That is invisible in
    training curves and quietly destroys the causality guarantee.  Normalising
    per time step keeps the property and costs nothing.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        xn = (x - mu) * torch.rsqrt(var + self.eps)
        return xn * self.weight[None, :, None] + self.bias[None, :, None]


class SqueezeExcite(nn.Module):
    """Channel gating from a causal running summary of the sequence.

    Uses a cumulative mean rather than a global average so that the gate at
    step t is a function of steps <= t only.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Conv1d(channels, hidden, 1)
        self.fc2 = nn.Conv1d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[-1]
        denom = torch.arange(1, t + 1, device=x.device, dtype=x.dtype)
        ctx = x.cumsum(dim=-1) / denom
        g = torch.sigmoid(self.fc2(F.gelu(self.fc1(ctx))))
        return x * g


class TCNBlock(nn.Module):
    """Residual dilated causal block: depthwise conv -> pointwise -> SE."""

    def __init__(self, channels: int, kernel_size: int, dilation: int,
                 dropout: float):
        super().__init__()
        self.dw = CausalConv1d(channels, channels, kernel_size,
                               dilation=dilation, groups=channels)
        self.pw1 = nn.Conv1d(channels, 2 * channels, 1)
        self.pw2 = nn.Conv1d(2 * channels, channels, 1)
        self.norm = ChannelNorm1d(channels)
        self.se = SqueezeExcite(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.dw(h)
        h = self.pw2(F.gelu(self.pw1(h)))
        h = self.se(h)
        return x + self.drop(h)


class TCNEncoder(nn.Module):
    """Stack of dilated causal blocks with an input projection."""

    def __init__(self, in_ch: int, hidden: int, dilations, kernel_size: int,
                 dropout: float):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, hidden, 1)
        self.blocks = nn.ModuleList(
            [TCNBlock(hidden, kernel_size, d, dropout) for d in dilations]
        )
        self.out_norm = ChannelNorm1d(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, H, T)
        h = self.proj(x)
        for b in self.blocks:
            h = b(h)
        return self.out_norm(h)


class CausalAttentionPool(nn.Module):
    """Attention pooling over the window, queried by the final time step.

    Flares are localised events; a plain mean over two hours dilutes the onset
    that matters.  Letting the last step choose what to attend to recovers it,
    and because the query is the last step the result stays causal.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, key_padding_mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        # h: (B, T, D)
        q = h[:, -1:, :]
        if key_padding_mask is not None:
            # Never let a fully-masked row produce NaN attention weights.
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, -1] = False
        out, _ = self.attn(q, h, h, key_padding_mask=key_padding_mask,
                           need_weights=False)
        return self.norm(out.squeeze(1) + q.squeeze(1))
