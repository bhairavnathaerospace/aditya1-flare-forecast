"""Interchangeable causal sequence encoders.

Every encoder here maps ``(B, C_in, T) -> (B, H, T)`` and is strictly causal,
so they can be swapped behind one interface and compared on identical data,
splits and losses.  Architecture claims are only meaningful under that kind of
controlled substitution -- comparing a paper's transformer to your own CNN
across different preprocessing tells you nothing.

Available: ``tcn`` (dilated convolutions), ``ssm`` (selective state space),
``gru`` (recurrent), ``transformer`` (causal self-attention), ``patchtst``
(patched causal transformer) and ``linear`` (deliberately trivial control).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .blocks import TCNEncoder, ChannelNorm1d
from .ssm import SSMEncoder


class GRUEncoder(nn.Module):
    """Unidirectional GRU.

    Included because on short sequences with few labels a plain recurrent net
    is often competitive, and a strong simple baseline is the fastest way to
    find out whether a fancier encoder is doing anything.

    Built as a stack of single-layer GRUs with ``nn.Dropout`` between them,
    rather than ``nn.GRU(num_layers=n, dropout=p)``. The two are equivalent -
    PyTorch's built-in inter-layer dropout applies dropout to each layer's
    output except the last - but the built-in form routes through cuDNN's RNN
    dropout state, and on Windows with torch 2.14 + CUDA 13 that state crashes
    the whole process at interpreter exit with STATUS_STACK_BUFFER_OVERRUN
    (0xC0000409). The crash is deterministic, reproduces with a bare
    ``nn.GRU(num_layers=2, dropout=0.1)`` and nothing else, and does not occur
    with dropout 0 or a single layer. It fires *after* all work has finished,
    so it did not corrupt results, but it made every run containing a GRU
    report failure. Keeping dropout outside cuDNN avoids it entirely.
    """

    def __init__(self, in_ch: int, hidden: int, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.GRU(in_ch if i == 0 else hidden, hidden, num_layers=1,
                   batch_first=True)
            for i in range(n_layers)
        )
        self.drop = nn.Dropout(dropout)
        self.out_norm = ChannelNorm1d(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        for i, rnn in enumerate(self.layers):
            h, _ = rnn(h)
            if i < len(self.layers) - 1:
                h = self.drop(h)
        return self.out_norm(h.transpose(1, 2))


class CausalTransformerEncoder(nn.Module):
    """Self-attention with a strict causal mask and sinusoidal positions.

    O(T^2) memory, so it is the wrong tool for hour-long 1 s sequences -- it is
    here as a reference point, not as the recommended encoder.
    """

    def __init__(self, in_ch: int, hidden: int, n_layers: int = 3,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, hidden, 1)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=2 * hidden,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        # norm_first makes the nested-tensor fast path unavailable; disabling
        # it explicitly keeps the warning out of every run log.
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers,
                                         enable_nested_tensor=False)
        self.out_norm = ChannelNorm1d(hidden)
        self.hidden = hidden

    def _pos(self, t: int, device, dtype) -> torch.Tensor:
        pos = torch.arange(t, device=device, dtype=dtype).unsqueeze(1)
        i = torch.arange(0, self.hidden, 2, device=device, dtype=dtype)
        div = torch.exp(-math.log(10000.0) * i / self.hidden)
        pe = torch.zeros(t, self.hidden, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x).transpose(1, 2)
        t = h.shape[1]
        h = h + self._pos(t, h.device, h.dtype)
        mask = torch.triu(torch.ones(t, t, device=h.device, dtype=torch.bool),
                          diagonal=1)
        h = self.enc(h, mask=mask)
        return self.out_norm(h.transpose(1, 2))


class LinearEncoder(nn.Module):
    """A pointwise projection with no temporal mixing at all.

    The control condition.  If a sequence model cannot beat this, the task is
    not actually using temporal structure and the extra capacity is decoration.
    """

    def __init__(self, in_ch: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 1),
        )
        self.out_norm = ChannelNorm1d(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_norm(self.net(x))


class PatchTSTEncoder(nn.Module):
    """Patch-and-transform encoder (PatchTST, Nie et al. 2023), made causal.

    The series is cut into non-overlapping patches of ``patch`` steps. Each
    channel's patch is embedded on its own (the channel independence that
    PatchTST is built on), a causal transformer attends over patches, and the
    per-channel patch representations are then mixed.

    **Causality.** A step may only see *completed* patches: step ``t`` is given
    the representation of patch ``t // patch - 1``, never its own, which would
    contain up to ``patch - 1`` steps of its future. Steps before the first
    completed patch get zeros. Attention over patches is causally masked too.

    Attention costs O((T/patch)^2) instead of O(T^2), which is why patching is
    worth trying at all on hour-long sequences.
    """

    def __init__(self, in_ch: int, hidden: int, patch: int = 20, n_layers: int = 3,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.patch = max(int(patch), 1)
        self.hidden = hidden
        self.embed = nn.Linear(self.patch, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=2 * hidden,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers,
                                         enable_nested_tensor=False)
        self.mix = nn.Linear(in_ch * hidden, hidden)
        self.out_norm = ChannelNorm1d(hidden)

    def _pos(self, n: int, device, dtype) -> torch.Tensor:
        pos = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
        i = torch.arange(0, self.hidden, 2, device=device, dtype=dtype)
        div = torch.exp(-math.log(10000.0) * i / self.hidden)
        pe = torch.zeros(n, self.hidden, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        p = self.patch
        n = t // p
        if n == 0:                      # sequence shorter than one patch
            return self.out_norm(torch.zeros(b, self.hidden, t, device=x.device, dtype=x.dtype))
        tok = self.embed(x[:, :, :n * p].reshape(b, c, n, p))          # (B, C, N, H)
        tok = tok + self._pos(n, x.device, tok.dtype)
        mask = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.enc(tok.reshape(b * c, n, self.hidden), mask=mask).reshape(b, c, n, self.hidden)
        z = self.mix(h.permute(0, 2, 1, 3).reshape(b, n, c * self.hidden))   # (B, N, H)
        # Every step takes the last completed patch; earlier steps get zeros.
        idx = torch.arange(t, device=x.device) // p - 1
        out = torch.zeros(b, t, self.hidden, device=x.device, dtype=z.dtype)
        keep = idx >= 0
        if keep.any():
            out[:, keep] = z[:, idx[keep].clamp(max=n - 1)]
        return self.out_norm(out.transpose(1, 2))


ENCODERS = ("tcn", "ssm", "gru", "transformer", "linear", "patchtst")


def build_encoder(kind: str, in_ch: int, hidden: int, cfg) -> nn.Module:
    """Construct one encoder by name, using the shared ModelConfig."""
    kind = kind.lower()
    dropout = getattr(cfg, "dropout", 0.1)
    if kind == "tcn":
        return TCNEncoder(in_ch, hidden, cfg.dilations, cfg.kernel_size, dropout)
    if kind == "ssm":
        return SSMEncoder(in_ch, hidden,
                          n_layers=getattr(cfg, "ssm_layers", 4),
                          d_state=getattr(cfg, "ssm_state", 16),
                          dropout=dropout)
    if kind == "gru":
        return GRUEncoder(in_ch, hidden,
                          n_layers=getattr(cfg, "rnn_layers", 2), dropout=dropout)
    if kind == "transformer":
        return CausalTransformerEncoder(
            in_ch, hidden, n_layers=getattr(cfg, "tf_layers", 3),
            n_heads=cfg.attn_heads, dropout=dropout)
    if kind == "linear":
        return LinearEncoder(in_ch, hidden, dropout)
    if kind == "patchtst":
        return PatchTSTEncoder(
            in_ch, hidden, patch=getattr(cfg, "patch_len", 20),
            n_layers=getattr(cfg, "tf_layers", 3), n_heads=cfg.attn_heads,
            dropout=dropout)
    raise ValueError(f"unknown encoder {kind!r}; choose from {ENCODERS}")
