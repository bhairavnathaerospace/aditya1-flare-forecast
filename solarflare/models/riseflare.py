"""RiseNet: predict a flare's outcome from its early rise.

Deliberately small and deliberately multi-head.  The four targets -- peak
magnitude, time to peak, threshold exceedance, long-duration character -- are
different views of the same underlying question (how much energy is being
released, and how fast), so sharing an encoder between them is a real
regulariser when events are scarce.

The encoder is pluggable (`--encoder tcn|ssm|gru|transformer|linear`) so the
architecture question can be settled by measurement on identical folds rather
than by preference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .blocks import CausalAttentionPool
from .zoo import build_encoder


class RiseNet(nn.Module):
    """(soft, hard, masks) -> peak magnitude / timing / class predictions."""

    def __init__(self, n_soft: int, n_hard: int, cfg: ModelConfig,
                 encoder: str = "tcn"):
        super().__init__()
        self.cfg = cfg
        self.encoder_kind = encoder
        hidden = cfg.hidden

        # +1 on each: the validity mask rides alongside its own modality.
        self.soft_enc = build_encoder(encoder, n_soft + 1, hidden, cfg)
        self.hard_enc = build_encoder(encoder, n_hard + 1, hidden, cfg)

        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden + 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)
        self.pool = CausalAttentionPool(hidden, cfg.attn_heads)

        d = 2 * hidden + 1  # pooled + last step + elapsed lead time

        def head(out: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(d, hidden), nn.GELU(), nn.Dropout(cfg.dropout),
                nn.Linear(hidden, out),
            )

        # Peak magnitude is predicted as three quantiles, not a point: the
        # operational question is "how bad could this get", which is an upper
        # quantile, not a mean.
        self.head_peak_q = head(3)
        self.head_ttp = head(1)
        self.head_exceed = head(1)
        self.head_lde = head(1)
        self.log_var = nn.Parameter(torch.zeros(4))

    def forward(self, soft: torch.Tensor, soft_mask: torch.Tensor,
                hard: torch.Tensor, hard_mask: torch.Tensor,
                lead: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training and self.cfg.modality_dropout > 0:
            b = soft.shape[0]
            p = self.cfg.modality_dropout
            keep_s = (torch.rand(b, device=soft.device) >= p).float().unsqueeze(1)
            keep_h = (torch.rand(b, device=soft.device) >= p).float().unsqueeze(1)
            both_gone = (keep_s.squeeze(1) == 0) & (keep_h.squeeze(1) == 0)
            keep_s[both_gone] = 1.0
            soft_mask = soft_mask * keep_s
            hard_mask = hard_mask * keep_h

        hs = self._encode(self.soft_enc, soft, soft_mask)
        hh = self._encode(self.hard_enc, hard, hard_mask)

        avail = torch.stack([soft_mask, hard_mask], dim=-1)
        h = self.norm(self.fuse(torch.cat([hs, hh, avail], dim=-1)))

        pad = torch.clamp(soft_mask + hard_mask, 0, 1) <= 0
        ctx = self.pool(h, key_padding_mask=pad)
        z = torch.cat([ctx, h[:, -1, :], lead.unsqueeze(-1)], dim=-1)

        q = self.head_peak_q(z)
        base = q[:, :1]
        deltas = torch.nn.functional.softplus(q[:, 1:])
        peak_q = torch.cat([base, base + torch.cumsum(deltas, dim=-1)], dim=-1)

        return {
            "peak_q": peak_q,                       # (B, 3) monotone quantiles
            "time_to_peak": self.head_ttp(z).squeeze(-1),
            "exceeds": self.head_exceed(z).squeeze(-1),
            "lde": self.head_lde(z).squeeze(-1),
            "log_var": self.log_var,
        }

    @staticmethod
    def _encode(enc: nn.Module, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x * mask.unsqueeze(-1)
        inp = torch.cat([x, mask.unsqueeze(-1)], dim=-1).transpose(1, 2)
        return enc(inp).transpose(1, 2)


class RiseLoss(nn.Module):
    """Pinball on peak magnitude, Huber on timing, focal BCE on the classes."""

    def __init__(self, quantiles=(0.1, 0.5, 0.9), focal_gamma: float = 2.0):
        super().__init__()
        self.register_buffer("q", torch.tensor(list(quantiles), dtype=torch.float32))
        self.gamma = focal_gamma

    def forward(self, out: dict[str, torch.Tensor],
                batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        from .losses import focal_bce

        err = batch["y_log_peak"].unsqueeze(-1) - out["peak_q"]
        l_peak = torch.maximum(self.q * err, (self.q - 1) * err).mean()

        l_ttp = nn.functional.huber_loss(
            out["time_to_peak"], batch["y_time_to_peak"], delta=0.5)
        l_exc = focal_bce(out["exceeds"], batch["y_exceeds"], gamma=self.gamma)
        l_lde = focal_bce(out["lde"], batch["y_lde"], gamma=self.gamma)

        lv = out["log_var"]
        terms = [l_peak, l_ttp, l_exc, l_lde]
        total = sum(torch.exp(-lv[i]) * t + 0.5 * lv[i] for i, t in enumerate(terms))

        return total, {
            "peak": float(l_peak.detach()),
            "ttp": float(l_ttp.detach()),
            "exceeds": float(l_exc.detach()),
            "lde": float(l_lde.detach()),
        }
