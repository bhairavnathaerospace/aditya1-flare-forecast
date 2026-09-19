"""SoLEXHEL-Net: dual-instrument causal network for flare nowcasting and
short-horizon forecasting.

    soft (SoLEXS, 24 ch) --> TCN encoder --\
                                            >-- gated cross-modal fusion --> TCN trunk --> heads
    hard (HEL1OS, 48 ch) --> TCN encoder --/     (12 per detector: CZT1, CZT2, CdTe1, CdTe2)

The fusion gate is conditioned on each modality's *observation mask*, and whole
modalities are randomly dropped during training.  That is what makes one network
usable on soft-only days, hard-only days, and (when they exist) genuinely
simultaneous ones -- instead of needing three separate models.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig, WindowConfig
from .blocks import TCNEncoder, CausalAttentionPool


@dataclass
class FluxAnchor:
    """How to read "calibrated SoLEXS flux now" back out of the normalised input.

    ``index`` is the ``log_goes_long`` column, ``mean``/``std`` undo the
    normaliser, and ``slope``/``intercept`` are the SoLEXS-rate -> log10 GOES
    flux calibration fitted on training data. ``default`` is used when SoLEXS
    was not observing at the forecast origin.
    """

    index: int
    mean: float
    std: float
    slope: float
    intercept: float
    default: float


def flux_anchor(model_cfg, pre_cfg, mean_soft, std_soft) -> FluxAnchor | None:
    """Build the anchor from a fitted normaliser, or None when switched off."""
    if not getattr(model_cfg, "anchor_flux", False):
        return None
    from ..preprocess.features import solexs_feature_names

    i = solexs_feature_names(pre_cfg).index("log_goes_long")
    return FluxAnchor(index=i, mean=float(mean_soft[i]), std=float(std_soft[i]),
                      slope=float(pre_cfg.flux_anchor_slope),
                      intercept=float(pre_cfg.flux_anchor_intercept),
                      default=float(pre_cfg.flux_anchor_default))


@dataclass
class HeadSpec:
    n_phase: int
    n_horizons: int
    n_quantiles: int
    n_occurrence: int


class ModalityEncoder(nn.Module):
    """Encoder for one instrument, with its validity mask as an input channel.

    Missing steps are zero-filled; the mask channel tells the network which
    zeros are real measurements and which are absences, so it never has to
    guess whether a quiet stretch means a quiet Sun or a dead link.
    """

    def __init__(self, n_features: int, cfg: ModelConfig):
        super().__init__()
        self.enc = TCNEncoder(
            in_ch=n_features + 1,  # +1 for the mask
            hidden=cfg.hidden,
            dilations=cfg.dilations,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F), mask: (B, T)
        x = x * mask.unsqueeze(-1)
        h = torch.cat([x, mask.unsqueeze(-1)], dim=-1).transpose(1, 2)
        return self.enc(h).transpose(1, 2)  # (B, T, H)


class GatedFusion(nn.Module):
    """Learned, mask-aware mixing of the two modality streams.

    The gate sees both representations and both availability fractions, so when
    HEL1OS is absent it can route the trunk entirely to the soft stream rather
    than averaging in a zero tensor.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * hidden),
        )
        self.cross = nn.Linear(2 * hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, hs: torch.Tensor, hh: torch.Tensor,
                ms: torch.Tensor, mh: torch.Tensor) -> torch.Tensor:
        # hs/hh: (B, T, H); ms/mh: (B, T)
        both = torch.cat([hs, hh], dim=-1)
        avail = torch.stack([ms, mh], dim=-1)
        g = torch.sigmoid(self.gate(torch.cat([both, avail], dim=-1)))
        gated = both * g
        return self.norm(self.cross(gated))


class SolexHelNet(nn.Module):
    def __init__(self, n_soft: int, n_hard: int, n_clock: int,
                 cfg: ModelConfig, win: WindowConfig,
                 anchor: FluxAnchor | None = None):
        super().__init__()
        self.cfg = cfg
        # Optional structural prior: predict flux as "what SoLEXS reads now,
        # calibrated" plus a learned change. It adds no parameters, so a
        # checkpoint trained with it loads into a model built without it --
        # which would silently drop the anchor, so load_model restores it.
        self.anchor = anchor
        self.soft_enc = ModalityEncoder(n_soft, cfg)
        self.hard_enc = ModalityEncoder(n_hard, cfg)
        self.fusion = GatedFusion(cfg.hidden)

        # Gated off by default -- see ModelConfig.use_clock for why.
        self.use_clock = bool(getattr(cfg, "use_clock", False))
        self.clock_proj = nn.Linear(n_clock, cfg.hidden) if self.use_clock else None
        self.trunk = TCNEncoder(
            in_ch=cfg.hidden, hidden=cfg.hidden,
            dilations=cfg.dilations[: max(len(cfg.dilations) // 2, 1)],
            kernel_size=cfg.kernel_size, dropout=cfg.dropout,
        )
        self.pool = CausalAttentionPool(cfg.hidden, cfg.attn_heads)

        self.spec = HeadSpec(
            n_phase=cfg.n_phase_classes,
            n_horizons=len(win.forecast_horizons_s),
            n_quantiles=len(win.quantiles),
            n_occurrence=len(win.occurrence_horizons_s),
        )
        d = 2 * cfg.hidden  # pooled context + last-step features

        def head(out_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(d, cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden, out_dim),
            )

        self.head_phase = head(self.spec.n_phase)
        self.head_inflare = head(1)
        self.head_nowcast = head(1)
        self.head_forecast = head(self.spec.n_horizons * self.spec.n_quantiles)
        self.head_occurrence = head(self.spec.n_occurrence)
        # Peak magnitude and time-to-peak of the ongoing / imminent event.
        self.head_peak = head(2)

        # Learned per-task log-variance (Kendall & Gal homoscedastic weighting):
        # with five heads on very different scales, hand-tuned weights are
        # guesswork, and this lets the optimiser balance them.
        self.log_var = nn.Parameter(torch.zeros(5))

    def forward(self, soft: torch.Tensor, soft_mask: torch.Tensor,
                hard: torch.Tensor, hard_mask: torch.Tensor,
                clock: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training and self.cfg.modality_dropout > 0:
            b = soft.shape[0]
            dev = soft.device
            p = self.cfg.modality_dropout
            drop_s = (torch.rand(b, device=dev) < p).float().unsqueeze(1)
            drop_h = (torch.rand(b, device=dev) < p).float().unsqueeze(1)
            # Never drop both: an empty input carries no supervision signal.
            both = ((drop_s > 0) & (drop_h > 0)).squeeze(1)
            drop_h[both] = 0.0
            soft_mask = soft_mask * (1 - drop_s)
            hard_mask = hard_mask * (1 - drop_h)

        hs = self.soft_enc(soft, soft_mask)
        hh = self.hard_enc(hard, hard_mask)
        fused = self.fusion(hs, hh, soft_mask, hard_mask)
        if self.clock_proj is not None:
            fused = fused + self.clock_proj(clock)

        h = self.trunk(fused.transpose(1, 2)).transpose(1, 2)

        any_mask = torch.clamp(soft_mask + hard_mask, 0, 1)
        pad = any_mask <= 0
        ctx = self.pool(h, key_padding_mask=pad)
        last = h[:, -1, :]
        z = torch.cat([ctx, last], dim=-1)

        # Calibrated SoLEXS flux at the forecast origin, or the training mean
        # where SoLEXS was not observing there.
        anchor = None
        if self.anchor is not None:
            a = self.anchor
            rate = torch.expm1(torch.clamp(soft[:, -1, a.index] * a.std + a.mean, max=30.0))
            val = a.intercept + a.slope * torch.log10(torch.clamp(rate, min=1e-3))
            anchor = torch.where(soft_mask[:, -1] > 0, val,
                                 torch.full_like(val, a.default))

        nq, nh = self.spec.n_quantiles, self.spec.n_horizons
        fore = self.head_forecast(z).view(-1, nh, nq)
        # Enforce monotone quantiles: q10 <= q50 <= q90 by construction, which
        # removes the crossing artefacts that make interval forecasts unusable.
        base = fore[..., :1]
        deltas = F.softplus(fore[..., 1:])
        fore = torch.cat([base, base + torch.cumsum(deltas, dim=-1)], dim=-1)

        nowcast = self.head_nowcast(z).squeeze(-1)
        if anchor is not None:
            nowcast = nowcast + anchor
            fore = fore + anchor[:, None, None]

        return {
            "phase": self.head_phase(z),
            "in_flare": self.head_inflare(z).squeeze(-1),
            "nowcast": nowcast,
            "forecast": fore,
            "occurrence": self.head_occurrence(z),
            "peak": self.head_peak(z),
            "log_var": self.log_var,
        }

    @torch.no_grad()
    def receptive_field(self) -> int:
        k = self.cfg.kernel_size
        rf = 1
        for d in self.cfg.dilations:
            rf += (k - 1) * d
        for d in self.cfg.dilations[: max(len(self.cfg.dilations) // 2, 1)]:
            rf += (k - 1) * d
        return rf


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
