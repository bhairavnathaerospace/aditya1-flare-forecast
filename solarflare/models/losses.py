"""Losses for the multi-task flare model.

All of them are mask-aware.  Masked targets are common here (a horizon that
runs off the end of an observation, a soft target on a hard-only day), and
quietly training on a zero-filled target teaches the model to predict zeros.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum()
    if denom <= 0:
        return x.sum() * 0.0
    return (x * mask).sum() / denom


def focal_bce(logits: torch.Tensor, target: torch.Tensor,
              mask: torch.Tensor | None = None, gamma: float = 2.0,
              alpha: float | None = None) -> torch.Tensor:
    """Binary focal loss.

    Flare-occurrence labels are heavily imbalanced and the easy negatives
    (long quiet stretches) dominate plain BCE; the focusing term keeps the
    gradient on the boundary cases that actually decide a forecast.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    loss = bce * (1 - p_t).clamp(min=1e-6) ** gamma
    if alpha is not None:
        a_t = alpha * target + (1 - alpha) * (1 - target)
        loss = loss * a_t
    if mask is None:
        return loss.mean()
    return _masked_mean(loss, mask)


def focal_ce(logits: torch.Tensor, target: torch.Tensor,
             weight: torch.Tensor | None = None,
             gamma: float = 2.0,
             mask: torch.Tensor | None = None) -> torch.Tensor:
    """Multi-class focal loss for the flare-phase head.

    ``mask`` matters more than it looks. Phase labels come from the soft X-ray
    curve; a window with no SoLEXS data has phase 0 *by construction*, not
    because the Sun was quiet. Unmasked, every HEL1OS-only stretch would teach
    "hard X-ray data without soft means quiet Sun" -- invisible on a sample
    whose one HEL1OS day happened to be quiet, systematically wrong on a real
    archive.
    """
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    tgt = target.unsqueeze(-1)
    logp_t = logp.gather(-1, tgt).squeeze(-1)
    p_t = p.gather(-1, tgt).squeeze(-1)
    loss = -((1 - p_t).clamp(min=1e-6) ** gamma) * logp_t
    if weight is not None:
        loss = loss * weight[target]
    if mask is None:
        return loss.mean()
    return _masked_mean(loss, mask)


def pinball(pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor,
            mask: torch.Tensor | None = None) -> torch.Tensor:
    """Quantile (pinball) loss.

    Point forecasts of flare flux are close to useless operationally -- what a
    space-weather user needs is a band.  Predicting q10/q50/q90 gives one, and
    the loss is what calibrates it.
    """
    # pred: (B, H, Q); target: (B, H)
    t = target.unsqueeze(-1)
    err = t - pred
    q = quantiles.view(1, 1, -1)
    loss = torch.maximum(q * err, (q - 1) * err)
    if mask is None:
        return loss.mean()
    m = mask.unsqueeze(-1).expand_as(loss)
    return _masked_mean(loss, m)


def masked_huber(pred: torch.Tensor, target: torch.Tensor,
                 mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    loss = F.huber_loss(pred, target, reduction="none", delta=delta)
    return _masked_mean(loss, mask)


class MultiTaskLoss(nn.Module):
    """Weighted sum of the five task losses with learned uncertainty.

    Each term is scaled by exp(-s_i) and penalised by s_i, so a task the model
    genuinely cannot fit (for instance the soft-flux heads on a hard-only
    segment) is down-weighted automatically instead of dragging the trunk.
    """

    def __init__(self, quantiles, focal_gamma: float = 2.0,
                 w_phase: float = 1.0, w_occurrence: float = 1.0,
                 w_nowcast: float = 1.0, w_forecast: float = 1.0,
                 phase_weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("quantiles", torch.as_tensor(list(quantiles),
                                                          dtype=torch.float32))
        self.gamma = focal_gamma
        self.w = dict(phase=w_phase, occurrence=w_occurrence,
                      nowcast=w_nowcast, forecast=w_forecast)
        if phase_weight is not None:
            self.register_buffer("phase_weight", phase_weight)
        else:
            self.phase_weight = None

    def forward(self, out: dict[str, torch.Tensor],
                batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        lv = out["log_var"]

        l_phase = focal_ce(out["phase"], batch["phase"],
                           weight=self.phase_weight, gamma=self.gamma,
                           mask=batch["nowcast_mask"])
        l_inflare = focal_bce(out["in_flare"], batch["in_flare"],
                              mask=batch["nowcast_mask"], gamma=self.gamma)
        l_now = masked_huber(out["nowcast"], batch["nowcast"],
                             batch["nowcast_mask"])
        l_fore = pinball(out["forecast"], batch["forecast"], self.quantiles,
                         mask=batch["forecast_mask"])
        l_occ = focal_bce(out["occurrence"], batch["occurrence"],
                          mask=batch["occurrence_mask"], gamma=self.gamma)
        l_peak = masked_huber(out["peak"], batch["peak"], batch["peak_mask"])

        terms = [
            self.w["phase"] * (l_phase + l_inflare),
            self.w["nowcast"] * l_now,
            self.w["forecast"] * l_fore,
            self.w["occurrence"] * l_occ,
            self.w["nowcast"] * l_peak,
        ]
        total = 0.0
        for i, t in enumerate(terms):
            total = total + torch.exp(-lv[i]) * t + 0.5 * lv[i]

        parts = {
            "phase": float(l_phase.detach()),
            "in_flare": float(l_inflare.detach()),
            "nowcast": float(l_now.detach()),
            "forecast": float(l_fore.detach()),
            "occurrence": float(l_occ.detach()),
            "peak": float(l_peak.detach()),
        }
        return total, parts
