"""Streaming inference: run the trained model along an observation.

Written the way an operational nowcaster would be: at every step the model sees
only the trailing window, so this same routine works on a live telemetry buffer
without modification.
"""

from __future__ import annotations

import csv
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .pipeline import prepare, resolve_device
from .preprocess.labels import PHASE_NAMES
from .train import load_model


@torch.no_grad()
def run_inference(cfg: Config, ckpt: Path, output: Path | None = None,
                  limit: int | None = None) -> list[dict]:
    device = resolve_device(cfg.train.device)
    prep = prepare(cfg, verbose=False)
    model = load_model(ckpt, cfg, device)
    L = cfg.steps_per_window
    norm = prep.norm

    if limit is None and prep.archive:
        # Streaming inference steps one window at a time, the way a live feed
        # would. Over an archive that is millions of forward passes; default to
        # the first day of each segment and say so, rather than silently
        # running for hours.
        limit = int(86400 / cfg.pre.dt_seconds)
        print(f"archive dataset: predicting the first {limit} steps (1 day) of each "
              f"segment; pass --limit to change")

    rows: list[dict] = []
    for seg in prep.segments:
        t = len(seg)
        if t < L:
            continue
        ends = range(L, t + 1)
        if limit:
            ends = list(ends)[:limit]
        for end in ends:
            sl = slice(end - L, end)
            soft = np.nan_to_num(norm.apply_soft(seg.soft[sl]), nan=0.0,
                                 posinf=0.0, neginf=0.0)
            hard = np.nan_to_num(norm.apply_hard(seg.hard[sl]), nan=0.0,
                                 posinf=0.0, neginf=0.0)
            out = model(
                torch.from_numpy(soft[None].astype(np.float32)).to(device),
                torch.from_numpy(seg.soft_mask[sl][None].astype(np.float32)).to(device),
                torch.from_numpy(hard[None].astype(np.float32)).to(device),
                torch.from_numpy(seg.hard_mask[sl][None].astype(np.float32)).to(device),
                torch.from_numpy(seg.clock[sl][None].astype(np.float32)).to(device),
            )
            ts = float(seg.time_unix[end - 1])
            phase = int(out["phase"][0].argmax())
            # The peak head is trained only on in-flare bins (masked elsewhere),
            # so its output during quiet Sun is undefined, not a prediction.
            # Emitting a number there would invite a consumer to act on noise.
            in_flare = phase > 0
            row = {
                "segment": seg.name,
                "time_utc": datetime.fromtimestamp(ts, UTC).isoformat(),
                "p_flare_now": float(torch.sigmoid(out["in_flare"])[0]),
                "phase": phase,
                "phase_name": PHASE_NAMES[phase],
                "nowcast_log_rate": float(out["nowcast"][0]),
                "pred_time_to_peak_min": (float(out["peak"][0, 0]) * 60.0
                                          if in_flare else ""),
                "pred_log_peak_rate": (float(out["peak"][0, 1])
                                       if in_flare else ""),
                "soft_available": float(seg.soft_mask[end - 1]),
                "hard_available": float(seg.hard_mask[end - 1]),
            }
            for i, hs in enumerate(cfg.win.occurrence_horizons_s):
                row[f"p_flare_within_{int(hs/60)}min"] = float(
                    torch.sigmoid(out["occurrence"])[0, i])
            for i, hs in enumerate(cfg.win.forecast_horizons_s):
                for j, q in enumerate(cfg.win.quantiles):
                    row[f"fcst_{int(hs/60)}min_q{int(q*100)}"] = float(
                        out["forecast"][0, i, j])
            rows.append(row)

    if output and rows:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} predictions to {output}")
    return rows
