"""Torch Dataset wrapper around the windowed segments."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config
from .preprocess.dataset import Segment, WindowIndex, Normalizer, build_targets


def shared_arrays(segments: list[Segment], cfg: Config, norm: Normalizer) -> dict:
    """Normalised inputs and targets for every segment, computed once.

    Every split's dataset reads from the same arrays. Built per dataset instead,
    an archive run duplicates them for train/val/test and again for each
    evaluation pass -- roughly 1 GB per copy at ~1000 days, ~10 GB at peak.
    """
    return {"targets": [build_targets(s, cfg) for s in segments],
            **normalized_inputs(segments, norm)}


def normalized_inputs(segments: list[Segment], norm: Normalizer) -> dict:
    """Normalised soft/hard input arrays per segment (no targets)."""
    def norm_f32(x, fn):
        return np.ascontiguousarray(
            np.nan_to_num(fn(x), nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)

    return {
        "soft": [norm_f32(s.soft, norm.apply_soft) for s in segments],
        "hard": [norm_f32(s.hard, norm.apply_hard) for s in segments],
    }


class FlareWindows(Dataset):
    """Materialises one window on demand.

    Segments stay as whole arrays in memory and windows are sliced per item, so
    overlapping windows are free rather than copying the same hours thousands of
    times. Normalisation happens once per segment (see ``shared_arrays``), not
    once per window: windows overlap by ~99%, and per-item normalisation turns
    the data loader, not the GPU, into the bottleneck.
    """

    def __init__(self, segments: list[Segment], windows: list[WindowIndex],
                 indices: list[int], cfg: Config, norm: Normalizer,
                 shared: dict | None = None):
        self.segments = segments
        self.windows = windows
        self.indices = list(indices)
        self.cfg = cfg
        self.norm = norm
        self.L = cfg.steps_per_window
        shared = shared if shared is not None else shared_arrays(segments, cfg, norm)
        self._targets = shared["targets"]
        self._soft = shared["soft"]
        self._hard = shared["hard"]

    def __len__(self) -> int:
        return len(self.indices)

    def label_stats(self) -> dict:
        """Class balance over this split, for weighting and for the report."""
        ph, inf, occ = [], [], []
        for i in self.indices:
            w = self.windows[i]
            t = self._targets[w.seg]
            j = w.end - 1
            # Windows without soft X-ray truth carry placeholder labels (phase 0,
            # no flare). Counting them would inflate "quiet" and skew the class
            # weights derived from these statistics.
            if t["nowcast_mask"][j] <= 0:
                continue
            ph.append(t["phase"][j])
            inf.append(t["in_flare"][j])
            occ.append(t["occurrence"][j])
        # int64 explicitly: an empty list would otherwise become float64, and
        # bincount refuses floats -- e.g. a split containing only HEL1OS time.
        ph = np.asarray(ph, dtype=np.int64)
        return {
            "n": len(ph),
            "phase_counts": np.bincount(ph, minlength=4).tolist(),
            "in_flare_rate": float(np.mean(inf)) if inf else 0.0,
            "occurrence_rate": np.mean(np.stack(occ), axis=0).tolist() if occ else [],
        }

    def __getitem__(self, k: int) -> dict[str, torch.Tensor]:
        w = self.windows[self.indices[k]]
        seg = self.segments[w.seg]
        tgt = self._targets[w.seg]
        sl = slice(w.end - self.L, w.end)
        j = w.end - 1  # prediction origin

        return {
            "soft": torch.from_numpy(self._soft[w.seg][sl].copy()),
            "soft_mask": torch.from_numpy(seg.soft_mask[sl].astype(np.float32)),
            "hard": torch.from_numpy(self._hard[w.seg][sl].copy()),
            "hard_mask": torch.from_numpy(seg.hard_mask[sl].astype(np.float32)),
            "clock": torch.from_numpy(seg.clock[sl].astype(np.float32)),
            "phase": torch.tensor(int(tgt["phase"][j])),
            "in_flare": torch.tensor(float(tgt["in_flare"][j])),
            "nowcast": torch.tensor(float(tgt["nowcast"][j])),
            "nowcast_mask": torch.tensor(float(tgt["nowcast_mask"][j])),
            "forecast": torch.from_numpy(tgt["forecast"][j].astype(np.float32)),
            "forecast_mask": torch.from_numpy(tgt["forecast_mask"][j].astype(np.float32)),
            "occurrence": torch.from_numpy(tgt["occurrence"][j].astype(np.float32)),
            "occurrence_mask": torch.from_numpy(tgt["occurrence_mask"][j].astype(np.float32)),
            "peak": torch.from_numpy(tgt["peak"][j].astype(np.float32)),
            "peak_mask": torch.from_numpy(tgt["peak_mask"][j].astype(np.float32)),
            "t_unix": torch.tensor(float(w.t_unix)),
            # Persistence baseline: the current value, carried forward.
            "persistence": torch.tensor(float(tgt["nowcast"][j])),
        }
