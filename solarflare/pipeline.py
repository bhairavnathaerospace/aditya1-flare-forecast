"""End-to-end assembly: mission products -> cache -> timelines -> windows -> loaders.

Kept separate from train.py so that evaluation, prediction and any notebook can
rebuild the exact same dataset without duplicating the wiring.

Every run goes through the per-file cache (preprocess/cache.py), whether the
data root holds two days or a thousand. One code path means the small sample
the tests use exercises exactly what the archive run does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from .config import Config
from .preprocess.cache import build_cache, index_sources, load_cached, Source
from .preprocess.dataset import (
    build_segments_from_raw, enumerate_windows, chronological_split,
    fit_normalizer, resolve_split_mode, span_days, thin_quiet_training_windows,
    Segment, WindowIndex, Normalizer,
)
from .torch_data import FlareWindows, shared_arrays


@dataclass
class Prepared:
    sources: list[Source]
    cache_entries: list[dict]
    segments: list[Segment]
    windows: list[WindowIndex]
    splits: dict[str, list[int]]
    norm: Normalizer
    meta: dict
    archive: bool = False
    data_root: Path = field(default_factory=lambda: Path("."))
    #: Normalised inputs + targets, built on first use and shared by every
    #: loader made from this dataset (see torch_data.shared_arrays).
    _shared: dict | None = field(default=None, repr=False)

    @property
    def n_soft(self) -> int:
        return self.segments[0].soft.shape[1]

    @property
    def n_hard(self) -> int:
        return self.segments[0].hard.shape[1]

    @property
    def n_clock(self) -> int:
        return self.segments[0].clock.shape[1]

    def load_inventory(self):
        """Raw observations (full spectra) -- only sensible for small samples."""
        from .io.discover import discover
        return discover(self.data_root)


def cache_dir_for(cfg: Config) -> Path:
    return Path(getattr(cfg, "cache_dir", None) or Path(cfg.out_dir) / "cache")


def prepare(cfg: Config, verbose: bool = True) -> Prepared:
    sources = index_sources([Path(cfg.data_root)])
    if not sources:
        raise RuntimeError(f"No SoLEXS or HEL1OS products found under {cfg.data_root}")

    # Spawning worker processes costs more than it saves on a handful of files.
    workers = cfg.pre.cache_workers if len(sources) >= 8 else 1
    entries = build_cache(sources, cfg.pre, cache_dir_for(cfg), workers=workers,
                          verbose=verbose)
    soft_raw = load_cached(entries, cache_dir_for(cfg), "solexs")
    hard_raw = load_cached(entries, cache_dir_for(cfg), "hel1os")
    if not soft_raw and not hard_raw:
        raise RuntimeError("Every source failed or was empty; see the cache manifest.")

    segments, meta = build_segments_from_raw(soft_raw, hard_raw, cfg)

    days = span_days(segments, cfg.pre.dt_seconds)
    archive = days >= cfg.win.large_data_days
    profile = {}
    if archive:
        # Applied to cfg itself so training, evaluation and the saved config all
        # see the values actually used.
        if cfg.win.stride_seconds < cfg.win.large_stride_seconds:
            profile["stride_seconds"] = (cfg.win.stride_seconds, cfg.win.large_stride_seconds)
            cfg.win.stride_seconds = cfg.win.large_stride_seconds
        if not cfg.train.max_batches_per_epoch:
            profile["max_batches_per_epoch"] = (0, cfg.train.large_max_batches_per_epoch)
            cfg.train.max_batches_per_epoch = cfg.train.large_max_batches_per_epoch

    windows = enumerate_windows(segments, cfg)
    if not windows:
        raise RuntimeError(
            "No usable windows. The observations are shorter than "
            f"input_seconds ({cfg.win.input_seconds:.0f} s) plus the longest "
            "horizon. Shorten the window or add more data."
        )

    mode = resolve_split_mode(segments, cfg)
    tr, va, te = chronological_split(windows, cfg, mode=mode)
    n_train_all = len(tr)
    if archive:
        tr = thin_quiet_training_windows(segments, windows, tr, cfg)
    norm = fit_normalizer(segments, windows, tr, cfg)

    meta = dict(meta)
    n_events = sum(len(s.events) for s in segments)
    meta.update({
        "observed_days": days,
        "archive_mode": archive,
        "archive_profile": {k: list(v) for k, v in profile.items()},
        "split_mode": mode,
        "n_sources": len(sources),
        "cache": {
            "ok": sum(e.get("status") == "ok" for e in entries),
            "empty": sum(e.get("status") == "empty" for e in entries),
            "failed": [
                {"path": e["source"]["path"], "error": e.get("error")}
                for e in entries if e.get("status") == "failed"
            ],
        },
        "n_windows": len(windows),
        "n_train": len(tr), "n_train_before_thinning": n_train_all,
        "n_val": len(va), "n_test": len(te),
        "n_events": n_events,
        "segments": [
            {
                "name": s.name,
                "steps": len(s),
                "soft_observed": float(s.soft_mask.mean()),
                "hard_observed": float(s.hard_mask.mean()),
                "n_events": len(s.events),
            }
            for s in segments
        ],
        "events": [
            {**e.as_dict(), "segment": s.name}
            for s in segments for e in s.events
        ],
    })
    if mode == "global" and tr and te:
        meta["split_dates"] = {
            "train_end": float(windows[tr[-1]].t_unix) if tr else None,
            "test_start": float(min(windows[i].t_unix for i in te)),
        }

    if verbose:
        _print_summary(segments, windows, tr, va, te, n_train_all, days, archive,
                       mode, profile, n_events, cfg)

    return Prepared(sources, entries, segments, windows,
                    {"train": tr, "val": va, "test": te}, norm, meta,
                    archive=archive, data_root=Path(cfg.data_root))


def _print_summary(segments, windows, tr, va, te, n_train_all, days, archive,
                   mode, profile, n_events, cfg) -> None:
    from datetime import datetime, UTC

    def d(t):
        return datetime.fromtimestamp(float(t), UTC).strftime("%Y-%m-%d")

    soft_days = sum(float(s.soft_mask.sum()) for s in segments) * cfg.pre.dt_seconds / 86400
    hard_days = sum(float(s.hard_mask.sum()) for s in segments) * cfg.pre.dt_seconds / 86400
    both = sum(float((s.soft_mask * s.hard_mask).sum()) for s in segments) * cfg.pre.dt_seconds / 86400
    print(f"\nData: {days:.1f} observed days in {len(segments)} segment(s) "
          f"[{d(segments[0].time_unix[0])} -> {d(segments[-1].time_unix[-1])}]")
    print(f"  SoLEXS {soft_days:.1f} d, HEL1OS {hard_days:.1f} d, simultaneous {both:.1f} d; "
          f"{n_events} flares detected")
    if archive:
        print(f"  archive mode (>= {cfg.win.large_data_days:g} days): "
              + ", ".join(f"{k} {a} -> {b}" for k, (a, b) in profile.items()))
    thin = f" (thinned from {n_train_all})" if len(tr) != n_train_all else ""
    print(f"Windows: {len(windows)}  split={mode}: train {len(tr)}{thin} / "
          f"val {len(va)} / test {len(te)}")
    if len(segments) <= 12:
        for s in segments:
            print(f"  {s.name}: {len(s)} steps, soft {100 * s.soft_mask.mean():.0f}% "
                  f"hard {100 * s.hard_mask.mean():.0f}%, {len(s.events)} events")


def make_loaders(prep: Prepared, cfg: Config
                 ) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    if prep._shared is None:
        prep._shared = shared_arrays(prep.segments, cfg, prep.norm)
    ds = {
        k: FlareWindows(prep.segments, prep.windows, idx, cfg, prep.norm,
                        shared=prep._shared)
        for k, idx in prep.splits.items()
    }
    stats = {k: v.label_stats() for k, v in ds.items()}

    pin = resolve_device(cfg.train.device).type == "cuda"

    def dl(key: str, shuffle: bool) -> DataLoader:
        cap = cfg.train.max_batches_per_epoch
        if shuffle and cap and len(ds[key]) > cap * cfg.train.batch_size:
            # A fresh random subset every epoch (without replacement within the
            # epoch), so the whole training set is still covered over training.
            sampler = RandomSampler(ds[key], replacement=False,
                                    num_samples=cap * cfg.train.batch_size)
            return DataLoader(ds[key], batch_size=cfg.train.batch_size,
                              sampler=sampler, drop_last=False, num_workers=0,
                              pin_memory=pin)
        return DataLoader(
            ds[key], batch_size=cfg.train.batch_size, shuffle=shuffle,
            drop_last=False, num_workers=0, pin_memory=pin,
        )

    return dl("train", True), dl("val", False), dl("test", False), stats


def resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def phase_class_weights(stats: dict, n_classes: int = 4) -> torch.Tensor:
    """Inverse-frequency weights, clipped so an absent class cannot explode."""
    counts = np.asarray(stats["train"]["phase_counts"], dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n_classes * counts)
    w = np.clip(w, 0.2, 10.0)
    return torch.tensor(w, dtype=torch.float32)
