"""Build windowed training samples from the gridded, labelled series.

Two things here are easy to get wrong and fatal if you do:

1. **Splitting.**  Windows overlap, so a random split puts near-duplicate
   samples in train and test and produces a model that looks excellent and
   forecasts nothing.  Splits are chronological, with an embargo gap at least
   as long as the longest forecast horizon between adjacent splits.

2. **Normalisation.**  Statistics come from the training split only and are
   frozen; validation, test and live inference reuse them.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

import numpy as np

from ..config import Config
from ..io.discover import DataInventory
from .features import (
    solexs_raw, solexs_derive, hel1os_product_raw, hel1os_derive, time_of_day_features,
)
from .timeline import stitch, uncovered_runs
from .labels import (
    detect_flares, set_event_times, phase_labels, magnitude_class,
    occurrence_labels, forecast_targets, FlareEvent, flux_target, goes_events_on_grid,
)


@dataclass
class Segment:
    """A contiguous stretch of common-grid time with everything aligned."""

    name: str
    time_unix: np.ndarray        # (T,)
    soft: np.ndarray             # (T, Fs)
    soft_mask: np.ndarray        # (T,) 1 where SoLEXS observed
    hard: np.ndarray             # (T, Fh)
    hard_mask: np.ndarray        # (T,) 1 where HEL1OS observed
    clock: np.ndarray            # (T, 2)
    phase: np.ndarray            # (T,)
    in_flare: np.ndarray         # (T,)
    magnitude: np.ndarray        # (T,)
    log_flux: np.ndarray         # (T,) nowcast regression target
    target_valid: np.ndarray     # (T,) 1 where the soft target is trustworthy
    events: list[FlareEvent]
    #: (T,) centred background of the GOES-long analogue, kept for plotting and
    #: diagnostics (labels only -- never a model input). None for hard-only.
    background: np.ndarray | None = None
    #: (T,) GOES-long analogue rate, cts/s. None for hard-only segments.
    goes_long: np.ndarray | None = None

    def __len__(self) -> int:
        return self.time_unix.size


@dataclass
class Normalizer:
    mean_soft: np.ndarray
    std_soft: np.ndarray
    mean_hard: np.ndarray
    std_hard: np.ndarray

    def apply_soft(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_soft) / self.std_soft

    def apply_hard(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_hard) / self.std_hard

    def save(self, path: Path) -> None:
        np.savez(path, **{k: v for k, v in asdict(self).items()})

    @staticmethod
    def load(path: Path) -> Normalizer:
        z = np.load(path)
        return Normalizer(z["mean_soft"], z["std_soft"], z["mean_hard"], z["std_hard"])


def build_segments(inv: DataInventory, cfg: Config) -> tuple[list[Segment], dict]:
    """In-memory entry point for small datasets (and tests): raw stage from
    loaded observations, then the same stitching path the cache uses."""
    soft_raw = [solexs_raw(o, cfg.pre) for o in inv.solexs]
    by_product: dict[str, list] = {}
    for h in inv.hel1os:
        by_product.setdefault(h.product, []).append(h)
    hard_raw = [hel1os_product_raw(obs, cfg.pre) for obs in by_product.values()]
    return build_segments_from_raw(soft_raw, hard_raw, cfg)


def _segment_name(prefix: str, grid: np.ndarray) -> str:
    from datetime import datetime, UTC
    a = datetime.fromtimestamp(float(grid[0]), UTC).strftime("%Y%m%d")
    b = datetime.fromtimestamp(float(grid[-1]), UTC).strftime("%Y%m%d")
    return f"{prefix}_{a}" if a == b else f"{prefix}_{a}_to_{b}"


def build_segments_from_raw(soft_raw: list, hard_raw: list, cfg: Config
                            ) -> tuple[list[Segment], dict]:
    """Stitch per-file raw rates into continuous timelines, then derive.

    When an instrument has no data covering a stretch its channels are zeroed
    and its mask is zero -- the model is trained to cope with exactly that, so
    soft-only and hard-only periods are both usable.
    """
    pre = cfg.pre
    dt = pre.dt_seconds
    segments: list[Segment] = []
    truth = None
    if pre.label_source == "goes":
        from ..io.goes import load_goes
        truth = load_goes(Path(pre.goes_dir))
    elif pre.label_source != "solexs":
        raise ValueError(f"unknown label_source {pre.label_source!r}; use 'solexs' or 'goes'")

    horizons = [max(int(h / dt), 1) for h in cfg.win.forecast_horizons_s]
    occ_steps = [max(int(h / dt), 1) for h in cfg.win.occurrence_horizons_s]

    soft_tl = stitch(soft_raw, dt, pre.max_stitch_gap_s)
    hard_tl = stitch(hard_raw, dt, pre.max_stitch_gap_s)
    hard_feats = [hel1os_derive(h, pre) for h in hard_tl]
    n_hard_features = hard_feats[0].values.shape[1] if hard_feats else 0
    hard_used = [np.zeros(len(hf), dtype=bool) for hf in hard_feats]

    for raw in soft_tl:
        sf = solexs_derive(raw, pre)
        grid = sf.series.time_unix
        soft = np.nan_to_num(sf.series.values, nan=0.0, posinf=0.0, neginf=0.0)
        soft_mask = (sf.series.coverage > 0).astype(np.float32)

        hard = np.zeros((grid.size, max(n_hard_features, 1)), dtype=np.float32)
        hard_mask = np.zeros(grid.size, dtype=np.float32)
        for k, hf in enumerate(hard_feats):
            lo, hi = hf.time_unix[0], hf.time_unix[-1]
            if hi < grid[0] or lo > grid[-1]:
                continue
            pos = np.rint((hf.time_unix - grid[0]) / dt).astype(np.int64)
            ok = (pos >= 0) & (pos < grid.size)
            if not ok.any():
                continue
            vals = np.nan_to_num(hf.values, nan=0.0, posinf=0.0, neginf=0.0)
            hard[pos[ok]] = vals[ok]
            hard_mask[pos[ok]] = (hf.coverage[ok] > 0).astype(np.float32)
            hard_used[k][ok] = True

        segments.append(_finish_segment(
            name=_segment_name("solexs", grid),
            grid=grid, soft=soft, soft_mask=soft_mask,
            hard=hard, hard_mask=hard_mask,
            goes_long=sf.goes_long, background=sf.background,
            observed=sf.series.observed, cfg=cfg, truth=truth,
        ))

    # Hard X-ray time with no SoLEXS counterpart becomes soft-less segments.
    # Only the uncovered *part* of a HEL1OS timeline goes here: discarding a
    # whole file because an hour of it overlapped SoLEXS would waste the rest.
    n_soft_features = segments[0].soft.shape[1] if segments else 0
    for k, hf in enumerate(hard_feats):
        for sl in uncovered_runs(hf.time_unix, hard_used[k], cfg.steps_per_window):
            grid = hf.time_unix[sl]
            hard = np.nan_to_num(hf.values[sl], nan=0.0, posinf=0.0, neginf=0.0)
            hard_mask = (hf.coverage[sl] > 0).astype(np.float32)
            soft = np.zeros((grid.size, max(n_soft_features, 1)), dtype=np.float32)
            soft_mask = np.zeros(grid.size, dtype=np.float32)
            segments.append(_finish_segment(
                name=_segment_name("hel1os", grid),
                grid=grid, soft=soft, soft_mask=soft_mask,
                hard=hard, hard_mask=hard_mask,
                goes_long=None, background=None,
                observed=hf.coverage[sl] > 0, cfg=cfg, truth=truth,
            ))

    segments.sort(key=lambda s: s.time_unix[0])
    meta = {
        "n_segments": len(segments),
        "soft_features": n_soft_features,
        "hard_features": n_hard_features,
        "horizons_steps": horizons,
        "occurrence_steps": occ_steps,
        "dt": dt,
        "label_source": pre.label_source,
        **({"goes_min_class": pre.goes_min_class, "goes_exceed_class": pre.goes_exceed_class,
            "goes_files": truth.source_files} if truth is not None else {}),
    }
    return segments, meta


def _finish_segment(name, grid, soft, soft_mask, hard, hard_mask,
                    goes_long, background, observed, cfg: Config,
                    truth=None) -> Segment:
    dt = cfg.pre.dt_seconds
    t = grid.size

    if truth is not None:
        # Independent truth: GOES events and GOES flux, for every segment --
        # including HEL1OS-only stretches, which had no truth before.
        xrsb = truth.flux_on_grid(grid)
        events = goes_events_on_grid(truth, grid, cfg.pre.goes_min_class, dt)
        phase = phase_labels(t, events, dt)
        log_flux = np.where(np.isfinite(xrsb), flux_target(xrsb, "goes"), 0.0).astype(np.float32)
        target_valid = np.isfinite(xrsb).astype(np.float32)
        magnitude = magnitude_class(events, t)
    elif goes_long is None:
        # Hard-only segment: no soft X-ray truth available, so every
        # soft-derived target is masked out rather than guessed.
        events: list[FlareEvent] = []
        phase = np.zeros(t, dtype=np.int64)
        log_flux = np.zeros(t, dtype=np.float32)
        target_valid = np.zeros(t, dtype=np.float32)
        magnitude = np.zeros(t, dtype=np.int64)
    else:
        events = detect_flares(goes_long, background, observed, dt, cfg.pre)
        set_event_times(events, grid)
        phase = phase_labels(t, events, dt)
        log_flux = np.log1p(np.clip(np.nan_to_num(goes_long, nan=0.0), 0, None)
                            ).astype(np.float32)
        target_valid = observed.astype(np.float32)
        magnitude = magnitude_class(events, t)

    in_flare = (phase > 0).astype(np.float32)
    return Segment(
        name=name, time_unix=grid,
        soft=soft.astype(np.float32), soft_mask=soft_mask,
        hard=hard.astype(np.float32), hard_mask=hard_mask,
        clock=time_of_day_features(grid).values,
        phase=phase, in_flare=in_flare, magnitude=magnitude,
        log_flux=log_flux, target_valid=target_valid, events=events,
        background=None if background is None else np.asarray(background, np.float32),
        goes_long=None if goes_long is None else np.asarray(goes_long, np.float32),
    )


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class WindowIndex:
    # slots: an archive yields several hundred thousand of these; without
    # slots each carries a per-instance __dict__ roughly tripling memory.
    seg: int
    end: int          # exclusive index of the last input step
    t_unix: float     # timestamp of the prediction origin


def enumerate_windows(segments: list[Segment], cfg: Config) -> list[WindowIndex]:
    """All windows whose input is sufficiently observed.

    The prediction origin is the *last* input step, so a window uses
    ``[end - L, end)`` and predicts from ``end - 1`` onwards.
    """
    L = cfg.steps_per_window
    stride = max(int(cfg.win.stride_seconds / cfg.pre.dt_seconds), 1)
    max_h = _max_horizon_steps(cfg)

    out: list[WindowIndex] = []
    for si, seg in enumerate(segments):
        t = len(seg)
        if t < L + max_h:
            continue
        any_mask = np.maximum(seg.soft_mask, seg.hard_mask)
        csum = np.concatenate([[0.0], np.cumsum(any_mask)])
        ends = np.arange(L, t - max_h + 1, stride)
        frac = (csum[ends] - csum[ends - L]) / L
        # Require the prediction origin itself to be observed.
        keep = (frac >= cfg.win.min_observed_fraction) & (any_mask[ends - 1] > 0)
        times = seg.time_unix[ends[keep] - 1]
        out.extend(WindowIndex(seg=si, end=int(e), t_unix=float(tt))
                   for e, tt in zip(ends[keep], times))
    return out


def _max_horizon_steps(cfg: Config) -> int:
    dt = cfg.pre.dt_seconds
    return max([int(h / dt) for h in cfg.win.forecast_horizons_s]
               + [int(h / dt) for h in cfg.win.occurrence_horizons_s])


def span_days(segments: list[Segment], dt: float) -> float:
    """Observed days across all segments (either instrument)."""
    return sum(float(np.maximum(s.soft_mask, s.hard_mask).sum()) for s in segments) * dt / 86400.0


def resolve_split_mode(segments: list[Segment], cfg: Config) -> str:
    mode = cfg.train.split_mode
    if mode != "auto":
        return mode
    return ("global" if span_days(segments, cfg.pre.dt_seconds) >= cfg.win.large_data_days
            else "per_segment")


def chronological_split(windows: list[WindowIndex], cfg: Config,
                        mode: str = "per_segment"
                        ) -> tuple[list[int], list[int], list[int]]:
    """Split window indices by time, with an embargo between the blocks.

    Two modes, because the right answer depends on the data:

    ``per_segment`` -- split each segment 60/20/20 internally. Right for a
    handful of disconnected days: a global sort handed each whole observation
    to one split, which with the original two-day sample left a test set of
    hard-only windows with no labels at all.

    ``global`` -- one cut on the calendar: train is everything before a date,
    test everything after, with an embargo either side. Right for an archive.
    Per-segment splitting across hundreds of segments interleaves the splits
    in time, so a model trains on April data and is tested on the March that
    preceded it -- leakage through slowly varying solar activity, not through
    any single window. Note the cost: with a single calendar cut, train and
    test sit at different phases of the solar cycle, so test skill reflects
    that shift as well as the model.
    """
    if not windows:
        return [], [], []
    if mode == "global":
        return _global_split(windows, cfg)

    f_tr, f_va, _ = cfg.train.split
    emb = cfg.train.embargo_s
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []

    by_seg: dict[int, list[int]] = {}
    for i, w in enumerate(windows):
        by_seg.setdefault(w.seg, []).append(i)

    for seg_id in sorted(by_seg):
        ids = sorted(by_seg[seg_id], key=lambda i: windows[i].t_unix)
        times = np.array([windows[i].t_unix for i in ids])
        n = len(ids)
        i_tr = int(n * f_tr)
        i_va = int(n * (f_tr + f_va))
        if i_tr == 0:
            continue
        t_tr_end = times[i_tr - 1]
        t_va_end = times[max(i_va - 1, i_tr - 1)]

        train += ids[:i_tr]
        val += [ids[i] for i in range(i_tr, i_va) if times[i] > t_tr_end + emb]
        test += [ids[i] for i in range(i_va, n) if times[i] > t_va_end + emb]

    return train, val, test


def _global_split(windows: list[WindowIndex], cfg: Config
                  ) -> tuple[list[int], list[int], list[int]]:
    f_tr, f_va, _ = cfg.train.split
    emb = cfg.train.embargo_s
    order = np.argsort([w.t_unix for w in windows], kind="stable")
    times = np.array([windows[i].t_unix for i in order])
    n = len(order)
    i_tr = max(int(n * f_tr), 1)
    i_va = max(int(n * (f_tr + f_va)), i_tr)
    t_tr_end = times[i_tr - 1]
    t_va_end = times[i_va - 1]
    train = [int(order[i]) for i in range(i_tr)]
    val = [int(order[i]) for i in range(i_tr, i_va) if times[i] > t_tr_end + emb]
    test = [int(order[i]) for i in range(i_va, n) if times[i] > t_va_end + emb]
    return train, val, test


def thin_quiet_training_windows(segments: list[Segment], windows: list[WindowIndex],
                                train: list[int], cfg: Config) -> list[int]:
    """Keep every flare-adjacent training window, subsample the quiet ones.

    Across an archive, quiet Sun is the overwhelming majority of windows and
    near-duplicates of each other; training on all of them costs hours and
    teaches little. A window is "active" if a flare is in progress anywhere in
    its input or within the longest forecast horizon after it -- those are all
    kept. Quiet windows are kept only on a coarse stride.

    Applied to the **training split only**. Thinning validation or test would
    change their base rates and make every probabilistic score (Brier,
    reliability, precision) describe a distribution the model never meets.
    """
    dt = cfg.pre.dt_seconds
    base = max(int(cfg.win.stride_seconds / dt), 1)
    stride_q = max(int(cfg.win.quiet_train_stride_seconds / dt), 1)
    keep_every = max(stride_q // base, 1)  # quiet windows: keep 1 in this many
    L = cfg.steps_per_window
    max_h = _max_horizon_steps(cfg)
    csums: dict[int, np.ndarray] = {}
    kept = []
    for i in train:
        w = windows[i]
        if w.seg not in csums:
            csums[w.seg] = np.concatenate([[0.0], np.cumsum(segments[w.seg].in_flare)])
        cs = csums[w.seg]
        hi = min(w.end + max_h, cs.size - 1)
        active = cs[hi] - cs[w.end - L] > 0
        if active or (w.end // base) % keep_every == 0:
            kept.append(i)
    return kept


def fit_normalizer(segments: list[Segment], windows: list[WindowIndex],
                   idx: list[int], cfg: Config) -> Normalizer:
    """Robust per-channel statistics from the training windows only.

    Median / IQR rather than mean / std: flare peaks are genuine outliers and
    would otherwise set the scale for the whole channel.
    """
    L = cfg.steps_per_window
    soft_chunks, hard_chunks = [], []
    take = idx if len(idx) <= 4000 else list(np.random.default_rng(0).choice(idx, 4000, replace=False))
    for i in take:
        w = windows[i]
        seg = segments[w.seg]
        sl = slice(w.end - L, w.end)
        if seg.soft_mask[sl].any():
            soft_chunks.append(seg.soft[sl][seg.soft_mask[sl] > 0])
        if seg.hard_mask[sl].any():
            hard_chunks.append(seg.hard[sl][seg.hard_mask[sl] > 0])

    def stats(chunks, n_feat):
        if not chunks:
            return np.zeros(n_feat, np.float32), np.ones(n_feat, np.float32)
        x = np.concatenate(chunks, axis=0)
        med = np.median(x, axis=0)
        q1, q3 = np.percentile(x, [25, 75], axis=0)
        scale = (q3 - q1) / 1.349
        scale = np.where(scale > 1e-6, scale, np.std(x, axis=0))
        scale = np.where(scale > 1e-6, scale, 1.0)
        return med.astype(np.float32), scale.astype(np.float32)

    ms, ss = stats(soft_chunks, segments[0].soft.shape[1])
    mh, sh = stats(hard_chunks, segments[0].hard.shape[1])
    return Normalizer(ms, ss, mh, sh)


def build_targets(seg: Segment, cfg: Config) -> dict[str, np.ndarray]:
    """Per-bin targets for a whole segment (sliced per window later)."""
    dt = cfg.pre.dt_seconds
    horizons = [max(int(h / dt), 1) for h in cfg.win.forecast_horizons_s]
    occ_steps = [max(int(h / dt), 1) for h in cfg.win.occurrence_horizons_s]

    y_fore, m_fore = forecast_targets(seg.log_flux, horizons)
    # A forecast target is only usable if the *future* bin was itself observed.
    for h, steps in enumerate(horizons):
        tv = np.zeros_like(seg.target_valid)
        if steps < len(seg):
            tv[:len(seg) - steps] = seg.target_valid[steps:]
        m_fore[:, h] &= tv > 0

    occ = occurrence_labels(seg.in_flare, occ_steps)
    occ_mask = np.ones_like(occ)
    for h, steps in enumerate(occ_steps):
        tv = np.ones(len(seg), dtype=bool)
        tv[len(seg) - steps:] = False
        occ_mask[:, h] = (tv & (seg.target_valid > 0)).astype(np.float32)

    peak, peak_mask = _peak_targets(seg, cfg)

    return {
        "phase": seg.phase.astype(np.int64),
        "in_flare": seg.in_flare.astype(np.float32),
        "nowcast": seg.log_flux.astype(np.float32),
        "nowcast_mask": seg.target_valid.astype(np.float32),
        "forecast": y_fore.astype(np.float32),
        "forecast_mask": m_fore.astype(np.float32),
        "occurrence": occ.astype(np.float32),
        "occurrence_mask": occ_mask.astype(np.float32),
        "peak": peak.astype(np.float32),
        "peak_mask": peak_mask.astype(np.float32),
    }


#: Time-to-peak is regressed in units of an hour so it lands on the same scale
#: as the log-flux targets and needs no separate loss weighting.
PEAK_TIME_SCALE_S = 3600.0


def _peak_targets(seg: Segment, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """(T, 2) = [time to this event's peak, log peak flux] and its mask.

    Defined only inside a flare: asking "when does it peak and how big will it
    get" is meaningless during quiet Sun, so those bins are masked out rather
    than given a sentinel the model would learn to reproduce.
    """
    t = len(seg)
    out = np.zeros((t, 2), dtype=np.float32)
    mask = np.zeros((t, 2), dtype=np.float32)
    dt = cfg.pre.dt_seconds
    for e in seg.events:
        lo, hi = e.start_idx, min(e.end_idx, t)
        if hi <= lo:
            continue
        idx = np.arange(lo, hi)
        out[lo:hi, 0] = (e.peak_idx - idx) * dt / PEAK_TIME_SCALE_S
        out[lo:hi, 1] = float(flux_target(e.peak_rate, cfg.pre.label_source))
        mask[lo:hi, :] = seg.target_valid[lo:hi, None]
    return out, mask


def save_meta(meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
