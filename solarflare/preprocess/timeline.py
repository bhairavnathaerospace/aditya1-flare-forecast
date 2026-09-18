"""Stitch per-file gridded series into continuous multi-day timelines.

Why this exists: backgrounds (6 h windows), trailing means and flare detection
are all window operations. Run per file, they restart at every midnight -- the
background for the first hours of a day is estimated from a handful of
samples, and a flare that peaks at 23:50 and decays past 00:00 is cut into two
events, or lost. Stitching first makes the day boundary invisible.

Every grid built by ``make_grid`` is snapped to a multiple of ``dt``, so files
align exactly and their bins can be placed by integer index.
"""

from __future__ import annotations

import numpy as np

from .grid import GriddedSeries


def stitch(series: list[GriddedSeries], dt: float, max_gap_s: float
           ) -> list[GriddedSeries]:
    """Merge time-sorted series into continuous segments.

    * Files separated by at most ``max_gap_s`` join one segment; the gap bins
      are NaN with zero coverage, which the window and background code already
      treat as unobserved.
    * A longer gap starts a new segment, so the model never sees a window that
      silently spans days of missing data.
    * If two files cover the same bin (a re-processed day, an overlapping
      telemetry dump), the higher ``priority`` wins -- the pipeline sets it
      from the processing version -- and between equal priorities the
      better-covered sample. HEL1OS ships the same interval as V111 and V211
      (2026-07-04) and as V211 and V212 (2026-08-10); coverage alone would
      pick whichever happened to have one more sample in that bin.
    * Series with different channel layouts are never merged.
    """
    if not series:
        return []
    ordered = sorted(series, key=lambda s: float(s.time_unix[0]))

    groups: list[list[GriddedSeries]] = [[ordered[0]]]
    group_end = float(ordered[0].time_unix[-1])
    for s in ordered[1:]:
        same_layout = list(s.names) == list(groups[-1][0].names)
        contiguous = float(s.time_unix[0]) - group_end <= max_gap_s + dt
        if same_layout and contiguous:
            groups[-1].append(s)
        else:
            groups.append([s])
            group_end = -np.inf
        group_end = max(group_end, float(s.time_unix[-1]))

    out: list[GriddedSeries] = []
    for g in groups:
        t0 = float(min(x.time_unix[0] for x in g))
        t1 = float(max(x.time_unix[-1] for x in g))
        n = int(round((t1 - t0) / dt)) + 1
        grid = t0 + dt * np.arange(n, dtype=np.float64)
        n_ch = g[0].values.shape[1]
        vals = np.full((n, n_ch), np.nan, dtype=np.float64)
        cov = np.zeros(n, dtype=np.float64)
        prio = np.full(n, -np.inf)
        for x in g:
            pos = np.rint((x.time_unix - t0) / dt).astype(np.int64)
            ok = (pos >= 0) & (pos < n)
            pos, xv, xc = pos[ok], x.values[ok], x.coverage[ok]
            p = float(getattr(x, "priority", 0.0))
            better = (xc > 0) & ((p > prio[pos]) | ((p == prio[pos]) & (xc > cov[pos])))
            vals[pos[better]] = xv[better]
            cov[pos[better]] = xc[better]
            prio[pos[better]] = p
        out.append(GriddedSeries(grid, vals, cov, list(g[0].names)))
    return out


def uncovered_runs(grid: np.ndarray, covered: np.ndarray, min_len: int
                   ) -> list[slice]:
    """Contiguous stretches where ``covered`` is False, at least ``min_len`` long."""
    free = ~covered
    if not free.any():
        return []
    edges = np.diff(np.concatenate([[0], free.astype(np.int8), [0]]))
    starts = np.where(edges == 1)[0]
    stops = np.where(edges == -1)[0]
    return [slice(int(a), int(b)) for a, b in zip(starts, stops) if b - a >= min_len]


def describe(segments: list[GriddedSeries], dt: float) -> str:
    if not segments:
        return "  (no data)"
    days = [s.time_unix.size * dt / 86400 for s in segments]
    observed = sum(float((s.coverage > 0).sum()) for s in segments) * dt / 86400
    return (f"  {len(segments)} segment(s), {sum(days):.1f} days spanned, "
            f"{observed:.1f} days observed; longest {max(days):.1f} d")
