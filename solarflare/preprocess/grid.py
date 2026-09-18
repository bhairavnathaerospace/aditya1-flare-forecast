"""Put both instruments on one common UTC time grid.

Rebinning is mask-aware throughout: a coarse bin averages only the fine samples
that actually carry data, and records what fraction that was.  That fraction is
fed to the network as a feature, so it can discount poorly-sampled bins instead
of mistaking sparse telemetry for a change in the Sun.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GriddedSeries:
    """A multi-channel series on a regular grid, with per-bin coverage."""

    time_unix: np.ndarray   # (T,) left edge of each bin
    values: np.ndarray      # (T, C) NaN where unobserved
    coverage: np.ndarray    # (T,) fraction of the bin that was sampled, 0..1
    names: list[str]
    #: Where two series cover the same bin when stitched, the higher priority
    #: wins outright (a newer processing version); equal priorities fall back
    #: to better coverage.
    priority: float = 0.0

    @property
    def observed(self) -> np.ndarray:
        return self.coverage > 0

    def __len__(self) -> int:
        return self.time_unix.size


def make_grid(t_start: float, t_stop: float, dt: float) -> np.ndarray:
    """Bin left-edges spanning [t_start, t_stop), snapped to a multiple of dt
    so that grids built from different files always align."""
    first = np.floor(t_start / dt) * dt
    n = int(np.ceil((t_stop - first) / dt))
    return first + dt * np.arange(max(n, 0), dtype=np.float64)


def rebin(
    time_unix: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    grid: np.ndarray,
    dt: float,
    min_valid_fraction: float = 0.0,
    expected_per_bin: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Average ``values`` into ``grid`` bins using only valid samples.

    Parameters
    ----------
    values
        (N,) or (N, C).
    valid
        (N,) or (N, C) boolean; same shape rules as ``values``.
    expected_per_bin
        Number of samples a fully-covered bin would contain.  Defaults to
        ``dt / median_sample_spacing``, which is what makes coverage meaningful
        for an irregularly sampled instrument like HEL1OS.

    Returns
    -------
    (T, C) means with NaN where coverage is insufficient, and (T,) coverage.
    """
    values = np.atleast_2d(values.T).T if values.ndim == 1 else values
    if valid.ndim == 1:
        valid = np.repeat(valid[:, None], values.shape[1], axis=1)

    t_ch = values.shape[1]
    idx = np.floor((time_unix - grid[0]) / dt).astype(np.int64)
    inside = (idx >= 0) & (idx < grid.size)

    out = np.full((grid.size, t_ch), np.nan, dtype=np.float64)

    if expected_per_bin is None:
        if time_unix.size > 1:
            spacing = float(np.median(np.diff(time_unix)))
            spacing = spacing if spacing > 0 else dt
        else:
            spacing = dt
        expected_per_bin = max(dt / spacing, 1.0)

    for c in range(t_ch):
        m = inside & valid[:, c] & np.isfinite(values[:, c])
        if not m.any():
            continue
        num = np.bincount(idx[m], weights=values[m, c], minlength=grid.size)
        den = np.bincount(idx[m], minlength=grid.size).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(den > 0, num / np.maximum(den, 1.0), np.nan)
        out[:, c] = mean

    # Coverage is a property of the *sample*, not of any one channel: a second
    # either carried telemetry or it did not.  HEL1OS bands have slightly
    # different valid masks (a band can read zero counts while its neighbour
    # reads some), so the row-level "any channel valid" is the honest
    # definition.  Counting one arbitrarily-chosen channel instead would make
    # coverage depend on which band happened to be busiest.
    row_valid = inside & (valid & np.isfinite(values)).any(axis=1)
    counts_tot = np.bincount(idx[row_valid], minlength=grid.size).astype(np.float64)

    coverage = np.clip(counts_tot / expected_per_bin, 0.0, 1.0)
    poor = coverage < min_valid_fraction
    out[poor, :] = np.nan
    coverage[poor] = 0.0
    return out, coverage


def common_grid(series: list[GriddedSeries], dt: float
                ) -> tuple[np.ndarray, list[GriddedSeries]]:
    """Reindex several series onto their shared union grid.

    Returns the union grid and the series padded with NaN / zero coverage where
    an instrument was not observing.  This is what lets a soft-only day and a
    hard-only day live in one dataset without pretending they overlap.
    """
    if not series:
        return np.zeros(0), []
    t0 = min(float(s.time_unix[0]) for s in series)
    t1 = max(float(s.time_unix[-1]) + dt for s in series)
    grid = make_grid(t0, t1, dt)

    out: list[GriddedSeries] = []
    for s in series:
        vals = np.full((grid.size, s.values.shape[1]), np.nan)
        cov = np.zeros(grid.size)
        pos = np.rint((s.time_unix - grid[0]) / dt).astype(np.int64)
        ok = (pos >= 0) & (pos < grid.size)
        vals[pos[ok]] = s.values[ok]
        cov[pos[ok]] = s.coverage[ok]
        out.append(GriddedSeries(grid, vals, cov, list(s.names)))
    return grid, out


#: Output rows processed per chunk by the rolling percentiles. nanpercentile
#: copies its (rows x window) input to sort it, so an unchunked call costs
#: rows * window * 8 bytes: 37 MB for one day at 20 s with a 6 h window, but
#: ~11 GB for a nine-month stitched segment. Chunking keeps the peak at
#: CHUNK * window * 8 bytes (~140 MB here) regardless of series length.
_PERCENTILE_CHUNK = 16384


#: Above this many output rows the sliding sorted-window algorithm is used.
#: Below it the vectorised numpy path is faster (no per-step Python overhead).
_SLIDING_MIN_ROWS = 50_000


def _sliding_percentile(padded: np.ndarray, n: int, w: int, q: float) -> np.ndarray:
    """Exact NaN-ignoring percentile of padded[i:i+w], by a sliding sorted window.

    ``nanpercentile`` re-sorts all ``w`` values at every step: O(n * w log w).
    Here the window is kept sorted and updated by one insertion and one
    deletion per step (bisect + memmove on a short list), which is ~10x faster
    for the 1080-sample, multi-million-row case of a stitched archive.

    Interpolation reproduces numpy's default ("linear") method, including the
    order of floating-point operations, so results match ``np.nanpercentile``.
    """
    import bisect
    import math

    out = np.empty(n, dtype=np.float64)
    vals = padded.tolist()
    window: list[float] = []
    qf = q / 100.0

    def add(v):
        if v == v:  # not NaN
            bisect.insort(window, v)

    def remove(v):
        if v == v:
            del window[bisect.bisect_left(window, v)]

    for v in vals[:w]:
        add(v)
    for i in range(n):
        k = len(window)
        if k == 0:
            out[i] = math.nan
        else:
            # numpy's "linear" method uses the direct form (n - 1) * q, not the
            # general alpha/beta expression -- the two round differently.
            vi = (k - 1) * qf
            lo = math.floor(vi)
            gamma = vi - lo
            lo = min(max(lo, 0), k - 1)
            hi = min(lo + 1, k - 1)
            a, b = window[lo], window[hi]
            out[i] = a + (b - a) * gamma
        if i + 1 < n:
            remove(vals[i])
            add(vals[i + w])
    return out


def _windowed_percentile(padded: np.ndarray, n: int, w: int, q: float) -> np.ndarray:
    """Percentile of padded[i : i + w] for i in 0..n-1, in bounded memory."""
    import warnings

    if n >= _SLIDING_MIN_ROWS:
        return _sliding_percentile(padded, n, w, q)

    out = np.empty(n, dtype=np.float64)
    step = padded.strides[0]
    for s in range(0, n, _PERCENTILE_CHUNK):
        e = min(s + _PERCENTILE_CHUNK, n)
        view = np.lib.stride_tricks.as_strided(
            padded[s:], shape=(e - s, w), strides=(step, step), writeable=False)
        # A window that falls entirely inside a data gap is all-NaN and the
        # correct answer is NaN. numpy warns about every such window, which
        # across a mission's worth of gaps would bury real warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[s:e] = np.nanpercentile(view, q, axis=1)
    return out


def trailing_percentile(x: np.ndarray, window: int, q: float) -> np.ndarray:
    """Backward-looking running percentile: bin t uses only bins <= t.

    This is the background that goes into **features**.  The centred version
    below is fine for building labels -- ground truth is allowed to look ahead
    -- but using it as a model input leaks up to half a window of the future
    into every sample, and a forecaster trained that way scores beautifully in
    validation and fails in flight.
    """
    n = x.size
    if n == 0:
        return x.copy()
    w = max(int(window), 1)
    # Left-pad by repeating the first value so early bins stay defined.
    padded = np.ascontiguousarray(
        np.concatenate([np.full(w - 1, x[0]), x]), dtype=np.float64)
    return _windowed_percentile(padded, n, w, q)


def running_percentile(x: np.ndarray, window: int, q: float) -> np.ndarray:
    """Centred running percentile, NaN-aware, with reflected edges.

    Used for the quiescent background **in label generation only** -- see
    ``trailing_percentile`` for the causal version used by features.  A
    percentile (not a mean) is essential: a mean over a window containing a
    flare is dragged up by the flare itself and the resulting background
    subtraction erases the event.
    """
    n = x.size
    if n == 0:
        return x.copy()
    half = max(int(window) // 2, 1)
    padded = np.ascontiguousarray(np.pad(x, half, mode="reflect"), dtype=np.float64)
    return _windowed_percentile(padded, n, 2 * half + 1, q)


def interpolate_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill NaN runs no longer than ``max_gap`` samples.

    Short dropouts (a few seconds of lost telemetry) are interpolated so they do
    not fragment a window; long ones stay NaN and are handled by the mask, since
    inventing an hour of solar activity would be worse than admitting ignorance.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    nan = ~np.isfinite(x)
    if not nan.any() or nan.all():
        return x
    idx = np.arange(x.size)
    edges = np.diff(nan.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if nan[0]:
        starts.insert(0, 0)
    if nan[-1]:
        ends.append(x.size)
    good = ~nan
    for s, e in zip(starts, ends):
        if e - s <= max_gap and s > 0 and e < x.size:
            x[s:e] = np.interp(idx[s:e], idx[good], x[good])
    return x
