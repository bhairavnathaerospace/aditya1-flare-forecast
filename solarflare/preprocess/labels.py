"""Flare detection and label construction.

There is no flare catalogue shipped with these files, so the labels are derived
from the data.  The detector follows the logic used operationally on GOES soft
X-ray curves -- a sustained rise above a slowly-varying background, significant
against Poisson noise -- rather than a bare threshold, because a bare threshold
either drowns in noise at B-level or misses everything during an elevated
background.

Labels produced here:

``phase_labels``       0 quiet / 1 rise / 2 peak / 3 decay, per time bin
``magnitude_class``    GOES-like magnitude bucket of the covering event
``occurrence_labels``  will a flare be in progress within H steps ahead
``forecast_targets``   future log-flux values and their validity mask
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import PreprocessConfig

PHASE_QUIET, PHASE_RISE, PHASE_PEAK, PHASE_DECAY = 0, 1, 2, 3
PHASE_NAMES = ["quiet", "rise", "peak", "decay"]


@dataclass
class FlareEvent:
    start_idx: int
    peak_idx: int
    end_idx: int
    start_unix: float
    peak_unix: float
    end_unix: float
    peak_rate: float        # at peak, background included: cts/s (SoLEXS) or W/m^2 (GOES)
    background: float       # same units, background at peak
    peak_excess: float      # same units, above background
    rise_time_s: float
    duration_s: float
    magnitude: float        # peak_excess / background, dimensionless
    goes_class: str = ""    # "M1.4" for GOES-labelled events, "" otherwise

    def as_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in self.__dict__.items()}


def flux_target(x, label_source: str):
    """The regression scale for a flux in truth units.

    SoLEXS counts: log1p(cts/s). GOES: log10(W/m^2), floored at 1e-9 (A0.1), so
    one unit is one decade, i.e. one GOES class step.
    """
    x = np.asarray(x, dtype=np.float64)
    if label_source == "goes":
        return np.log10(np.clip(np.nan_to_num(x, nan=1e-9), 1e-9, None))
    return np.log1p(np.clip(np.nan_to_num(x, nan=0.0), 0.0, None))


def goes_events_on_grid(truth, grid: np.ndarray, min_class: str, dt: float
                        ) -> list[FlareEvent]:
    """GOES-listed flares at or above ``min_class`` that overlap ``grid``.

    Indices are clipped to the grid, but the ``*_unix`` fields keep the true
    GOES times, so a consumer can tell when a rise began before the grid did.
    """
    from ..io.goes import class_flux

    if grid.size == 0:
        return []
    floor = class_flux(min_class)
    t0, t1 = float(grid[0]), float(grid[-1]) + dt
    n = grid.size
    out: list[FlareEvent] = []
    for f in truth.flares:
        if not (f.peak_flux >= floor) or f.end_unix < t0 or f.start_unix >= t1:
            continue
        s = int(np.clip(np.searchsorted(grid, f.start_unix, side="left"), 0, n - 1))
        p = int(np.clip(np.floor((f.peak_unix - grid[0]) / dt), 0, n - 1))
        e = int(np.clip(np.searchsorted(grid, f.end_unix, side="right"), 1, n))
        s = min(s, p)
        e = max(e, p + 1)
        bg = f.background_flux if np.isfinite(f.background_flux) and f.background_flux > 0 else np.nan
        out.append(FlareEvent(
            start_idx=s, peak_idx=p, end_idx=e,
            start_unix=f.start_unix, peak_unix=f.peak_unix, end_unix=f.end_unix,
            peak_rate=f.peak_flux,
            background=float(bg) if bg == bg else float("nan"),
            peak_excess=float(f.peak_flux - bg) if bg == bg else float("nan"),
            rise_time_s=float(f.peak_unix - f.start_unix),
            duration_s=float(f.end_unix - f.start_unix),
            magnitude=float(f.peak_flux / bg) if bg == bg else float("nan"),
            goes_class=f.goes_class,
        ))
    return out


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    k = np.ones(win) / win
    v = np.where(np.isfinite(x), x, 0.0)
    m = np.isfinite(x).astype(np.float64)
    num = np.convolve(v, k, mode="same")
    den = np.convolve(m, k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def detect_flares(
    rate: np.ndarray,
    background: np.ndarray,
    observed: np.ndarray,
    dt: float,
    cfg: PreprocessConfig,
) -> list[FlareEvent]:
    """Detect flares in a background-subtracted soft X-ray light curve.

    ``rate`` is in counts/s; Poisson significance is evaluated on counts, i.e.
    on ``rate * dt``, which is why dt has to be passed explicitly.
    """
    n = rate.size
    if n == 0:
        return []

    win = max(int(cfg.smooth_s / dt), 1)
    sm = _smooth(rate, win)
    bg = np.maximum(background, 1e-9)

    # Poisson sigma of the *excess*, in rate units.
    sigma = np.sqrt(np.maximum(bg, 1e-9) / max(dt, 1e-9))
    excess = sm - bg

    above = (sm >= cfg.flare_rise_factor * bg) & (excess >= cfg.flare_rise_sigma * sigma)
    above &= observed & np.isfinite(sm)

    min_rise = max(int(cfg.flare_min_rise_s / dt), 1)

    # Candidate segments: runs of `above` at least min_rise long.
    events: list[FlareEvent] = []
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        if j - i >= min_rise:
            events.append(_build_event(i, j, sm, bg, observed, dt, cfg))
        i = j

    events = [e for e in events if e is not None]
    events = _merge_events(events, sm, bg, observed, dt, cfg)
    events = [e for e in events if e.duration_s >= cfg.flare_min_duration_s]
    return events


def _build_event(i: int, j: int, sm: np.ndarray, bg: np.ndarray,
                 observed: np.ndarray, dt: float,
                 cfg: PreprocessConfig) -> FlareEvent | None:
    n = sm.size
    seg = sm[i:j] - bg[i:j]
    if seg.size == 0 or not np.isfinite(seg).any():
        return None
    peak_off = int(np.nanargmax(seg))
    peak = i + peak_off
    peak_excess = float(sm[peak] - bg[peak])
    if peak_excess <= 0:
        return None

    # Walk backwards to the true onset: the last bin before the rise where the
    # curve was still at background level.  The walk stops at a data gap --
    # extending an event across unobserved time would invent a rise phase.
    thr_start = bg[peak] + 0.1 * peak_excess
    s = i
    while (s > 0 and observed[s - 1] and np.isfinite(sm[s - 1])
           and sm[s - 1] > thr_start):
        s -= 1

    # Walk forwards until the excess has decayed to the configured fraction.
    thr_end = bg[peak] + cfg.flare_end_fraction * peak_excess
    e = j
    while (e < n - 1 and observed[e] and np.isfinite(sm[e])
           and sm[e] > thr_end):
        e += 1

    return FlareEvent(
        start_idx=s, peak_idx=peak, end_idx=e,
        start_unix=float(s), peak_unix=float(peak), end_unix=float(e),
        peak_rate=float(sm[peak]),
        background=float(bg[peak]),
        peak_excess=peak_excess,
        rise_time_s=float((peak - s) * dt),
        duration_s=float((e - s) * dt),
        magnitude=float(peak_excess / max(bg[peak], 1e-9)),
    )


def _merge_events(events: list[FlareEvent], sm: np.ndarray, bg: np.ndarray,
                  observed: np.ndarray, dt: float,
                  cfg: PreprocessConfig) -> list[FlareEvent]:
    if not events:
        return []
    gap = max(int(cfg.flare_merge_gap_s / dt), 1)
    merged = [events[0]]
    for ev in events[1:]:
        prev = merged[-1]
        if ev.start_idx - prev.end_idx <= gap:
            i, j = prev.start_idx, max(prev.end_idx, ev.end_idx)
            new = _build_event(i, j, sm, bg, observed, dt, cfg)
            merged[-1] = new if new is not None else prev
        else:
            merged.append(ev)
    return merged


def set_event_times(events: list[FlareEvent], grid: np.ndarray) -> None:
    """Replace index-valued time fields with real unix timestamps."""
    for e in events:
        e.start_unix = float(grid[min(e.start_idx, grid.size - 1)])
        e.peak_unix = float(grid[min(e.peak_idx, grid.size - 1)])
        e.end_unix = float(grid[min(e.end_idx, grid.size - 1)])


def phase_labels(n: int, events: list[FlareEvent], dt: float,
                 peak_halfwidth_s: float = 120.0) -> np.ndarray:
    """Per-bin flare phase.

    The peak class is a short window bracketing the maximum rather than a single
    bin: with one positive sample per event the class would be untrainable.
    """
    phase = np.full(n, PHASE_QUIET, dtype=np.int64)
    half = max(int(peak_halfwidth_s / dt), 1)
    for e in events:
        phase[e.start_idx:e.end_idx] = PHASE_DECAY
        phase[e.start_idx:e.peak_idx] = PHASE_RISE
        lo = max(e.peak_idx - half, 0)
        hi = min(e.peak_idx + half + 1, n)
        phase[lo:hi] = PHASE_PEAK
    return phase


# GOES-like magnitude buckets, expressed as excess-over-background ratio because
# these are counts, not calibrated W/m^2.  The "-like" names are deliberate: they
# are *not* GOES classes.  To map onto real classes, cross-match these events
# against a GOES catalogue for the same days and re-fit these edges -- this
# tuple is the only place that needs to change.
MAGNITUDE_EDGES = (1.0, 3.0, 10.0, 30.0)
MAGNITUDE_NAMES = ("sub-A", "A-like", "B-like", "C-like", "M+-like")


def magnitude_class(events: list[FlareEvent], n: int) -> np.ndarray:
    """Per-bin magnitude bucket of the covering event (0 where quiet)."""
    out = np.zeros(n, dtype=np.int64)
    for e in events:
        cls = int(np.searchsorted(MAGNITUDE_EDGES, e.magnitude, side="right"))
        out[e.start_idx:e.end_idx] = cls
    return out


def occurrence_labels(in_flare: np.ndarray, horizons_steps: list[int]) -> np.ndarray:
    """(T, H) -- will a flare be in progress at any point within H steps ahead?

    Computed with a forward-looking maximum filter.  This is a *label*, so
    looking ahead is correct; the input features never do.
    """
    t = in_flare.size
    out = np.zeros((t, len(horizons_steps)), dtype=np.float32)
    x = in_flare.astype(np.float32)
    for h, steps in enumerate(horizons_steps):
        steps = max(int(steps), 1)
        # Forward-looking sliding-window maximum via a strided view.
        padded = np.concatenate([x, np.zeros(steps)])
        shape = (t, steps + 1)
        strides = (padded.strides[0], padded.strides[0])
        win = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides,
                                              writeable=False)
        out[:, h] = win.max(axis=1)
    return out


def forecast_targets(log_flux: np.ndarray, horizons_steps: list[int]
                     ) -> tuple[np.ndarray, np.ndarray]:
    """(T, H) future log-flux values and a validity mask.

    Bins whose target falls off the end of the series, or whose target is
    unobserved, are masked rather than filled -- an invented target is worse
    than a smaller training set.
    """
    t = log_flux.size
    y = np.full((t, len(horizons_steps)), np.nan, dtype=np.float32)
    for h, steps in enumerate(horizons_steps):
        steps = max(int(steps), 1)
        if steps < t:
            y[:t - steps, h] = log_flux[steps:]
    mask = np.isfinite(y)
    return np.nan_to_num(y, nan=0.0), mask
