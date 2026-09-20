"""Turn raw counts into features a network can learn flare physics from.

Design rules followed here:

* Everything is **causal or instantaneous**.  No feature peeks forward, because
  the same code path runs in real-time inference.
* Rates go through ``log1p``.  Flare soft X-ray flux spans two or three decades;
  in linear space the loss is dominated entirely by the few peak samples.
* Spectral *shape* is separated from spectral *amplitude*.  Hardness ratios and
  the counts-weighted mean energy carry the plasma-temperature information that
  distinguishes a genuine flare onset from a rate bump, and they stay
  informative even when the absolute calibration is uncertain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import (
    PreprocessConfig,
    SOLEXS_BANDS_BY_SCALE,
    SOLEXS_CH_LO,
    GOES_LONG_KEV,
    GOES_SHORT_BY_SCALE,
    HEL1OS_DETECTORS,
)
from ..io.solexs import SolexsObservation, channel_energies
from ..io.hel1os import Hel1osObservation
from .grid import (GriddedSeries, make_grid, rebin, running_percentile,
                   trailing_percentile)

EPS = 1e-9


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / (b + EPS)


def _log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x, 0.0, None))


def _causal_diff(x: np.ndarray, lag: int) -> np.ndarray:
    """x[t] - x[t-lag], with the first ``lag`` samples set to 0."""
    out = np.zeros_like(x)
    if lag < x.size:
        out[lag:] = x[lag:] - x[:-lag]
    return out


def _causal_mean(x: np.ndarray, win: int) -> np.ndarray:
    """Trailing mean over ``win`` samples, NaN-aware."""
    if win <= 1:
        return x.copy()
    v = np.where(np.isfinite(x), x, 0.0)
    m = np.isfinite(x).astype(np.float64)
    cs = np.concatenate([[0.0], np.cumsum(v)])
    cm = np.concatenate([[0.0], np.cumsum(m)])
    i = np.arange(x.size) + 1
    lo = np.maximum(i - win, 0)
    num = cs[i] - cs[lo]
    den = cm[i] - cm[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1.0), np.nan)


# ---------------------------------------------------------------------------
# SoLEXS (soft X-ray)
# ---------------------------------------------------------------------------

@dataclass
class SolexsFeatures:
    series: GriddedSeries
    goes_long: np.ndarray       # (T,) 1-8 A analogue rate, cts/s
    background: np.ndarray      # (T,) centred background -- LABELS ONLY
    excess: np.ndarray          # (T,) goes_long - background, for detection
    mean_energy: np.ndarray     # (T,) keV
    causal_background: np.ndarray = None  # (T,) trailing background, features


def solexs_raw_names(scale: str = "sarwade2025") -> list[str]:
    """Columns of the per-file raw SoLEXS product that the cache stores.
    Everything the model sees is derived from these, on stitched timelines."""
    return ([f"slx_{lo:g}_{hi:g}keV" for lo, hi in SOLEXS_BANDS_BY_SCALE[scale]]
            + ["slx_goes_long", "slx_goes_short", "slx_total", "slx_mean_energy"])


#: Raw columns under the default (published) energy scale.
SOLEXS_RAW_NAMES: list[str] = solexs_raw_names()


def _band_edges_from_names(names: list[str]) -> list[tuple[float, float]]:
    """(lo, hi) keV of the soft-band columns, in order, from names like slx_2_3keV."""
    out = []
    for n in names:
        if n.startswith("slx_") and n.endswith("keV"):
            lo, hi = n.removeprefix("slx_").removesuffix("keV").split("_")
            out.append((float(lo), float(hi)))
    return out


def solexs_raw(obs: SolexsObservation, cfg: PreprocessConfig,
               grid: np.ndarray | None = None) -> GriddedSeries:
    """Stage 1 (per file, cacheable): 340-channel spectra -> gridded band rates.

    Nothing here depends on neighbouring days, so the result can be computed
    once per downloaded file and cached. Everything window-based (backgrounds,
    derivatives, trailing means) lives in `solexs_derive`, which runs on
    stitched timelines so that a flare crossing midnight is one flare.
    """
    dt = cfg.dt_seconds
    if grid is None:
        grid = make_grid(obs.time_unix[0], obs.time_unix[-1] + 1.0, dt)

    scale = cfg.solexs_energy_scale
    energies = channel_energies(scale, obs.detector)
    bands = SOLEXS_BANDS_BY_SCALE[scale]

    native: list[np.ndarray] = [obs.band_rate(lo, hi, energies) for lo, hi in bands]
    goes_l_native = obs.band_rate(*GOES_LONG_KEV, energies)
    goes_s_native = obs.band_rate(*GOES_SHORT_BY_SCALE[scale], energies)
    total_native = obs.total_rate()

    # Counts-weighted mean energy: a compact proxy for plasma temperature that
    # rises sharply at flare onset, ahead of the total rate.
    sel = energies >= bands[0][0]
    sel[:SOLEXS_CH_LO] = False  # the electronic noise peak is never signal
    spec = obs.spectra[:, sel].astype(np.float64)
    e_sel = energies[sel]
    tot = spec.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_e_native = np.where(tot > 0, (spec * e_sel).sum(axis=1) / np.maximum(tot, EPS),
                                 np.nan)
    del spec

    stack = np.column_stack(native + [goes_l_native, goes_s_native,
                                      total_native, mean_e_native])
    vals, coverage = rebin(
        obs.time_unix, stack, obs.valid, grid, dt,
        min_valid_fraction=cfg.min_valid_fraction, expected_per_bin=dt,
    )
    return GriddedSeries(grid, vals, coverage, solexs_raw_names(scale))


def solexs_features(obs: SolexsObservation, cfg: PreprocessConfig,
                    grid: np.ndarray | None = None) -> SolexsFeatures:
    """Single-file convenience: raw stage then derive stage."""
    return solexs_derive(solexs_raw(obs, cfg, grid), cfg)


def solexs_derive(raw: GriddedSeries, cfg: PreprocessConfig) -> SolexsFeatures:
    """Stage 2 (per stitched timeline): band rates -> model features.

    ``raw`` may span many days. Every window-based quantity is computed here
    so it sees across file boundaries.
    """
    dt = cfg.dt_seconds
    grid = raw.time_unix
    vals = raw.values
    coverage = raw.coverage
    names = list(raw.names)

    col = {n: i for i, n in enumerate(names)}
    goes_long = vals[:, col["slx_goes_long"]]
    mean_energy = vals[:, col["slx_mean_energy"]]

    # --- backgrounds -----------------------------------------------------
    # Two of them, deliberately:
    #   `background`        centred  -> used to DETECT flares (labels may look
    #                                   ahead; a centred estimate is the least
    #                                   biased one for defining truth).
    #   `causal_background` trailing -> used as a model FEATURE, so no input
    #                                   ever contains information from after
    #                                   the prediction origin.
    bg_win = max(int(cfg.background_window_s / dt), 3)
    background = np.maximum(
        running_percentile(goes_long, bg_win, cfg.background_percentile), EPS)
    causal_background = np.maximum(
        trailing_percentile(goes_long, bg_win, cfg.background_percentile), EPS)
    excess = goes_long - background
    causal_excess = goes_long - causal_background

    # --- assemble the feature matrix ------------------------------------
    feats: list[np.ndarray] = []
    fnames: list[str] = []

    # Band layout is read from the raw columns, so a cache built under either
    # energy scale derives consistently.
    bands = _band_edges_from_names(names)
    for lo, hi in bands:
        k = f"slx_{lo:g}_{hi:g}keV"
        feats.append(_log1p(vals[:, col[k]]))
        fnames.append(f"log_{k}")

    feats += [_log1p(goes_long), _log1p(vals[:, col["slx_goes_short"]]),
              _log1p(vals[:, col["slx_total"]])]
    fnames += ["log_goes_long", "log_goes_short", "log_total"]

    feats.append(_log1p(np.clip(causal_excess, 0, None)))
    fnames.append("log_excess")
    feats.append(np.log1p(np.clip(_safe_ratio(goes_long, causal_background), 0, None)))
    fnames.append("log_ratio_to_background")

    # Hardness ratios between adjacent soft bands.
    for i in range(len(bands) - 1):
        a = vals[:, col[f"slx_{bands[i + 1][0]:g}_{bands[i + 1][1]:g}keV"]]
        b = vals[:, col[f"slx_{bands[i][0]:g}_{bands[i][1]:g}keV"]]
        feats.append(_safe_ratio(a - b, a + b))
        fnames.append(f"slx_hr{i}")

    # GOES-style ratio: the standard temperature diagnostic.
    feats.append(_safe_ratio(vals[:, col["slx_goes_short"]], goes_long))
    fnames.append("slx_goes_ratio")

    feats.append(np.nan_to_num(mean_energy, nan=0.0))
    fnames.append("slx_mean_energy")

    # Causal derivatives of log flux over 1 / 5 / 15 minutes.  The rise rate is
    # the single most predictive quantity for short-horizon nowcasting.
    lf = _log1p(goes_long)
    for secs in (60.0, 300.0, 900.0):
        lag = max(int(secs / dt), 1)
        feats.append(_causal_diff(np.nan_to_num(lf, nan=0.0), lag))
        fnames.append(f"slx_dlog_{int(secs)}s")

    for secs in (300.0, 1800.0):
        win = max(int(secs / dt), 1)
        feats.append(np.nan_to_num(_causal_mean(lf, win), nan=0.0))
        fnames.append(f"slx_logmean_{int(secs)}s")

    feats.append(coverage)
    fnames.append("slx_coverage")

    matrix = np.column_stack(feats).astype(np.float32)
    series = GriddedSeries(grid, matrix, coverage, fnames)
    return SolexsFeatures(series, goes_long, background, excess, mean_energy,
                          causal_background)


# ---------------------------------------------------------------------------
# HEL1OS (hard X-ray)
# ---------------------------------------------------------------------------

def hel1os_features(obs: Hel1osObservation, cfg: PreprocessConfig,
                    grid: np.ndarray | None = None) -> GriddedSeries:
    """Single-file convenience: raw stage then derive stage."""
    return hel1os_derive(hel1os_raw(obs, cfg, grid), cfg)


def _hls_band_name(det: str, lo: float, hi: float) -> str:
    return f"hls_{det}_{lo:g}_{hi:g}keV"


def _hls_cov_name(det: str) -> str:
    return f"hls_{det}_cov"


def _hel1os_layout() -> list[tuple[str, list[tuple[float, float]], list[int], int]]:
    """(detector, bands, band column indices, coverage column index) in raw order."""
    out, col = [], 0
    for det, (_, bands) in HEL1OS_DETECTORS.items():
        idx = list(range(col, col + len(bands)))
        out.append((det, list(bands), idx, col + len(bands)))
        col += len(bands) + 1
    return out


#: Columns of the per-product raw HEL1OS series the cache stores: for every
#: detector its band rates, then that detector's own bin coverage. A fixed
#: layout, so products missing a detector still stitch with complete ones.
HEL1OS_RAW_NAMES: list[str] = [
    n for det, bands, _, _ in _hel1os_layout()
    for n in [_hls_band_name(det, lo, hi) for lo, hi in bands] + [_hls_cov_name(det)]
]


def hel1os_raw(obs: Hel1osObservation, cfg: PreprocessConfig,
               grid: np.ndarray | None = None) -> GriddedSeries:
    """Stage 1 (per detector file): masked 1 s band rates -> gridded rates in
    the fixed ``HEL1OS_RAW_NAMES`` layout. Only this detector's columns are
    filled; the rest are NaN with zero coverage. Combine the detectors of one
    product with ``hel1os_merge``."""
    dt = cfg.dt_seconds
    if grid is None:
        grid = make_grid(obs.time_unix[0], obs.time_unix[-1] + 1.0, dt)

    layout = {det: (bands, idx, cov) for det, bands, idx, cov in _hel1os_layout()}
    key = obs.detector_key
    if key not in layout:
        raise ValueError(f"unknown HEL1OS detector {obs.detector!r}; "
                         f"expected one of {sorted(layout)}")
    bands, idx, cov_col = layout[key]

    # HEL1OS is sampled every few seconds, so a 10 s bin holds only ~2 real
    # samples.  expected_per_bin uses the true median spacing rather than dt,
    # otherwise every bin would look 80% empty.
    spacing = float(np.median(np.diff(obs.time_unix))) if obs.n > 1 else 1.0
    sampled_spacing = spacing
    any_valid = obs.any_valid
    if any_valid.sum() > 1:
        sampled_spacing = float(np.median(np.diff(obs.time_unix[any_valid])))
    expected = max(dt / max(sampled_spacing, 1e-6), 1.0)

    vals, coverage = rebin(
        obs.time_unix, obs.rates.astype(np.float64), obs.valid, grid, dt,
        min_valid_fraction=cfg.min_valid_fraction, expected_per_bin=expected,
    )

    out = np.full((grid.size, len(HEL1OS_RAW_NAMES)), np.nan, dtype=np.float64)
    matched = 0
    for i, (lo, hi) in enumerate(obs.bands_kev):
        j = next((k for k, (blo, bhi) in enumerate(bands)
                  if abs(blo - lo) < 1e-6 and abs(bhi - hi) < 1e-6), None)
        if j is not None:
            out[:, idx[j]] = vals[:, i]
            matched += 1
    if not matched:
        raise ValueError(f"{obs.detector}: none of the bands {obs.bands_kev.tolist()} "
                         f"match the expected layout {bands}")
    out[:, cov_col] = coverage
    return GriddedSeries(grid, out, coverage, list(HEL1OS_RAW_NAMES))


def hel1os_merge(raws: list[GriddedSeries], dt: float) -> GriddedSeries:
    """Combine per-detector raw series into one series, detector by detector.

    Each detector's columns come only from that detector's own data -- where
    two inputs cover the same detector and bin, the better-covered one wins
    for that detector alone. Row coverage is the best detector coverage.
    """
    if not raws:
        raise ValueError("nothing to merge")
    t0 = min(float(r.time_unix[0]) for r in raws)
    t1 = max(float(r.time_unix[-1]) for r in raws) + dt
    grid = make_grid(t0, t1, dt)
    n_col = len(HEL1OS_RAW_NAMES)
    out = np.full((grid.size, n_col), np.nan, dtype=np.float64)
    layout = _hel1os_layout()
    cov_cols = [c for *_, c in layout]
    out[:, cov_cols] = 0.0
    for r in raws:
        if list(r.names) != HEL1OS_RAW_NAMES:
            raise ValueError("hel1os_merge needs series in the HEL1OS_RAW_NAMES layout")
        pos = np.rint((r.time_unix - grid[0]) / dt).astype(np.int64)
        ok = (pos >= 0) & (pos < grid.size)
        pos, v = pos[ok], r.values[ok]
        for _, _, idx, c in layout:
            incoming = np.nan_to_num(v[:, c], nan=0.0)
            better = incoming > out[pos, c]
            if better.any():
                rows = pos[better]
                out[np.ix_(rows, idx)] = v[better][:, idx]
                out[rows, c] = incoming[better]
    coverage = out[:, cov_cols].max(axis=1)
    return GriddedSeries(grid, out, coverage, list(HEL1OS_RAW_NAMES))


def hel1os_product_raw(observations: list[Hel1osObservation],
                       cfg: PreprocessConfig) -> GriddedSeries:
    """All detectors of one HEL1OS product -> one raw series, with its
    processing version (``..._V212``) as stitching priority."""
    import re

    merged = hel1os_merge([hel1os_raw(o, cfg) for o in observations], cfg.dt_seconds)
    m = re.search(r"_V(\d+)$", Path(observations[0].product).name) if observations else None
    merged.priority = float(m.group(1)) if m else 0.0
    return merged


def solexs_feature_names(cfg: PreprocessConfig) -> list[str]:
    """Names of the columns ``solexs_derive`` produces, in order."""
    names = solexs_raw_names(cfg.solexs_energy_scale)
    empty = GriddedSeries(np.arange(8, dtype=np.float64) * cfg.dt_seconds,
                          np.full((8, len(names)), 1.0), np.ones(8), names)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base = list(solexs_derive(empty, cfg).series.names)
    if getattr(cfg, "sharp_dir", ""):
        from ..io.sharp import FEATURE_NAMES
        base += list(FEATURE_NAMES) + ["sharp_fresh"]
    return base


def hel1os_feature_names(cfg: PreprocessConfig) -> list[str]:
    """Names of the columns ``hel1os_derive`` produces, in order."""
    empty = GriddedSeries(np.arange(3, dtype=np.float64) * cfg.dt_seconds,
                          np.full((3, len(HEL1OS_RAW_NAMES)), np.nan), np.zeros(3),
                          list(HEL1OS_RAW_NAMES))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return hel1os_derive(empty, cfg).names


def hel1os_derive(raw: GriddedSeries, cfg: PreprocessConfig) -> GriddedSeries:
    """Stage 2 (per stitched timeline): per-detector band rates -> features.

    Every feature is computed from a single detector's columns, so no feature
    ever mixes detectors with different responses. Per detector:
    log band rates, hardness ratios of adjacent narrow bands, wide-band ratio
    to a trailing background, wide-band log derivatives, and coverage.
    """
    if list(raw.names) != HEL1OS_RAW_NAMES:
        raise ValueError("HEL1OS raw series is not in the current layout; "
                         "rebuild the cache (python -m solarflare cache)")
    dt = cfg.dt_seconds
    grid = raw.time_unix
    vals = raw.values
    bg_win = max(int(cfg.background_window_s / dt), 3)
    smooth = int(round(getattr(cfg, "hel1os_smooth_s", 0.0) / dt))
    if smooth > 1:
        # Readout batches: average rates (never coverage) over a trailing window.
        vals = vals.copy()
        cov_cols = {c for _, _, _, c in _hel1os_layout()}
        for j in range(vals.shape[1]):
            if j not in cov_cols:
                vals[:, j] = _causal_mean(vals[:, j], smooth)

    feats: list[np.ndarray] = []
    fnames: list[str] = []
    for det, bands, idx, cov_col in _hel1os_layout():
        edges = np.asarray(bands, dtype=np.float64)
        wide = int(np.argmax(edges[:, 1] - edges[:, 0]))
        rates = vals[:, idx]

        for j, (lo, hi) in enumerate(bands):
            feats.append(_log1p(rates[:, j]))
            fnames.append(f"log_{_hls_band_name(det, lo, hi)}")

        # Hardness ratios: non-thermal signatures live here, and they lead the
        # soft X-ray peak (Neupert effect) -- the lead time a nowcaster needs.
        narrow = [j for j in range(len(bands)) if j != wide]
        for a, b in zip(narrow[1:], narrow[:-1]):
            feats.append(_safe_ratio(rates[:, a] - rates[:, b], rates[:, a] + rates[:, b]))
            fnames.append(f"hls_{det}_hr_{a}_{b}")

        wide_rate = rates[:, wide]
        # Feature, therefore trailing -- same reasoning as the soft X-ray case.
        bg = np.maximum(trailing_percentile(wide_rate, bg_win,
                                            cfg.background_percentile), EPS)
        feats.append(np.log1p(np.clip(_safe_ratio(wide_rate, bg), 0, None)))
        fnames.append(f"hls_{det}_log_ratio_to_background")

        # Differences only where both ends were observed. Filling gaps with 0
        # first turned every return from a data gap into a jump of log(rate).
        lw = _log1p(wide_rate)
        for secs in (60.0, 300.0):
            lag = max(int(secs / dt), 1)
            feats.append(np.nan_to_num(_causal_diff(lw, lag), nan=0.0))
            fnames.append(f"hls_{det}_dlog_{int(secs)}s")

        feats.append(np.nan_to_num(vals[:, cov_col], nan=0.0))
        fnames.append(f"hls_{det}_coverage")

    matrix = np.column_stack(feats).astype(np.float32)
    return GriddedSeries(grid, matrix, raw.coverage, fnames)


def time_of_day_features(grid: np.ndarray) -> GriddedSeries:
    """Cyclic clock features.

    WARNING: these are computed but **not fed to the model by default**
    (``ModelConfig.use_clock = False``).  The intent was to absorb
    spacecraft-periodic systematics, but on a short dataset time-of-day
    uniquely identifies each sample and the network uses it as a lookup table
    rather than reading the Sun.  Measured consequences are documented at
    ``ModelConfig.use_clock``.  Only enable with many days of data.
    """
    sec = np.mod(grid, 86400.0)
    ang = 2 * np.pi * sec / 86400.0
    vals = np.column_stack([np.sin(ang), np.cos(ang)]).astype(np.float32)
    return GriddedSeries(grid, vals, np.ones(grid.size), ["tod_sin", "tod_cos"])
