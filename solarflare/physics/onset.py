"""Per-flare physics diagnostics: hot onset and the Neupert effect.

Two signatures from the literature that the network is never given explicitly:

* **Hot onset** (Hudson et al. 2021): soft X-ray emission is already hot in
  the first minute of a flare, before the impulsive phase. Without a response
  matrix (the SoLEXS L1 zips carry none) temperature is tracked by a
  background-subtracted *hardness* -- net 6-8 keV over net 3-4 keV counts,
  which rises steeply with temperature because the 6-8 keV band holds the
  Fe XXV 6.7 keV complex -- and cross-checked with the GOES XRS short/long
  ratio. Converting either to MK needs a thermal model; XSPEC fits of a few
  flares can calibrate the hardness later.
* **Neupert effect** (Neupert 1968; Veronig et al. 2002): hard X-rays from
  accelerated electrons track the *time derivative* of the soft X-ray flux.
  Measured as the correlation and lag between HEL1OS 20-40 keV net counts and
  d(SoLEXS)/dt.

Everything runs on the 20 s grid of the cache. Features meant for prediction
are computed only from data before a stated decision time, ``t_dec`` = the
causal SoLEXS onset + ``decision_s``; ``tests/test_physics.py`` checks that
changing data after ``t_dec`` leaves them unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io.goes import GOES_EPOCH_UNIX

#: Hold a linear background at its value this long after the pre-flare window.
MAX_EXTRAPOLATE_S = 600.0


def linear_background(t: np.ndarray, y: np.ndarray, ok: np.ndarray, t_lo: float, t_hi: float,
                      min_points: int = 8) -> tuple[np.ndarray, float] | None:
    """Pre-flare background over [t_lo, t_hi), carried through the flare.

    A decaying earlier flare is followed (negative slope, extrapolated for at
    most ``MAX_EXTRAPOLATE_S`` and then held); a rising pre-flare trend is not
    extrapolated upward -- it is usually the flare itself starting, and
    projecting it forward would subtract the flare. Returns (background,
    residual standard deviation), or None without enough pre-flare samples.
    """
    sel = ok & np.isfinite(y) & (t >= t_lo) & (t < t_hi)
    n = int(sel.sum())
    if n < min_points:
        return None
    x = t[sel] - t_hi
    if n >= 2 * min_points:
        slope, level = np.polyfit(x, y[sel], 1)
        slope = min(float(slope), 0.0)
        level = float(np.median(y[sel] - slope * x))
    else:
        slope, level = 0.0, float(np.median(y[sel]))
    dt_after = np.clip(t - t_hi, None, MAX_EXTRAPOLATE_S)
    bg = np.maximum(level + slope * dt_after, 0.0)
    resid = y[sel] - (level + slope * x)
    sd = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    return bg, max(sd, 1e-9)


def causal_onset(t: np.ndarray, net: np.ndarray, ok: np.ndarray, sd: float, t_lo: float, t_hi: float,
                 n_sigma: float = 3.0, n_consec: int = 3) -> tuple[int, int] | None:
    """First run of ``n_consec`` observed bins above ``n_sigma`` * sd within [t_lo, t_hi].

    Returns (onset index, detection index): the onset is the first bin of the
    run, but it is only *known* at the last one, which is what a live system
    could act on.
    """
    run = 0
    for i in np.flatnonzero((t >= t_lo) & (t <= t_hi)):
        if ok[i] and np.isfinite(net[i]) and net[i] > n_sigma * sd:
            run += 1
            if run == n_consec:
                return i - n_consec + 1, i
        else:
            run = 0
    return None


def hardness(net_hi: np.ndarray, net_lo: np.ndarray, ok: np.ndarray, i0: int, i1: int, dt: float,
             min_lo_counts: float = 200.0, min_hi_counts: float = 20.0) -> float:
    """Summed net counts hi / lo over bins [i0, i1); NaN when either is too faint."""
    i0, i1 = max(i0, 0), min(i1, net_lo.size)
    if i1 <= i0:
        return float("nan")
    m = ok[i0:i1] & np.isfinite(net_hi[i0:i1]) & np.isfinite(net_lo[i0:i1])
    hi = float(np.sum(net_hi[i0:i1][m])) * dt
    lo = float(np.sum(net_lo[i0:i1][m])) * dt
    if lo < min_lo_counts or hi < min_hi_counts:
        return float("nan")
    return hi / lo


def smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    """Centred running mean ignoring NaN (edges use what is available)."""
    x = np.asarray(x, float)
    v = np.where(np.isfinite(x), x, 0.0)
    w = np.isfinite(x).astype(float)
    ker = np.ones(k)
    num = np.convolve(v, ker, mode="same")
    den = np.convolve(w, ker, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def derivative(x: np.ndarray, dt: float, k: int = 3) -> np.ndarray:
    """d/dt of the smoothed series, per second (central differences)."""
    s = smooth(x, k)
    return np.gradient(s, dt)


def impulsive_index(net_sxr: np.ndarray, dt: float, i0: int, i1: int) -> int | None:
    """Index of the steepest soft X-ray rise in [i0, i1] -- the Neupert proxy for
    the impulsive phase when no hard X-rays are available."""
    if i1 <= i0:
        return None
    d = derivative(net_sxr, dt)[i0:i1 + 1]
    if not np.isfinite(d).any():
        return None
    return i0 + int(np.nanargmax(d))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def lagged_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[int, float, float]:
    """Best lag (bins) and its Pearson r, plus r at zero lag.

    A positive lag means ``y`` follows ``x`` (y[i + lag] pairs with x[i]).
    """
    best_lag, best_r = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r = pearson(x[:x.size - lag], y[lag:])
        else:
            r = pearson(x[-lag:], y[:y.size + lag])
        if np.isfinite(r) and r > best_r:
            best_lag, best_r = lag, r
    return best_lag, (float(best_r) if np.isfinite(best_r) else float("nan")), pearson(x, y)


def goes_minutes(goes_t: np.ndarray, goes_x: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Boolean selector of GOES 1-min records fully inside [t0, t1)."""
    return (goes_t >= t0) & (goes_t + 60.0 <= t1) & np.isfinite(goes_x)


def flare_record(t: np.ndarray, gl: np.ndarray, lo: np.ndarray, hi: np.ndarray, ok: np.ndarray,
                 flare: dict, goes_t: np.ndarray, goes_a: np.ndarray, goes_b: np.ndarray,
                 hxr: np.ndarray | None = None, hxr_ok: np.ndarray | None = None,
                 dt: float = 20.0, pre_s: float = 900.0, decision_s: float = 120.0) -> dict:
    """Diagnostics of one flare on a 20 s window around it.

    ``gl``/``lo``/``hi`` are SoLEXS rates (GOES-long analogue, 3-4 keV, 6-8 keV),
    ``flare`` holds GOES ``start``/``peak``/``end`` (Unix s) and ``peak_flux``.
    """
    rec: dict = {"status": "ok"}
    start, peak, end = flare["start"], flare["peak"], flare["end"]
    pre_lo, pre_hi = start - pre_s, start - 60.0

    obs_rise = ok[(t >= start) & (t <= peak)]
    rec["solexs_rise_coverage"] = float(obs_rise.mean()) if obs_rise.size else 0.0
    if rec["solexs_rise_coverage"] < 0.8:
        rec["status"] = "solexs_gap"
        return rec
    fits = [linear_background(t, y, ok, pre_lo, pre_hi) for y in (gl, lo, hi)]
    if any(f is None for f in fits):
        rec["status"] = "no_background"
        return rec
    (bg_gl, sd_gl), (bg_lo, _), (bg_hi, _) = fits
    n_gl, n_lo, n_hi = gl - bg_gl, lo - bg_lo, hi - bg_hi

    on = causal_onset(t, n_gl, ok, sd_gl, start - 600.0, peak)
    if on is None:
        rec["status"] = "no_onset"
        return rec
    i_on, i_det = on
    t_on = float(t[i_on])
    t_dec = t_on + decision_s
    i_dec = int(np.searchsorted(t, t_dec))          # first bin at/after the decision time
    i_end = int(np.searchsorted(t, min(end, peak + 600.0), side="right")) - 1
    seg = np.where(ok[i_on:i_end + 1], n_gl[i_on:i_end + 1], -np.inf)
    i_pk = i_on + int(np.argmax(seg))
    i_imp = impulsive_index(np.where(ok, n_gl, np.nan), dt, i_on, i_pk)

    rec.update({
        "t_onset": t_on, "detect_delay_s": float(t[i_det] - t_on),
        "onset_minus_goes_start_s": t_on - start,
        "t_peak_solexs": float(t[i_pk]),
        "t_impulsive": float(t[i_imp]) if i_imp is not None else float("nan"),
        "rise_min_goes": (peak - start) / 60.0,
        "onset_to_peak_min": (float(t[i_pk]) - t_on) / 60.0,
        "onset_to_impulsive_min": (float(t[i_imp]) - t_on) / 60.0 if i_imp is not None else float("nan"),
        "hardness_background": float(bg_hi[i_on] / bg_lo[i_on]) if bg_lo[i_on] > 0 else float("nan"),
        "hardness_onset": hardness(n_hi, n_lo, ok, i_on, i_dec, dt),
        "hardness_impulsive": (hardness(n_hi, n_lo, ok, i_imp - 3, i_imp + 3, dt)
                               if i_imp is not None else float("nan")),
        "hardness_peak": hardness(n_hi, n_lo, ok, i_pk - 3, i_pk + 3, dt),
        "log_peak_flux": float(np.log10(flare["peak_flux"])),
    })

    # --- GOES XRS short/long ratio, same background idea at 1-min cadence ---
    gb = linear_background(goes_t, goes_b, np.isfinite(goes_b), pre_lo, pre_hi, min_points=5)
    ga = linear_background(goes_t, goes_a, np.isfinite(goes_a), pre_lo, pre_hi, min_points=5)
    if gb is not None and ga is not None:
        na, nb = goes_a - ga[0], goes_b - gb[0]

        def ratio(t0, t1):
            m = goes_minutes(goes_t, goes_b, t0, t1) & np.isfinite(goes_a)
            s_b = float(np.sum(nb[m]))
            return float(np.sum(na[m]) / s_b) if m.any() and s_b > 0 else float("nan")

        rec["goes_ratio_onset"] = ratio(t_on, t_dec)
        rec["goes_ratio_peak"] = ratio(peak - 60.0, peak + 120.0)
    else:
        rec["goes_ratio_onset"] = rec["goes_ratio_peak"] = float("nan")

    # --- decision-time features: data before t_dec only ----------------------
    before = (t < t_dec) & ok
    span = min(60.0, decision_s / 2.0)     # "first" and "last" never overlap
    last = np.flatnonzero(before & (t >= t_dec - span))
    first = np.flatnonzero(before & (t >= t_on) & (t < t_on + span))
    now_net = float(np.mean(n_gl[last])) if last.size else float("nan")
    first_net = float(np.mean(n_gl[first])) if first.size else float("nan")
    gm = goes_minutes(goes_t, goes_b, t_dec - 120.0, t_dec)
    rec.update({
        "t_decision": t_dec,
        "dec_log_net_rate": float(np.log10(now_net)) if now_net > 0 else float("nan"),
        "dec_log_background_rate": float(np.log10(bg_gl[i_on])) if bg_gl[i_on] > 0 else float("nan"),
        "dec_rise_dex": (float(np.log10(now_net / first_net)) if now_net > 0 and first_net > 0
                         else float("nan")),
        "dec_goes_log_flux": float(np.log10(goes_b[gm][-1])) if gm.any() and goes_b[gm][-1] > 0 else float("nan"),
        "peak_after_decision": bool(peak > t_dec + 60.0),
    })

    # --- Neupert effect ------------------------------------------------------
    rec["hel1os"] = False
    if hxr is not None and hxr_ok is not None:
        win = (t >= t_on - 120.0) & (t <= float(t[i_pk]) + 120.0)
        cov = float(hxr_ok[win].mean()) if win.any() else 0.0
        pre = hxr_ok & np.isfinite(hxr) & (t >= pre_lo) & (t < pre_hi)
        if cov >= 0.8 and pre.sum() >= 8:
            base = float(np.median(hxr[pre]))
            hsd = max(float(1.4826 * np.median(np.abs(hxr[pre] - base))), 1e-9)
            n_hxr = np.where(hxr_ok, hxr - base, np.nan)
            dsxr = derivative(np.where(ok, n_gl, np.nan), dt)
            idx = np.flatnonzero(win)
            x, y = n_hxr[idx], dsxr[idx]
            lag, r_best, r0 = lagged_correlation(x, y, max_lag=6)
            i_hpk = idx[int(np.nanargmax(np.where(np.isfinite(x), x, -np.inf)))]
            early = hxr_ok & (t >= t_on) & (t < t_dec) & np.isfinite(n_hxr)
            rec.update({
                "hel1os": True, "hel1os_coverage": cov,
                "hxr_peak_sigma": float(np.nanmax(x) / hsd),
                "hxr_peak_net_rate": float(np.nanmax(x)),
                "neupert_r0": r0, "neupert_r_best": r_best, "neupert_lag_s": lag * dt,
                "hxr_peak_minus_impulsive_s": (float(t[i_hpk]) - float(t[i_imp])) if i_imp is not None
                else float("nan"),
                "dec_hxr_net_counts": float(np.sum(n_hxr[early]) * dt) if early.any() else float("nan"),
                "dec_hxr_sigma": float(np.max(n_hxr[early]) / hsd) if early.any() else float("nan"),
            })
    return rec


def load_goes_xrs(goes_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GOES-R XRS-A (0.5-4 A) and XRS-B (1-8 A) 1-min fluxes; flagged or
    non-positive samples are NaN. Returns (unix time, xrsa, xrsb)."""
    import h5py

    ts, as_, bs = [], [], []
    for p in sorted(Path(goes_dir).glob("*avg1m*.nc")):
        with h5py.File(p, "r") as f:
            t = f["time"][:].astype(np.float64) + GOES_EPOCH_UNIX
            a = f["xrsa_flux"][:].astype(np.float64)
            b = f["xrsb_flux"][:].astype(np.float64)
            a[(f["xrsa_flag"][:] != 0) | ~(a > 0)] = np.nan
            b[(f["xrsb_flag"][:] != 0) | ~(b > 0)] = np.nan
        ts.append(t)
        as_.append(a)
        bs.append(b)
    t = np.concatenate(ts)
    order = np.argsort(t, kind="stable")
    t, a, b = t[order], np.concatenate(as_)[order], np.concatenate(bs)[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    return t[keep], a[keep], b[keep]
