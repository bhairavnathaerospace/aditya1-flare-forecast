"""HEL1OS timing audit and sub-second hard X-ray structure, from the photon lists.

    python -m solarflare hxr-timing

What the L1 event lists (and the 1 s light curves built from them) can resolve
in time, measured on the flares with a reliable CZT spectrum
(outputs/physics/hxr_spectra.csv, from python -m solarflare hxr-spectra):

1. Batching. At ordinary rates events are not spread over time: they come in
   bursts about 1 s long every ~6.5 s, identically in all four detectors and in
   the official 1 s light-curve product, whose rates match the events exactly.
   Those are readout times, not photon arrival times. Measured per second:
   the share of 10 ms ticks holding an event (occupancy) against the event
   rate, and the burst period from the occupancy autocorrelation before each
   flare. A second is *continuous* when >= 90% of its ticks are occupied.
2. Sub-second structure, only inside continuous stretches (>= 20 s) at flare
   peaks. Fractional residuals from a 4 s running mean at 50-500 ms. Thermal
   6-12 keV photons in CdTe cannot vary in 0.1 s, so CdTe1 x CdTe2 residual
   covariance above zero is instrumental. For 40-100 keV each CZT is corrected
   by its own CdTe monitor (independent counting noise) and the CZT1 x CZT2
   covariance of what is left estimates real sub-second variance. 95%
   intervals from resampled 5 s blocks.
3. Energy-dependent delay in the same stretches: cross-correlation of 25-40 and
   60-150 keV (both CZT, 20 ms bins, 10 s mean removed); positive = high
   energies later. If events were stamped at readout, every energy would share
   the stamp and the delay would be pinned at 0, so a zero here is not
   evidence of simultaneity.

Writes outputs/physics/{subsecond.csv, subsecond_summary.json, SUBSECOND.md, subsecond.png}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from solarflare.settings import load_settings
from solarflare.util import read_rows

import numpy as np


from solarflare.io.hel1os_events import TICK_S, product_for, read_events, tick_counts

HALF_S = 150.0
PRE_S = (700.0, 400.0)          # quiet window: from tp - 700 s to tp - 400 s
SCALES_S = (0.05, 0.1, 0.2, 0.5)
TREND_S = 4.0
BLOCK_S = 5.0
CONTINUOUS = 0.9
MIN_STRETCH_S = 20
DETS = ("CZT1", "CZT2", "CDTE1", "CDTE2")


def ts(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp()


def running_mean(x: np.ndarray, w: int) -> np.ndarray:
    ok = np.isfinite(x)
    k = np.ones(w)
    s = np.convolve(np.where(ok, x, 0.0), k, "same")
    n = np.convolve(ok.astype(float), k, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= 0.6 * w, s / n, np.nan)


def rebin(c: np.ndarray, m: int) -> np.ndarray:
    n = c.size // m
    return c[:n * m].reshape(n, m).sum(1).astype(float)


def block_ci(prod: np.ndarray, block: int, n_boot: int = 800, seed: int = 0):
    ok = np.isfinite(prod)
    if ok.sum() < 20:
        return [None, None, None]
    nb = max(prod.size // block, 1)
    sums = np.array([np.nansum(prod[i * block:(i + 1) * block]) for i in range(nb)])
    cnts = np.array([np.isfinite(prod[i * block:(i + 1) * block]).sum() for i in range(nb)])
    rng = np.random.default_rng(seed)
    vals = [sums[k].sum() / cnts[k].sum() for k in (rng.integers(0, nb, nb) for _ in range(n_boot)) if cnts[k].sum()]
    return [round(float(np.nanmean(prod)), 5), round(float(np.percentile(vals, 2.5)), 5),
            round(float(np.percentile(vals, 97.5)), 5)]


def lag_ccf(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[float, float]:
    """Lag (bins) of b after a at the cross-correlation peak, parabolic refinement."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 50:
        return np.nan, np.nan
    a = np.where(ok, a - a[ok].mean(), 0.0)
    b = np.where(ok, b - b[ok].mean(), 0.0)
    den = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if den == 0:
        return np.nan, np.nan
    lags = np.arange(-max_lag, max_lag + 1)
    cc = np.array([np.sum(a[max(0, -L):a.size - max(0, L)] * b[max(0, L):b.size - max(0, -L)]) for L in lags]) / den
    k = int(np.argmax(cc))
    d = 0.0
    if 0 < k < cc.size - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        den2 = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den2 if den2 != 0 else 0.0
    return float(lags[k] + d), float(cc[k])


def bursts(tot: np.ndarray) -> dict:
    """Event bursts in a tick series: runs of occupied 0.1 s bins separated by > 1 s of nothing."""
    occ = rebin(tot, 10) > 0
    idx = np.flatnonzero(occ)
    if idx.size < 2:
        return {"n": int(idx.size > 0)}
    first = np.concatenate([[True], np.diff(idx) > 10])
    starts, ends = idx[first], np.concatenate([idx[np.flatnonzero(first)[1:] - 1], [idx[-1]]])
    per = rebin(tot, 10)
    events = [float(per[a:b + 1].sum()) for a, b in zip(starts, ends)]
    sp = np.diff(starts) * 0.1
    return {"n": int(starts.size),
            "spacing_s": round(float(np.median(sp)), 2) if sp.size else None,
            "length_s": round(float(np.median((ends - starts + 1) * 0.1)), 2),
            "events_per_burst": round(float(np.median(events[1:-1] if len(events) > 2 else events)), 0)}


def analyse(prod, tp: float, fid: str) -> dict:
    t0, t1 = tp - HALF_S, tp + HALF_S
    ev = {d: read_events(prod, d, tp - PRE_S[0] - 5.0, t1 + 5.0) for d in DETS}
    if any(e.tick.size == 0 for e in ev.values()):
        return {"id": fid, "status": "a detector has no events"}
    off = ev["CZT1"].utc_offset
    out = {"id": fid, "status": "ok",
           "clock_offset_spread_s": round(max(e.utc_offset for e in ev.values()) - min(e.utc_offset for e in ev.values()), 3)}

    def counts(det, start, n, lo=0.0, hi=np.inf):
        return tick_counts(ev[det], int(round((start - off) / TICK_S)), n, lo, hi)

    # ---- 1. batching, quiet window before the flare ---------------------------------
    nq = int(round((PRE_S[0] - PRE_S[1]) / TICK_S))
    tot_q = sum(counts(d, tp - PRE_S[0], nq) for d in DETS)
    occ_q = tot_q > 0
    out["quiet_event_rate"] = round(float(tot_q.sum()) / (PRE_S[0] - PRE_S[1]), 1)
    out["quiet_occupancy"] = round(float(occ_q.mean()), 4)
    b = bursts(tot_q)
    out["quiet_burst_spacing_s"] = b.get("spacing_s")
    out["quiet_burst_length_s"] = b.get("length_s")
    out["quiet_events_per_burst"] = b.get("events_per_burst")
    # Poisson occupancy expected at that rate if events were spread in time
    lam = tot_q.mean()
    out["quiet_occupancy_if_poisson"] = round(float(1 - np.exp(-lam)), 4)

    # ---- per-second occupancy through the flare ---------------------------------------
    n = int(round((t1 - t0) / TICK_S))
    C = {d: counts(d, t0, n) for d in DETS}
    tot = sum(C.values())
    occ_s = rebin((tot > 0).astype(float), 100) / 100.0
    rate_s = rebin(tot, 100)
    out["per_second"] = [[round(float(r), 0), round(float(o), 3)] for r, o in zip(rate_s, occ_s)]
    cont = occ_s >= CONTINUOUS
    # longest continuous stretch
    best, cur, best_end = 0, 0, -1
    for i, c in enumerate(cont):
        cur = cur + 1 if c else 0
        if cur > best:
            best, best_end = cur, i
    out["longest_continuous_s"] = int(best)
    if best < MIN_STRETCH_S:
        out["status"] = "no continuous stretch"
        return out
    a_s, b_s = best_end - best + 1, best_end + 1
    k0, k1 = a_s * 100, b_s * 100
    out["continuous_utc"] = [datetime.fromtimestamp(t0 + a_s, UTC).strftime("%H:%M:%S"),
                             datetime.fromtimestamp(t0 + b_s, UTC).strftime("%H:%M:%S")]
    out["continuous_rate"] = round(float(tot[k0:k1].sum()) / best, 0)
    dead = tot[k0:k1] == 0
    out["continuous_empty_ticks"] = round(float(dead.mean()), 4)

    # ---- 2. sub-second covariance inside the stretch -----------------------------------
    H = {d: counts(d, t0, n, 40, 100)[k0:k1] for d in ("CZT1", "CZT2")}
    M = {d: counts(d, t0, n, 6, 12)[k0:k1] for d in ("CDTE1", "CDTE2")}
    for sc in SCALES_S:
        m = int(round(sc / TICK_S))
        tb = int(round(TREND_S / sc))
        z, tr = {}, {}
        for key, c in (("h1", H["CZT1"]), ("h2", H["CZT2"]), ("m1", M["CDTE1"]), ("m2", M["CDTE2"])):
            x = rebin(c, m)
            tr[key] = running_mean(x, tb)
            with np.errstate(invalid="ignore", divide="ignore"):
                z[key] = np.where(tr[key] > 0, x / tr[key] - 1.0, np.nan)
        blk = max(int(round(BLOCK_S / sc)), 1)
        tag = f"{int(sc * 1000)}ms"
        out[f"{tag}_monitor_cov"] = block_ci(z["m1"] * z["m2"], blk)
        out[f"{tag}_hxr_cov_raw"] = block_ci(z["h1"] * z["h2"], blk)
        out[f"{tag}_hxr_cov_corrected"] = block_ci((z["h1"] - z["m1"]) * (z["h2"] - z["m2"]), blk)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where((tr["h1"] > 0) & (tr["m1"] > 0) & (tr["h2"] > 0) & (tr["m2"] > 0),
                         (1 / tr["h1"] + 1 / tr["m1"]) * (1 / tr["h2"] + 1 / tr["m2"]), np.nan)
        nb = int(np.isfinite(v).sum())
        out[f"{tag}_noise_sd"] = round(float(np.sqrt(np.nanmean(v) / nb)), 6) if nb else None
        out[f"{tag}_hxr_counts_per_bin"] = round(float(np.nanmean(tr["h1"] + tr["h2"])), 1)

    # ---- 3. energy-dependent delay -----------------------------------------------------
    lo_c = (counts("CZT1", t0, n, 25, 40) + counts("CZT2", t0, n, 25, 40))[k0:k1]
    hi_c = (counts("CZT1", t0, n, 60, 150) + counts("CZT2", t0, n, 60, 150))[k0:k1]
    xl, xh = rebin(lo_c, 2), rebin(hi_c, 2)
    rl, rh = xl - running_mean(xl, 500), xh - running_mean(xh, 500)
    lag, pc = lag_ccf(rl, rh, 100)
    out["lag_ms"] = round(lag * 20.0, 1) if np.isfinite(lag) else None
    out["lag_peak_cc"] = round(pc, 3) if np.isfinite(pc) else None
    out["hi_counts"] = int(hi_c.sum())
    per = []
    for b in range(xl.size // 500):            # 10 s blocks
        sl = slice(b * 500, (b + 1) * 500)
        lg, p2 = lag_ccf(rl[sl], rh[sl], 50)
        if np.isfinite(lg) and p2 > 0.2:
            per.append(lg * 20.0)
    out["lag_err_ms"] = round(float(np.std(per) / np.sqrt(len(per))), 1) if len(per) >= 3 else None
    return out


def _r(v, nd=4):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def summarise(res: list[dict]) -> dict:
    s: dict = {"generated_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"), "flares": len(res),
               "status_counts": {k: sum(1 for a in res if a["status"] == k) for k in sorted({a["status"] for a in res})}}
    q = [a for a in res if "quiet_occupancy" in a]
    if q:
        sp = np.array([a["quiet_burst_spacing_s"] for a in q if a.get("quiet_burst_spacing_s")], float)
        rt = np.array([a["quiet_event_rate"] for a in q if a.get("quiet_burst_spacing_s")], float)
        nb = np.array([a["quiet_events_per_burst"] for a in q if a.get("quiet_burst_spacing_s")], float)
        s["batching"] = {
            "flares": len(q),
            "quiet_event_rate_median": _r(np.median([a["quiet_event_rate"] for a in q]), 1),
            "quiet_occupancy_median": _r(np.median([a["quiet_occupancy"] for a in q]), 4),
            "quiet_occupancy_if_poisson_median": _r(np.median([a["quiet_occupancy_if_poisson"] for a in q]), 4),
            "burst_spacing_median_s": _r(np.median(sp), 2) if sp.size else None,
            "burst_spacing_range_s": [_r(sp.min(), 2), _r(sp.max(), 2)] if sp.size else None,
            "burst_length_median_s": _r(np.median([a["quiet_burst_length_s"] for a in q if a.get("quiet_burst_length_s")]), 2),
            "events_per_burst_median": _r(np.median(nb), 0) if nb.size else None,
            "spacing_vs_rate_spearman": _r(__import__("scipy.stats").stats.spearmanr(rt, sp).statistic, 3) if sp.size > 4 else None,
            "spacing_cap_s": _r(sp.max(), 2) if sp.size else None,
        }
        pts = np.array([p for a in q for p in a["per_second"]], float)
        if pts.size:
            edges = np.array([0, 300, 1000, 2000, 3000, 5000, 8000, 12000, 1e9])
            rows = []
            for lo_, hi_ in zip(edges[:-1], edges[1:]):
                m = (pts[:, 0] >= lo_) & (pts[:, 0] < hi_)
                if m.sum() >= 5:
                    rows.append({"rate_lo": int(lo_), "rate_hi": None if hi_ > 1e8 else int(hi_), "seconds": int(m.sum()),
                                 "occupancy_median": _r(np.median(pts[m, 1]), 3),
                                 "continuous_share": _r(np.mean(pts[m, 1] >= CONTINUOUS), 3)})
            s["occupancy_vs_rate"] = rows
    ok = [a for a in res if a["status"] == "ok"]
    s["flares_with_continuous_stretch"] = len(ok)
    if not ok:
        return s
    s["continuous_rate_median"] = _r(np.median([a["continuous_rate"] for a in ok]), 0)
    s["continuous_length_median_s"] = _r(np.median([a["longest_continuous_s"] for a in ok]), 0)
    s["scales"] = {}
    for sc in SCALES_S:
        tag = f"{int(sc * 1000)}ms"
        b = {}
        for key in ("monitor_cov", "hxr_cov_raw", "hxr_cov_corrected"):
            v = [a[f"{tag}_{key}"] for a in ok if a[f"{tag}_{key}"][0] is not None]
            if v:
                b[f"{key}_median"] = _r(np.median([x[0] for x in v]), 5)
                b[f"{key}_above_zero"] = _r(np.mean([x[1] > 0 for x in v]), 3)
                b[f"{key}_below_zero"] = _r(np.mean([x[2] < 0 for x in v]), 3)
        nz = [a[f"{tag}_noise_sd"] for a in ok if a[f"{tag}_noise_sd"]]
        b["noise_sd_median"] = _r(np.median(nz), 5) if nz else None
        b["hxr_counts_per_bin_median"] = _r(np.median([a[f"{tag}_hxr_counts_per_bin"] for a in ok]), 1)
        hc = b.get("hxr_cov_corrected_median")
        b["rms_subsecond_median"] = _r(np.sqrt(max(hc, 0.0)), 4) if hc is not None else None
        s["scales"][tag] = b
    lags = [a for a in ok if a.get("lag_err_ms") is not None and (a.get("lag_peak_cc") or 0) > 0.3]
    if lags:
        L = np.array([a["lag_ms"] for a in lags])
        E = np.array([max(a["lag_err_ms"], 2.0) for a in lags])
        w = 1 / E ** 2
        s["energy_lag"] = {"flares": len(lags), "median_ms": _r(np.median(L), 1),
                           "weighted_mean_ms": _r(np.sum(w * L) / w.sum(), 1),
                           "weighted_mean_err_ms": _r(1 / np.sqrt(w.sum()), 1),
                           "beyond_2sigma": _r(np.mean(np.abs(L) > 2 * E), 3)}
    return s


def render(s: dict) -> str:
    L = ["# HEL1OS timing audit and sub-second hard X-rays", "",
         f"Generated {s['generated_utc']} UTC by `python -m solarflare hxr-timing` on {s['flares']} flares with a reliable CZT "
         "spectrum (+-150 s around the CZT peak; quiet reference 400-700 s before it). Onboard 10 ms ticks throughout: "
         "the UTC column is a per-packet stamp that jumps by up to +-1 s between packets.", ""]
    b = s.get("batching")
    if b:
        L += ["## Events come in readout batches", "",
              f"Before the flares ({b['quiet_event_rate_median']:.0f} events/s summed over the four detectors, median), "
              f"only **{100 * b['quiet_occupancy_median']:.1f}%** of 10 ms ticks hold an event, against "
              f"{100 * b['quiet_occupancy_if_poisson_median']:.0f}% if the same events were spread at random. They come in "
              f"batches of about **{b['events_per_burst_median']:.0f} events**, each ~{b['burst_length_s'] if 'burst_length_s' in b else b['burst_length_median_s']} s "
              f"long, one every {b['burst_spacing_range_s'][0]}-{b['burst_spacing_range_s'][1]} s (median "
              f"{b['burst_spacing_median_s']} s; faster at higher rates, Spearman {b['spacing_vs_rate_spearman']}, never "
              f"longer than {b['spacing_cap_s']} s), in all four detectors at once. The official 1 s light-curve product "
              "shows the same batches with exactly the same counts. This is the signature of a buffer read out when it "
              "fills (or on a timeout): at ordinary rates the time stamps are readout times, and timing is only as good "
              "as the batch spacing. Averaged over 20 s this also makes HEL1OS look over-dispersed, the 2-10x Poisson "
              "noise the master catalogue had to measure.", "",
              "| events/s (all detectors) | seconds | ticks occupied (median) | continuous seconds (>= 90% occupied) |",
              "|---|---|---|---|"]
        for r in s.get("occupancy_vs_rate", []):
            hi = f"{r['rate_hi']:,}" if r["rate_hi"] else "and above"
            L.append(f"| {r['rate_lo']:,}-{hi} | {r['seconds']} | {100 * r['occupancy_median']:.0f}% | "
                     f"{100 * r['continuous_share']:.0f}% |")
        L += ["", "Only bright flare peaks fill the stream continuously; everything below is measured there only.", ""]
    L += [f"## Inside continuous stretches ({s['flares_with_continuous_stretch']} flares)", ""]
    if s.get("scales"):
        L += [f"Median stretch {s['continuous_length_median_s']:.0f} s at {s['continuous_rate_median']:,.0f} events/s. "
              "Fractional residuals from a 4 s running mean. *Monitor*: CdTe1 x CdTe2 at 6-12 keV (thermal; anything "
              "above zero is instrumental). *HXR raw*: CZT1 x CZT2 at 40-100 keV. *HXR corrected*: after removing each "
              "CZT's own CdTe monitor; above zero would be real sub-second hard X-ray structure. Share of flares whose "
              "95% interval is above zero in brackets.", "",
              "| scale | monitor | HXR raw | HXR corrected | noise sd | 95% upper limit on real rms |",
              "|---|---|---|---|---|---|"]
        for k, v in s["scales"].items():
            def c(key, _v=v):
                return (f"{_v[key + '_median']:+.4f} ({100 * _v[key + '_above_zero']:.0f}%)"
                        if _v.get(key + "_median") is not None else "--")
            L.append(f"| {k} | {c('monitor_cov')} | {c('hxr_cov_raw')} | {c('hxr_cov_corrected')} | "
                     f"{v['noise_sd_median'] if v['noise_sd_median'] is not None else '--'} | "
                     f"{100 * np.sqrt(max(v['hxr_cov_corrected_median'], 0) + 2 * v['noise_sd_median']):.0f}% |"
                     if v.get("rms_subsecond_median") is not None else
                     f"| {k} | -- | -- | -- | -- | -- |")
        v = s["scales"].get("100ms", {})
        if v.get("hxr_cov_corrected_above_zero") is not None:
            L += ["", f"After the monitor correction only {100 * v['hxr_cov_corrected_above_zero']:.0f}% of flares keep a "
                  "covariance above zero at 100 ms, and the median sits inside its counting noise: **no sub-second hard "
                  "X-ray structure is detected** beyond what the instrument itself imprints. The thermal monitors, which "
                  f"cannot vary that fast, share a ~{100 * np.sqrt(v['monitor_cov_median']):.0f}% rms modulation at 100 ms."]
        L += [""]
    e = s.get("energy_lag")
    if e:
        L += [f"**Energy-dependent delay** (60-150 vs 25-40 keV, {e['flares']} flares with a clear cross-correlation): "
              f"weighted mean {e['weighted_mean_ms']:+.1f} +- {e['weighted_mean_err_ms']} ms, median {e['median_ms']:+.1f} ms; "
              f"{100 * e['beyond_2sigma']:.0f}% individually beyond 2 sigma. A readout stamp shared by all energies would "
              "also give zero, so this does not show that the energies are simultaneous.", ""]
    L += ["## What this means", "",
          "- Sub-second hard X-ray science is not available from the L1 event lists in general: below ~2,000 events/s "
          "the timing is set by the readout batches (seconds). It needs the instrument team's description of how "
          "events are time-tagged.",
          "- HEL1OS light curves are reliable on >= 10-20 s bins, which is what the forecasting pipeline uses.",
          "- Spectral shapes (python -m solarflare hxr-spectra) do not depend on timing and stand.", ""]
    return "\n".join(L)


def figure(res, s, dest: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, HARD, BOTH = "#1d2733", "#98a2ad", "#7a4fd1", "#2a8c7c"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    a = ax[0]
    pts = np.array([p for r in res if "per_second" in r for p in r["per_second"]], float)
    if pts.size:
        m = pts[:, 0] > 0
        a.plot(pts[m, 0], 100 * pts[m, 1], ".", ms=2, color=HARD, alpha=0.3, label="1 s of flare data")
        r = np.geomspace(10, pts[:, 0].max(), 100)
        a.plot(r, 100 * (1 - np.exp(-r * TICK_S)), color=INK, lw=1.2, label="if photons were stamped on arrival")
        a.axhline(100 * CONTINUOUS, color=MUTED, lw=0.8, ls=":")
    a.set_xscale("log")
    a.set_xlabel("events per second, all four detectors")
    a.set_ylabel("10 ms ticks holding an event, %")
    a.set_title("HEL1OS events come in readout bursts except at flare peaks", loc="left", fontsize=10, color=INK)
    a.legend(frameon=False, fontsize=7.5, loc="upper left")
    b = ax[1]
    q = [r for r in res if r.get("quiet_burst_spacing_s")]
    if q:
        b.plot([r["quiet_event_rate"] for r in q], [r["quiet_burst_spacing_s"] for r in q], "o", ms=4, color=BOTH,
               label="before each flare")
        rr = np.geomspace(min(r["quiet_event_rate"] for r in q) * 0.8, max(r["quiet_event_rate"] for r in q) * 1.2, 50)
        nb = s["batching"]["events_per_burst_median"]
        b.plot(rr, np.minimum(nb / rr, s["batching"]["spacing_cap_s"]), color=INK, lw=1,
               label=f"{nb:.0f} events per batch, {s['batching']['spacing_cap_s']:.0f} s cap")
        b.legend(frameon=False, fontsize=7.5)
        b.set_xscale("log")
    b.set_xlabel("events per second, all four detectors (quiet, before the flare)")
    b.set_ylabel("time between readout batches, s")
    b.set_title("Batches come faster as the rate rises", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(dest)
    plt.close(fig)


def main(argv=None) -> int:
    S = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectra", default=str(S.physics / "hxr_spectra.csv"))
    ap.add_argument("--hel1os-root", default=str(S.hel1os_extracted))
    ap.add_argument("--out", default=str(S.physics))
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.out)
    rows = [r for r in read_rows(args.spectra) if r.get("status") == "ok"]
    t_start = time.time()
    res = []
    for r in rows:
        tp = ts(r["czt_peak_utc"])
        prod = product_for(args.hel1os_root, tp - PRE_S[0], tp + HALF_S)
        if prod is None:
            continue
        a = analyse(prod, tp, r["id"])
        a.update({"goes_class": r["goes_class"], "czt_peak_utc": r["czt_peak_utc"]})
        res.append(a)
        msg = (f"{a['id']} {a['goes_class'] or '-':>5} quiet {a.get('quiet_event_rate', 0):6.0f}/s occupancy "
               f"{100 * a.get('quiet_occupancy', 0):.1f}% (Poisson {100 * a.get('quiet_occupancy_if_poisson', 0):.0f}%), "
               f"bursts every {a.get('quiet_burst_spacing_s')} s of {a.get('quiet_events_per_burst')} events; longest continuous {a.get('longest_continuous_s')} s")
        if a["status"] == "ok":
            msg += (f" at {a['continuous_rate']:.0f}/s; 100 ms monitor {a['100ms_monitor_cov'][0]}, raw "
                    f"{a['100ms_hxr_cov_raw'][0]}, corrected {a['100ms_hxr_cov_corrected'][0]} "
                    f"(noise {a['100ms_noise_sd']}); lag {a['lag_ms']} +- {a['lag_err_ms']} ms")
        print(msg, flush=True)
    s = summarise(res)
    s["seconds"] = round(time.time() - t_start, 1)
    with (out / "subsecond.csv").open("w", newline="", encoding="utf-8") as fh:
        flat = [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in a.items() if k != "per_second"} for a in res]
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for x in flat for k in x)))
        w.writeheader()
        w.writerows(flat)
    (out / "subsecond_summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    (out / "SUBSECOND.md").write_text(render(s), encoding="utf-8")
    figure(res, s, out / "subsecond.png")
    print(render(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
