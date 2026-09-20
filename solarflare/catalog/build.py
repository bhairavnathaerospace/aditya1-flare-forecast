"""Master flare catalogue: SoLEXS and HEL1OS detected independently, merged, checked against GOES.

    python -m solarflare catalog

1. SoLEXS: every archive day on one 1-minute axis, converted to GOES flux with
   the training-period calibration, run through the SWPC-style rule
   (solarflare/catalog.py). Each flare gets a class from SoLEXS alone.
2. HEL1OS: CdTe 5-20 keV through the same rule, CZT 20-40 keV through the
   coincidence burst detector, higher CZT bands attached.
3. Merge into one catalogue: soft+hard / soft only / hard only, with whether
   the other instrument was observing.
4. Validate against the GOES-18 flare list (never used for detection):
   recall by GOES class for each detector and combined, precision, false
   alarms per day, class accuracy, alert time before the GOES peak; the same
   again on the model's test period alone, which the flux calibration never saw. The rule run on GOES's own flux gives the ceiling.

Writes <out>/master_catalog.csv, <out>/catalog_summary.json and <out>/CATALOG.md.
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

import numpy as np


from solarflare.catalog.detect import (
    HARD_BEFORE_SOFT_S, PiecewiseCalibration, as_row, attach_bands, hxr_bursts, merge, noaa_events,
    flux_class, rise_events_as_bursts, solexs_to_goes_flux, to_minutes, trailing_noise,
)
from solarflare.config import PreprocessConfig
from solarflare.io.goes import load_goes
from solarflare.preprocess.cache import load_cached
from solarflare.preprocess.timeline import stitch

DT = 20.0
#: The trained model's split (outputs/model/reports/data_meta.json), set in main():
#: the flux calibration is fitted before TRAIN_END, scores are also given from TEST_START.
TRAIN_END = float("nan")
TEST_START = float("nan")
PEAK_MATCH_S = 300.0
CLASSES = ("B", "C", "M", "X")
CZT_BANDS = (("20_40", True), ("40_60", False), ("60_80", False), ("80_150", False))


def utc(x) -> str:
    if x is None or not np.isfinite(x):
        return ""
    return datetime.fromtimestamp(float(x), UTC).strftime("%Y-%m-%d %H:%M:%S")


class Coverage:
    """Which minutes an instrument observed; ``observing`` asks about an interval."""

    def __init__(self, minutes: np.ndarray):
        self.m = np.unique(np.asarray(minutes, dtype=np.float64))

    def fraction(self, t0: float, t1: float) -> float:
        a = np.floor(t0 / 60.0) * 60.0
        b = np.floor(t1 / 60.0) * 60.0
        n = int((b - a) / 60.0) + 1
        got = np.searchsorted(self.m, b, side="right") - np.searchsorted(self.m, a, side="left")
        return float(got) / max(n, 1)

    def observing(self, t0: float, t1: float) -> bool:
        return self.fraction(t0, t1) >= 0.5

    def days(self, t0: float = -np.inf, t1: float = np.inf) -> float:
        return float(((self.m >= t0) & (self.m < t1)).sum()) / 1440.0


#: Hours that PRADAN day files copy from the previous day (python -m solarflare quality).
DUPLICATES = load_settings().copied_days


def duplicate_mask(t: np.ndarray, path: Path = DUPLICATES) -> np.ndarray:
    """True for samples inside a copied interval; all False if the list does not exist."""
    bad = np.zeros(t.size, bool)
    if path.exists():
        for d in json.loads(path.read_text("utf-8"))["duplicates"]:
            for a, b in d["intervals_unix"]:
                bad |= (t >= a) & (t < b)
    return bad


def load_solexs(cache: Path):
    entries = json.loads((cache / "manifest.json").read_text("utf-8"))
    ts, rs, his, oks = [], [], [], []
    for e in entries:
        if e["source"]["kind"] != "solexs" or e.get("status") != "ok":
            continue
        with np.load(cache / f"{e['key']}.npz") as z:
            names = [str(n) for n in z["names"]]
            v = z["values"][:, names.index("slx_goes_long")]
            hi = z["values"][:, names.index("slx_6_8keV")] + z["values"][:, names.index("slx_8_12keV")]
            ts.append(z["time_unix"].astype(np.float64))
            rs.append(v.astype(np.float64))
            his.append(hi.astype(np.float64))
            oks.append((z["coverage"] > 0) & np.isfinite(v))
    t = np.concatenate(ts)
    r = np.concatenate(rs)
    hi = np.concatenate(his)
    ok = np.concatenate(oks)
    order = np.argsort(t, kind="stable")
    t, r, hi, ok = t[order], r[order], hi[order], ok[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, r, hi, ok = t[keep], r[keep], hi[keep], ok[keep]
    return entries, t, r, hi, ok & ~duplicate_mask(t)


def hot_excess(t: np.ndarray, hi: np.ndarray, ok: np.ndarray, tp: float) -> float | None:
    """Significance of a SoLEXS 6-12 keV rise within 2 min of ``tp``, against
    the median and measured scatter of the 30-10 min before it."""
    a, b = np.searchsorted(t, [tp - 1800.0, tp + 121.0])
    tt, hh, kk = t[a:b], hi[a:b], ok[a:b]
    pre = (tt < tp - 600.0) & kk
    at = (tt >= tp - 120.0) & kk
    if pre.sum() < 10 or at.sum() < 3:
        return None
    base = float(np.median(hh[pre]))
    sd = max(1.4826 * float(np.median(np.abs(np.diff(hh[pre])))) / np.sqrt(2.0),
             np.sqrt(max(base, 0.05) / DT))
    return (float(hh[at].max()) - base) / sd


def hel1os_events(entries, cache: Path, hard_sigma: float):
    """HEL1OS events and the minutes HEL1OS observed, timeline by timeline."""
    timelines = stitch(load_cached(entries, cache, "hel1os"), DT, 21600.0)
    events, observed, n_cdte, n_czt = [], [], 0, 0
    for h in timelines:
        names = list(h.names)

        def col(n, _h=h, _names=names):
            return _h.values[:, _names.index(n)]

        c1, c2 = col("hls_cdte1_cov"), col("hls_cdte2_cov")
        r1, r2 = col("hls_cdte1_5_20keV"), col("hls_cdte2_5_20keV")
        both = (c1 > 0) & (c2 > 0) & np.isfinite(r1) & np.isfinite(r2)
        tm, rm, cm, vm = to_minutes(h.time_unix, r1 + r2, both, DT)
        # CdTe is over-dispersed: guard the rise with its measured scatter,
        # never less than counting noise
        noise = np.fmax(trailing_noise(tm, rm, vm), np.sqrt(np.maximum(rm, 0.0) / 60.0))
        cdte = noaa_events(tm, rm, vm, sigma=hard_sigma, noise=noise)
        n_cdte += len(cdte)
        observed.append(tm[vm])

        k1, k2 = col("hls_czt1_cov"), col("hls_czt2_cov")
        others = []
        for band, standalone in CZT_BANDS:
            b = hxr_bursts(h.time_unix, [col(f"hls_czt1_{band}keV"), col(f"hls_czt2_{band}keV")],
                           [k1, k2], DT, band=f"czt_{band}", names=["czt1", "czt2"])
            if band == "20_40":
                n_czt += len(b)
            others.append((b, standalone))
        czt_on = ((k1 > 0) | (k2 > 0))
        tmz, _, _, vz = to_minutes(h.time_unix, np.ones(h.time_unix.size), czt_on, DT)
        observed.append(tmz[vz])
        events += attach_bands(rise_events_as_bursts(cdte, "cdte_5_20"), others)
    events.sort(key=lambda e: e.start_unix)
    return events, Coverage(np.concatenate(observed) if observed else np.zeros(0)), len(timelines), n_cdte, n_czt


def nearest(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each a, the index of the nearest b and the distance."""
    if b.size == 0:
        return np.full(a.size, -1), np.full(a.size, np.inf)
    i = np.clip(np.searchsorted(b, a), 1, max(b.size - 1, 1))
    lo = np.clip(i - 1, 0, b.size - 1)
    use_lo = np.abs(a - b[lo]) <= np.abs(a - b[np.minimum(i, b.size - 1)])
    j = np.where(use_lo, lo, np.minimum(i, b.size - 1))
    return j, np.abs(a - b[j])


def rate(num, den) -> float | None:
    return None if den == 0 else round(float(num) / float(den), 4)


def main(argv=None) -> int:
    S = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(S.cache))
    ap.add_argument("--goes-dir", default=str(S.goes_dir))
    ap.add_argument("--out", default=str(S.catalog))
    ap.add_argument("--run-dir", default=None, help="trained run whose split to use (default: the final model)")
    ap.add_argument("--calibration", choices=("piecewise", "powerlaw"), default="piecewise",
                    help="SoLEXS -> GOES flux map, both fitted on the training period only")
    ap.add_argument("--no-peak-correction", action="store_true",
                    help="classify from the minute-level calibration alone")
    ap.add_argument("--hard-match", choices=("rise", "peak", "end"), default="rise",
                    help="a HEL1OS event counts for a GOES flare if it peaks from 5 min before the GOES "
                         "start to 5 min after the GOES peak (peak) or to the GOES end (end)")
    ap.add_argument("--shift-hard-s", type=float, default=0.0,
                    help="control run: move every HEL1OS event by this many seconds to measure "
                         "chance coincidences")
    ap.add_argument("--hard-sigma", type=float, default=5.0,
                    help="noise guard for the CdTe rise rule, in measured standard deviations")
    args = ap.parse_args(argv)
    global TRAIN_END, TEST_START
    split = S.split_dates(Path(args.run_dir) if args.run_dir else None)
    TRAIN_END, TEST_START = split["train_end"], split["test_start"]
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)
    pre = PreprocessConfig()
    t_start = time.time()

    # ---- 1. SoLEXS -------------------------------------------------------
    entries, t, r, hi, ok = load_solexs(cache)
    tm, rm, cm, vm = to_minutes(t, r, ok, DT)
    truth = load_goes(Path(args.goes_dir))
    gmin = truth.flux_on_grid(tm)
    fit_on = vm & (tm <= TRAIN_END) & np.isfinite(gmin) & (gmin > 0) & (rm > 0)
    cal = PiecewiseCalibration.fit(np.log10(rm[fit_on]), np.log10(gmin[fit_on]))
    if args.calibration == "piecewise":
        flux = cal(rm)
        cal_text = f"piecewise, {len(cal.x)} knots, fitted on {int(fit_on.sum())} training minutes"
    else:
        flux = solexs_to_goes_flux(rm, pre.flux_anchor_intercept, pre.flux_anchor_slope)
        cal_text = (f"log10 F = {pre.flux_anchor_intercept} + {pre.flux_anchor_slope} log10 rate "
                    f"(training period)")
    soft = noaa_events(tm, flux, vm, counts=cm)
    soft_cov = Coverage(tm[vm])
    print(f"SoLEXS: {soft_cov.days():.1f} observed days, {len(soft)} flares")

    # ---- 2. HEL1OS -------------------------------------------------------
    hard, hard_cov, n_tl, n_cdte, n_czt = hel1os_events(entries, cache, args.hard_sigma)
    if args.shift_hard_s:
        for h in hard:
            h.start_unix += args.shift_hard_s
            h.peak_unix += args.shift_hard_s
            h.end_unix += args.shift_hard_s
            h.detect_unix += args.shift_hard_s
            h.peak_time = {k: v + args.shift_hard_s for k, v in h.peak_time.items()}
        hard.sort(key=lambda h: h.start_unix)
    print(f"HEL1OS: {n_tl} timelines, {hard_cov.days():.1f} observed days, {len(hard)} events "
          f"({n_cdte} CdTe 5-20 keV rises, {n_czt} CZT 20-40 keV bursts)")

    # ---- 3. merge --------------------------------------------------------
    master = merge(soft, hard, hard_cov.observing, soft_cov.observing)

    # ---- 4. GOES ---------------------------------------------------------
    t0, t1 = float(tm[0]), float(tm[-1]) + 60.0
    G = [g for g in truth.flares if t0 <= g.peak_unix <= t1 and g.goes_class[:1] in CLASSES]
    gpk = np.array([g.peak_unix for g in G])
    gcls = np.array([g.goes_class[:1] for g in G])

    # Peak-level class correction. Flare peaks are hotter than the average
    # minute the flux calibration was fitted on, and SoLEXS's band gains more
    # from hot plasma than GOES 1-8 A does, so peaks read high (B +0.13 dex,
    # C +0.07 dex). Refit on training-period flares only: SoLEXS peak -> GOES
    # peak, matched within 5 min.
    j0, d0 = nearest(np.array([e.peak_unix for e in soft]), gpk)
    tr = [(e.peak_flux, G[j0[k]].peak_flux) for k, e in enumerate(soft)
          if d0[k] <= PEAK_MATCH_S and e.peak_unix <= TRAIN_END]
    peak_cal = PiecewiseCalibration.fit(np.log10([a for a, _ in tr]), np.log10([b for _, b in tr]),
                                        width=0.2, min_n=20)
    for e in soft:
        e.peak_flux_minute = e.peak_flux
        if not args.no_peak_correction:
            e.peak_flux = float(peak_cal(np.array([e.peak_flux]))[0])
            e.goes_class = flux_class(e.peak_flux)

    # GOES validity per minute, for "could GOES have listed it"
    gt, gx = truth.time_unix, truth.xrsb
    goes_cov = Coverage(gt[np.isfinite(gx)])

    speaks = np.array([e.peak_unix for e in soft])
    j_soft, d_soft = nearest(gpk, speaks)
    hpk = np.array([h.peak_unix for h in hard])
    hst = np.array([h.start_unix for h in hard])

    def hard_window_end(g) -> float:
        return g.end_unix if args.hard_match == "end" else g.peak_unix + PEAK_MATCH_S

    def hard_for(g, shift: float = 0.0):
        """The HEL1OS event for a GOES flare: it peaks from 5 min before the
        flare start to 5 min after its peak and (``rise``) begins during the
        rise. ``shift`` moves every HEL1OS event, for the chance control."""
        pk, st = hpk + shift, hst + shift
        lo = np.searchsorted(pk, g.start_unix - HARD_BEFORE_SOFT_S, side="left")
        hi = np.searchsorted(pk, hard_window_end(g), side="right")
        cand = [k for k in range(lo, hi)
                if args.hard_match != "rise"
                or g.start_unix - HARD_BEFORE_SOFT_S <= st[k] <= g.peak_unix]
        if not cand:
            return None
        return min(cand, key=lambda k: abs(pk[k] - g.peak_unix))

    def build_per_flare(shift: float = 0.0) -> list[dict]:
        out_ = []
        for gi, g in enumerate(G):
            so = soft_cov.observing(g.start_unix, g.peak_unix)
            ho = hard_cov.observing(g.start_unix, g.peak_unix)
            sf = bool(d_soft[gi] <= PEAK_MATCH_S)
            hk = hard_for(g, shift)
            hf = hk is not None
            s_det = soft[j_soft[gi]].detect_unix if sf else np.nan
            h_det = hard[hk].detect_unix + shift if hf else np.nan
            out_.append({"cls": gcls[gi], "test": g.peak_unix >= TEST_START, "so": so, "ho": ho,
                         "sf": sf and so, "hf": hf and ho, "s_det": s_det, "h_det": h_det,
                         "peak": g.peak_unix, "start": g.start_unix,
                         "nonthermal": bool(hf and ho and "czt_20_40" in hard[hk].bands)})
        return out_

    per_flare = build_per_flare()
    # Chance control: HEL1OS events moved 2 h either way still land in some
    # flare windows; that share of any "combined" gain is coincidence.
    controls = [build_per_flare(s_) for s_ in (-7200.0, 7200.0)]

    def recall_block(test_only: bool, pf: list[dict] | None = None) -> dict:
        rows = [p for p in (per_flare if pf is None else pf) if (p["test"] or not test_only)]
        res = {}
        for c in CLASSES:
            rc = [p for p in rows if p["cls"] == c]
            s_obs = [p for p in rc if p["so"]]
            h_obs = [p for p in rc if p["ho"]]
            both = [p for p in rc if p["so"] and p["ho"]]
            res[c] = {
                "soft_n": len(s_obs), "soft_recall": rate(sum(p["sf"] for p in s_obs), len(s_obs)),
                "hard_n": len(h_obs), "hard_recall": rate(sum(p["hf"] for p in h_obs), len(h_obs)),
                "hard_nonthermal_fraction": rate(sum(p["nonthermal"] for p in h_obs), len(h_obs)),
                "both_observed_n": len(both),
                "both_soft_only_recall": rate(sum(p["sf"] for p in both), len(both)),
                "both_combined_recall": rate(sum(p["sf"] or p["hf"] for p in both), len(both)),
            }
        if pf is None:
            ctl = [recall_block(test_only, c_) for c_ in controls]
            for c in CLASSES:
                for k in ("hard_recall", "both_combined_recall"):
                    v = [x[c][k] for x in ctl if x[c][k] is not None]
                    res[c][f"{k}_chance"] = round(float(np.mean(v)), 4) if v else None
        return res

    def lead_block(test_only: bool) -> dict:
        """Alert time before the GOES peak, per class, for flares both instruments saw."""
        res = {}
        for c in CLASSES:
            rows = [p for p in per_flare if p["cls"] == c and p["so"] and p["ho"] and (p["test"] or not test_only)]
            s = np.array([p["peak"] - p["s_det"] for p in rows if p["sf"]]) / 60.0
            comb = np.array([p["peak"] - np.nanmin([p["s_det"], p["h_det"]])
                             for p in rows if p["sf"] or p["hf"]]) / 60.0
            gain = np.array([p["s_det"] - p["h_det"] for p in rows if p["sf"] and p["hf"]]) / 60.0
            res[c] = {
                "soft_median_min_before_peak": None if s.size == 0 else round(float(np.median(s)), 2),
                "soft_alert_before_peak": rate((s > 0).sum(), s.size),
                "combined_median_min_before_peak": None if comb.size == 0 else round(float(np.median(comb)), 2),
                "combined_alert_before_peak": rate((comb > 0).sum(), comb.size),
                "hard_first_fraction": rate((gain > 0).sum(), gain.size),
                "hard_first_median_gain_min": None if gain.size == 0 else round(float(np.median(gain)), 2),
                "n": int(comb.size),
            }
        return res

    # soft precision / false alarms: SoLEXS flares while GOES was valid
    def soft_precision(test_only: bool) -> dict:
        j, d = nearest(speaks, gpk)
        res = {}
        observed_days = soft_cov.days(TEST_START if test_only else -np.inf)
        for floor in ("B", "C", "M"):
            sel = [k for k, e in enumerate(soft)
                   if e.goes_class[:1] in CLASSES[CLASSES.index(floor):]
                   and (e.peak_unix >= TEST_START or not test_only)
                   and goes_cov.observing(e.peak_unix - 300, e.peak_unix + 300)]
            hit = [k for k in sel if d[k] <= PEAK_MATCH_S]
            res[f">={floor}"] = {"n": len(sel), "precision": rate(len(hit), len(sel)),
                                 "unmatched_per_day": round((len(sel) - len(hit)) / max(observed_days, 1e-9), 3)}
        return res

    def class_block(test_only: bool, minute: bool = False) -> dict:
        j, d = nearest(speaks, gpk)
        pairs = [(e, G[j[k]]) for k, e in enumerate(soft)
                 if d[k] <= PEAK_MATCH_S and (e.peak_unix >= TEST_START or not test_only) and e.goes_class]
        if not pairs:
            return {}
        pf = [(e.peak_flux_minute if minute else e.peak_flux) for e, _ in pairs]
        dl = np.array([np.log10(f) - np.log10(g.peak_flux) for f, (_, g) in zip(pf, pairs)])
        letters = [(flux_class(f)[:1], g.goes_class[:1]) for f, (_, g) in zip(pf, pairs)]
        conf = {gc: {sc: sum(1 for a, b in letters if a == sc and b == gc) for sc in ("A",) + CLASSES}
                for gc in CLASSES}
        by = {}
        for c in CLASSES:
            m = np.array([b == c for _, b in letters])
            if m.any():
                by[c] = {"n": int(m.sum()), "letter_agreement": rate(sum(a == b for a, b in np.array(letters)[m]), m.sum()),
                         "median_abs_dex": round(float(np.median(np.abs(dl[m]))), 3),
                         "bias_dex": round(float(np.median(dl[m])), 3)}
        return {"n": len(pairs), "letter_agreement": rate(sum(a == b for a, b in letters), len(letters)),
                "median_abs_dex": round(float(np.median(np.abs(dl))), 3),
                "bias_dex": round(float(np.median(dl)), 3),
                "within_factor_1.5": rate((np.abs(dl) <= np.log10(1.5)).sum(), dl.size),
                "by_goes_class": by, "confusion_goes_rows_solexs_cols": conf}

    def hard_precision(test_only: bool) -> dict:
        gs = np.array([g.start_unix for g in G])
        ge = np.array([hard_window_end(g) for g in G])
        sel = [h for h in hard if (h.peak_unix >= TEST_START or not test_only)
               and goes_cov.observing(h.peak_unix - 300, h.peak_unix + 300)]
        def matched(h):
            k = np.flatnonzero((gs - HARD_BEFORE_SOFT_S <= h.peak_unix) & (h.peak_unix <= ge))
            return k.size > 0
        hit = [h for h in sel if matched(h)]
        nt = [h for h in sel if "czt_20_40" in h.bands]
        nt_hit = [h for h in nt if matched(h)]
        return {"n": len(sel), "matched_goes_flare": rate(len(hit), len(sel)),
                "nonthermal_n": len(nt), "nonthermal_matched_goes_flare": rate(len(nt_hit), len(nt))}

    # ceiling: the same rule on GOES's own 1-minute flux over the same period
    mg = (gt >= t0) & (gt <= t1)
    gi_ = np.rint((gt[mg] - gt[mg][0]) / 60.0).astype(np.int64)
    gtm = gt[mg][0] + 60.0 * np.arange(gi_[-1] + 1)
    gfm = np.full(gtm.size, np.nan)
    gfm[gi_] = gx[mg]
    gev = noaa_events(gtm, gfm, np.isfinite(gfm))
    gep = np.array([e.peak_unix for e in gev])
    _, dg = nearest(gpk, gep)
    _, dgp = nearest(gep, gpk)
    ceiling = {"n_events": len(gev), "n_goes": len(G),
               "recall": {c: rate((dg[gcls == c] <= PEAK_MATCH_S).sum(), (gcls == c).sum()) for c in CLASSES},
               "precision": rate((dgp <= PEAK_MATCH_S).sum(), dgp.size)}

    # master rows with GOES match
    rows = []
    gs_arr = np.array([g.start_unix for g in G])
    ge_arr = np.array([hard_window_end(g) for g in G])
    for m in master:
        r = as_row(m)
        if m.soft is not None:
            k, dd = nearest(np.array([m.soft.peak_unix]), gpk)
            k = int(k[0]) if dd[0] <= PEAK_MATCH_S else -1
        else:
            cand = np.flatnonzero((gs_arr - HARD_BEFORE_SOFT_S <= m.hard.peak_unix) & (m.hard.peak_unix <= ge_arr))
            k = int(cand[np.argmin(np.abs(gpk[cand] - m.hard.peak_unix))]) if cand.size else -1
        g = G[k] if k >= 0 else None
        row = {"id": f"AL1-{datetime.fromtimestamp(m.start_unix, UTC):%Y%m%d-%H%M}",
               "origin": m.origin,
               "start_utc": utc(m.start_unix), "peak_utc": utc(m.peak_unix),
               "end_utc": utc(m.soft.end_unix if m.soft else m.hard.end_unix),
               "alert_utc": utc(m.detect_unix),
               "class_solexs": m.soft.goes_class if m.soft else "",
               "peak_flux_solexs_Wm2": f"{m.soft.peak_flux:.3e}" if m.soft else "",
               "peak_flux_minute_calibration_Wm2": f"{m.soft.peak_flux_minute:.3e}" if m.soft else "",
               "soft_alert_utc": utc(m.soft.detect_unix) if m.soft else "",
               "soft_end_reason": m.soft.end_reason if m.soft else "",
               "hard_bands": r.get("hard_bands", ""),
               "hard_alert_utc": utc(m.hard.detect_unix) if m.hard else "",
               "hard_peak_utc": utc(m.hard.peak_unix) if m.hard else "",
               "hard_peak_cdte_5_20_cps": round(r["hard_excess_cdte_5_20"], 1) if "hard_excess_cdte_5_20" in r else "",
               "hard_peak_czt_20_40_cps": round(r["hard_excess_czt_20_40"], 1) if "hard_excess_czt_20_40" in r else "",
               "hard_peak_czt_20_40_utc": utc(r.get("hard_peak_unix_czt_20_40")) if "hard_peak_unix_czt_20_40" in r else "",
               "max_energy_keV": max([int(b.split("_")[-1]) for b in m.hard.bands]) if m.hard else "",
               "n_hard": m.n_hard,
               "hard_observed": "" if m.hard_observed is None else int(m.hard_observed),
               "soft_observed": "" if m.soft_observed is None else int(m.soft_observed),
               "goes_class": g.goes_class if g else "",
               "goes_peak_utc": utc(g.peak_unix) if g else "",
               "class_error_dex": round(float(np.log10(m.soft.peak_flux) - np.log10(g.peak_flux)), 3)
               if (g and m.soft) else "",
               "alert_min_before_goes_peak": round((g.peak_unix - m.detect_unix) / 60.0, 1) if g else "",
               "test_period": int(m.peak_unix >= TEST_START)}
        rows.append(row)

    with (out / "master_catalog.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    origin_counts = {o: sum(1 for m in master if m.origin == o) for o in ("soft+hard", "soft", "hard")}
    hard_only = [m for m in master if m.origin == "hard"]

    # Are HEL1OS-only events real? While SoLEXS was observing, look for a rise
    # in SoLEXS's own 6-12 keV band at the HEL1OS peak -- an independent
    # instrument -- and at random quiet times (>= 30 min from any catalogue
    # entry, within 6 h) as the control.
    rng = np.random.default_rng(0)
    all_peaks = np.sort(np.array([m.peak_unix for m in master]))
    z_ev, z_ctrl = [], []
    for m in hard_only:
        if not m.soft_observed:
            continue
        v = hot_excess(t, hi, ok, m.hard.peak_unix)
        if v is not None:
            z_ev.append(v)
        for _ in range(3):
            tc = m.hard.peak_unix + rng.uniform(-6.0, 6.0) * 3600.0
            _, dd = nearest(np.array([tc]), all_peaks)
            if dd[0] < 1800.0:
                continue
            v = hot_excess(t, hi, ok, tc)
            if v is not None:
                z_ctrl.append(v)
    z_ev, z_ctrl = np.array(z_ev), np.array(z_ctrl)
    hard_only_check = {
        "hard_only_while_solexs_observing": int(sum(1 for m in hard_only if m.soft_observed)),
        "in_goes_list": int(sum(1 for r_ in rows if r_["origin"] == "hard" and r_["soft_observed"] == 1
                                and r_["goes_class"])),
        "checked": int(z_ev.size), "control_n": int(z_ctrl.size),
        "solexs_6_12keV_rise_gt3sigma": rate((z_ev > 3).sum(), z_ev.size),
        "solexs_6_12keV_rise_gt5sigma": rate((z_ev > 5).sum(), z_ev.size),
        "control_gt3sigma": rate((z_ctrl > 3).sum(), z_ctrl.size),
        "control_gt5sigma": rate((z_ctrl > 5).sum(), z_ctrl.size),
    }
    summary = {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        "period": [utc(t0), utc(t1)], "test_period_start": utc(TEST_START),
        "calibration": cal_text,
        "calibration_knots": {"log10_rate": [round(v, 4) for v in cal.x], "log10_flux": [round(v, 4) for v in cal.y]},
        "peak_correction": None if args.no_peak_correction else {
            "fitted_on_training_flares": len(tr),
            "log10_minute_peak": [round(v, 4) for v in peak_cal.x],
            "log10_goes_peak": [round(v, 4) for v in peak_cal.y]},
        "hard_match": args.hard_match, "shift_hard_s": args.shift_hard_s,
        "rule": {"n_rise_minutes": 5, "confirm_ratio": 1.4, "soft_poisson_sigma": 3.0,
                 "hard_cdte_poisson_sigma": args.hard_sigma, "czt": "5 sigma summed, 2 sigma each detector, 3 bins",
                 "peak_match_s": PEAK_MATCH_S},
        "observed_days": {"solexs": round(soft_cov.days(), 1), "hel1os": round(hard_cov.days(), 1)},
        "counts": {"solexs_flares": len(soft), "hel1os_events": len(hard), "master": len(master),
                   "by_origin": origin_counts,
                   "hard_only_while_solexs_observing": sum(1 for m in hard_only if m.soft_observed),
                   "hard_only_while_solexs_off": sum(1 for m in hard_only if not m.soft_observed)},
        "hard_only_check": hard_only_check,
        "solexs_class_counts": {c: sum(1 for e in soft if e.goes_class[:1] == c) for c in ("A",) + CLASSES},
        "ceiling_rule_on_goes_flux": ceiling,
        "all": {"recall": recall_block(False), "lead": lead_block(False),
                "soft_precision": soft_precision(False), "hard_precision": hard_precision(False),
                "class": class_block(False), "class_minute_calibration_only": class_block(False, minute=True)},
        "test": {"recall": recall_block(True), "lead": lead_block(True),
                 "soft_precision": soft_precision(True), "hard_precision": hard_precision(True),
                 "class": class_block(True), "class_minute_calibration_only": class_block(True, minute=True)},
        "seconds": round(time.time() - t_start, 1),
    }
    (out / "catalog_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "CATALOG.md").write_text(render(summary), encoding="utf-8")
    print(render(summary))
    print(f"wrote {out / 'master_catalog.csv'} ({len(rows)} rows), catalog_summary.json, CATALOG.md "
          f"in {summary['seconds']} s")
    return 0


def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def render(s: dict) -> str:
    L = []
    L.append("# Aditya-L1 master flare catalogue\n")
    L.append(f"Generated {s['generated_utc']} UTC by `python -m solarflare catalog`. Period {s['period'][0]} → "
             f"{s['period'][1]}; SoLEXS observed {s['observed_days']['solexs']} days, HEL1OS "
             f"{s['observed_days']['hel1os']} days. GOES-18 is used only to score, never to detect.\n")
    c = s["counts"]
    L.append(f"**{c['master']} catalogue entries**: {c['by_origin']['soft+hard']} seen by both instruments, "
             f"{c['by_origin']['soft']} by SoLEXS only, {c['by_origin']['hard']} by HEL1OS only "
             f"({c['hard_only_while_solexs_off']} of them while SoLEXS was not observing).\n")
    h = s["hard_only_check"]
    L.append(f"Are the HEL1OS-only flares real? {h['hard_only_while_solexs_observing']} occurred while SoLEXS was "
             f"observing; only {h['in_goes_list']} are in the GOES list, yet {pct(h['solexs_6_12keV_rise_gt3sigma'])} "
             f"show a simultaneous >3 sigma rise in SoLEXS 6-12 keV ({pct(h['solexs_6_12keV_rise_gt5sigma'])} "
             f"above 5 sigma), against {pct(h['control_gt3sigma'])} ({pct(h['control_gt5sigma'])}) at random quiet "
             f"times: about half are real small hot flares that neither the GOES list nor the SoLEXS rule records.\n")
    cc = s["solexs_class_counts"]
    L.append("SoLEXS-derived classes: " + ", ".join(f"{k} {v}" for k, v in cc.items()) + ".\n")
    for key, title in (("all", "Whole archive"), ("test", f"Test period only (from {s['test_period_start'][:10]})")):
        b = s[key]
        L.append(f"\n## {title}\n")
        L.append("### Detection rate (recall) by GOES class\n")
        L.append("| GOES class | SoLEXS | HEL1OS (chance) | HEL1OS saw non-thermal (≥20 keV) | "
                 "both observing: SoLEXS alone | both observing: combined (chance) |")
        L.append("|---|---|---|---|---|---|")
        for cl, v in b["recall"].items():
            L.append(f"| {cl} | {pct(v['soft_recall'])} (n={v['soft_n']}) | {pct(v['hard_recall'])} "
                     f"({pct(v.get('hard_recall_chance'))}) (n={v['hard_n']}) | "
                     f"{pct(v['hard_nonthermal_fraction'])} | {pct(v['both_soft_only_recall'])} (n={v['both_observed_n']}) | "
                     f"{pct(v['both_combined_recall'])} ({pct(v.get('both_combined_recall_chance'))}) |")
        L.append("\n*Chance*: the same numbers with every HEL1OS event moved 2 h earlier or later (mean of "
                 "both). HEL1OS's real contribution is the gap between a figure and its chance level.")
        if key == "all":
            ce = s["ceiling_rule_on_goes_flux"]
            L.append("\nCeiling — the same rule on GOES's own flux: " +
                     ", ".join(f"{k} {pct(v)}" for k, v in ce["recall"].items()) +
                     f"; precision {pct(ce['precision'])}.\n")
        p = b["soft_precision"]
        L.append("\n### False alarms (SoLEXS flares with no GOES-listed flare within 5 min)\n")
        L.append("| SoLEXS class | flares | matched to GOES | unmatched per observed day |")
        L.append("|---|---|---|---|")
        for k, v in p.items():
            L.append(f"| {k} | {v['n']} | {pct(v['precision'])} | {v['unmatched_per_day']} |")
        hp = b["hard_precision"]
        L.append(f"\nHEL1OS events inside a GOES flare: {pct(hp['matched_goes_flare'])} of {hp['n']}; "
                 f"with non-thermal (CZT 20–40 keV) emission: {pct(hp['nonthermal_matched_goes_flare'])} of {hp['nonthermal_n']}.\n")
        k = b["class"]
        if k:
            L.append("\n### Class from SoLEXS alone vs GOES\n")
            L.append(f"{k['n']} matched flares: same letter {pct(k['letter_agreement'])}, median error "
                     f"{k['median_abs_dex']} dex (bias {k['bias_dex']:+}), within a factor 1.5: {pct(k['within_factor_1.5'])}.")
            k0 = b.get("class_minute_calibration_only") or {}
            if k0 and s.get("peak_correction"):
                L.append(f"Without the peak-level correction: same letter {pct(k0['letter_agreement'])}, "
                         f"median error {k0['median_abs_dex']} dex (bias {k0['bias_dex']:+}).")
            L.append("")
            L.append("| GOES class | n | same letter | median error (dex) | bias (dex) |")
            L.append("|---|---|---|---|---|")
            for cl, v in k["by_goes_class"].items():
                L.append(f"| {cl} | {v['n']} | {pct(v['letter_agreement'])} | {v['median_abs_dex']} | {v['bias_dex']:+} |")
        L.append("\n### Alert time before the GOES peak (flares both instruments observed)\n")
        L.append("| GOES class | SoLEXS alert: median min before peak | alerted before peak | combined alert: median | before peak | HEL1OS fired first | by (median min) |")
        L.append("|---|---|---|---|---|---|---|")
        for cl, v in b["lead"].items():
            L.append(f"| {cl} | {v['soft_median_min_before_peak']} | {pct(v['soft_alert_before_peak'])} | "
                     f"{v['combined_median_min_before_peak']} | {pct(v['combined_alert_before_peak'])} | "
                     f"{pct(v['hard_first_fraction'])} | {v['hard_first_median_gain_min']} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
