"""Phase 2: per-flare hot-onset and Neupert catalogue, and whether it helps forecast peaks.

    python -m solarflare onset-study

For every GOES-18 flare >= C1.0 that SoLEXS observed (published energy scale):

1. hot onset -- background-subtracted SoLEXS hardness (6-8 / 3-4 keV) and GOES
   XRS short/long ratio in the first 2 minutes vs around the peak;
2. timing -- causal SoLEXS onset, steepest rise (impulsive-phase proxy), peak;
3. Neupert effect where HEL1OS observed -- HXR 20-40 keV vs d(SoLEXS)/dt
   correlation and lag.

Then a leakage-free test: two minutes after the causal onset, does adding the
onset hardness (or the GOES ratio, or early hard X-rays) to "the flux seen so
far" reduce the error of the final GOES peak? Rolling-origin folds in time,
gradient-boosted trees, paired day-block bootstrap.

Writes <out>/flare_catalog.csv, <out>/physics_summary.json and prints a summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, UTC
from functools import lru_cache
from pathlib import Path

from solarflare.settings import load_settings

import numpy as np


from solarflare.io.goes import class_flux, load_goes
from solarflare.physics.onset import flare_record, load_goes_xrs
from solarflare.preprocess.cache import load_cached
from solarflare.preprocess.timeline import stitch

DT = 20.0
BEFORE_S, AFTER_S = 1500.0, 900.0


def main(argv=None) -> int:
    S = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--goes-dir", default=str(S.goes_dir))
    ap.add_argument("--cache-dir", default=str(S.cache))
    ap.add_argument("--out", default=str(S.physics))
    ap.add_argument("--min-class", default="C1.0")
    ap.add_argument("--decision-s", type=float, default=120.0,
                    help="features use data up to this long after the causal onset")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)

    entries = json.loads((cache / "manifest.json").read_text("utf-8"))
    slx = []
    for e in entries:
        if e["source"]["kind"] != "solexs" or e.get("status") != "ok":
            continue
        with np.load(cache / f"{e['key']}.npz") as z:
            names = [str(n) for n in z["names"]]
        if "slx_3_4keV" in names and "slx_6_8keV" in names:      # published energy scale only
            slx.append((e["t_start"], e["t_stop"], e["key"]))
    slx.sort()
    print(f"SoLEXS days on the published energy scale: {len(slx)}")

    @lru_cache(maxsize=8)
    def day(key):
        z = np.load(cache / f"{key}.npz")
        names = [str(n) for n in z["names"]]
        v = z["values"]
        return (z["time_unix"], v[:, names.index("slx_goes_long")], v[:, names.index("slx_3_4keV")],
                v[:, names.index("slx_6_8keV")], z["coverage"])

    hard = stitch(load_cached(entries, cache, "hel1os"), DT, 21600.0)
    print(f"HEL1OS timelines: {len(hard)}")
    gt, ga, gb = load_goes_xrs(Path(args.goes_dir))
    truth = load_goes(Path(args.goes_dir))
    floor = class_flux(args.min_class)
    flares = [f for f in truth.flares if f.peak_flux >= floor
              and any(a <= f.start_unix and f.start_unix <= b for a, b, _ in slx)]
    print(f"GOES flares >= {args.min_class} starting inside a SoLEXS day: {len(flares)}")

    rows = []
    for n_done, f in enumerate(flares):
        w0 = np.floor((f.start_unix - BEFORE_S) / DT) * DT
        w1 = f.end_unix + AFTER_S
        t = w0 + DT * np.arange(int((w1 - w0) / DT) + 1)
        gl, lo, hi = (np.full(t.size, np.nan) for _ in range(3))
        ok = np.zeros(t.size, bool)
        for a, b, key in slx:
            if b < w0 or a > w1:
                continue
            dt_, dgl, dlo, dhi, dcov = day(key)
            pos = np.rint((dt_ - w0) / DT).astype(np.int64)
            m = (pos >= 0) & (pos < t.size)
            gl[pos[m]], lo[pos[m]], hi[pos[m]] = dgl[m], dlo[m], dhi[m]
            ok[pos[m]] = (dcov[m] > 0) & np.isfinite(dgl[m]) & np.isfinite(dlo[m]) & np.isfinite(dhi[m])

        hxr = hxr_ok = None
        for h in hard:
            if h.time_unix[-1] < w0 or h.time_unix[0] > w1:
                continue
            names = list(h.names)
            pos = np.rint((h.time_unix - w0) / DT).astype(np.int64)
            m = (pos >= 0) & (pos < t.size)
            if hxr is None:
                hxr, hxr_ok = np.full(t.size, np.nan), np.zeros(t.size, bool)
            for det in ("czt2", "czt1"):                 # czt1 wins where both exist
                v = h.values[m, names.index(f"hls_{det}_20_40keV")]
                c = h.values[m, names.index(f"hls_{det}_cov")]
                good = np.isfinite(v) & (c > 0)
                hxr[pos[m][good]] = v[good]
                hxr_ok[pos[m][good]] = True

        g = (gt >= w0 - 60.0) & (gt <= w1)
        rec = flare_record(t, gl, lo, hi, ok, {"start": f.start_unix, "peak": f.peak_unix, "end": f.end_unix,
                                               "peak_flux": f.peak_flux},
                           gt[g], ga[g], gb[g], hxr, hxr_ok, dt=DT, decision_s=args.decision_s)
        rec.update({"flare_id": int(f.flare_id), "goes_class": str(f.goes_class),
                    "start_utc": datetime.fromtimestamp(f.start_unix, UTC).strftime("%Y-%m-%d %H:%M"),
                    "peak_unix": f.peak_unix, "start_unix": f.start_unix})
        rows.append(rec)
        if (n_done + 1) % 1000 == 0:
            print(f"  {n_done + 1}/{len(flares)}", flush=True)

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("flare_id", "goes_class", "start_utc", "status"), k))
    tag = f"dec{int(args.decision_s)}s"
    with (out / f"flare_catalog_{tag}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = {"n_flares": len(rows), "decision_s": args.decision_s, "status": {}}
    for r in rows:
        summary["status"][r["status"]] = summary["status"].get(r["status"], 0) + 1
    good = [r for r in rows if r["status"] == "ok"]
    summary["hot_onset"] = hot_onset_stats(good)
    summary["neupert"] = neupert_stats(good)
    summary["peak_forecast"] = forecast_test(good, args.decision_s)
    summary["becomes_M_or_X"] = exceedance_test(good)
    ex = [r for r in good if r["start_utc"].startswith("2024-05-05 11:4")]
    summary["example_2024_05_05_X1.2"] = ex[0] if ex else None
    (out / f"physics_summary_{tag}.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print_summary(summary)
    print(f"\nwrote {out / f'flare_catalog_{tag}.csv'} and {out / f'physics_summary_{tag}.json'}")
    return 0


def _f(r, k):
    v = r.get(k, float("nan"))
    return float(v) if v is not None else float("nan")


def hot_onset_stats(good):
    out = {}
    for letter in ("C", "M", "X"):
        rs = [r for r in good if r["goes_class"].startswith(letter)]
        if not rs:
            continue
        h_on = np.array([_f(r, "hardness_onset") for r in rs])
        h_pk = np.array([_f(r, "hardness_peak") for r in rs])
        h_bg = np.array([_f(r, "hardness_background") for r in rs])
        g_on = np.array([_f(r, "goes_ratio_onset") for r in rs])
        g_pk = np.array([_f(r, "goes_ratio_peak") for r in rs])
        m = np.isfinite(h_on) & np.isfinite(h_pk)
        mg = np.isfinite(g_on) & np.isfinite(g_pk) & (g_pk > 0)
        out[letter] = {
            "n": len(rs),
            "n_with_onset_hardness": int(m.sum()),
            "median_hardness_background": float(np.nanmedian(h_bg)),
            "median_hardness_onset": float(np.nanmedian(h_on[m])) if m.any() else None,
            "median_hardness_peak": float(np.nanmedian(h_pk[m])) if m.any() else None,
            "median_onset_over_peak_hardness": float(np.median(h_on[m] / h_pk[m])) if m.any() else None,
            "median_onset_over_impulsive_hardness": _ratio_median(rs, "hardness_onset", "hardness_impulsive"),
            "frac_onset_at_least_80pct_of_peak_hardness": float(np.mean(h_on[m] >= 0.8 * h_pk[m])) if m.any() else None,
            "frac_onset_harder_than_background": float(np.mean(h_on[m] > h_bg[m])) if m.any() else None,
            "median_goes_ratio_onset_over_peak": float(np.median(g_on[mg] / g_pk[mg])) if mg.any() else None,
            "frac_goes_onset_at_least_80pct_of_peak": float(np.mean(g_on[mg] >= 0.8 * g_pk[mg])) if mg.any() else None,
            "median_onset_to_impulsive_min": float(np.nanmedian([_f(r, "onset_to_impulsive_min") for r in rs])),
            "median_onset_to_peak_min": float(np.nanmedian([_f(r, "onset_to_peak_min") for r in rs])),
            "median_onset_minus_goes_start_min": float(np.nanmedian([_f(r, "onset_minus_goes_start_s") for r in rs]) / 60),
        }
    return out


def neupert_stats(good):
    rs = [r for r in good if r.get("hel1os") in (True, "True") and _f(r, "hxr_peak_sigma") >= 5]
    if not rs:
        return {"n": 0}
    r_best = np.array([_f(r, "neupert_r_best") for r in rs])
    lag = np.array([_f(r, "neupert_lag_s") for r in rs])
    dpk = np.array([_f(r, "hxr_peak_minus_impulsive_s") for r in rs])
    n_obs = sum(1 for r in good if r.get("hel1os") in (True, "True"))
    return {
        "n_hel1os_observed": n_obs,
        "n_significant_hxr": len(rs),
        "median_r_best": float(np.nanmedian(r_best)),
        "frac_r_best_above_0.5": float(np.nanmean(r_best > 0.5)),
        "median_lag_s": float(np.nanmedian(lag)),
        "lag_s_p25_p75": [float(np.nanpercentile(lag, 25)), float(np.nanpercentile(lag, 75))],
        "median_hxr_peak_minus_steepest_sxr_rise_s": float(np.nanmedian(dpk)),
        "by_class_n": {c: sum(1 for r in rs if r["goes_class"].startswith(c)) for c in "CMX"},
    }


BASE = ["dec_log_net_rate", "dec_log_background_rate", "dec_rise_dex"]
SETS = {
    "flux so far (SoLEXS)": BASE,
    "+ onset hardness": BASE + ["log_hardness_onset", "log_hardness_background"],
    "+ GOES ratio at onset": BASE + ["goes_ratio_onset"],
}


def _usable(X, tr):
    """Zero out columns with no finite value in the training rows (e.g. hardness
    that is never bright enough 40 s after onset) -- the trees cannot bin them."""
    dead = ~np.isfinite(X[tr]).any(axis=0)
    if not dead.any():
        return X
    X = X.copy()
    X[:, dead] = 0.0
    return X


def _ratio_median(rs, a, b):
    x = np.array([_f(r, a) for r in rs])
    y = np.array([_f(r, b) for r in rs])
    m = np.isfinite(x) & np.isfinite(y) & (y > 0)
    return float(np.median(x[m] / y[m])) if m.any() else None


def _prepare(good):
    """Flares whose peak is still ahead at the decision time, in time order."""
    rows = [r for r in good if r.get("peak_after_decision") in (True, "True")
            and np.isfinite(_f(r, "dec_log_net_rate")) and np.isfinite(_f(r, "dec_goes_log_flux"))]
    for r in rows:
        for k in ("hardness_onset", "hardness_background"):
            v = _f(r, k)
            r["log_" + k] = float(np.log10(v)) if v > 0 else float("nan")
    rows.sort(key=lambda r: r["t_onset"])
    return rows


def exceedance_test(good, n_blocks=5, n_boot=300):
    """Among flares still below M1 at the decision time: which will reach >= M1?"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    rows = [r for r in _prepare(good) if _f(r, "dec_goes_log_flux") < -5.0]
    y = np.array([r["log_peak_flux"] >= -5.0 for r in rows]).astype(int)
    days = np.array([r["start_utc"][:10] for r in rows])
    blocks = np.array_split(np.arange(len(rows)), n_blocks)
    te = np.concatenate(blocks[1:])
    score = {}
    for name, feats in SETS.items():
        X = np.array([[_f(r, k) for k in feats] for r in rows])
        p = np.full(len(rows), np.nan)
        for k in range(1, n_blocks):
            tr = np.concatenate(blocks[:k])
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3,
                                               min_samples_leaf=20, random_state=0)
            Xk = _usable(X, tr)
            m.fit(Xk[tr], y[tr])
            p[blocks[k]] = m.predict_proba(Xk[blocks[k]])[:, 1]
        score[name] = p
    score["current GOES level (reference)"] = np.array([_f(r, "dec_goes_log_flux") for r in rows])
    res = {"n_flares": len(rows), "n_test": int(te.size), "n_test_positive": int(y[te].sum()),
           "auc": {k: float(roc_auc_score(y[te], v[te])) for k, v in score.items()}}
    rng = np.random.default_rng(1)
    mem = [te[days[te] == d] for d in np.unique(days[te])]
    res["auc_gain_vs_flux_so_far"] = {}
    for k in ("+ onset hardness", "+ GOES ratio at onset", "current GOES level (reference)"):
        diffs = []
        for _ in range(n_boot):
            pick = np.concatenate([mem[i] for i in rng.integers(0, len(mem), len(mem))])
            if y[pick].min() == y[pick].max():
                continue
            diffs.append(roc_auc_score(y[pick], score[k][pick])
                         - roc_auc_score(y[pick], score["flux so far (SoLEXS)"][pick]))
        res["auc_gain_vs_flux_so_far"][k] = {"gain": res["auc"][k] - res["auc"]["flux so far (SoLEXS)"],
                                             "ci95": [float(np.percentile(diffs, 2.5)),
                                                      float(np.percentile(diffs, 97.5))]}
    return res


def forecast_test(good, decision_s, n_blocks=5, n_boot=1000):
    from sklearn.ensemble import HistGradientBoostingRegressor

    base, sets = BASE, SETS
    rows = _prepare(good)
    y = np.array([r["log_peak_flux"] for r in rows])
    cur = np.array([_f(r, "dec_goes_log_flux") for r in rows])
    days = np.array([r["start_utc"][:10] for r in rows])
    cls = np.array([r["goes_class"][0] for r in rows])
    blocks = np.array_split(np.arange(len(rows)), n_blocks)
    test_idx = np.concatenate(blocks[1:])

    def run(feats, pool=None):
        """Rolling-origin predictions; ``pool`` (indices, time-ordered) gets its own blocks."""
        X = np.array([[_f(r, k) for k in feats] for r in rows])
        pred = np.full(len(rows), np.nan)
        bl = blocks if pool is None else np.array_split(pool, n_blocks)
        for k in range(1, n_blocks):
            tr, te = np.concatenate(bl[:k]), bl[k]
            if len(tr) < 50 or len(te) == 0:
                continue
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3,
                                              min_samples_leaf=20, random_state=0)
            Xk = _usable(X, tr)
            m.fit(Xk[tr], y[tr])
            pred[te] = m.predict(Xk[te])
        return pred

    rng = np.random.default_rng(0)

    def paired(err_a, err_b, idx):
        ok = idx[np.isfinite(err_a[idx]) & np.isfinite(err_b[idx])]
        if ok.size == 0:
            return float("nan"), [float("nan"), float("nan")], 0
        ud = np.unique(days[ok])
        mem = [ok[days[ok] == d] for d in ud]
        diffs = []
        for _ in range(n_boot):
            pick = np.concatenate([mem[i] for i in rng.integers(0, len(mem), len(mem))])
            diffs.append(np.mean(err_a[pick]) - np.mean(err_b[pick]))
        return float(np.mean(err_a[ok]) - np.mean(err_b[ok])), [float(np.percentile(diffs, 2.5)),
                                                                 float(np.percentile(diffs, 97.5))], int(ok.size)

    res = {"n_flares": len(rows), "n_test": int(test_idx.size), "decision": f"causal SoLEXS onset + {decision_s:g} s",
           "target": "log10 GOES peak flux", "folds": f"rolling origin, {n_blocks} time blocks"}
    err = {}
    for name, feats in sets.items():
        p = run(feats)
        err[name] = np.abs(p - y)
    err["current GOES level (reference)"] = np.abs(cur - y)
    err["training median (reference)"] = np.abs(np.median(y[blocks[0]]) - y)
    res["mae_dex"] = {k: float(np.nanmean(v[test_idx])) for k, v in err.items()}
    res["mae_dex_by_class"] = {c: {k: float(np.nanmean(v[test_idx][cls[test_idx] == c]))
                                   for k, v in err.items()} for c in "CMX" if np.any(cls[test_idx] == c)}
    a = err["flux so far (SoLEXS)"]
    res["gain_vs_flux_so_far"] = {}
    for k in ("+ onset hardness", "+ GOES ratio at onset", "current GOES level (reference)"):
        d, ci, n = paired(a, err[k], test_idx)
        res["gain_vs_flux_so_far"][k] = {"mae_reduction_dex": d, "ci95": ci, "n": n}

    hel = np.array([r.get("hel1os") in (True, "True") and np.isfinite(_f(r, "dec_hxr_net_counts")) for r in rows])
    pool = np.flatnonzero(hel)
    if pool.size >= 150:
        pa = run(base, pool=pool)
        pb = run(base + ["dec_hxr_net_counts", "dec_hxr_sigma"], pool=pool)
        ea, eb = np.abs(pa - y), np.abs(pb - y)
        te = np.concatenate(np.array_split(pool, n_blocks)[1:])
        d, ci, n = paired(ea, eb, te)
        res["hel1os_subset"] = {"n_flares": int(pool.size), "n_test": n,
                                "mae_flux_so_far": float(np.nanmean(ea[te])),
                                "mae_plus_early_hxr": float(np.nanmean(eb[te])),
                                "mae_current_goes_level": float(np.nanmean(np.abs(cur - y)[te])),
                                "mae_reduction_dex": d, "ci95": ci}
    else:
        res["hel1os_subset"] = {"n_flares": int(pool.size), "note": "too few HEL1OS flares"}
    return res


def print_summary(s):
    print("\nstatus:", s["status"])
    print("\nHOT ONSET (hardness = net 6-8 / 3-4 keV counts)")
    for c, v in s["hot_onset"].items():
        print(f"  {c}: n={v['n']} (hardness at onset: {v['n_with_onset_hardness']}) | background {v['median_hardness_background']:.3f}"
              f" onset {v['median_hardness_onset']} peak {v['median_hardness_peak']} | onset/peak {v['median_onset_over_peak_hardness']}"
              f" | onset >= 80% of peak: {v['frac_onset_at_least_80pct_of_peak_hardness']} | GOES onset/peak {v['median_goes_ratio_onset_over_peak']}"
              f" | onset/impulsive {v['median_onset_over_impulsive_hardness']}"
              f" | onset->steepest rise {v['median_onset_to_impulsive_min']:.1f} min, ->peak {v['median_onset_to_peak_min']:.1f} min")
    print("\nNEUPERT:", json.dumps(s["neupert"]))
    pf = s["peak_forecast"]
    print(f"\nPEAK FORECAST at {pf['decision']} ({pf['n_flares']} flares, {pf['n_test']} tested)")
    for k, v in pf["mae_dex"].items():
        print(f"  {k:34s} MAE {v:.3f} dex")
    for k, v in pf["gain_vs_flux_so_far"].items():
        print(f"  gain of {k:30s} {v['mae_reduction_dex']:+.4f} dex  95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    print("  HEL1OS subset:", json.dumps(pf["hel1os_subset"]))
    ex_ = s["becomes_M_or_X"]
    print(f"\nWILL IT REACH >= M1? (below M1 at decision; {ex_['n_test']} tested, {ex_['n_test_positive']} did)")
    for k, v in ex_["auc"].items():
        print(f"  {k:34s} AUC {v:.3f}")
    for k, v in ex_["auc_gain_vs_flux_so_far"].items():
        print(f"  gain of {k:30s} {v['gain']:+.4f}  95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    ex = s.get("example_2024_05_05_X1.2")
    if ex:
        keys = ["goes_class", "start_utc", "onset_minus_goes_start_s", "onset_to_impulsive_min", "onset_to_peak_min",
                "hardness_background", "hardness_onset", "hardness_impulsive", "hardness_peak",
                "goes_ratio_onset", "goes_ratio_peak"]
        print("\n2024-05-05 X1.2:", {k: ex.get(k) for k in keys})


if __name__ == "__main__":
    raise SystemExit(main())
