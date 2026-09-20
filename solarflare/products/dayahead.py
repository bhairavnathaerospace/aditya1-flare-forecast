"""Will a flare happen in the next 2-24 hours? Soft X-ray activity, magnetic fields, or both.

    python -m solarflare dayahead

The network forecasts 1-60 min ahead from the last 2 h of light curve.
Operational services forecast a day ahead, from sunspot-region magnetic data.
This asks what each source is worth at hours to a day, on the same hours:

Origins: every hour on the hour with SoLEXS data for >= 50% of the previous 6 h
         (and, when SHARP files are present, a SHARP hour an operator would have
         had, at most 6 h old).
Targets: a GOES flare >= C1 / >= M1 peaks in (t, t+H], H = 2, 6, 12, 24 h,
         kept only where GOES observed >= 80% of the window.
Feature sets, all known at t:
  xray   SoLEXS minute flux on the training-period GOES calibration: flux now;
         1/6/24 h minimum (background), maximum and mean; 1 h and 6 h change;
         background hardness (6-12 keV / GOES band, quietest minutes of the last
         hour); master-catalogue flares >= B, >= C, >= M in the last 6/24/72 h
         (known 5 min after their peak); hours since the last >= C and >= M
         flare; largest flare of the last 24 and 72 h.
  sharp  SDO/HMI SHARP whole-disk indicators (solarflare.io.sharp) at
         t - LATENCY_S, and the 24 h change of total unsigned flux and current
         helicity (flux emergence).
  both   the two together.
Models: logistic regression and gradient-boosted trees per set, fitted on the
training period, the better one (validation AUC) and its threshold chosen on
validation, scored on the test period. Reference: "persistence", a logistic fit
on one number, flares of the class SoLEXS saw in the last 24 h. Paired
differences (both minus xray: what SHARP adds; xray minus persistence) come with
95% intervals that resample whole weeks (hourly origins with day-long targets
overlap).

Writes outputs/dayahead/{DAYAHEAD.md, dayahead_summary.json, dayahead.png}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from solarflare.settings import load_settings
from solarflare.util import read_rows, ts, utc

#: The model's split (outputs/model/reports/data_meta.json), set in main().
TRAIN_END = float("nan")
TEST_START = float("nan")
HORIZONS_H = (2, 6, 12, 24)
CLASSES = {"C": 1e-6, "M": 1e-5}
SHARP_MAX_AGE_S = 6 * 3600.0
SETS = ("xray", "sharp", "both")
SET_NAMES = {"xray": "SoLEXS activity", "sharp": "SHARP magnetic", "both": "SoLEXS + SHARP"}


def xray_features(args):
    """Hourly SoLEXS activity features, GOES targets, and which origins have enough SoLEXS."""
    from solarflare.catalog.build import load_solexs
    from solarflare.catalog.detect import PiecewiseCalibration, to_minutes
    from solarflare.io.goes import load_goes

    truth = load_goes(Path(args.goes_dir))
    _, t, r, hi, ok = load_solexs(Path(args.cache_dir))
    tm, rm, _, vm = to_minutes(t, r, ok, 20.0)
    _, hm, _, vh = to_minutes(t, hi, ok, 20.0)
    gmin = truth.flux_on_grid(tm)
    fit_on = vm & (tm <= TRAIN_END) & np.isfinite(gmin) & (gmin > 0) & (rm > 0)
    cal = PiecewiseCalibration.fit(np.log10(rm[fit_on]), np.log10(gmin[fit_on]))
    with np.errstate(divide="ignore", invalid="ignore"):
        lf = np.log10(cal(np.where(vm, rm, np.nan)))
        hard = np.where(vm & vh & (rm > 0), hm / rm, np.nan)
    idx = pd.to_datetime(tm + 60.0, unit="s", utc=True)          # a minute is known when it ends
    s = pd.DataFrame({"f": lf, "hard": hard}, index=idx)

    origins = pd.date_range(idx[0].ceil("h"), idx[-1].floor("h"), freq="h")
    feat = pd.DataFrame(index=origins)

    def roll(col, win, how):
        r_ = getattr(s[col].rolling(win, min_periods=max(1, int(pd.Timedelta(win).total_seconds() / 60 * 0.3))), how)()
        return r_.reindex(origins, method="ffill", tolerance=pd.Timedelta("2min"))

    feat["flux_now"] = s["f"].reindex(origins, method="ffill", tolerance=pd.Timedelta("5min"))
    for w in ("1h", "6h", "24h"):
        feat[f"flux_min_{w}"] = roll("f", w, "min")
        feat[f"flux_max_{w}"] = roll("f", w, "max")
        feat[f"flux_mean_{w}"] = roll("f", w, "mean")
    feat["change_1h"] = feat["flux_now"] - s["f"].reindex(origins - pd.Timedelta("1h"), method="ffill",
                                                          tolerance=pd.Timedelta("5min")).to_numpy()
    feat["change_6h"] = feat["flux_mean_1h"] - s["f"].rolling("1h", min_periods=10).mean().reindex(
        origins - pd.Timedelta("6h"), method="ffill", tolerance=pd.Timedelta("5min")).to_numpy()
    feat["hardness_background_1h"] = roll("hard", "1h", "median")
    cov6 = s["f"].notna().astype(float).rolling("6h").sum().reindex(origins, method="ffill",
                                                                     tolerance=pd.Timedelta("2min")) / 360.0

    # catalogue flares (SoLEXS), known 5 min after their peak
    cat = [r_ for r_ in read_rows(args.catalog)
           if r_["origin"] in ("soft", "soft+hard") and r_["peak_flux_solexs_Wm2"]]
    known = np.array([ts(r_["peak_utc"]) + 300.0 for r_ in cat])
    pf = np.array([float(r_["peak_flux_solexs_Wm2"]) for r_ in cat])
    order = np.argsort(known)
    known, pf = known[order], pf[order]
    o = ((origins - pd.Timestamp(0, tz="UTC")) / pd.Timedelta("1s")).to_numpy(dtype=np.float64)
    for lab, lo in (("B", 1e-7), ("C", 1e-6), ("M", 1e-5)):
        kk = known[pf >= lo]
        for h in (6, 24, 72):
            feat[f"n_{lab}_{h}h"] = np.searchsorted(kk, o, side="right") - np.searchsorted(kk, o - 3600.0 * h, side="right")
        if lab in ("C", "M"):
            j = np.searchsorted(kk, o, side="right") - 1
            feat[f"hours_since_{lab}"] = np.where(j >= 0, np.minimum((o - kk[np.maximum(j, 0)]) / 3600.0, 168.0), 168.0)
    for h in (24, 72):
        mx = np.full(o.size, -8.0)
        for i, t_ in enumerate(o):
            a, b = np.searchsorted(known, [t_ - 3600.0 * h, t_], side="right")
            if b > a:
                mx[i] = np.log10(pf[a:b].max())
        feat[f"max_flare_{h}h"] = mx

    # targets
    fl = [f for f in truth.flares if f.goes_class[:1] in "CMX"]
    gp = np.array([f.peak_unix for f in fl])
    gf = np.array([f.peak_flux for f in fl])
    gv = np.isfinite(truth.xrsb).astype(float)
    gcs = np.concatenate([[0.0], np.cumsum(gv)])
    gt = truth.time_unix
    tgt = {}
    for c, lo in CLASSES.items():
        pk = np.sort(gp[gf >= lo])
        for H in HORIZONS_H:
            n = np.searchsorted(pk, o + 3600.0 * H, side="right") - np.searchsorted(pk, o, side="right")
            a, b = np.searchsorted(gt, o), np.searchsorted(gt, o + 3600.0 * H)
            cover = (gcs[b] - gcs[a]) / (60.0 * H)
            tgt[(c, H)] = (n > 0).astype(float), cover >= 0.8
    return feat, cov6.to_numpy() >= 0.5, o, tgt


def sharp_features(sharp_dir: Path, o: np.ndarray):
    """SHARP indicators at each origin, as an operator would have had them; None without files."""
    from solarflare.io.sharp import LATENCY_S, load_sharp

    sh = load_sharp(sharp_dir)
    if sh is None:
        return None, None
    now = sh.at(o - LATENCY_S, max_age_s=SHARP_MAX_AGE_S)
    day = sh.at(o - LATENCY_S - 86400.0, max_age_s=SHARP_MAX_AGE_S)
    feat = pd.DataFrame(now, columns=sh.names)
    for k in ("sharp_log_usflux_sum", "sharp_log_totusjh_sum"):
        j = sh.names.index(k)
        feat[k.replace("sharp_log_", "sharp_d24_log_")] = now[:, j] - day[:, j]
    info = {"rows_read": sh.n_rows_read, "rows_kept": sh.n_rows_kept, "hours": int(sh.time_unix.size),
            "first": utc(sh.time_unix[0]), "last": utc(sh.time_unix[-1]), "latency_h": LATENCY_S / 3600.0}
    return feat, info


def week_ci(y, p, stat, weeks, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    u = np.unique(weeks)
    groups = {w: np.flatnonzero(weeks == w) for w in u}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[w] for w in rng.choice(u, u.size)])
        if len(np.unique(y[idx])) == 2:
            vals.append(stat(y[idx], p[idx]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)] if vals else None


def tss_at(y, p, thr):
    pred = p >= thr
    tp, fn = np.sum(pred & (y == 1)), np.sum(~pred & (y == 1))
    fp, tn = np.sum(pred & (y == 0)), np.sum(~pred & (y == 0))
    return tp / max(tp + fn, 1) - fp / max(fp + tn, 1)


def best_tss_threshold(y, p):
    cands = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    vals = [tss_at(y, p, c) for c in cands]
    return float(cands[int(np.argmax(vals))])


def _models():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {"logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
            "trees": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                    min_samples_leaf=40, l2_regularization=1.0, random_state=0)}


def fit_score(X, y, tr, va, te, weeks, clim):
    """Both model families on one feature set; the better on validation is kept."""
    from sklearn.metrics import brier_score_loss, roc_auc_score

    fams, preds = {}, {}
    yt = y[te]
    for name, m in _models().items():
        m.fit(X[tr], y[tr])
        pv, pt = m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1]
        thr = best_tss_threshold(y[va], pv)
        bs, bs_c = brier_score_loss(yt, pt), brier_score_loss(yt, np.full_like(pt, clim))
        fams[name] = {"val_AUC": round(float(roc_auc_score(y[va], pv)), 3),
                      "AUC": round(float(roc_auc_score(yt, pt)), 3),
                      "AUC_ci": week_ci(yt, pt, roc_auc_score, weeks),
                      "TSS": round(float(tss_at(yt, pt, thr)), 3),
                      "TSS_ci": week_ci(yt, pt, lambda a, b, _t=thr: tss_at(a, b, _t), weeks),
                      "BSS_vs_training_climatology": round(1 - bs / bs_c, 3),
                      "threshold": round(thr, 3)}
        preds[name] = (m, pt)
    sel = max(fams, key=lambda k: fams[k]["val_AUC"])       # chosen on validation, never on test
    return {"selected": sel, **fams[sel], "families": fams}, preds[sel]


def paired_gain(yt, a, b, weeks):
    from sklearn.metrics import roc_auc_score

    return {"AUC_gain": round(float(roc_auc_score(yt, a) - roc_auc_score(yt, b)), 3),
            "ci": week_ci(yt, np.stack([a, b], 1),
                          lambda yy, pp: roc_auc_score(yy, pp[:, 0]) - roc_auc_score(yy, pp[:, 1]), weeks)}


def main(argv=None) -> int:
    S = load_settings()
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--cache-dir", default=str(S.cache))
    ap.add_argument("--goes-dir", default=str(S.goes_dir))
    ap.add_argument("--sharp-dir", default=str(S.sharp_dir))
    ap.add_argument("--catalog", default=str(S.catalog / "master_catalog.csv"))
    ap.add_argument("--run-dir", default=None, help="trained run whose split to use (default: the final model)")
    ap.add_argument("--out", default=str(S.dayahead))
    args = ap.parse_args(argv)
    global TRAIN_END, TEST_START
    split = S.split_dates(Path(args.run_dir) if args.run_dir else None)
    TRAIN_END, TEST_START = split["train_end"], split["test_start"]
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    xf, have, o, tgt = xray_features(args)
    sf, sharp_info = sharp_features(Path(args.sharp_dir), o)
    cols = {"xray": list(xf.columns)}
    X = xf.to_numpy(dtype=np.float64)
    if sf is not None:
        cols["sharp"] = list(sf.columns)
        cols["both"] = cols["xray"] + cols["sharp"]
        X = np.hstack([X, sf.to_numpy(dtype=np.float64)])
    names = cols["xray"] + cols.get("sharp", [])
    idx = {k: [names.index(n) for n in v] for k, v in cols.items()}
    sets = [k for k in SETS if k in idx]
    finite = np.isfinite(X).all(1)
    print(f"{len(o)} hourly origins, {have.sum()} with SoLEXS, {finite.sum()} with every feature"
          + (f"; SHARP {sharp_info['first']} -> {sharp_info['last']}" if sharp_info else "; no SHARP files")
          + f" ({time.time() - t0:.0f} s)", flush=True)
    weeks_all = np.floor(o / (7 * 86400)).astype(int)
    summary: dict = {"generated_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
                     "split": {"train_end": utc(TRAIN_END), "test_start": utc(TEST_START)},
                     "sharp": sharp_info, "feature_sets": cols, "results": {}}
    for (c, H), (y, gok) in tgt.items():
        base = have & gok & finite           # every set scored on the same hours
        tr = base & (o + 3600.0 * H <= TRAIN_END)
        va = base & (o > TRAIN_END) & (o + 3600.0 * H <= TEST_START)
        te = base & (o >= TEST_START)
        if min(tr.sum(), va.sum(), te.sum()) < 50 or len(np.unique(y[te])) < 2 or len(np.unique(y[va])) < 2:
            continue
        clim = float(y[tr].mean())
        yt, weeks = y[te], weeks_all[te]
        res = {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum()),
               "base_rate_train": round(clim, 3), "base_rate_test": round(float(yt.mean()), 3), "sets": {}}
        pt = {}
        fitted = {}
        for k in sets:
            Xk = X[:, idx[k]]
            res["sets"][k], (fitted[k], pt[k]) = fit_score(Xk, y, tr, va, te, weeks, clim)
        pj = names.index(f"n_{c}_24h")
        pm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr][:, [pj]], y[tr])
        pt["persistence"] = pm.predict_proba(X[te][:, [pj]])[:, 1]
        pv = pm.predict_proba(X[va][:, [pj]])[:, 1]
        thr = best_tss_threshold(y[va], pv)
        res["persistence"] = {"AUC": round(float(roc_auc_score(yt, pt["persistence"])), 3),
                              "AUC_ci": week_ci(yt, pt["persistence"], roc_auc_score, weeks),
                              "TSS": round(float(tss_at(yt, pt["persistence"], thr)), 3)}
        res["gains"] = {"xray_minus_persistence": paired_gain(yt, pt["xray"], pt["persistence"], weeks)}
        if "both" in pt:
            res["gains"]["sharp_added"] = paired_gain(yt, pt["both"], pt["xray"], weeks)
            res["gains"]["sharp_minus_xray"] = paired_gain(yt, pt["sharp"], pt["xray"], weeks)
        # the set used operationally is also chosen on validation
        best = max(sets, key=lambda k: res["sets"][k]["val_AUC"])
        res["best_set"] = best
        pi = permutation_importance(fitted[best], X[te][:, idx[best]], yt, scoring="roc_auc", n_repeats=5,
                                    random_state=0)
        top = np.argsort(-pi.importances_mean)[:6]
        res["top_features"] = [[cols[best][k], round(float(pi.importances_mean[k]), 4)] for k in top]
        summary["results"][f">={c}1 within {H} h"] = res
        line = " | ".join(f"{k} {res['sets'][k]['AUC']}" for k in sets)
        print(f">= {c}1 within {H:2d} h: base rate {res['base_rate_test']:.2f}; AUC {line} | persistence "
              f"{res['persistence']['AUC']}" + (f"; SHARP adds {res['gains']['sharp_added']['AUC_gain']:+.3f} "
                                                f"{res['gains']['sharp_added']['ci']}" if "both" in pt else ""),
              flush=True)
    summary["seconds"] = round(time.time() - t0, 1)
    (out / "dayahead_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = render(summary)
    (out / "DAYAHEAD.md").write_text(md, encoding="utf-8")
    figure(summary, out / "dayahead.png")
    print(md)
    return 0


def _ci(ci) -> str:
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "[--]"


def render(s: dict) -> str:
    R = s["results"]
    sets = [k for k in SETS if k in s["feature_sets"]]
    sh = s.get("sharp")
    L = ["# Forecasting flares hours ahead", "",
         f"Generated {s['generated_utc']} UTC by `python -m solarflare dayahead`. Hourly origins; fitted on the "
         f"training period (to {s['split']['train_end'][:10]}), model family and threshold chosen on validation, "
         f"scored on the test period (from {s['split']['test_start'][:10]}). Every feature set is scored on the same "
         "hours. 95% intervals resample whole weeks.", ""]
    if sh:
        L += [f"SHARP: {sh['hours']} hours with regions from {sh['first'][:10]} to {sh['last'][:10]}, "
              f"{sh['rows_kept']} of {sh['rows_read']} region rows kept (QUALITY 0, within 70 deg of central "
              f"meridian), used {sh['latency_h']:g} h after observation.", ""]
    else:
        L += ["No SHARP files were found, so only SoLEXS activity is scored "
              "(scripts/download_sharp.py fetches them).", ""]
    head = " | ".join(f"{SET_NAMES[k]} AUC [95% CI]" for k in sets)
    L += [f"| target | test hours | base rate | {head} | persistence AUC | best set (validation) |",
          "|---|---|---|" + "---|" * len(sets) + "---|---|"]
    for k, r in R.items():
        cells = " | ".join(f"{r['sets'][x]['AUC']} {r['sets'][x]['AUC_ci']}" for x in sets)
        L.append(f"| {k} | {r['test']} | {r['base_rate_test']:.2f} | {cells} | {r['persistence']['AUC']} | "
                 f"{SET_NAMES[r['best_set']]} |")
    L += ["", "## Paired differences (same hours)", "",
          "| target | SoLEXS minus persistence | " + ("SHARP added to SoLEXS | SHARP alone minus SoLEXS |" if "both" in sets
                                                    else "") ,
          "|---|---|" + ("---|---|" if "both" in sets else "")]
    for k, r in R.items():
        g = r["gains"]
        row = f"| {k} | {g['xray_minus_persistence']['AUC_gain']:+.3f} {_ci(g['xray_minus_persistence']['ci'])} |"
        if "sharp_added" in g:
            row += (f" {g['sharp_added']['AUC_gain']:+.3f} {_ci(g['sharp_added']['ci'])} | "
                    f"{g['sharp_minus_xray']['AUC_gain']:+.3f} {_ci(g['sharp_minus_xray']['ci'])} |")
        L.append(row)
    L += ["", "TSS at the validation-chosen threshold and Brier skill against the training base rate:", "",
          "| target | " + " | ".join(f"{SET_NAMES[k]} TSS / BSS" for k in sets) + " | persistence TSS |",
          "|---|" + "---|" * len(sets) + "---|"]
    for k, r in R.items():
        L.append(f"| {k} | " + " | ".join(f"{r['sets'][x]['TSS']} / {r['sets'][x]['BSS_vs_training_climatology']}"
                                          for x in sets) + f" | {r['persistence']['TSS']} |")
    L += ["", "## What carries the signal", "", "Drop in test AUC when one feature is shuffled (best set):", ""]
    for k, r in R.items():
        L.append(f"- {k} ({SET_NAMES[r['best_set']]}): " +
                 ", ".join(f"{n} ({v:+.3f})" for n, v in r["top_features"][:4]))
    sig_x = [k for k, r in R.items() if (r["gains"]["xray_minus_persistence"]["ci"] or [0])[0] > 0]
    low_x = [k for k, r in R.items() if (r["gains"]["xray_minus_persistence"]["ci"] or [0, 0])[1] < 0]
    L += ["", "## Reading it", "",
          f"- SoLEXS activity beats persistence (interval above zero) for: {', '.join(sig_x) or 'none'}; "
          f"falls behind it for: {', '.join(low_x) or 'none'}."]
    if "both" in sets:
        up = [k for k, r in R.items() if (r["gains"]["sharp_added"]["ci"] or [0])[0] > 0]
        down = [k for k, r in R.items() if (r["gains"]["sharp_added"]["ci"] or [0, 0])[1] < 0]
        L.append(f"- Adding SHARP to SoLEXS helps (interval above zero) for: {', '.join(up) or 'none'}; "
                 f"hurts for: {', '.join(down) or 'none'}.")
    rates = [r["base_rate_test"] for r in R.values()]
    if rates:
        L.append(f"- Test base rates run {min(rates):.2f}-{max(rates):.2f}; where nearly every window holds a flare "
                 "there is little left to forecast, and AUC says more than TSS.")
    shift = [r["base_rate_test"] - r["base_rate_train"] for r in R.values()]
    if shift:
        L.append(f"- BSS is against the training base rate; the test base rate differs from training by "
                 f"{min(shift):+.2f} to {max(shift):+.2f}, which costs every model calibration, not ranking.")
    L += [""]
    return "\n".join(L)


def figure(s: dict, dest: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED = "#1d2733", "#98a2ad"
    col = {"xray": "#4a6fa5", "sharp": "#b5563c", "both": "#2a8c7c", "persistence": "#98a2ad"}
    sets = [k for k in SETS if k in s["feature_sets"]] + ["persistence"]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    for i, c in enumerate(("C", "M")):
        a = ax[i]
        for j, k in enumerate(sets):
            hs, auc, lo, hi = [], [], [], []
            for H in HORIZONS_H:
                r = s["results"].get(f">={c}1 within {H} h")
                if not r:
                    continue
                m = r["persistence"] if k == "persistence" else r["sets"][k]
                ci = m["AUC_ci"] or [m["AUC"], m["AUC"]]
                hs.append(H)
                auc.append(m["AUC"])
                lo.append(m["AUC"] - ci[0])
                hi.append(ci[1] - m["AUC"])
            if hs:
                a.errorbar(np.array(hs) * (1 + 0.03 * j), auc, yerr=[lo, hi], fmt="o-", ms=4, color=col[k],
                           label=SET_NAMES.get(k, k), capsize=2)
        a.axhline(0.5, color=MUTED, lw=0.8, ls=":")
        a.set_xscale("log")
        a.minorticks_off()
        a.set_xticks(list(HORIZONS_H))
        a.set_xticklabels([f"{h} h" for h in HORIZONS_H])
        a.set_xlabel("forecast window")
        a.set_ylabel("ROC AUC, test period")
        a.set_title(f"A >= {c}1 flare within the window", loc="left", fontsize=10, color=INK)
        a.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(dest)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
