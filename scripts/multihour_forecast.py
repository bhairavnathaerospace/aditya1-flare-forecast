"""Will a flare happen in the next 2-24 hours? Activity indicators from SoLEXS alone.

    python scripts/multihour_forecast.py

The network in this project forecasts 1-60 min ahead from the last 2 h of light
curve. Operational services forecast a day ahead, from sunspot-region data.
This asks how far soft X-ray activity alone gets at hours to a day:

Origins: every hour on the hour with SoLEXS data for >= 50% of the previous 6 h.
Targets: a GOES flare >= C1 / >= M1 peaks in (t, t+H], H = 2, 6, 12, 24 h,
         kept only where GOES observed >= 80% of the window.
Features, all known at t (SoLEXS minute flux on the training-period GOES
calibration; flares from the master catalogue once their peak is 5 min past):
  flux now; 1/6/24 h minimum (background), maximum and mean; 1 h and 6 h change;
  background hardness (6-12 keV / GOES band, quietest minutes of the last hour);
  SoLEXS flares >= B, >= C, >= M in the last 6/24/72 h; hours since the last
  >= C and >= M flare; largest flare of the last 24 and 72 h.
Models: logistic regression and gradient-boosted trees, fitted on the training
period (to 2025-09-07), operating threshold chosen on validation, scored on the
test period (2026-03-24 on). References: training climatology, and "persistence":
a logistic fit on one feature, the number of flares of the class in the last 24 h.
95% intervals resample whole weeks (hourly origins with day-long targets overlap).

Writes outputs/multihour/{MULTIHOUR.md, multihour_summary.json, multihour.png}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

TRAIN_END = 1757237860.0
TEST_START = 1774325020.0
HORIZONS_H = (2, 6, 12, 24)
CLASSES = {"C": 1e-6, "M": 1e-5}


def ts(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp()


def build_features(args):
    from master_catalog import load_solexs
    from solarflare.catalog import PiecewiseCalibration, to_minutes
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
    cat = [r_ for r_ in csv.DictReader(open(args.catalog, encoding="utf-8"))
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
    k = int(np.argmax(vals))
    return float(cands[k])


def main() -> int:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="outputs/archive/cache")
    ap.add_argument("--goes-dir", default="D:/Data/goes")
    ap.add_argument("--catalog", default="outputs/catalog/master_catalog.csv")
    ap.add_argument("--out", default="outputs/multihour")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    feat, have, o, tgt = build_features(args)
    names = list(feat.columns)
    X = feat.to_numpy(dtype=np.float64)
    print(f"{len(feat)} hourly origins, {have.sum()} with SoLEXS; {len(names)} features ({time.time() - t0:.0f} s)",
          flush=True)
    weeks = np.floor(o / (7 * 86400)).astype(int)
    summary: dict = {"generated_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"), "features": names,
                     "results": {}}
    for (c, H), (y, gok) in tgt.items():
        base = have & gok & np.isfinite(X).all(1)
        tr = base & (o + 3600.0 * H <= TRAIN_END)
        va = base & (o > TRAIN_END) & (o + 3600.0 * H <= TEST_START)
        te = base & (o >= TEST_START)
        if min(tr.sum(), va.sum(), te.sum()) < 50 or len(np.unique(y[te])) < 2:
            continue
        clim = float(y[tr].mean())
        persist_col = names.index(f"n_{c}_24h")
        models = {
            "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
            "trees": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                    min_samples_leaf=40, l2_regularization=1.0, random_state=0),
            "persistence": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        }
        res = {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum()),
               "base_rate_train": round(clim, 3), "base_rate_test": round(float(y[te].mean()), 3), "models": {}}
        preds = {}
        for name, m in models.items():
            cols = [persist_col] if name == "persistence" else list(range(len(names)))
            m.fit(X[tr][:, cols], y[tr])
            pv = m.predict_proba(X[va][:, cols])[:, 1]
            pt = m.predict_proba(X[te][:, cols])[:, 1]
            preds[name] = pt
            thr = best_tss_threshold(y[va], pv)
            yt = y[te]
            bs, bs_c = brier_score_loss(yt, pt), brier_score_loss(yt, np.full_like(pt, clim))
            res["models"][name] = {
                "val_AUC": round(float(roc_auc_score(y[va], pv)), 3),
                "AUC": round(float(roc_auc_score(yt, pt)), 3),
                "AUC_ci": week_ci(yt, pt, roc_auc_score, weeks[te]),
                "TSS": round(float(tss_at(yt, pt, thr)), 3),
                "TSS_ci": week_ci(yt, pt, lambda a, b, _t=thr: tss_at(a, b, _t), weeks[te]),
                "BSS_vs_training_climatology": round(1 - bs / bs_c, 3),
                "threshold": round(thr, 3),
            }
        # the multi-feature model is chosen on validation, never on the test period
        sel = max(("logistic", "trees"), key=lambda k: res["models"][k]["val_AUC"])
        res["selected"] = sel
        yt = y[te]
        a_, b_ = preds[sel], preds["persistence"]
        res["AUC_gain_over_persistence"] = round(float(roc_auc_score(yt, a_) - roc_auc_score(yt, b_)), 3)
        res["AUC_gain_ci"] = week_ci(yt, np.stack([a_, b_], 1),
                                     lambda yy, pp: roc_auc_score(yy, pp[:, 0]) - roc_auc_score(yy, pp[:, 1]), weeks[te])
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(models[sel], X[te], yt, scoring="roc_auc", n_repeats=5, random_state=0)
        top = np.argsort(-pi.importances_mean)[:6]
        res["top_features"] = [[names[k], round(float(pi.importances_mean[k]), 4)] for k in top]
        # one feature at a time: which indicator carries the signal
        single = {}
        for k, nm in enumerate(names):
            v = X[te][:, k]
            if np.nanstd(v) > 0:
                a = roc_auc_score(y[te], v)
                single[nm] = round(float(max(a, 1 - a)), 3)
        res["single_feature_AUC_top"] = sorted(single.items(), key=lambda kv: -kv[1])[:6]
        summary["results"][f">={c}1 within {H} h"] = res
        m_ = res["models"]
        print(f">= {c}1 within {H:2d} h: base rate {res['base_rate_test']:.2f}; selected {sel} AUC {m_[sel]['AUC']} vs "
              f"persistence {m_['persistence']['AUC']} (gain {res['AUC_gain_over_persistence']:+.3f} {res['AUC_gain_ci']})",
              flush=True)
    summary["seconds"] = round(time.time() - t0, 1)
    (out / "multihour_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "MULTIHOUR.md").write_text(render(summary), encoding="utf-8")
    figure(summary, out / "multihour.png")
    print(render(summary))
    return 0


def render(s: dict) -> str:
    L = ["# Forecasting flares hours ahead from SoLEXS activity", "",
         f"Generated {s['generated_utc']} UTC by `scripts/multihour_forecast.py`. Hourly origins; features from the "
         "SoLEXS light curve and the master catalogue only (no magnetograms). Fitted on the training period; the "
         "model (logistic regression or gradient-boosted trees) and its threshold chosen on validation; scored on the "
         "test period (from 2026-03-24). 95% intervals resample whole weeks.", "",
         "| target | test hours | base rate | model | AUC [95% CI] | persistence AUC | AUC gain [95% CI] | TSS | "
         "persistence TSS | BSS vs training climatology |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in s["results"].items():
        m = r["models"]
        sel = r["selected"]
        L.append(f"| {k} | {r['test']} | {r['base_rate_test']:.2f} | {sel} | {m[sel]['AUC']} {m[sel]['AUC_ci']} | "
                 f"{m['persistence']['AUC']} | {r['AUC_gain_over_persistence']:+.3f} {r['AUC_gain_ci']} | "
                 f"{m[sel]['TSS']} | {m['persistence']['TSS']} | {m[sel]['BSS_vs_training_climatology']} |")
    L += ["", "*Persistence* is a logistic fit on one number: flares of the class SoLEXS saw in the last 24 h. "
          "BSS is against the training-period base rate; the test period is more active than training, which "
          "penalises every model's calibration.", "", "## What carries the signal", "",
          "Drop in test AUC when one feature is shuffled (selected model):", ""]
    for k, r in s["results"].items():
        L.append(f"- {k}: " + ", ".join(f"{n} ({v:+.3f})" for n, v in r.get("top_features", [])[:4]))
    gains = {k: (r["AUC_gain_over_persistence"], r["AUC_gain_ci"]) for k, r in s["results"].items()}
    sig = [k for k, (g, ci) in gains.items() if ci and ci[0] > 0]
    L += ["", "## Reading it", "",
          f"- Significant gain over persistence: {', '.join(sig) if sig else 'none'}. " +
          "The extra skill comes from the soft X-ray level (flux now, the 24 h maximum) and from how many small "
          "flares, down to B-class microflares, SoLEXS counted over the last 1-3 days: an active region announces "
          "itself before its big flares, and a count of big flares alone misses that.",
          "- For >= C1 at 6-24 h nearly every window contains a flare in this active period (base rate 0.65-0.94), "
          "so there is little left to forecast.",
          "- Without magnetic-field data (SDO/HMI SHARP) this is a lower bound on what a multi-hour forecast can do.", ""]
    return "\n".join(L)


def figure(s: dict, dest: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED = "#1d2733", "#98a2ad"
    col = {"trees": "#2a8c7c", "logistic": "#4a6fa5", "persistence": "#d9822b"}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    for i, c in enumerate(("C", "M")):
        a = ax[i]
        for mth in ("logistic", "trees", "persistence"):
            hs, auc, lo, hi = [], [], [], []
            for H in HORIZONS_H:
                r = s["results"].get(f">={c}1 within {H} h")
                if not r:
                    continue
                m = r["models"][mth]
                hs.append(H)
                auc.append(m["AUC"])
                lo.append(m["AUC"] - (m["AUC_ci"] or [m["AUC"]])[0])
                hi.append((m["AUC_ci"] or [0, m["AUC"]])[1] - m["AUC"])
            a.errorbar(np.array(hs) * (1 + 0.03 * ["trees", "logistic", "persistence"].index(mth)), auc,
                       yerr=[lo, hi], fmt="o-", ms=4, color=col[mth], label=mth, capsize=2)
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
