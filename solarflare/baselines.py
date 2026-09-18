"""Non-deep baselines on identical splits and identical features.

A deep model is only worth its complexity if it beats the simple things. These
run on exactly the same windows, targets and splits as the network, so the
comparison is apples to apples:

* **climatology** -- always predict the training base rate.
* **persistence** -- the current value, carried forward.
* **logistic regression** on hand-summarised window statistics.
* **gradient boosting** on the same statistics.

The summary statistics deliberately capture what a domain expert would look at
(current level, ratio to background, recent slope, hardness), so the deep model
has to earn its keep against informed features, not against noise.
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .pipeline import Prepared
from .preprocess.dataset import build_targets
from .metrics import skill_scores, best_threshold, roc_auc, brier_skill_score, \
    regression_scores, skill_vs_reference, forecast_score_mask


def window_summary(seg, sl: slice) -> np.ndarray:
    """Compact hand-engineered description of one window.

    Last value, recent means over three timescales, slopes, and dispersion --
    per channel, for both instruments, plus the availability fractions.
    """
    feats: list[float] = []
    for x, mask in ((seg.soft[sl], seg.soft_mask[sl]),
                    (seg.hard[sl], seg.hard_mask[sl])):
        n = x.shape[0]
        m = mask > 0
        if not m.any():
            feats.extend([0.0] * (x.shape[1] * 6 + 1))
            continue
        xv = np.where(m[:, None], x, np.nan)
        with np.errstate(invalid="ignore"):
            last = np.nan_to_num(x[-1], nan=0.0)
            m_short = np.nan_to_num(np.nanmean(xv[-max(n // 12, 1):], axis=0), nan=0.0)
            m_mid = np.nan_to_num(np.nanmean(xv[-max(n // 4, 1):], axis=0), nan=0.0)
            m_long = np.nan_to_num(np.nanmean(xv, axis=0), nan=0.0)
            std = np.nan_to_num(np.nanstd(xv, axis=0), nan=0.0)
        slope = m_short - m_mid
        feats.extend(np.concatenate([last, m_short, m_mid, m_long, std, slope]).tolist())
        feats.append(float(m.mean()))
    return np.asarray(feats, dtype=np.float32)


def build_matrix(prep: Prepared, cfg: Config, idx: list[int]):
    L = cfg.steps_per_window
    targets = {i: build_targets(s, cfg) for i, s in enumerate(prep.segments)}
    X, rows = [], []
    for i in idx:
        w = prep.windows[i]
        seg = prep.segments[w.seg]
        t = targets[w.seg]
        j = w.end - 1
        X.append(window_summary(seg, slice(w.end - L, w.end)))
        rows.append({
            "in_flare": float(t["in_flare"][j]),
            "nowcast": float(t["nowcast"][j]),
            "nowcast_mask": float(t["nowcast_mask"][j]),
            "forecast": t["forecast"][j].copy(),
            "forecast_mask": t["forecast_mask"][j].copy(),
            "occurrence": t["occurrence"][j].copy(),
            "occurrence_mask": t["occurrence_mask"][j].copy(),
        })
    X = np.stack(X) if X else np.zeros((0, 1), np.float32)
    out = {
        "X": np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0),
        "in_flare": np.array([r["in_flare"] for r in rows]),
        "nowcast": np.array([r["nowcast"] for r in rows]),
        "nowcast_mask": np.array([r["nowcast_mask"] for r in rows]),
        "forecast": np.stack([r["forecast"] for r in rows]) if rows else np.zeros((0, 1)),
        "forecast_mask": np.stack([r["forecast_mask"] for r in rows]) if rows else np.zeros((0, 1)),
        "occurrence": np.stack([r["occurrence"] for r in rows]) if rows else np.zeros((0, 1)),
        "occurrence_mask": np.stack([r["occurrence_mask"] for r in rows]) if rows else np.zeros((0, 1)),
    }
    return out


def run_baselines(prep: Prepared, cfg: Config, verbose: bool = True) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    tr = build_matrix(prep, cfg, prep.splits["train"])
    va = build_matrix(prep, cfg, prep.splits["val"])
    te = build_matrix(prep, cfg, prep.splits["test"])
    if te["X"].shape[0] == 0:
        return {"error": "empty test split"}

    scaler = StandardScaler().fit(tr["X"])
    Xtr, Xva, Xte = (scaler.transform(d["X"]) for d in (tr, va, te))

    report: dict = {}

    # --- flare in progress -------------------------------------------
    mtr = tr["nowcast_mask"] > 0
    mte = te["nowcast_mask"] > 0
    mva = va["nowcast_mask"] > 0
    ytr, yte = tr["in_flare"][mtr], te["in_flare"][mte]

    base_rate = float(ytr.mean()) if ytr.size else 0.0
    report["climatology"] = {
        "in_flare": {
            "Brier": float(np.mean((base_rate - yte) ** 2)) if yte.size else float("nan"),
            "TSS": 0.0,
            "predicted_rate": base_rate,
        }
    }

    if ytr.size and len(np.unique(ytr)) > 1:
        for name, mk in (("logistic", LogisticRegression(max_iter=2000, C=0.5,
                                                         class_weight="balanced")),
                         ("gbdt", HistGradientBoostingClassifier(
                             max_iter=200, learning_rate=0.08, max_depth=4,
                             l2_regularization=1.0, random_state=0))):
            mk.fit(Xtr[mtr], ytr)
            p_va = mk.predict_proba(Xva[mva])[:, 1] if mva.sum() else np.zeros(0)
            thr = 0.5
            if p_va.size and len(np.unique(va["in_flare"][mva])) > 1:
                thr, _ = best_threshold(va["in_flare"][mva], p_va, "TSS")
            p_te = mk.predict_proba(Xte[mte])[:, 1]
            s = skill_scores(yte, p_te >= thr)
            s.update({"AUC": roc_auc(yte, p_te),
                      "BSS_vs_climatology": brier_skill_score(yte, p_te),
                      "threshold": float(thr)})
            report.setdefault(name, {})["in_flare"] = s

    # --- occurrence within H -----------------------------------------
    for name, make in (("logistic", lambda: LogisticRegression(
                            max_iter=2000, C=0.5, class_weight="balanced")),
                       ("gbdt", lambda: HistGradientBoostingClassifier(
                            max_iter=200, learning_rate=0.08, max_depth=4,
                            l2_regularization=1.0, random_state=0))):
        occ_out = {}
        for h, hs in enumerate(cfg.win.occurrence_horizons_s):
            a = tr["occurrence_mask"][:, h] > 0
            b = te["occurrence_mask"][:, h] > 0
            c = va["occurrence_mask"][:, h] > 0
            if a.sum() == 0 or b.sum() == 0 or len(np.unique(tr["occurrence"][a, h])) < 2:
                continue
            mk = make()
            mk.fit(Xtr[a], tr["occurrence"][a, h])
            thr = 0.5
            if c.sum() and len(np.unique(va["occurrence"][c, h])) > 1:
                thr, _ = best_threshold(va["occurrence"][c, h],
                                        mk.predict_proba(Xva[c])[:, 1], "TSS")
            p = mk.predict_proba(Xte[b])[:, 1]
            y = te["occurrence"][b, h]
            s = skill_scores(y, p >= thr)
            s.update({"AUC": roc_auc(y, p),
                      "BSS_vs_climatology": brier_skill_score(y, p)})
            occ_out[f"{int(hs/60)}min"] = s
        report.setdefault(name, {})["occurrence"] = occ_out

    # --- forecast regression, vs persistence -------------------------
    persist_te = te["nowcast"]
    fr_persist, fr_gbdt = {}, {}
    for h, hs in enumerate(cfg.win.forecast_horizons_s):
        a = tr["forecast_mask"][:, h] > 0
        # Scored only where the origin value is real, as in evaluate.py.
        b = forecast_score_mask(te["forecast_mask"][:, h], te["nowcast_mask"])
        if b.sum() == 0:
            continue
        key = f"{int(hs/60)}min"
        fr_persist[key] = regression_scores(te["forecast"][:, h], persist_te, b)
        if a.sum() > 20:
            reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08,
                                                max_depth=4, random_state=0)
            reg.fit(Xtr[a], tr["forecast"][a, h])
            pred = reg.predict(Xte)
            r = regression_scores(te["forecast"][:, h], pred, b)
            r["skill_vs_persistence"] = skill_vs_reference(
                te["forecast"][:, h], pred, persist_te, b)
            fr_gbdt[key] = r
    report["persistence"] = {"forecast": fr_persist}
    report.setdefault("gbdt", {})["forecast"] = fr_gbdt

    if verbose:
        _print(report, cfg)
    return report


def _print(r: dict, cfg: Config) -> None:
    print("\n" + "=" * 74)
    print("BASELINES (same splits, same windows)")
    print("=" * 74)
    print("\nFlare in progress:")
    print(f"  {'model':>12}  {'TSS':>6}  {'HSS':>6}  {'AUC':>6}  {'BSS':>7}")
    for name in ("logistic", "gbdt"):
        s = r.get(name, {}).get("in_flare")
        if s:
            print(f"  {name:>12}  {s['TSS']:6.3f}  {s['HSS']:6.3f}  "
                  f"{s['AUC']:6.3f}  {s['BSS_vs_climatology']:7.3f}")
    c = r.get("climatology", {}).get("in_flare")
    if c:
        print(f"  {'climatology':>12}  {0.0:6.3f}  {0.0:6.3f}  "
              f"{0.5:6.3f}  {0.0:7.3f}")

    print("\nFlare occurrence within horizon (TSS):")
    hs = [f"{int(h/60)}min" for h in cfg.win.occurrence_horizons_s]
    print(f"  {'model':>12}  " + "  ".join(f"{h:>7}" for h in hs))
    for name in ("logistic", "gbdt"):
        occ = r.get(name, {}).get("occurrence", {})
        if occ:
            print(f"  {name:>12}  " + "  ".join(
                f"{occ.get(h, {}).get('TSS', float('nan')):7.3f}" for h in hs))

    print("\nForecast RMSE (log flux):")
    fh = [f"{int(h/60)}min" for h in cfg.win.forecast_horizons_s]
    print(f"  {'model':>12}  " + "  ".join(f"{h:>7}" for h in fh))
    for name, key in (("persistence", "forecast"), ("gbdt", "forecast")):
        d = r.get(name, {}).get(key, {})
        if d:
            print(f"  {name:>12}  " + "  ".join(
                f"{d.get(h, {}).get('RMSE', float('nan')):7.4f}" for h in fh))
