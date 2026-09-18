"""Forecast verification metrics.

Accuracy is meaningless for flare forecasting: a model that always says "no
flare" scores 90%+ on a quiet week.  The operational standard is the True Skill
Statistic, which is insensitive to class balance, together with the Heidke
Skill Score; both are reported here alongside the usual POD/FAR/CSI and
probabilistic scores.
"""

from __future__ import annotations

import numpy as np


def contingency(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    return {
        "TP": int(np.sum(y_true & y_pred)),
        "FP": int(np.sum(~y_true & y_pred)),
        "FN": int(np.sum(y_true & ~y_pred)),
        "TN": int(np.sum(~y_true & ~y_pred)),
    }


def skill_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    c = contingency(y_true, y_pred)
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    n = tp + fp + fn + tn

    def d(a, b):
        return a / b if b else 0.0

    pod = d(tp, tp + fn)                 # recall / hit rate
    pofd = d(fp, fp + tn)                # false alarm rate
    far = d(fp, tp + fp)                 # false alarm ratio
    csi = d(tp, tp + fp + fn)            # critical success index
    tss = pod - pofd                     # true skill statistic

    exp_correct = d((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp), n) if n else 0.0
    hss = d((tp + tn) - exp_correct, n - exp_correct) if n else 0.0

    precision = d(tp, tp + fp)
    f1 = d(2 * precision * pod, precision + pod)

    return {
        **{k: float(v) for k, v in c.items()},
        "accuracy": float(d(tp + tn, n)),
        "POD": float(pod), "POFD": float(pofd), "FAR": float(far),
        "CSI": float(csi), "TSS": float(tss), "HSS": float(hss),
        "precision": float(precision), "F1": float(f1),
        "base_rate": float(d(tp + fn, n)),
    }


def best_threshold(y_true: np.ndarray, prob: np.ndarray,
                   metric: str = "TSS") -> tuple[float, dict]:
    """Pick the operating point that maximises a skill score.

    0.5 is the right threshold only when the classes are balanced, which they
    never are here -- so the threshold is a fitted quantity, chosen on
    validation data and then held fixed for the test set.
    """
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5, skill_scores(y_true, prob >= 0.5)
    cands = np.unique(np.round(np.clip(prob, 0, 1), 3))
    cands = np.unique(np.concatenate([cands, np.linspace(0.02, 0.98, 49)]))
    best, best_s, best_m = 0.5, -np.inf, {}
    for t in cands:
        s = skill_scores(y_true, prob >= t)
        if s[metric] > best_s:
            best, best_s, best_m = float(t), s[metric], s
    return best, best_m


def brier_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((prob - y_true) ** 2))


def brier_skill_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Brier score relative to always predicting the climatological rate."""
    if y_true.size == 0:
        return float("nan")
    base = float(np.mean(y_true))
    ref = float(np.mean((base - y_true) ** 2))
    if ref <= 0:
        return float("nan")
    return float(1.0 - brier_score(y_true, prob) / ref)


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = y_true.astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # Average ranks over ties, otherwise AUC is biased for coarse scores.
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
        i = j + 1
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def reliability(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> dict:
    """Reliability (calibration) curve.

    A forecast that says 30% should verify 30% of the time; without this the
    probabilities are just rankings.
    """
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    out = {"bin_centre": [], "predicted": [], "observed": [], "count": []}
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        out["bin_centre"].append(float((edges[b] + edges[b + 1]) / 2))
        out["predicted"].append(float(prob[m].mean()))
        out["observed"].append(float(y_true[m].mean()))
        out["count"].append(int(m.sum()))
    return out


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray,
                      mask: np.ndarray | None = None) -> dict[str, float]:
    if mask is not None:
        m = mask.astype(bool)
        y_true, y_pred = y_true[m], y_pred[m]
    if y_true.size == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def forecast_score_mask(future_valid: np.ndarray, origin_valid: np.ndarray) -> np.ndarray:
    """Windows on which a flux forecast is scored: the future target *and* the
    value at the forecast origin must both be real.

    Where the truth is missing the stored target is 0 (a GOES gap, a stretch
    with no soft X-ray truth). Scored there, persistence "forecasts" log flux 0
    -- six decades off in GOES W/m^2 -- and the model's skill over it is
    inflated. Model, persistence and climatology share this one sample.
    """
    return (np.asarray(future_valid) > 0) & (np.asarray(origin_valid) > 0)


def skill_vs_reference(y_true: np.ndarray, y_pred: np.ndarray,
                       y_ref: np.ndarray, mask: np.ndarray | None = None
                       ) -> float:
    """Fractional RMSE reduction against a reference forecast.

    For short horizons persistence is a genuinely strong baseline; a flare
    forecaster that cannot beat it has not earned its complexity, so this is
    reported next to every regression score.
    """
    if mask is not None:
        m = mask.astype(bool)
        y_true, y_pred, y_ref = y_true[m], y_pred[m], y_ref[m]
    if y_true.size == 0:
        return float("nan")
    rmse_m = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    rmse_r = float(np.sqrt(np.mean((y_ref - y_true) ** 2)))
    if rmse_r <= 0:
        return float("nan")
    return float(1.0 - rmse_m / rmse_r)


def multiclass_scores(y_true: np.ndarray, y_pred: np.ndarray,
                      n_classes: int) -> dict:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    per_class = {}
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_class[c] = {
            "precision": float(prec), "recall": float(rec),
            "f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
            "support": int(cm[c, :].sum()),
        }
    acc = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
    macro_f1 = float(np.mean([per_class[c]["f1"] for c in range(n_classes)]))
    return {"confusion": cm.tolist(), "accuracy": acc,
            "macro_f1": macro_f1, "per_class": per_class}
