"""Probability calibration: make "70 %" mean 70 %.

The network ranks flares well (AUC ~0.9) but its raw probabilities are not
calibrated: on the archive run the Brier skill of the occurrence heads was
negative. Isotonic regression fitted on the validation split fixes that
(+0.34 Brier skill at 15 min) and, being monotone, cannot change AUC or which
windows a threshold separates.

A calibrator is stored as knots ``{"x": [...], "y": [...]}`` and applied by linear
interpolation, so a frozen model carries it without pickling sklearn objects.
"""

from __future__ import annotations

import numpy as np

IDENTITY = {"x": [0.0, 1.0], "y": [0.0, 1.0]}


def fit(p: np.ndarray, y: np.ndarray) -> dict:
    """Isotonic fit of outcome ``y`` (0/1) on predicted ``p``; identity if it cannot be fitted."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    if ok.sum() < 50 or len(np.unique(y[ok])) < 2:
        return dict(IDENTITY)
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p[ok], y[ok])
    x, yy = np.asarray(iso.X_thresholds_, float), np.asarray(iso.y_thresholds_, float)
    if x.size < 2:
        return dict(IDENTITY)
    return {"x": x.round(6).tolist(), "y": yy.round(6).tolist()}


def apply(cal: dict | None, p: np.ndarray) -> np.ndarray:
    if not cal:
        return np.asarray(p, dtype=np.float64)
    return np.interp(np.asarray(p, dtype=np.float64), cal["x"], cal["y"])
