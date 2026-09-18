"""Tests for the per-flare physics diagnostics (solarflare/physics/onset.py).

What is checked:

* a decaying earlier flare is followed as background, a rising pre-flare trend
  is never projected into the flare, and extrapolation is capped;
* the causal onset is the first bin of a run but only known at its last bin;
* hardness refuses to divide faint counts;
* the Neupert lag comes out with the right sign and size on a synthetic flare
  whose soft X-rays are the delayed integral of its hard X-rays;
* decision-time features do not change when data after the decision time do
  -- the leakage guard for using them in forecasting.

    python -m tests.test_physics
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarflare.physics.onset import (  # noqa: E402
    MAX_EXTRAPOLATE_S, causal_onset, flare_record, hardness, lagged_correlation, linear_background,
)

FAILURES: list[str] = []
DT = 20.0
T0 = 1.715e9


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def synthetic_flare(lag_bins: int = 2, seed: int = 0):
    """2 h at 20 s: decaying background, a flare whose GOES-long analogue is the
    delayed integral of a hard X-ray pulse, hotter at onset than at peak."""
    rng = np.random.default_rng(seed)
    n = 360
    t = T0 + DT * np.arange(n)
    start = T0 + 3600.0
    k = np.arange(n)
    pulse = np.exp(-0.5 * ((k - 190) / 6.0) ** 2) + 0.4 * np.exp(-0.5 * ((k - 184) / 3.0) ** 2)
    pulse[t < start] = 0.0
    hxr = 30.0 + 400.0 * pulse + rng.normal(0, 2.0, n)
    shifted = np.concatenate([np.zeros(lag_bins), pulse[:n - lag_bins]])
    sxr = 40.0 * np.cumsum(shifted)
    decay = 200.0 * np.exp(-(t - T0) / 5000.0)
    gl = 800.0 + decay + sxr + rng.normal(0, 3.0, n)
    frac_hot = np.clip(1.0 - (k - 180) / 60.0, 0.6, 1.0)          # hottest at onset
    lo = 500.0 + 0.5 * decay + 0.5 * sxr + rng.normal(0, 2.0, n)
    hi = 50.0 + 0.05 * decay + 0.25 * sxr * frac_hot + rng.normal(0, 1.0, n)
    ok = np.ones(n, bool)
    peak_i = int(np.argmax(sxr + decay))
    flare = {"start": start, "peak": float(t[peak_i]), "end": float(t[min(peak_i + 60, n - 1)]),
             "peak_flux": 2.5e-5}
    gt = T0 + 60.0 * np.arange(n // 3)
    gb = 1e-6 + 1e-5 * np.interp(gt, t, sxr) / sxr.max()
    ga = 0.08e-6 + 0.3e-5 * np.interp(gt, t, sxr) / sxr.max()
    return t, gl, lo, hi, ok, flare, gt, ga, gb, hxr


def test_background():
    t = T0 + DT * np.arange(200)
    y = 1000.0 - 0.2 * (t - T0)
    ok = np.ones(t.size, bool)
    bg, sd = linear_background(t, y, ok, T0, T0 + 1800.0)
    i = int(np.searchsorted(t, T0 + 1800.0 + 300.0))
    check("a decaying pre-flare background is followed into the flare", abs(bg[i] - y[i]) < 1e-6, f"{bg[i]} vs {y[i]}")
    j = int(np.searchsorted(t, T0 + 1800.0 + MAX_EXTRAPOLATE_S + 900.0))
    check("extrapolation is held after the cap", abs(bg[j] - (1000.0 - 0.2 * (1800.0 + MAX_EXTRAPOLATE_S))) < 1e-6)
    rising = 500.0 + 0.3 * (t - T0)
    bg2, _ = linear_background(t, rising, ok, T0, T0 + 1800.0)
    check("a rising pre-flare trend is not projected into the flare", np.ptp(bg2) == 0.0)
    check("too few pre-flare samples gives no background",
          linear_background(t, y, np.zeros(t.size, bool), T0, T0 + 1800.0) is None)


def test_causal_onset():
    t = T0 + DT * np.arange(50)
    net = np.zeros(50)
    net[[10, 11]] = 10.0     # a two-bin blip: not an onset
    net[20:] = 10.0
    ok = np.ones(50, bool)
    on = causal_onset(t, net, ok, sd=1.0, t_lo=T0, t_hi=T0 + 1000.0)
    check("onset is the first bin of the first 3-bin run", on == (20, 22), str(on))
    ok2 = ok.copy()
    ok2[21] = False
    check("an unobserved bin breaks the run", causal_onset(t, net, ok2, 1.0, T0, T0 + 1000.0) == (22, 24))


def test_hardness_needs_counts():
    hi, lo, ok = np.full(10, 0.2), np.full(10, 1.0), np.ones(10, bool)
    check("faint counts give NaN", np.isnan(hardness(hi, lo, ok, 0, 3, DT)))
    check("bright counts give the ratio", abs(hardness(hi * 10, lo * 10, ok, 0, 10, DT) - 0.2) < 1e-12)


def test_lagged_correlation_sign():
    x = np.exp(-0.5 * ((np.arange(100) - 50) / 4.0) ** 2)
    y = np.roll(x, 3)
    lag, r, r0 = lagged_correlation(x, y, 6)
    check("y following x by 3 bins gives lag +3", lag == 3 and r > 0.999, f"{lag} {r}")
    check("zero-lag correlation is lower than the best", r0 < r)


def test_flare_record_end_to_end():
    t, gl, lo, hi, ok, flare, gt, ga, gb, hxr = synthetic_flare(lag_bins=3)
    rec = flare_record(t, gl, lo, hi, ok, flare, gt, ga, gb, hxr, np.ones(t.size, bool), dt=DT)
    check("record computed", rec["status"] == "ok", rec["status"])
    check("onset found near the injected start",
          abs(rec["onset_minus_goes_start_s"]) <= 5 * 60, str(rec.get("onset_minus_goes_start_s")))
    check("hot onset: onset harder than peak", rec["hardness_onset"] > rec["hardness_peak"],
          f"{rec['hardness_onset']:.3f} vs {rec['hardness_peak']:.3f}")
    check("GOES ratio at onset recovered (injected 0.3)", abs(rec["goes_ratio_onset"] - 0.3) < 0.03,
          str(rec["goes_ratio_onset"]))
    check("HEL1OS diagnostics present", rec["hel1os"])
    # A central difference of a 3-bin-delayed integral peaks 2.5 bins late: 40 or 60 s.
    check("Neupert: d(SXR)/dt lags HXR by the injected delay", rec["neupert_lag_s"] in (40.0, 60.0),
          str(rec["neupert_lag_s"]))
    check("Neupert correlation is strong", rec["neupert_r_best"] > 0.9, str(rec["neupert_r_best"]))


def test_decision_features_do_not_leak():
    t, gl, lo, hi, ok, flare, gt, ga, gb, hxr = synthetic_flare()
    a = flare_record(t, gl, lo, hi, ok, flare, gt, ga, gb, hxr, np.ones(t.size, bool), dt=DT)
    t_dec = a["t_decision"]
    after, gafter = t >= t_dec, gt + 60.0 > t_dec
    scramble = np.random.default_rng(5).uniform(0.2, 5.0, t.size)
    gl2, lo2, hi2, hxr2 = (np.where(after, v * scramble, v) for v in (gl, lo, hi, hxr))
    ga2, gb2 = np.where(gafter, ga * 7.0, ga), np.where(gafter, gb * 0.3, gb)
    b = flare_record(t, gl2, lo2, hi2, ok, flare, gt, ga2, gb2, hxr2, np.ones(t.size, bool), dt=DT)
    keys = [k for k in a if k.startswith("dec_")] + ["t_onset", "t_decision", "hardness_onset",
                                                     "hardness_background", "goes_ratio_onset"]
    def equal(u, v):
        return u == v or (isinstance(u, float) and isinstance(v, float) and np.isnan(u) and np.isnan(v))

    same = all(equal(a[k], b[k]) for k in keys)
    check("decision-time features ignore everything after the decision time", same,
          str({k: (a[k], b[k]) for k in keys if not equal(a[k], b[k])}))
    check("and the scrambled future does change the hindsight diagnostics",
          a["hardness_peak"] != b["hardness_peak"] or a["t_peak_solexs"] != b["t_peak_solexs"])


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} physics test groups\n")
    for fn in tests:
        print(f"{fn.__name__}:")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            FAILURES.append(f"{fn.__name__} (exception: {exc})")
        print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("All physics checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
