"""Tests for the master-catalogue detectors (solarflare/catalog.py).

What is checked:

* GOES class strings, including the rounding edge (9.97e-7 is C1.0);
* the piecewise calibration recovers a bent rate -> flux law and stays monotone;
* minute averaging keeps only minutes with enough observed bins;
* the soft rule: start at the first of five rising minutes, alert only once
  the flux is 1.4x the start, half-way end, an unconfirmed bump folded into
  the decay, a second flare during a decay split off, Poisson noise at low
  counts not passed as flares;
* the hard burst detector: no bursts from over-dispersed noise, a burst in
  both detectors found, a one-detector glitch rejected, the background causal;
* attaching bands and merging instruments keep contiguous flares apart and
  record whether the other instrument was observing.

    python -m tests.test_catalog
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarflare.catalog import (  # noqa: E402
    HardBurst, PiecewiseCalibration, SoftEvent, attach_bands, flux_class, hxr_bursts, merge,
    noaa_events, to_minutes, trailing_background,
)

FAILURES: list[str] = []
T0 = 1749999960.0                      # a whole minute


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def flare_curve(n: int = 240, bg: float = 1e-6, peak: float = 5e-6, start: int = 60,
                rise: int = 10, decay: float = 20.0) -> np.ndarray:
    """1-minute GOES-like flux: flat background, linear rise, exponential decay."""
    k = np.arange(n, dtype=np.float64)
    f = np.full(n, bg)
    up = (k >= start) & (k < start + rise)
    f[up] = bg + (peak - bg) * (k[up] - start + 1) / rise
    down = k >= start + rise
    f[down] = bg + (peak - bg) * np.exp(-(k[down] - start - rise + 1) / decay)
    return f


def test_flux_class():
    check("M1.2", flux_class(1.23e-5) == "M1.2")
    check("C1.0 at the boundary", flux_class(1e-6) == "C1.0")
    check("9.97e-7 rounds up to C1.0, not B10.0", flux_class(9.97e-7) == "C1.0", flux_class(9.97e-7))
    check("9.94e-7 stays B9.9", flux_class(9.94e-7) == "B9.9")
    check("X keeps counting past 10", flux_class(1.2e-3) == "X12.0")
    check("below A1 or NaN gives no class", flux_class(5e-9) == "" and flux_class(float("nan")) == "")


def test_piecewise_calibration():
    rng = np.random.default_rng(1)
    lr = rng.uniform(0.0, 4.5, 20000)
    # a power law that bends upward above log rate 3 (saturating detector)
    lf = -6.7 + 0.64 * lr + np.where(lr > 3.0, 0.4 * (lr - 3.0), 0.0) + rng.normal(0, 0.03, lr.size)
    cal = PiecewiseCalibration.fit(lr, lf)
    for x, want in ((1.0, -6.06), (3.0, -4.78), (4.2, -3.53)):
        got = float(np.log10(cal(np.array([10 ** x]))[0]))
        check(f"recovers the law at log rate {x}", abs(got - want) < 0.03, f"{got:.3f} vs {want:.3f}")
    check("knots are monotone", bool(np.all(np.diff(cal.y) >= 0)))
    ext = float(np.log10(cal(np.array([10 ** 5.0]))[0]))
    check("extrapolates past the last knot with the end slope", ext > float(np.log10(cal(np.array([10 ** 4.4]))[0])))
    check("zero or negative rate gives NaN", bool(np.isnan(cal(np.array([0.0, -1.0]))).all()))


def test_to_minutes():
    t = T0 + 20.0 * np.arange(9)                   # three whole minutes at 20 s
    r = np.array([1, 2, 3, 10, 10, 10, 5, 5, 5], dtype=float)
    ok = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0], bool)
    tm, rm, cm, vm = to_minutes(t, r, ok, 20.0)
    check("minute means over observed bins", np.allclose(rm[[0, 2]], [2.0, 5.0]))
    check("a minute with one observed bin is invalid", not vm[1] and vm[0] and vm[2])
    check("counts are per minute", np.isclose(cm[0], 120.0))


def test_soft_rule_single_flare():
    f = flare_curve()
    t = T0 + 60.0 * np.arange(f.size)
    ev = noaa_events(t, f, np.ones(f.size, bool))
    check("one flare", len(ev) == 1, str(len(ev)))
    e = ev[0]
    check("starts at the first rising minute", e.start_unix == t[59], f"{(e.start_unix - T0) / 60}")
    check("peaks at the maximum", e.peak_unix == t[69])
    # flux reaches 1.4 x start (1.4e-6) at minute 60 already, but the rule needs
    # the five-minute rise first: alert at the end of minute 63
    check("alert waits for the full rise", e.detect_unix == t[63] + 60.0, f"{(e.detect_unix - T0) / 60}")
    half = 0.5 * (f[69] + f[59])
    first_below = 70 + int(np.argmax(f[70:] <= half))
    check("ends half-way down", e.end_unix == t[first_below] and e.end_reason == "decay")
    check("class from the peak flux", e.goes_class == "C5.0", e.goes_class)


def test_soft_rule_small_rise_is_not_a_flare():
    f = flare_curve(peak=1.3e-6)                   # rises strictly but only 1.3x
    t = T0 + 60.0 * np.arange(f.size)
    check("a rise below 1.4x is not confirmed", noaa_events(t, f, np.ones(f.size, bool)) == [])


def test_soft_rule_bump_and_second_flare():
    f = flare_curve(n=400)
    bump = flare_curve(n=400, bg=0.0, peak=0.25e-6, start=85, rise=5, decay=5)
    f2 = f + bump                                  # small re-brightening in the decay
    t = T0 + 60.0 * np.arange(f.size)
    ev = noaa_events(t, f2, np.ones(f.size, bool))
    check("an unconfirmed bump stays part of the flare", len(ev) == 1, str(len(ev)))
    second = flare_curve(n=400, bg=0.0, peak=6e-6, start=75, rise=8, decay=20)   # before half-way
    ev = noaa_events(t, f + second, np.ones(f.size, bool))
    check("a new flare during the decay is split off", len(ev) == 2, str(len(ev)))
    if len(ev) == 2:
        check("the first ends where the second starts",
              ev[0].end_unix == ev[1].start_unix and ev[0].end_reason == "next_flare")


def test_soft_rule_poisson_guard():
    rng = np.random.default_rng(3)
    n = 20000
    counts = rng.poisson(30.0, n).astype(float)    # ~B-level SoLEXS: 30 counts a minute
    t = T0 + 60.0 * np.arange(n)
    rate = counts / 60.0
    loose = noaa_events(t, rate, np.ones(n, bool))
    guarded = noaa_events(t, rate, np.ones(n, bool), counts=counts)
    check("pure Poisson noise yields false rises without the guard", len(loose) > 0, str(len(loose)))
    check("and almost none with it", len(guarded) <= max(1, len(loose) // 20),
          f"{len(guarded)} vs {len(loose)}")


def _hxr(n=2000, dt=20.0, scale=10.0, seed=0):
    """Two CZT-like detectors: 70 cts/s with scatter `scale` x Poisson."""
    rng = np.random.default_rng(seed)
    t = T0 + dt * np.arange(n)
    sd = scale * np.sqrt(70.0 / dt)
    return t, [70.0 + rng.normal(0, sd, n), 70.0 + rng.normal(0, sd, n)], [np.ones(n), np.ones(n)]


def test_hxr_bursts():
    t, r, c = _hxr()
    check("no bursts from over-dispersed noise", hxr_bursts(t, r, c, 20.0, "czt_20_40", ["a", "b"]) == [])
    k = np.arange(t.size)
    pulse = 400.0 * np.exp(-0.5 * ((k - 1200) / 4.0) ** 2)
    both = hxr_bursts(t, [r[0] + pulse, r[1] + pulse], c, 20.0, "czt_20_40", ["a", "b"])
    check("a burst in both detectors is found", len(both) == 1, str(len(both)))
    if both:
        check("at the right time", abs(both[0].peak_unix - t[1200]) <= 40.0)
        check("detected before the peak has passed by much", both[0].detect_unix <= t[1206])
    one = hxr_bursts(t, [r[0] + pulse, r[1]], c, 20.0, "czt_20_40", ["a", "b"])
    check("a glitch in one detector only is rejected", one == [], str(len(one)))


def test_background_is_causal():
    t, r, c = _hxr(n=600)
    a = trailing_background(t, r[0], c[0] > 0)
    r2 = r[0].copy()
    r2[400:] += 1000.0
    b = trailing_background(t, r2, c[0] > 0)
    check("changing data after t leaves the background at t alone", np.allclose(a[:400], b[:400], equal_nan=True))
    check("and the guard keeps the next two minutes out too", np.allclose(a[:406], b[:406], equal_nan=True))


def _burst(band, s, p, e, ex=100.0):
    return HardBurst(band, T0 + s, T0 + p, T0 + e, T0 + s + 60, ex, 1.0, 10.0, 0.0, "")


def test_attach_and_merge():
    prim = [_burst("cdte_5_20", 0, 300, 900), _burst("cdte_5_20", 900, 1200, 2000)]
    czt = [_burst("czt_20_40", 1000, 1100, 1150), _burst("czt_20_40", 9000, 9100, 9200)]
    hi = [_burst("czt_40_60", 20000, 20100, 20200)]
    evs = attach_bands(prim, [(czt, True), (hi, False)])
    check("contiguous primary events stay separate", len([e for e in evs if "cdte_5_20" in e.bands]) == 2)
    host = [e for e in evs if e.start_unix == T0 + 900][0]
    check("a CZT burst joins the flare it peaks in", host.bands == ["cdte_5_20", "czt_20_40"])
    check("an unattached standalone band starts its own event", any(e.bands == ["czt_20_40"] for e in evs))
    check("an unattached non-standalone band is dropped", not any("czt_40_60" in e.bands for e in evs))

    soft = [SoftEvent(T0 + 800, T0 + 1300, T0 + 2500, T0 + 1000, 1e-6, 5e-6, "C5.0", "decay"),
            SoftEvent(T0 + 50000, T0 + 50500, T0 + 51000, T0 + 50200, 1e-6, 3e-6, "C3.0", "decay")]
    master = merge(soft, evs, hard_observing=lambda a, b: a < T0 + 30000,
                   soft_observing=lambda a, b: False)
    origins = [m.origin for m in master]
    check("soft flare with a HEL1OS peak inside becomes soft+hard", "soft+hard" in origins)
    lone = [m for m in master if m.origin == "soft"]
    check("a soft flare HEL1OS did not observe says so", len(lone) == 1 and lone[0].hard_observed is False)
    hard_only = [m for m in master if m.origin == "hard"]
    check("unmatched HEL1OS events stay, flagged unobserved by SoLEXS",
          len(hard_only) >= 1 and all(m.soft_observed is False for m in hard_only))
    both = [m for m in master if m.origin == "soft+hard"][0]
    check("the combined alert is the earlier of the two", both.detect_unix == min(both.soft.detect_unix,
                                                                                both.hard.detect_unix))


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} catalogue test groups\n")
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
    print("All catalogue checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
