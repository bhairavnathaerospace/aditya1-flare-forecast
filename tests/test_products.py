"""Tests for the physical-product extractions (lead time, HXR spectra, timing, temperature, multi-hour).

What is checked, each on synthetic data with a known answer:

* HEL1OS event lists: the product covering an interval wins, the higher
  version wins a tie; events are sorted from packet order, filtered by time and
  energy, disabled pixels dropped, and binned on onboard ticks even when the
  per-packet UTC stamp jitters by a second;
* lead-time helpers: alert episodes merge short gaps, "next alert minute",
  flare-interval overlap queries, minute marking, and the best-TSS threshold;
* HXR spectra: the Poisson power-law fit recovers a known index over a
  background, its interval covers the truth, the fit range stops where the
  excess fades, and the Am-241 line centroid is found;
* timing: readout batches are measured, a known delay is recovered by the
  cross-correlation, and independent noise gives a covariance interval that
  contains zero;
* temperature: the continuum fit recovers a known temperature through the
  silicon efficiency, and the efficiency falls above 10 keV;
* data quality: hours a SoLEXS day file copies from the previous day are masked;
* day-ahead: TSS at a threshold and its best threshold.

    python -m tests.test_products
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solarflare.io import hel1os_events as he  # noqa: E402
from solarflare.products import dayahead as multihour_forecast  # noqa: E402
from solarflare.products import hxr_spectra, hxr_timing  # noqa: E402
from solarflare.products import leadtime as lead_time  # noqa: E402
from solarflare.products import temperature as solexs_temperature  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------- HEL1OS events

def _write_product(root: Path, name: str, rows: dict, disabled: list[int] | None = None) -> Path:
    from astropy.io import fits

    d = root / "2026" / "07" / "03" / name
    (d / "events").mkdir(parents=True)
    (d / "aux" / "cztdis").mkdir(parents=True)
    (d / "aux" / "cztdis" / "czt1dispix.txt").write_text("\n".join(str(p) for p in (disabled or [])))
    hdus = [fits.PrimaryHDU()]
    for det, (obt, mjd, ener, pix) in rows.items():
        cols = [fits.Column("mjd", "D", array=mjd), fits.Column("hlsobt", "D", array=obt),
                fits.Column("ener", "D", array=ener)]
        if pix is not None:
            cols.append(fits.Column("pix", "B", array=pix))
        hdus.append(fits.BinTableHDU.from_columns(cols, name=f"{det}-EVENTS"))
    fits.HDUList(hdus).writeto(d / "events" / "evt.fits")
    return d


def test_event_products_and_reading():
    t_start = 1783080000.0            # 2026-07-03 12:00:00 UTC
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        n = 400
        rng = np.random.default_rng(1)
        obt = 1000.0 + np.round(np.sort(rng.uniform(0, 40, n)), 2)     # 10 ms ticks
        # packet-ordered: swap neighbours in blocks
        perm = np.arange(n)
        for k in range(0, n - 1, 7):
            perm[k], perm[k + 1] = perm[k + 1], perm[k]
        offset = t_start + 5.0 - 1000.0
        jitter = np.repeat(rng.uniform(-1, 1, n // 20 + 1), 20)[:n]      # per-packet UTC jumps
        mjd = (obt + offset + jitter) / 86400.0 + he.MJD_UNIX0
        ener = rng.uniform(15, 150, n)
        pix = rng.integers(1, 10, n).astype(np.uint8)
        rows = {"CZT1": (obt[perm], mjd[perm], ener[perm], pix[perm])}
        _write_product(root, "HLS_20260703_120000_43200sec_lev1_V111", rows, disabled=[3])
        _write_product(root, "HLS_20260703_120000_43200sec_lev1_V211", rows, disabled=[3])
        _write_product(root, "HLS_20260703_000000_43210sec_lev1_V211", rows)
        he.product_index.cache_clear()
        p = he.product_for(root, t_start + 10.0, t_start + 30.0)
        check("the product covering the interval is chosen", p is not None and p.t_start == t_start)
        check("a tie in coverage goes to the higher version", p is not None and p.version == 211)
        ev = he.read_events(p, "CZT1", t_start + 5.0 + 10.0, t_start + 5.0 + 30.0, 20.0, 100.0)
        check("events come back time-sorted", bool(np.all(np.diff(ev.tick) >= 0)))
        # the UTC window maps to ticks through the estimated offset (good to the ~1 s packet jitter)
        tk = np.rint(obt / he.TICK_S).astype(np.int64)
        k0 = int(np.floor((t_start + 15.0 - ev.utc_offset) / he.TICK_S))
        k1 = int(np.ceil((t_start + 35.0 - ev.utc_offset) / he.TICK_S))
        keep = (tk >= k0) & (tk < k1) & (ener >= 20) & (ener < 100) & (pix != 3)
        check("time, energy and disabled-pixel cuts applied", ev.tick.size == int(keep.sum()),
              f"{ev.tick.size} vs {int(keep.sum())}")
        check("ticks come from the onboard clock, not the jittered UTC",
              np.array_equal(np.sort(ev.tick), np.sort(np.rint(obt[keep] / he.TICK_S).astype(np.int64))))
        check("UTC offset is the median over the interval (good to the jitter)", abs(ev.utc_offset - offset) < 1.0)
        c = he.tick_counts(ev, int(ev.tick.min()), 500)
        check("tick counts add up", int(c.sum()) == int((ev.tick < ev.tick.min() + 500).sum()))


# ---------------------------------------------------------------- lead time

def test_lead_time_helpers():
    on = np.zeros(40, bool)
    on[[2, 3, 4, 8, 9, 20, 30, 31]] = True
    s, e = lead_time.episodes(on)
    check("episodes merge gaps of <= 5 minutes", list(zip(s.tolist(), e.tolist())) == [(2, 9), (20, 20), (30, 31)],
          str(list(zip(s.tolist(), e.tolist()))))
    nx = lead_time.next_on(on)
    check("next alert minute", nx[0] == 2 and nx[5] == 8 and nx[21] == 30 and nx[35] == 40)
    fl = lead_time.Flares(np.array([100.0, 500.0]), np.array([200.0, 520.0]))
    got = fl.overlaps(np.array([150.0, 250.0, 510.0, 0.0]), np.array([160.0, 300.0, 600.0, 99.0]))
    check("flare overlap queries", got.tolist() == [True, False, True, False], str(got.tolist()))
    g = lead_time.Grid(0.0, 600.0)
    m = g.mark(np.array([60.0]), np.array([180.0]))
    check("minute marking is inclusive", m.tolist()[:5] == [False, True, True, True, False])
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.3
    s_ = np.where(y, rng.normal(2, 1, y.size), rng.normal(0, 1, y.size))
    thr, best = lead_time.tss_threshold(y.astype(float), s_)
    check("best-TSS threshold sits between the classes", 0.4 < thr < 1.6 and best > 0.6, f"{thr:.2f} {best:.2f}")


# ---------------------------------------------------------------- HXR spectra

def test_power_law_fit_recovers_index():
    rng = np.random.default_rng(3)
    lo, hi = hxr_spectra.EDGES[:-1], hxr_spectra.EDGES[1:]
    use = np.flatnonzero(lo >= 30.0)
    truth = 3.7
    bkg = np.full(use.size, 40.0)
    model = 2000.0 * hxr_spectra.shape(truth, lo[use], hi[use]) / hxr_spectra.shape(truth, lo[use], hi[use]).sum()
    hits = 0
    gammas = []
    for _ in range(40):
        src = rng.poisson(bkg + model).astype(float)
        f = hxr_spectra.fit_power_law(src, bkg, lo[use], hi[use])
        gammas.append(f["gamma"])
        hits += f["gamma_lo"] <= truth <= f["gamma_hi"]
    check("the power-law index is recovered on average", abs(np.mean(gammas) - truth) < 0.1,
          f"{np.mean(gammas):.3f}")
    check("the 68% profile interval covers the truth about 68% of the time", 0.5 <= hits / 40 <= 0.9, str(hits))
    src = np.array([500, 400, 300, 50, 41, 39, 40], float)
    bk = np.full(7, 40.0)
    edges_src = np.zeros(hxr_spectra.EDGES.size - 1)
    edges_bkg = np.zeros_like(edges_src)
    k0 = int(np.flatnonzero(lo >= 30.0)[0])
    edges_src[k0:k0 + 7], edges_bkg[k0:k0 + 7] = src, bk
    r = hxr_spectra.fit_range(edges_src, edges_bkg, 30.0)
    check("the fit range stops at the first bin without a 3-sigma excess", r.tolist() == list(range(k0, k0 + 3)),
          str(r.tolist()))


def test_am241_centroid():
    class E:
        def __init__(self, e):
            self.energy = e

        def utc(self):
            return np.zeros(self.energy.size)

    rng = np.random.default_rng(4)
    e = np.concatenate([rng.uniform(50, 70, 3000), rng.normal(59.3, 0.8, 2000)])
    c = hxr_spectra.am241_line([E(e)], -1.0, 1.0)
    check("Am-241 line centroid found", c is not None and abs(c - 59.3) < 0.3, str(c))


# ---------------------------------------------------------------- timing

def test_timing_helpers():
    tot = np.zeros(6000)
    for k in range(0, 6000, 700):                # a batch of events every 7 s, 0.4 s long
        tot[k:k + 40] = 12.0
    b = hxr_timing.bursts(tot)
    check("readout batches: spacing and length measured", b["spacing_s"] == 7.0 and abs(b["length_s"] - 0.4) < 0.11,
          str(b))
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 3000)
    x = np.convolve(x, np.ones(5) / 5, "same")
    y = np.roll(x, 5) + rng.normal(0, 0.05, x.size)
    lag, cc = hxr_timing.lag_ccf(x, y, 20)
    check("a known 5-bin delay is recovered", abs(lag - 5) < 0.3 and cc > 0.8, f"{lag:.2f}")
    z1, z2 = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
    m, lo, hi = hxr_timing.block_ci(z1 * z2, 50)
    check("independent noise: covariance interval contains zero", lo < 0 < hi, f"[{lo}, {hi}]")


# ---------------------------------------------------------------- temperature

def test_temperature_fit_recovers_truth():
    rng = np.random.default_rng(6)
    e = np.arange(4.0, 12.5, 0.05)
    win = ((e >= 4.3) & (e < 6.2)) | ((e >= 8.6) & (e < 12.0))
    e, de = e[win], np.full(win.sum(), 0.05)
    eff = solexs_temperature.si_efficiency(e)
    truth_mk = 20.0
    kt = truth_mk * solexs_temperature.KEV_PER_MK
    mu = solexs_temperature.continuum(e, de, 20.0, 0.0, kt, eff=eff)
    mu *= 30000.0 / mu.sum()
    bkg = np.full(e.size, 2.0)
    ts_ = []
    for _ in range(20):
        f = solexs_temperature.fit_temperature(rng.poisson(mu + bkg).astype(float), bkg, e, de, 20.0, eff=eff)
        ts_.append(f["T_MK"])
    check("the continuum temperature is recovered", abs(np.mean(ts_) - truth_mk) < 0.5, f"{np.mean(ts_):.2f} MK")
    check("its error is realistic", 0.3 < np.std(ts_) / f["T_err_MK"] < 2.0, f"{np.std(ts_):.2f} vs {f['T_err_MK']:.2f}")
    eff2 = solexs_temperature.si_efficiency(np.array([5.0, 10.0, 15.0]))
    check("silicon efficiency ~1 at 5 keV and falls above 10 keV", eff2[0] > 0.99 and eff2[0] > eff2[1] > eff2[2])
    few = solexs_temperature.fit_temperature(np.full(e.size, 2.0), bkg, e, de, 20.0, eff=eff)
    check("no fit without a net signal", few is None)


# ---------------------------------------------------------------- data quality

def test_duplicate_mask():
    import json

    from solarflare.catalog import build as master_catalog

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "dups.json"
        p.write_text(json.dumps({"duplicates": [{"intervals_unix": [[100.0, 200.0], [500.0, 520.0]]}]}))
        t = np.array([50.0, 100.0, 150.0, 199.9, 200.0, 510.0, 600.0])
        m = master_catalog.duplicate_mask(t, p)
        check("copied intervals are masked, half-open", m.tolist() == [False, True, True, True, False, True, False],
              str(m.tolist()))
        check("no list, no mask", not master_catalog.duplicate_mask(t, Path(tmp) / "missing.json").any())
        # the training pipeline's version (--exclude-intervals): blanks values and coverage
        from solarflare.preprocess.cache import mask_intervals
        from solarflare.preprocess.timeline import GriddedSeries
        s = GriddedSeries(t.copy(), np.ones((t.size, 2), np.float32), np.ones(t.size, np.float32), ["a", "b"])
        n = mask_intervals([s], p)
        check("training mask blanks the copied samples only",
              n == 4 and np.isnan(s.values[1:4]).all() and np.isnan(s.values[5]).all()
              and s.coverage[[0, 4, 6]].tolist() == [1, 1, 1] and not np.isnan(s.values[[0, 4, 6]]).any(),
              f"{n} {s.coverage.tolist()}")


# ---------------------------------------------------------------- multi-hour

def test_multihour_scores():
    y = np.array([0, 0, 1, 1, 1, 0], float)
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.6])
    check("TSS at a threshold", abs(multihour_forecast.tss_at(y, p, 0.5) - (2 / 3 - 1 / 3)) < 1e-9)
    thr = multihour_forecast.best_tss_threshold(y, p)
    check("the best threshold separates the classes as well as possible",
          multihour_forecast.tss_at(y, p, thr) >= multihour_forecast.tss_at(y, p, 0.5))


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} product test groups\n")
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
    print("All product checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
