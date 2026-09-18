"""Tests for reading the HEL1OS mission archive.

Every check here is a failure seen on the real products, not a hypothetical:

* CdTe band extensions of one file have different lengths and sub-second
  phases; the first reader dropped every band of a different length.
* STAT_ERR is sqrt(counts), so a zero-count sample looks exactly like "no
  telemetry"; deciding per band biased faint CdTe bands up to 10x.
* CZT1 and CZT2 wrote identical band names, so stitching mixed the two
  detectors bin by bin although CZT2 reads 0.9-1.5x CZT1.
* The same interval ships as V111 and V211; the newer version must win.
* Several overlapping products per day must become one source each, not one
  per detector file.

    python -m tests.test_hel1os
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarflare.config import Config, PreprocessConfig  # noqa: E402

FAILURES: list[str] = []
T0_MJD = 61294.5  # 2026-09-11 12:00 UTC


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


CZT_BANDS = ((20.0, 40.0), (40.0, 60.0), (60.0, 80.0), (80.0, 150.0), (18.0, 160.0))
CDTE_BANDS = ((5.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, 60.0), (1.8, 90.0))


def write_lc(path: Path, detnam: str, bands, n: int, rate_of_band, every: int = 6,
             start_s=None, phase_s=None, t0_mjd: float = T0_MJD) -> Path:
    """A light curve file in the L1 layout.

    ``rate_of_band(i)`` gives the counts written at sampled slots of band i
    (0 writes a zero-count sample). ``start_s``/``phase_s`` offset each band's
    grid, as CdTe files do.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    hdus = [fits.PrimaryHDU()]
    hdus[0].header["ISOSTART"] = "2026-09-11T12:00:00"
    for i, (lo, hi) in enumerate(bands):
        s = int(start_s[i]) if start_s else 0
        ph = float(phase_s[i]) if phase_s else 0.0
        m = n - s
        sec = s + ph + np.arange(m, dtype=np.float64)
        ctr = np.zeros(m)
        sampled = (np.arange(m) + s) % every == 0
        ctr[sampled] = rate_of_band(i)
        err = np.sqrt(ctr)
        h = fits.BinTableHDU.from_columns(fits.ColDefs([
            fits.Column(name="MJD", format="D", array=t0_mjd + sec / 86400.0),
            fits.Column(name="ISOT", format="30A", array=np.array(["x"] * m)),
            fits.Column(name="CTR", format="D", array=ctr),
            fits.Column(name="STAT_ERR", format="D", array=err),
        ]), name=f"{detnam.upper()}_LC_BAND_{lo:.2f}KEV_TO_{hi:.2f}KEV")
        h.header["ELOW"] = lo
        h.header["EHIGH"] = hi
        h.header["DETNAM"] = detnam
        hdus.append(h)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def write_product(root: Path, name: str, n: int = 7200, czt1=100.0, czt2=50.0,
                  cdte=4.0, t0_mjd: float = T0_MJD) -> Path:
    day = name.split("_")[1]
    prod = root / day[:4] / day[4:6] / day[6:] / name
    write_lc(prod / "czt/lightcurve_czt1.fits", "CZT1", CZT_BANDS, n, lambda i: czt1, t0_mjd=t0_mjd)
    write_lc(prod / "czt/lightcurve_czt2.fits", "CZT2", CZT_BANDS, n, lambda i: czt2, every=5, t0_mjd=t0_mjd)
    write_lc(prod / "cdte/lightcurve_cdte1.fits", "CdTe1", CDTE_BANDS, n, lambda i: cdte,
             every=8, start_s=(0, 32, 16, 0, 0), phase_s=(0.726, 0.953, 0.745, 0.726, 0.726),
             t0_mjd=t0_mjd)
    (prod / "aux").mkdir(parents=True, exist_ok=True)
    return prod


# ---------------------------------------------------------------------------

def test_cdte_bands_of_different_length_and_phase_all_kept():
    from solarflare.io.hel1os import read_hel1os_lightcurve
    with tempfile.TemporaryDirectory() as d:
        p = write_lc(Path(d) / "lightcurve_cdte1.fits", "CdTe1", CDTE_BANDS, 800, lambda i: 3.0,
                     every=8, start_s=(0, 32, 16, 0, 0),
                     phase_s=(0.726, 0.953, 0.745, 0.726, 0.09))
        obs = read_hel1os_lightcurve(p)
    check("every CdTe band is kept with its real energy edges",
          obs.bands_kev.tolist() == [list(b) for b in CDTE_BANDS], str(obs.bands_kev.tolist()))
    # 800 s of data; mapping a 0.09 s-phase band onto a 0.73 s grid may add
    # one slot at an edge. A union of phases would give ~2000 rows.
    check("bands are aligned on one 1 s grid, not a union of phases",
          obs.n in (800, 801), f"n={obs.n}")
    sampled_rows = np.flatnonzero(obs.any_valid)
    same_slots = all(np.array_equal(np.flatnonzero(obs.valid[:, i]), sampled_rows) for i in range(5))
    check("a sample lands in the same slot in every band despite phase offsets", same_slots)


def test_zero_count_samples_are_real_zeros():
    from solarflare.io.hel1os import read_hel1os_lightcurve
    with tempfile.TemporaryDirectory() as d:
        # Band 0 sees 5 counts per sample, band 1 sees none: genuine zeros.
        p = write_lc(Path(d) / "lightcurve_cdte1.fits", "CdTe1", CDTE_BANDS[:2], 600,
                     lambda i: 5.0 if i == 0 else 0.0, every=6)
        obs = read_hel1os_lightcurve(p)
    check("slots are sampled at the telemetry cadence (1 in 6)",
          int(obs.any_valid.sum()) == 100, f"{int(obs.any_valid.sum())}")
    check("a silent band at a sampled slot is a valid zero, not missing",
          int(obs.valid[:, 1].sum()) == 100 and float(np.nanmean(obs.rates[:, 1])) == 0.0,
          f"valid={int(obs.valid[:, 1].sum())} mean={np.nanmean(obs.rates[:, 1])}")
    check("fill slots (all bands zero, no error) stay missing",
          int((~obs.any_valid).sum()) == 500)


def test_fill_rule_still_rejects_fill_rows():
    """The original 83%-fill finding must survive the per-slot rule."""
    from solarflare.io.hel1os import read_hel1os_lightcurve
    with tempfile.TemporaryDirectory() as d:
        p = write_lc(Path(d) / "lightcurve_czt1.fits", "CZT1", CZT_BANDS, 600, lambda i: 150.0)
        obs = read_hel1os_lightcurve(p)
    w = obs.wide_band_index
    check("masked mean is the sampled rate, not diluted by fill rows",
          abs(float(np.nanmean(obs.rates[:, w])) - 150.0) < 1e-6)


def test_raw_layout_keeps_detectors_apart():
    from solarflare.io.hel1os import read_hel1os_product
    from solarflare.preprocess.features import HEL1OS_RAW_NAMES, hel1os_product_raw, hel1os_raw
    cfg = PreprocessConfig()
    with tempfile.TemporaryDirectory() as d:
        prod = write_product(Path(d), "HLS_20260911_120000_7200sec_lev1_V111")
        obs = read_hel1os_product(prod)
        check("all three detector files are read", sorted(o.detector_key for o in obs)
              == ["cdte1", "czt1", "czt2"], str([o.detector for o in obs]))
        czt1 = next(o for o in obs if o.detector_key == "czt1")
        single = hel1os_raw(czt1, cfg)
        col = {n: i for i, n in enumerate(HEL1OS_RAW_NAMES)}
        check("a CZT1 file fills only CZT1 columns",
              np.isfinite(single.values[:, col["hls_czt1_18_160keV"]]).any()
              and not np.isfinite(single.values[:, col["hls_czt2_18_160keV"]]).any())
        raw = hel1os_product_raw(obs, cfg)
    c1 = np.nanmedian(raw.values[:, col["hls_czt1_18_160keV"]])
    c2 = np.nanmedian(raw.values[:, col["hls_czt2_18_160keV"]])
    check("merged product keeps CZT1 (100) and CZT2 (50) in separate columns",
          abs(c1 - 100) < 1e-6 and abs(c2 - 50) < 1e-6, f"czt1={c1} czt2={c2}")
    check("CdTe is merged in too", np.isfinite(raw.values[:, col["hls_cdte1_5_20keV"]]).mean() > 0.9)
    check("CZT2 columns never contain CZT1 values",
          np.nanmax(raw.values[:, col["hls_czt2_18_160keV"]]) < 51)
    check("product version becomes stitching priority", raw.priority == 111.0, str(raw.priority))
    check("absent detector (CdTe2) has zero coverage, not NaN",
          float(np.nanmax(raw.values[:, col["hls_cdte2_cov"]])) == 0.0)


def test_unknown_detector_is_an_error_not_a_silent_merge():
    from solarflare.io.hel1os import read_hel1os_lightcurve
    from solarflare.preprocess.features import hel1os_raw
    with tempfile.TemporaryDirectory() as d:
        p = write_lc(Path(d) / "lightcurve_czt9.fits", "CZT9", CZT_BANDS, 120, lambda i: 1.0)
        obs = read_hel1os_lightcurve(p)
    try:
        hel1os_raw(obs, PreprocessConfig())
        check("unknown detector raises", False)
    except ValueError:
        check("unknown detector raises", True)


def test_stitch_prefers_newer_version():
    from solarflare.preprocess.grid import GriddedSeries
    from solarflare.preprocess.timeline import stitch
    dt = 20.0
    t = np.arange(10) * dt
    old = GriddedSeries(t, np.full((10, 1), 1.0), np.full(10, 1.0), ["x"], priority=111)
    new = GriddedSeries(t[3:7], np.full((4, 1), 2.0), np.full(4, 0.3), ["x"], priority=211)
    for order in ((old, new), (new, old)):
        s = stitch(list(order), dt, 3600)[0]
        v = s.values[:, 0]
        ok = np.array_equal(v, [1, 1, 1, 2, 2, 2, 2, 1, 1, 1])
        check(f"V211 wins where present even with lower coverage (order {order[0].priority:g} first)",
              ok, str(v))
    a = GriddedSeries(t, np.full((10, 1), 1.0), np.full(10, 0.5), ["x"])
    b = GriddedSeries(t, np.full((10, 1), 2.0), np.full(10, 0.9), ["x"])
    check("equal priority still falls back to better coverage",
          np.all(stitch([a, b], dt, 3600)[0].values[:, 0] == 2.0))


def test_index_groups_detectors_into_products():
    from solarflare.preprocess.cache import index_sources
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        write_product(root, "HLS_20260911_120000_7200sec_lev1_V111")
        write_product(root, "HLS_20260911_120000_7200sec_lev1_V211")
        half = write_product(root / ".tmp_x", "HLS_20260912_000000_7200sec_lev1_V111")
        srcs = [s for s in index_sources([root]) if s.kind == "hel1os"]
        half_written = half.exists()
    check("one source per product, not per detector file", len(srcs) == 2, str(srcs))
    check("versions parsed", sorted(s.version for s in srcs) == ["V111", "V211"])
    check("half-extracted products are skipped", half_written
          and all(".tmp_" not in s.path for s in srcs))


def test_end_to_end_cache_to_features():
    from solarflare.preprocess.cache import build_cache, index_sources, load_cached
    from solarflare.preprocess.dataset import build_segments_from_raw
    from solarflare.preprocess.features import hel1os_feature_names
    cfg = Config()
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        write_product(root / "data", "HLS_20260911_120000_7200sec_lev1_V111", czt1=100, czt2=50)
        # The same interval reprocessed, with different rates: must replace V111.
        write_product(root / "data", "HLS_20260911_120000_7200sec_lev1_V211", czt1=300, czt2=150)
        srcs = index_sources([root / "data"])
        entries = build_cache(srcs, cfg.pre, root / "cache", workers=1, verbose=False)
        check("both products cached", sum(e["status"] == "ok" for e in entries) == 2,
              str([e.get("error") for e in entries]))
        again = build_cache(srcs, cfg.pre, root / "cache", workers=1, verbose=False)
        check("second build reuses the cache", [e["key"] for e in again] == [e["key"] for e in entries])
        hard = load_cached(entries, root / "cache", "hel1os")
    segs, meta = build_segments_from_raw([], hard, cfg)
    names = hel1os_feature_names(cfg.pre)
    check("48 hard features, named", meta["hard_features"] == len(names) == 48,
          f"{meta['hard_features']} vs {len(names)}")
    h = segs[0].hard
    m = segs[0].hard_mask > 0
    c1 = np.expm1(np.median(h[m, names.index("log_hls_czt1_18_160keV")]))
    c2 = np.expm1(np.median(h[m, names.index("log_hls_czt2_18_160keV")]))
    check("features come from the newer version and keep detectors apart",
          abs(c1 - 300) < 1 and abs(c2 - 150) < 1, f"czt1={c1:.1f} czt2={c2:.1f}")
    check("no NaN or inf reaches the model input", np.isfinite(h).all())


def test_sliding_percentile_matches_numpy():
    """The fast archive path must equal np.nanpercentile, gaps and ties included."""
    import solarflare.preprocess.grid as g
    rng = np.random.default_rng(3)
    x = rng.lognormal(0, 2, 6000)
    x[rng.random(6000) < 0.3] = np.nan
    x[2000:4300] = np.nan          # a gap longer than the window
    x[5000:5200] = 7.0             # ties
    saved = g._SLIDING_MIN_ROWS
    worst = 0.0
    try:
        for fn in (g.trailing_percentile, g.running_percentile):
            for q in (0.0, 10.0, 37.3, 100.0):
                g._SLIDING_MIN_ROWS = 10**12
                ref = fn(x, 1080, q)
                g._SLIDING_MIN_ROWS = 0
                got = fn(x, 1080, q)
                if not np.array_equal(np.isnan(ref), np.isnan(got)):
                    worst = np.inf
                    break
                f = np.isfinite(ref)
                worst = max(worst, float(np.max(np.abs(got[f] - ref[f]) / np.abs(ref[f]).clip(1e-300))))
    finally:
        g._SLIDING_MIN_ROWS = saved
    check("sliding percentile == numpy to rounding (<=1e-12), same NaN pattern",
          worst <= 1e-12, f"worst relative difference {worst:.2e}")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} HEL1OS test groups\n")
    for t in tests:
        print(f"{t.__name__}:")
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  ERROR {t.__name__}: {type(exc).__name__}: {exc}")
            FAILURES.append(f"{t.__name__} (exception)")
        print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("All HEL1OS checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
