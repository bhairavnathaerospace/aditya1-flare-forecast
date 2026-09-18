"""Tests for GOES-labelled training (solarflare/io/goes.py, label_source="goes").

Why GOES truth exists: on the 2024-2026 archive the project's own SoLEXS
detector marked 66% of observed time as "in a flare" (GOES-18: 9.5%) and found
only ~46% of GOES C-class flares. What is checked here:

* the flare summary is read the way NOAA writes it -- START/PEAK/END records
  tied by flare_id, epoch 2000-01-01 12:00 UTC, some flares with no END;
* flagged or non-positive 1-minute fluxes are never used as truth;
* with label_source="goes" the events ARE the GOES list (>= min class), not
  whatever the SoLEXS detector finds, and flux targets are log10 W/m^2;
* "large flare" is the GOES M1.0 boundary, and rise targets use GOES peaks;
* rises cut by the start of a data segment are not used for peak forecasting;
* two configurations sharing a cache folder keep each other's entries.

    python -m tests.test_goes
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarflare.config import Config, PreprocessConfig  # noqa: E402
from tests.test_robustness import _make_solexs  # noqa: E402

FAILURES: list[str] = []
EPOCH = 946728000.0
T0 = 1.789e9


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def write_flsum(path: Path, flares: list[tuple]) -> None:
    """flares: (flare_id, start, peak, end or None, class, background) in unix s."""
    import h5py
    t, status, cls, fid, bg, flux = [], [], [], [], [], []
    for i, s, p, e, c, b in flares:
        recs = [(s, "EVENT_START", ""), (p, "EVENT_PEAK", c)]
        if e is not None:
            recs += [(e, "EVENT_END", ""), (e + 60, "POST_EVENT", "")]
        for tt, st, cc in recs:
            t.append(tt - EPOCH)
            status.append(st)
            cls.append(cc)
            fid.append(i)
            bg.append(b)
            flux.append(1e-6)
    order = np.argsort(t, kind="stable")
    sdt = h5py.string_dtype()
    with h5py.File(path, "w") as h:
        h["time"] = np.asarray(t)[order]
        h.create_dataset("status", data=np.asarray(status, dtype=object)[order], dtype=sdt)
        h.create_dataset("flare_class", data=np.asarray(cls, dtype=object)[order], dtype=sdt)
        h["flare_id"] = np.asarray(fid, dtype=np.int64)[order]
        h["background_flux"] = np.asarray(bg, dtype=np.float32)[order]
        h["xrsb_flux"] = np.asarray(flux, dtype=np.float32)[order]


def write_avg1m(path: Path, t0: float, n_min: int, flux_fn, bad: set[int] = frozenset()) -> None:
    import h5py
    t = t0 + 60.0 * np.arange(n_min)
    f = np.array([flux_fn(x) for x in t], dtype=np.float32)
    flag = np.zeros(n_min, dtype=np.uint8)
    for i in bad:
        flag[i] = 1
    with h5py.File(path, "w") as h:
        h["time"] = t - EPOCH
        h["xrsb_flux"] = f
        h["xrsb_flag"] = flag


# ---------------------------------------------------------------------------

def test_flare_summary_reader():
    from solarflare.io.goes import read_flare_summary
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sci_xrsf-l2-flsum_test.nc"
        write_flsum(p, [
            (1, T0 + 0, T0 + 600, T0 + 1500, "C2.0", 1e-6),
            (2, T0 + 3000, T0 + 3300, None, "M1.5", 2e-6),       # no END; next starts at 3500
            (3, T0 + 3500, T0 + 3700, T0 + 4200, "B5.0", 8e-7),
        ])
        fl = read_flare_summary(p)
    check("three flares, sorted by peak", [f.flare_id for f in fl] == [1, 2, 3], str(fl))
    check("GOES epoch (2000-01-01 12:00 UTC) converted exactly",
          fl[0].start_unix == T0 and fl[0].peak_unix == T0 + 600)
    check("class and peak flux parsed", fl[1].goes_class == "M1.5" and abs(fl[1].peak_flux - 1.5e-5) < 1e-12)
    check("a missing end is estimated and flagged, never past the next flare's start",
          fl[1].end_estimated and fl[1].end_unix == T0 + 3500, f"{fl[1].end_unix - T0}")
    check("a present end is used as is", not fl[0].end_estimated and fl[0].end_unix == T0 + 1500)


def test_xrs_1min_reader_and_grid():
    from solarflare.io.goes import GoesTruth, read_xrs_1min
    with tempfile.TemporaryDirectory() as d:
        a = Path(d) / "a.nc"
        b = Path(d) / "b.nc"
        write_avg1m(a, T0, 10, lambda t: 1e-6, bad={3})
        write_avg1m(b, T0 + 600, 10, lambda t: -1.0 if t == T0 + 660 else 2e-6)
        t, f = read_xrs_1min([b, a])
    check("files are merged in time order", np.all(np.diff(t) > 0) and t.size == 20)
    check("flagged samples are not truth", np.isnan(f[3]))
    check("non-positive fluxes are not truth", np.isnan(f[11]))
    g = GoesTruth([], t, f, [])
    grid = T0 + 20.0 * np.arange(-3, 70)
    on = g.flux_on_grid(grid)
    check("each 20 s bin takes the 1-min record containing it",
          on[3] == np.float64(np.float32(1e-6)) and on[3 + 32] == np.float64(np.float32(2e-6)))
    check("bins outside GOES coverage are NaN", np.isnan(on[0]) and np.isnan(on[-1]))


def test_events_on_grid():
    from solarflare.io.goes import GoesFlare, GoesTruth
    from solarflare.preprocess.labels import goes_events_on_grid
    fl = [GoesFlare(1, T0 - 300, T0 + 120, T0 + 900, False, "C3.0", 3e-6, 1e-6),   # starts before grid
          GoesFlare(2, T0 + 2000, T0 + 2300, T0 + 2600, False, "B9.0", 9e-7, 5e-7),
          GoesFlare(3, T0 + 4000, T0 + 4400, T0 + 5000, False, "M2.0", 2e-5, 1e-6)]
    grid = T0 + 20.0 * np.arange(600)
    ev = goes_events_on_grid(GoesTruth(fl, np.zeros(0), np.zeros(0), []), grid, "C1.0", 20.0)
    check("below-threshold (B9.0) flares are not events at C1.0", [e.goes_class for e in ev] == ["C3.0", "M2.0"])
    e0, e1 = ev
    check("a flare starting before the grid is clipped but keeps its true start time",
          e0.start_idx == 0 and e0.start_unix == T0 - 300)
    check("indices follow the GOES times", e1.start_idx == 200 and e1.peak_idx == 220 and e1.end_idx >= 250,
          f"{e1.start_idx} {e1.peak_idx} {e1.end_idx}")
    check("magnitude is peak / background", abs(e1.magnitude - 20.0) < 1e-9)


def test_goes_labels_end_to_end():
    from solarflare.pipeline import prepare
    from solarflare.preprocess.events import build_rise_dataset
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        data = root / "data"
        # SoLEXS sees two strong flares (peaks at 3600 s and 9000 s); GOES lists the
        # first as M1.5, a third one SoLEXS barely shows (C2.0 at 14000 s), and a
        # B-class event. The second SoLEXS flare is NOT in the GOES list.
        _make_solexs(data / "slx", n=20000, flares=((3600, 600, 900, 120), (9000, 600, 900, 90)), t0=T0)
        gdir = root / "goes"
        gdir.mkdir()
        write_flsum(gdir / "sci_xrsf-l2-flsum_g18_test.nc", [
            (10, T0 + 3000, T0 + 3600, T0 + 4500, "M1.5", 1e-6),
            (11, T0 + 13700, T0 + 14000, T0 + 14600, "C2.0", 1e-6),
            (12, T0 + 16000, T0 + 16200, T0 + 16500, "B6.0", 5e-7),
        ])

        def flux(t):
            x = t - T0
            if 3000 <= x < 4500:
                return 1.5e-5
            if 13700 <= x < 14600:
                return 2e-6
            return 1e-6
        write_avg1m(gdir / "sci_xrsf-l2-avg1m_g18_test.nc", T0 - 600, 360, flux, bad={100})

        cfg = Config(data_root=data, out_dir=root / "out")
        cfg.pre.dt_seconds = 60.0
        cfg.pre.background_window_s = 7200.0
        cfg.win.input_seconds = 1200.0
        cfg.win.stride_seconds = 120.0
        cfg.win.forecast_horizons_s = (60.0, 300.0)
        cfg.win.occurrence_horizons_s = (300.0,)
        cfg.train.device = "cpu"
        cfg.pre.label_source = "goes"
        cfg.pre.goes_dir = str(gdir)
        prep = prepare(cfg, verbose=False)

        events = [e for s in prep.segments for e in s.events]
        check("events are exactly the GOES list at >= C1.0, not SoLEXS detections",
              sorted(e.goes_class for e in events) == ["C2.0", "M1.5"], str([e.goes_class for e in events]))
        seg = prep.segments[0]
        i = int(np.searchsorted(seg.time_unix, T0 + 3300))
        check("flux targets are log10(GOES W/m^2)", abs(seg.log_flux[i] - np.log10(1.5e-5)) < 1e-4,
              f"{seg.log_flux[i]}")
        bad_t = T0 - 600 + 60 * 100
        j = int(np.searchsorted(seg.time_unix, bad_t))
        check("bins without valid GOES flux have no target",
              seg.target_valid[j] == 0 and seg.target_valid[i] == 1)
        k = int(np.searchsorted(seg.time_unix, T0 + 9000))
        check("a SoLEXS flare GOES did not list is not a flare", seg.in_flare[k] == 0)
        check("meta records the label source and files", prep.meta.get("label_source") == "goes"
              and len(prep.meta.get("goes_files", [])) == 2)

        ds = build_rise_dataset(prep.segments, cfg)
        check("large-flare threshold is GOES M1.0", abs(ds.threshold_rate - 1e-5) < 1e-15, str(ds.threshold_rate))
        m15 = [s for s in ds.samples if abs(s.y_log_peak - np.log10(1.5e-5)) < 1e-9]
        c20 = [s for s in ds.samples if abs(s.y_log_peak - np.log10(2e-6)) < 1e-9]
        check("rise targets use log10 of the GOES peak flux", len(m15) > 0 and len(c20) > 0,
              f"{len(m15)} {len(c20)} {set(round(s.y_log_peak, 3) for s in ds.samples)}")
        check("the M1.5 flare exceeds, the C2.0 does not",
              all(s.y_exceeds == 1 for s in m15) and all(s.y_exceeds == 0 for s in c20))

        # Solexs labels on the same data still come from the detector.
        cfg2 = Config(data_root=data, out_dir=root / "out2")
        for k2, v in vars(cfg.pre).items():
            setattr(cfg2.pre, k2, v)
        cfg2.win = cfg.win
        cfg2.pre.label_source = "solexs"
        prep2 = prepare(cfg2, verbose=False)
        n2 = sum(len(s.events) for s in prep2.segments)
        check("label_source='solexs' still uses the detector (both injected flares)", n2 >= 2, str(n2))


def test_truncated_rise_skipped():
    from solarflare.io.goes import GoesFlare
    from solarflare.preprocess.events import build_rise_dataset
    from solarflare.preprocess.labels import goes_events_on_grid
    from solarflare.preprocess.dataset import Segment
    from solarflare.io.goes import GoesTruth
    cfg = Config()
    cfg.pre.label_source = "goes"
    cfg.pre.dt_seconds = 20.0
    n = 400
    grid = T0 + 20.0 * np.arange(n)
    fl = [GoesFlare(1, T0 - 600, T0 + 400, T0 + 1200, False, "M1.0", 1e-5, 1e-6),
          GoesFlare(2, T0 + 3000, T0 + 3600, T0 + 4200, False, "C5.0", 5e-6, 1e-6)]
    ev = goes_events_on_grid(GoesTruth(fl, np.zeros(0), np.zeros(0), []), grid, "C1.0", 20.0)
    z = np.zeros(n, np.float32)
    seg = Segment("s", grid, np.zeros((n, 1), np.float32), np.ones(n, np.float32),
                  np.zeros((n, 1), np.float32), z, np.zeros((n, 2), np.float32),
                  np.zeros(n, np.int64), z, np.zeros(n, np.int64), z, np.ones(n, np.float32), ev)
    ds = build_rise_dataset([seg], cfg)
    ids = {s.event_id for s in ds.samples}
    check("a rise cut off by the segment start is not used for peak forecasting",
          ids == {1}, str(ids))


def test_shared_cache_keeps_both_configurations():
    from solarflare.preprocess.cache import build_cache, index_sources
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_solexs(root / "data" / "slx", n=3000, t0=T0)
        srcs = index_sources([root / "data"])
        a = PreprocessConfig(solexs_energy_scale="legacy_linear", dt_seconds=60.0)
        b = PreprocessConfig(dt_seconds=60.0)
        build_cache(srcs, a, root / "cache", workers=1, verbose=False)
        build_cache(srcs, b, root / "cache", workers=1, verbose=False)
        again = build_cache(srcs, a, root / "cache", workers=1, verbose=False)
        import json
        man = json.loads((root / "cache" / "manifest.json").read_text("utf-8"))
    check("switching configuration back does not rebuild (index kept both)",
          all(e.get("status") == "ok" for e in again) and len(man) == 2 * len(srcs),
          f"{len(man)} entries")


def test_unknown_label_source_is_an_error():
    from solarflare.preprocess.dataset import build_segments_from_raw
    cfg = Config()
    cfg.pre.label_source = "made_up"
    try:
        build_segments_from_raw([], [], cfg)
        check("unknown label source raises", False)
    except ValueError:
        check("unknown label source raises", True)


def test_persistence_scored_only_on_real_origins():
    from solarflare.metrics import forecast_score_mask, regression_scores, skill_vs_reference

    # Four windows of GOES log flux; the last origin is a GOES gap stored as 0.
    y_future = np.array([-5.0, -5.2, -4.8, -5.1], np.float32)
    origin = np.array([-5.1, -5.2, -4.9, 0.0], np.float32)
    origin_valid = np.array([1, 1, 1, 0], np.float32)
    future_valid = np.ones(4, np.float32)
    model = y_future + 0.05
    old = future_valid > 0
    new = forecast_score_mask(future_valid, origin_valid)
    check("a gap at the origin is excluded from forecast scoring", new.tolist() == [True, True, True, False])
    check("scored on every window, persistence looked five decades wrong",
          regression_scores(y_future, origin, old)["RMSE"] > 2.0)
    rmse_p = regression_scores(y_future, origin, new)["RMSE"]
    check("on real origins persistence error is what it really is", abs(rmse_p - np.sqrt(0.02 / 3)) < 1e-6, str(rmse_p))
    check("skill over persistence is no longer inflated",
          skill_vs_reference(y_future, model, origin, new) < skill_vs_reference(y_future, model, origin, old))


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} GOES test groups\n")
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
    print("All GOES checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
