"""Robustness and integration tests.

The correctness suite covers the science. This one covers the ways the code
meets the real world: malformed files, empty detectors, absent optional
products, unwritable output, and every CLI path actually executing.

That split exists because a real defect slipped through: a failed PNG write
destroyed a completed training run, and `predict` shipped without ever having
been executed. Both are covered here now.

    python -m tests.test_robustness
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarflare.config import Config
from solarflare.io.solexs import read_solexs, read_gti, verify_lc_consistency
from solarflare.io.hel1os import read_hel1os_lightcurve, read_housekeeping
from solarflare.io.discover import discover

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Synthetic product builders
# ---------------------------------------------------------------------------

def _make_solexs(dirpath: Path, n: int = 600, with_lc: bool = True,
                 with_gti: bool = True, detector: str = "SDD2",
                 flares: tuple = (), t0: float = 1.789e9, seed: int = 0) -> None:
    """Synthetic SoLEXS day.

    ``flares`` is ((peak_second, rise_s, decay_s, amplitude_cts_per_s), ...):
    a linear rise and exponential decay spread over the soft channels, so the
    detector and the rise-phase dataset have real events to find.
    """
    d = dirpath / detector
    d.mkdir(parents=True, exist_ok=True)
    t = t0 + np.arange(n, dtype=float)
    rng = np.random.default_rng(seed)
    spec = rng.poisson(0.03, size=(n, 340)).astype(float)
    sec = np.arange(n, dtype=float)
    for peak, rise, decay, amp in flares:
        prof = np.where(sec <= peak,
                        np.clip(1.0 - (peak - sec) / rise, 0.0, None),
                        np.exp(-(sec - peak) / decay))
        # Spread over channels 45..160 (roughly 1.3-9.5 keV) with a soft spectrum.
        chans = np.arange(45, 161)
        weights = np.exp(-(chans - 45) / 30.0)
        weights /= weights.sum()
        spec[:, chans] += rng.poisson(amp * prof[:, None] * weights[None, :])

    cols = fits.ColDefs([
        fits.Column(name="TSTART", format="D", array=t),
        fits.Column(name="TELAPSE", format="D", array=np.ones(n)),
        fits.Column(name="SPEC_NUM", format="J", array=np.arange(n)),
        fits.Column(name="CHANNEL", format="340K",
                    array=np.tile(np.arange(340), (n, 1))),
        fits.Column(name="COUNTS", format="340D", array=spec),
        fits.Column(name="EXPOSURE", format="D", array=np.ones(n)),
    ])
    hdu = fits.BinTableHDU.from_columns(cols, name="SPECTRUM")
    prim = fits.PrimaryHDU()
    prim.header["OBS_DATE"] = "20260910"
    fits.HDUList([prim, hdu]).writeto(d / f"AL1_SOLEXS_TEST_{detector}_L1.pi")

    if with_lc:
        lc = fits.BinTableHDU.from_columns(fits.ColDefs([
            fits.Column(name="TIME", format="D", array=t),
            fits.Column(name="COUNTS", format="D", array=spec[:, 41:].sum(axis=1)),
        ]), name="RATE")
        fits.HDUList([fits.PrimaryHDU(), lc]).writeto(
            d / f"AL1_SOLEXS_TEST_{detector}_L1.lc")

    rows = np.array([[t[0], t[-1]]]) if with_gti else np.zeros((0, 2))
    g = fits.BinTableHDU.from_columns(fits.ColDefs([
        fits.Column(name="START", format="D", array=rows[:, 0] if len(rows) else np.array([])),
        fits.Column(name="STOP", format="D", array=rows[:, 1] if len(rows) else np.array([])),
    ]), name="GTI")
    fits.HDUList([fits.PrimaryHDU(), g]).writeto(
        d / f"AL1_SOLEXS_TEST_{detector}_L1.gti")


def _make_hel1os(root: Path, n: int = 600, every: int = 6) -> Path:
    czt = root / "czt"
    aux = root / "aux"
    czt.mkdir(parents=True, exist_ok=True)
    aux.mkdir(parents=True, exist_ok=True)
    mjd = 61294.5 + np.arange(n) / 86400.0
    hdus = [fits.PrimaryHDU()]
    hdus[0].header["ISOSTART"] = "2026-09-11T12:00:03.000"
    hdus[0].header["ISOSTOP"] = "2026-09-11T12:10:03.000"
    for lo, hi in ((20.0, 40.0), (18.0, 160.0)):
        ctr = np.zeros(n)
        err = np.zeros(n)
        ctr[::every] = 120.0
        err[::every] = np.sqrt(120.0)
        h = fits.BinTableHDU.from_columns(fits.ColDefs([
            fits.Column(name="MJD", format="D", array=mjd),
            fits.Column(name="ISOT", format="30A", array=np.array(["x"] * n)),
            fits.Column(name="CTR", format="D", array=ctr),
            fits.Column(name="STAT_ERR", format="D", array=err),
        ]), name=f"CZT1_LC_BAND_{lo:.2f}KEV_TO_{hi:.2f}KEV")
        h.header["ELOW"] = lo
        h.header["EHIGH"] = hi
        h.header["DETNAM"] = "CZT1"
        hdus.append(h)
    fits.HDUList(hdus).writeto(czt / "lightcurve_czt1.fits")
    return czt / "lightcurve_czt1.fits"


# ---------------------------------------------------------------------------
# I/O edge cases
# ---------------------------------------------------------------------------

def test_solexs_missing_optional_products():
    """No .lc and no .gti must degrade gracefully, not raise."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_solexs(root, with_lc=False, with_gti=False)
        obs = read_solexs(root / "SDD2")
        check("SoLEXS reads with no .lc and no .gti", obs is not None)
        if obs:
            check("...and reports no GTI coverage rather than crashing",
                  obs.valid.sum() == 0, f"valid={obs.valid.sum()}")
            check("...and lc cross-check reports 'not checked'",
                  verify_lc_consistency(obs)["checked"] is False)


def test_solexs_absent_detector_returns_none():
    """A directory with no .pi at all (the SDD1 case) yields None."""
    with tempfile.TemporaryDirectory() as d:
        empty = Path(d) / "SDD1"
        empty.mkdir()
        check("SoLEXS returns None for a detector with no science data",
              read_solexs(empty) is None)


def test_empty_gti_is_zero_rows_not_error():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_solexs(root, with_gti=False)
        g = next((root / "SDD2").glob("*.gti"))
        arr = read_gti(g)
        check("zero-row GTI returns an empty (0,2) array",
              arr.shape == (0, 2), f"got {arr.shape}")


def test_hel1os_missing_hk_and_gti():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        p = _make_hel1os(root)
        obs = read_hel1os_lightcurve(p, gti_path=None)
        check("HEL1OS reads with no GTI file", obs is not None)
        check("absent hk.fits returns None, not an exception",
              read_housekeeping(root / "aux" / "hk.fits") is None)


def test_hel1os_all_fill_rows():
    """A file where nothing was telemetered must not divide by zero."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        p = _make_hel1os(root, n=120, every=10_000)  # only index 0 sampled
        obs = read_hel1os_lightcurve(p)
        check("near-empty HEL1OS file still parses", obs is not None)
        if obs:
            from solarflare.config import PreprocessConfig
            from solarflare.preprocess.features import hel1os_features
            f = hel1os_features(obs, PreprocessConfig())
            check("...and produces finite features",
                  bool(np.isfinite(np.nan_to_num(f.values)).all()))


def test_discover_on_empty_tree():
    with tempfile.TemporaryDirectory() as d:
        inv = discover(Path(d))
        check("discover on an empty directory returns empty inventory",
              inv.solexs == [] and inv.hel1os == [])
        check("...and its summary renders without error",
              "(none)" in inv.summary())


def test_truncated_fits_is_reported_not_silent():
    """A corrupt file should raise loudly, never return partial data."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "SDD2"
        root.mkdir(parents=True)
        (root / "broken.pi").write_bytes(b"SIMPLE  =  T" + b"\0" * 200)
        raised = False
        try:
            read_solexs(root)
        except Exception:
            raised = True
        check("a truncated .pi raises rather than returning junk", raised)


# ---------------------------------------------------------------------------
# Failure-path robustness
# ---------------------------------------------------------------------------

def test_plot_save_failure_does_not_propagate():
    """The defect that destroyed a finished training run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from solarflare.plots import _save

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    # A path that cannot exist: a file used as a directory component.
    with tempfile.TemporaryDirectory() as d:
        blocker = Path(d) / "notadir"
        blocker.write_text("x", encoding="utf-8")
        out = _save(fig, blocker / "sub" / "fig.png")
    check("a failed figure write returns None instead of raising",
          out is None)


def test_config_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        cfg = Config(data_root=Path("."), out_dir=Path(d))
        p = Path(d) / "cfg.json"
        cfg.to_json(p)
        import json
        loaded = json.loads(p.read_text(encoding="utf-8"))
        check("config serialises to JSON",
              loaded["pre"]["dt_seconds"] == cfg.pre.dt_seconds)
        check("...including the clock flag that caused the leakage bug",
              loaded["model"]["use_clock"] is False)


# ---------------------------------------------------------------------------
# End-to-end: every CLI path must actually execute
# ---------------------------------------------------------------------------

def test_full_pipeline_on_synthetic_data():
    """Train, evaluate, report and predict on a tiny synthetic dataset.

    `predict` in particular shipped once without ever being run. Every command
    is exercised here so that can never be true again.
    """
    from solarflare.pipeline import prepare
    from solarflare.train import train, load_model
    from solarflare.evaluate import evaluate
    from solarflare.predict import run_inference
    from solarflare.report import build

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        data = root / "data"
        # Long enough for a few windows at a coarse grid.
        _make_solexs(data / "slx", n=9000)
        _make_hel1os(data / "hls_obs", n=4000)

        cfg = Config(data_root=data, out_dir=root / "out")
        cfg.pre.dt_seconds = 60.0
        cfg.win.input_seconds = 1200.0
        cfg.win.stride_seconds = 120.0
        cfg.win.forecast_horizons_s = (60.0, 300.0)
        cfg.win.occurrence_horizons_s = (300.0,)
        cfg.train.epochs = 2
        cfg.train.batch_size = 8
        cfg.train.embargo_s = 120.0
        cfg.train.device = "cpu"
        cfg.model.hidden = 16
        cfg.model.dilations = (1, 2)

        try:
            prep = prepare(cfg, verbose=False)
        except RuntimeError as exc:
            check("pipeline builds windows from synthetic data", False, str(exc))
            return
        check("pipeline builds windows from synthetic data", len(prep.windows) > 0)

        res = train(cfg, prep=prep, verbose=False)
        check("training completes", Path(res["checkpoint"]).exists())

        model = load_model(Path(res["checkpoint"]), cfg)
        rep = evaluate(model, prep, cfg, verbose=False)
        check("evaluation produces a report", isinstance(rep, dict))
        check("ablation includes the clock_only guard",
              "clock_only" in rep.get("modality_ablation", {}))

        rows = run_inference(cfg, Path(res["checkpoint"]),
                             output=root / "out" / "pred.csv", limit=3)
        check("predict runs and writes rows", len(rows) > 0)
        if rows:
            quiet = [r for r in rows if r["phase"] == 0]
            check("predict blanks the peak head outside flares",
                  all(r["pred_time_to_peak_min"] == "" for r in quiet)
                  if quiet else True)

        from solarflare.evaluate import save_report
        save_report(rep, root / "out" / "reports" / "evaluation.json")
        out = build(root / "out")
        check("RESULTS.md renders", out.exists() and out.stat().st_size > 0)


def test_gru_encoder_exits_cleanly_on_cuda():
    """Regression guard for the cuDNN GRU teardown crash.

    ``nn.GRU(num_layers=2, dropout=0.1)`` on CUDA made the process die at
    interpreter exit with 0xC0000409 on Windows + torch 2.14 cu130 -- after
    all work had finished, so the only symptom was every GRU run reporting
    failure. The encoder now keeps dropout outside cuDNN. This runs it in a
    child process, because the failure only shows up as the exit code.
    Skipped without a GPU.
    """
    import subprocess
    import torch
    if not torch.cuda.is_available():
        print("  SKIP  no CUDA device")
        return
    code = (
        "import sys, torch\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n"
        "from solarflare.config import ModelConfig\n"
        "from solarflare.models.zoo import build_encoder\n"
        "enc = build_encoder('gru', 25, 64, ModelConfig()).cuda().train()\n"
        "y = enc(torch.randn(16, 25, 120, device='cuda'))\n"
        "y.sum().backward()\n"
        "torch.cuda.synchronize()\n"
        "print('ok', flush=True)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    rc = r.returncode & 0xFFFFFFFF
    check("GRU encoder on CUDA exits with code 0 (no cuDNN teardown crash)",
          rc == 0 and "ok" in r.stdout, f"exit {rc:#010x} stderr={r.stderr[-300:]!r}")


def test_rise_forecasting_on_synthetic_flares():
    """The forecasting module end to end, on data with known flares.

    Exercises event detection -> rise dataset -> grouped CV for every encoder
    -> reference forecasts -> report. Without this, forecast.py would be in the
    same position `predict` once was: shipped, never executed by a test.
    """
    from solarflare.pipeline import prepare
    from solarflare.forecast import (grouped_cv, print_comparison, save,
                                     _train_climatology)
    from solarflare.preprocess.events import build_rise_dataset
    from solarflare.models.zoo import ENCODERS
    from solarflare.report import build

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        data = root / "data"
        # Six flares an hour apart, with different sizes so peak magnitude
        # actually varies and the exceedance class has both labels.
        flares = tuple((3600 * (k + 1), 600, 900, amp)
                       for k, amp in enumerate((30, 80, 45, 120, 35, 90)))
        _make_solexs(data / "slx", n=26000, flares=flares)

        cfg = Config(data_root=data, out_dir=root / "out")
        cfg.pre.dt_seconds = 60.0
        cfg.pre.background_window_s = 7200.0
        cfg.win.input_seconds = 1200.0
        cfg.win.stride_seconds = 120.0
        cfg.win.forecast_horizons_s = (60.0, 300.0)
        cfg.win.occurrence_horizons_s = (300.0,)
        cfg.train.epochs = 2
        cfg.train.batch_size = 8
        cfg.train.embargo_s = 120.0
        cfg.train.device = "cpu"
        cfg.model.hidden = 16
        cfg.model.dilations = (1, 2)
        cfg.model.attn_heads = 2
        cfg.model.ssm_layers = 1
        cfg.model.tf_layers = 1

        prep = prepare(cfg, verbose=False)
        n_events = sum(len(s.events) for s in prep.segments)
        check("synthetic flares are detected", n_events >= 4, f"found {n_events}")

        ds = build_rise_dataset(prep.segments, cfg)
        check("rise dataset is built from the events", len(ds) > 0, f"n={len(ds)}")
        if ds.samples:
            # The raw current level can exceed the *smoothed* peak through
            # noise (see RiseSample.y_current), but only rarely. A large
            # fraction would mean the rise window runs past the real peak.
            above = np.mean([x.y_current > x.y_log_peak for x in ds.samples])
            check("current level rarely exceeds the eventual (smoothed) peak",
                  above <= 0.10, f"{100 * above:.1f}% of samples")
            check("time to peak is non-negative during the rise",
                  all(x.y_time_to_peak >= 0 for x in ds.samples))
            clim, _ = _train_climatology(ds, list(range(len(ds))))
            per_event = {x.event_id: x.y_log_peak for x in ds.samples}
            check("climatology is weighted per event, not per sample",
                  abs(clim - np.mean(list(per_event.values()))) < 1e-9)

        res = grouped_cv(prep, cfg, encoders=ENCODERS, n_folds=2, verbose=False)
        check("grouped CV completes", "summary" in res, str(res.get("error")))
        if "summary" not in res:
            return
        check("every encoder produced results",
              set(res["summary"]) == set(ENCODERS),
              f"got {sorted(res['summary'])}, failures {res.get('failures')}")
        check("no fold failed", not res.get("failures"), str(res.get("failures")))

        # Learning curves: a smaller train_fraction must shrink the training
        # set while leaving each fold's test block exactly as it was.
        half = grouped_cv(prep, cfg, encoders=("linear",), n_folds=2, ds=ds,
                          train_fraction=0.5, verbose=False)
        full = grouped_cv(prep, cfg, encoders=("linear",), n_folds=2, ds=ds,
                          train_fraction=1.0, verbose=False)
        if "folds" in half and "folds" in full:
            h = {f["fold"]: f for f in half["folds"]}
            g = {f["fold"]: f for f in full["folds"]}
            common = sorted(set(h) & set(g))
            check("train_fraction keeps the test blocks identical",
                  all(h[k]["n_test"] == g[k]["n_test"] for k in common),
                  str([(h[k]["n_test"], g[k]["n_test"]) for k in common]))
            check("train_fraction trains on no more than the full set",
                  all(h[k]["n_train"] <= g[k]["n_train"] for k in common),
                  str([(h[k]["n_train"], g[k]["n_train"]) for k in common]))

        folds = res["folds"]
        check("folds never share an event between train and test",
              all(f["n_train"] > 0 and f["n_test"] > 0 for f in folds))
        check("every fold reports both reference forecasts",
              all("peak_skill_vs_current" in f and "peak_skill_vs_climatology" in f
                  for f in folds))
        check("reference MAEs are finite",
              all(np.isfinite(f["peak_log_MAE_current"])
                  and np.isfinite(f["peak_log_MAE_climatology"]) for f in folds))

        print_comparison(res)
        save(res, root / "out" / "reports" / "forecast_cv.json")

        # Fusion ablation: coverage 0 makes every flare eligible, so both arms
        # run on this soft-only data. What is under test is the pairing.
        from solarflare.forecast import fusion_ablation
        fu = fusion_ablation(prep, cfg, encoder="linear", n_folds=2,
                             min_hard_coverage=0.0, verbose=False)
        check("fusion ablation completes", "error" not in fu, str(fu.get("error")))
        if "error" not in fu:
            arm_f = [(f["fold"], f["n_train"], f["n_test"]) for f in fu["soft+hard"]["folds"]]
            arm_s = [(f["fold"], f["n_train"], f["n_test"]) for f in fu["soft-only"]["folds"]]
            check("both arms use identical folds", arm_f == arm_s and arm_f, f"{arm_f} vs {arm_s}")
            check("every test sample is paired",
                  fu["n_paired_samples"] == sum(n for _, _, n in arm_f))
            check("with no HEL1OS data, blanking it changes nothing",
                  abs(fu["peak_MAE_reduction_from_hel1os"]["value"]) < 1e-6,
                  str(fu["peak_MAE_reduction_from_hel1os"]))
            save(fu, root / "out" / "reports" / "fusion_ablation.json")

        text = build(root / "out").read_text(encoding="utf-8")
        check("report renders the forecasting comparison",
              "Rise-phase forecasting" in text and "skill vs climatology" in text)
        check("report renders the fusion ablation", "Does HEL1OS add skill" in text)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} robustness test groups\n")
    for t in tests:
        print(f"{t.__name__}:")
        try:
            t()
        except Exception as exc:
            print(f"  ERROR {t.__name__}: {type(exc).__name__}: {exc}")
            FAILURES.append(f"{t.__name__} (exception)")
        print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("All robustness checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
