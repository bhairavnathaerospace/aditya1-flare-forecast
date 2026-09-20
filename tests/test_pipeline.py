"""Tests for the settings, training safeguards and the pipeline runner.

What is checked, each on synthetic data with a known answer:

* settings: project.toml paths resolve, the final model configuration follows
  the [model] section, ``--set`` overrides parse to the field's type and an
  unknown key is refused;
* a run's saved configuration round-trips (Config.from_json / run_config);
* probability calibration: isotonic knots improve the Brier score of
  miscalibrated forecasts, never reorder them, and fall back to identity when
  there is too little to fit;
* model selection on a trailing mean of the validation score, so one lucky
  epoch is not "best";
* per-group gradient clipping bounds every part of the network separately and
  leaves small gradients alone;
* HEL1OS readout smoothing is causal: changing the future never changes the past;
* the flux anchor is fitted on training samples only and recovers a known line;
* SHARP: T_REC in TAI, quality and limb cuts, disk sums and maxima, an empty
  month, "latest hour at or before t" and the operator latency;
* the pipeline runner: stages in dependency order, dependants of a redone
  stage, a frozen model counted as done unless stale, SHARP chosen on
  validation with the winner copied and nothing deleted;
* the one-page summary builds with no products at all;
* the console's alert rules are written from validation operating points.

    python -m tests.test_pipeline
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solarflare import probcal  # noqa: E402
from solarflare.config import Config, run_config  # noqa: E402
from solarflare.settings import Settings, apply_overrides, load_settings, model_config  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def _settings(tmp: Path, **model) -> Settings:
    out = tmp / "outputs"
    return Settings(root=tmp, data_root=tmp / "data", goes_dir=tmp / "goes", sharp_dir=tmp / "sharp",
                    hel1os_extracted=tmp / "hel1os", cache=tmp / "cache", outputs=out, min_free_gb=0.0,
                    model=model, pipeline={"seeds": [1, 2], "sharp_min_gain": 0.01})


# ---------------------------------------------------------------- settings

def test_settings_and_overrides():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "p.toml"
        f.write_text('[paths]\ndata_root = "D:/X"\noutputs = "out"\n[model]\nhel1os_smooth_s = 30\n'
                     'select_smooth_epochs = 4\nuse_sharp = true\n', encoding="utf-8")
        s = load_settings(str(f))
        check("absolute paths kept, relative ones under the project", s.data_root == Path("D:/X")
              and s.outputs == s.root / "out", f"{s.data_root} {s.outputs}")
        check("output layout", s.frozen_dir == s.outputs / "model" / "forward" / "final"
              and s.copied_days == s.outputs / "quality" / "solexs_duplicates.json")
        cfg = model_config(s)
        check("final configuration follows [model]", cfg.pre.hel1os_smooth_s == 30.0
              and cfg.train.select_smooth_epochs == 4 and cfg.pre.label_source == "goes"
              and cfg.pre.fit_flux_anchor and cfg.pre.sharp_dir == str(s.sharp_dir))
        check("copied-day mask only when the list exists", cfg.pre.exclude_intervals == "")
        cfg2 = model_config(s, sharp=False)
        check("an explicit sharp=False wins over the file", cfg2.pre.sharp_dir == "")
    cfg = apply_overrides(Config(), ["train.balance_head_gradients=false", "train.epochs=7", "train.lr=0.002",
                                     "win.occurrence_horizons_s=[600, 1200]", "pre.goes_min_class=M1.0"])
    check("overrides parse to the field's type", cfg.train.balance_head_gradients is False and cfg.train.epochs == 7
          and cfg.train.lr == 0.002 and cfg.win.occurrence_horizons_s == (600, 1200)
          and cfg.pre.goes_min_class == "M1.0")
    try:
        apply_overrides(Config(), ["train.no_such_thing=1"])
        check("an unknown setting is refused", False)
    except SystemExit:
        check("an unknown setting is refused", True)


def test_config_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "run"
        cfg = Config(data_root=Path(tmp), out_dir=Path(tmp) / "elsewhere")
        cfg.pre.hel1os_smooth_s = 45.0
        cfg.win.forecast_horizons_s = (60.0, 300.0)
        cfg.train.select_smooth_epochs = 3
        cfg.to_json(run / "reports" / "config.json")
        back = run_config(run)
        check("a run's configuration round-trips", back.pre.hel1os_smooth_s == 45.0
              and back.win.forecast_horizons_s == (60.0, 300.0) and back.train.select_smooth_epochs == 3)
        check("run_config points at the run where it now lives", back.out_dir == run)
        try:
            run_config(Path(tmp) / "nothing")
            check("a folder without a config is not a run", False)
        except FileNotFoundError:
            check("a folder without a config is not a run", True)


# ---------------------------------------------------------------- calibration

def test_probcal():
    rng = np.random.default_rng(0)
    truth = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, truth.size) < truth).astype(float)
    p = truth ** 3                                     # ranks right, badly calibrated
    cal = probcal.fit(p[:10000], y[:10000])
    q = probcal.apply(cal, p[10000:])
    b0 = np.mean((p[10000:] - y[10000:]) ** 2)
    b1 = np.mean((q - y[10000:]) ** 2)
    check("calibration lowers the Brier score out of sample", b1 < b0 - 0.01, f"{b0:.4f} -> {b1:.4f}")
    order = np.argsort(p[10000:])
    check("calibration never reorders forecasts", np.all(np.diff(q[order]) >= -1e-12))
    check("knots are plain lists (a frozen model stores them as JSON)", isinstance(cal["x"], list))
    check("too few samples: identity", probcal.fit(p[:20], y[:20]) == probcal.IDENTITY)
    check("one class only: identity", probcal.fit(p[:500], np.zeros(500)) == probcal.IDENTITY)
    check("no calibrator: unchanged", np.allclose(probcal.apply(None, p[:5]), p[:5]))


# ---------------------------------------------------------------- training safeguards

def test_smoothed_selection():
    from solarflare.train import smoothed_score

    hist = [{"score": s} for s in (0.50, 0.70, 0.52)]
    check("k=1 is the raw score", smoothed_score(hist, 0.53, 1) == 0.53)
    check("k=3 averages this epoch and the two before", abs(smoothed_score(hist, 0.53, 3) - np.mean([0.70, 0.52, 0.53]))
          < 1e-12)
    check("first epoch has nothing to average", smoothed_score([], 0.6, 3) == 0.6)
    # a lone spike loses to a sustained level once smoothed
    raw = [0.50, 0.62, 0.50, 0.51, 0.57, 0.58, 0.58]
    h, sm = [], []
    for r in raw:
        sm.append(smoothed_score(h, r, 3))
        h.append({"score": r})
    check("one lucky epoch is not selected", int(np.argmax(raw)) == 1 and int(np.argmax(sm)) >= 5, str(sm))


def test_clip_by_group():
    from solarflare.train import clip_by_group

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.soft_enc = torch.nn.Linear(4, 4)
            self.head_peak = torch.nn.Linear(4, 1)
            self.head_phase = torch.nn.Linear(4, 1)

    m = Net()
    for p in m.soft_enc.parameters():
        p.grad = torch.full_like(p, 0.01)
    for p in m.head_peak.parameters():
        p.grad = torch.full_like(p, 10.0)
    for p in m.head_phase.parameters():
        p.grad = torch.full_like(p, 3.0)
    small = torch.cat([p.grad.flatten() for p in m.soft_enc.parameters()]).clone()
    clip_by_group(m, 1.0)

    def norm(mod):
        return float(torch.cat([p.grad.flatten() for p in mod.parameters()]).norm())

    check("each large group clipped to the limit", abs(norm(m.head_peak) - 1.0) < 1e-4 and abs(norm(m.head_phase) - 1.0)
          < 1e-4, f"{norm(m.head_peak)} {norm(m.head_phase)}")
    check("a small group is left alone (a global clip would have shrunk it ~20x)",
          torch.allclose(torch.cat([p.grad.flatten() for p in m.soft_enc.parameters()]), small))


def test_hel1os_smoothing_is_causal():
    from solarflare.config import PreprocessConfig
    from solarflare.preprocess.features import HEL1OS_RAW_NAMES, _causal_mean, _hel1os_layout, hel1os_derive
    from solarflare.preprocess.grid import GriddedSeries

    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    check("trailing mean, NaN-aware", np.allclose(_causal_mean(x, 3), [1.0, 1.5, 1.5, 3.0, 4.5, 5.0]))
    rng = np.random.default_rng(1)
    n, k = 400, len(HEL1OS_RAW_NAMES)
    t = 1.7e9 + 2.0 * np.arange(n)
    vals = rng.gamma(2.0, 50.0, (n, k))
    cov_cols = [c for *_, c in _hel1os_layout()]
    vals[:, cov_cols] = 1.0
    pre = PreprocessConfig()
    pre.dt_seconds = 2.0
    pre.hel1os_smooth_s = 60.0
    a = hel1os_derive(GriddedSeries(t, vals.copy(), np.ones(n, np.float32), list(HEL1OS_RAW_NAMES)), pre)
    v2 = vals.copy()
    v2[300:] *= 50.0                                     # a burst in the "future"
    b = hel1os_derive(GriddedSeries(t, v2, np.ones(n, np.float32), list(HEL1OS_RAW_NAMES)), pre)
    same = np.allclose(np.nan_to_num(a.values[:300]), np.nan_to_num(b.values[:300]))
    check("smoothed HEL1OS features before t do not see data after t", same)
    check("but they do respond after it", not np.allclose(np.nan_to_num(a.values[300:]), np.nan_to_num(b.values[300:])))


def test_flux_anchor_fit():
    from types import SimpleNamespace

    from solarflare.pipeline import fit_flux_anchor

    rng = np.random.default_rng(2)
    n = 5000
    t = np.arange(n, dtype=float)
    gl = 10 ** rng.uniform(-7, -4, n)
    lf = 0.8 + 0.95 * np.log10(gl) + rng.normal(0, 0.02, n)
    lf[t > 3000] += 5.0                                  # after train_end: must be ignored
    seg = SimpleNamespace(goes_long=gl, soft_mask=np.ones(n), target_valid=np.ones(n), log_flux=lf, time_unix=t)
    fit = fit_flux_anchor([seg], train_end=3000.0)
    check("anchor recovers a known line from training samples only",
          fit is not None and abs(fit["slope"] - 0.95) < 0.01 and abs(fit["intercept"] - 0.8) < 0.05, str(fit))
    check("too few samples: no fit", fit_flux_anchor([seg], train_end=500.0) is None)


# ---------------------------------------------------------------- SHARP

def test_sharp_reader():
    from solarflare.io import sharp as sh
    from solarflare.preprocess.dataset import sharp_columns

    check("T_REC is TAI: 37 s ahead of UTC", sh.parse_trec("2024.05.01_12:00:00_TAI")
          == 1714564800.0 - 37.0)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        rows = ["T_REC,HARPNUM,QUALITY,LON_FWT,USFLUX,TOTUSJH,TOTPOT,SAVNCPP,R_VALUE,SHRGT45,AREA_ACR",
                "2024.05.01_12:00:00_TAI,1,0,10,1e22,100,1e23,1e12,3.0,20,100",
                "2024.05.01_12:00:00_TAI,2,0x00000000,-30,3e22,300,3e23,3e12,4.5,40,300",
                "2024.05.01_12:00:00_TAI,3,0,80,9e22,900,9e23,9e12,5.0,60,900",      # beyond 70 deg
                "2024.05.01_12:00:00_TAI,4,0x00000400,0,9e22,900,9e23,9e12,5.0,60,900",  # bad quality
                "2024.05.01_13:00:00_TAI,1,0,11,2e22,200,2e23,2e12,3.2,25,200"]
        (d / "sharp_2024-05.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        (d / "sharp_2024-06.csv").write_text("\n", encoding="utf-8")                  # month not published yet
        s = sh.load_sharp(d)
        check("an empty month is skipped, not fatal", s is not None and s.time_unix.size == 2)
        check("quality and limb cuts", s.n_rows_read == 5 and s.n_rows_kept == 3, f"{s.n_rows_read} {s.n_rows_kept}")
        v = s.values[0]
        name = {n: i for i, n in enumerate(s.names)}
        check("disk sums and maxima", v[name["sharp_n_regions"]] == 2
              and abs(v[name["sharp_log_usflux_sum"]] - np.log10(4e22)) < 1e-9
              and abs(v[name["sharp_log_usflux_max"]] - np.log10(3e22)) < 1e-9 and v[name["sharp_r_value_max"]] == 4.5)
        t12 = sh.parse_trec("2024.05.01_12:00:00_TAI")
        at = s.at(np.array([t12 - 1.0, t12, t12 + 1800.0, t12 + 4 * 3600.0 + 3600.0]), max_age_s=3 * 3600.0)
        check("latest hour at or before t, never after", np.isnan(at[0, 0]) and at[1, 0] == 2 and at[2, 0] == 2)
        check("too old: missing", np.isnan(at[3, 0]))
        cols = sharp_columns(s, np.array([t12 + 1800.0, t12 + sh.LATENCY_S + 60.0]))
        check("the network sees SHARP only after the operator latency", cols[0, 0] == 0.0 and cols[0, -1] == 0.0
              and cols[1, 0] == 2 and cols[1, -1] == 1.0, str(cols[:, [0, -1]]))
        check("no files: None", sh.load_sharp(d / "nothing") is None)


# ---------------------------------------------------------------- pipeline runner

def test_runner_order_and_dependants():
    from solarflare import runall

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "sharp").mkdir()
        (tmp / "sharp" / "sharp_2024-05.csv").write_text("x\n")
        for mode, s in (("auto", _settings(tmp, use_sharp="auto")), ("off", _settings(tmp, use_sharp=False))):
            st = runall.stages(s)
            names = [x.name for x in st]
            check(f"[{mode}] stage names are unique", len(names) == len(set(names)))
            check(f"[{mode}] every stage comes after what it needs",
                  all(names.index(n) < names.index(x.name) for x in st for n in x.needs), str(names))
            check(f"[{mode}] the SHARP ablation only in auto mode", ("train-sharp" in names) == (mode == "auto"))
        s = _settings(tmp, use_sharp="auto")
        st = runall.stages(s)
        dep = runall.dependents("train", st)
        check("redoing training redoes freeze, alerts and the report", {"freeze", "alerts", "report"} <= dep
              and "cache" not in dep and "hel1os-seed-1" not in dep)
        freeze = next(x for x in st if x.name == "freeze")
        state = runall.State(tmp / "state.json")
        check("a missing frozen model is not done", not runall.is_done(freeze, state))
        s.frozen_dir.mkdir(parents=True)
        (s.frozen_dir / "frozen.pt").write_bytes(b"x")
        check("an existing frozen model counts as done", runall.is_done(freeze, state))
        state.set("freeze", status="stale")
        check("unless what it was built from changed", not runall.is_done(freeze, state))
        check("state is written to disk", json.loads((tmp / "state.json").read_text())["stages"]["freeze"]["status"]
              == "stale")


def _fake_run(run: Path, score: float) -> None:
    (run / "reports").mkdir(parents=True)
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "best.pt").write_bytes(b"w")
    (run / "reports" / "evaluation.json").write_text(json.dumps({"training": {"best_score": score, "best_epoch": 9}}))
    (run / "reports" / "config.json").write_text(json.dumps({"out_dir": str(run)}))
    (run / "reports" / "live.json").write_text("{}")


def test_choose_inputs():
    from solarflare import runall

    for sx, ss, expect in ((0.55, 0.555, False), (0.55, 0.57, True)):
        with tempfile.TemporaryDirectory() as tmp:
            s = _settings(Path(tmp), use_sharp="auto")
            xr, sr = runall.sharp_runs(s)
            _fake_run(xr, sx)
            _fake_run(sr, ss)
            old = s.model_dir
            (old / "reports").mkdir(parents=True)
            (old / "reports" / "marker.txt").write_text("previous model")
            runall.choose_inputs(s)
            dec = json.loads((s.ablations / "sharp" / "decision.json").read_text())
            check(f"SHARP {'kept' if expect else 'left out'} at gain {ss - sx:+.3f} (needs +0.01)",
                  dec["use_sharp"] is expect)
            cfg = json.loads((s.model_dir / "reports" / "config.json").read_text())
            check("the winner is copied to outputs/model and points there", cfg["out_dir"] == str(s.model_dir)
                  and (s.model_dir / "checkpoints" / "best.pt").exists()
                  and not (s.model_dir / "reports" / "live.json").exists())
            check("both trained runs are still there", (xr / "reports" / "evaluation.json").exists()
                  and (sr / "reports" / "evaluation.json").exists())
            moved = list((Path(tmp) / "archive" / "superseded").glob("model_*/reports/marker.txt"))
            check("the previous outputs/model is moved aside, not deleted", len(moved) == 1)


def test_summary_without_products():
    from solarflare.summary import build

    with tempfile.TemporaryDirectory() as tmp:
        s = _settings(Path(tmp))
        out = build(s)
        txt = out.read_text("utf-8")
        check("summary builds with nothing run yet", out.exists() and txt.count("Not available yet") >= 6)


def test_alert_rules():
    from solarflare.products.leadtime import PRIMARY, write_alert_rules

    keys = ("TPR", "chance", "event_TSS", "median_lead_min", "false_per_day")
    res = {k: 0.5 for k in keys}
    summary = {"results": {"C": {"model": {f"fa_{PRIMARY['C']:g}": res}},
                           "M": {"combined": {f"fa_{PRIMARY['M']:g}": res}}}}
    ops = {"C": {"model": {f"fa_{PRIMARY['C']:g}": {"threshold": 0.31, "val_false_per_day": 1.9}}},
           "M": {"combined": {f"fa_{PRIMARY['M']:g}": {"threshold": -5.2, "val_false_per_day": 0.48}}}}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "alert_rules.json"
        write_alert_rules(p, summary, ops, "abc")
        r = json.loads(p.read_text())
        check("alert rules carry the validation thresholds", r["C"]["threshold"] == 0.31 and r["M"]["threshold"] == -5.2
              and r["model_sha256"] == "abc" and r["M"]["false_alarms_per_day_validation"] == 0.48)
        check("and the test scores next to them", r["C"]["test"]["TPR"] == 0.5)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} pipeline test groups\n")
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
    print("All pipeline checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
