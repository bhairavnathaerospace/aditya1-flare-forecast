"""Tests for the prospective forward test (solarflare/forward.py).

A forward test is only worth anything if it is impossible to cheat by
accident. What is checked:

* the frozen model carries the normaliser it was trained with, not one refit
  on data that arrived later (refitting shifts every input);
* an old checkpoint without a normaliser refuses to freeze once the data changed;
* a frozen model is never overwritten;
* forecasts exist only for moments after the data cutoff, and re-running
  predict never rewrites or duplicates them;
* forecasts whose truth is not yet final are held back, not scored;
* scoring against an independent flare list works and reports label agreement.

    python -m tests.test_forward
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarflare.config import Config  # noqa: E402
from tests.test_robustness import _make_solexs  # noqa: E402

FAILURES: list[str] = []
T0 = 1.789e9


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def _cfg(data: Path, out: Path) -> Config:
    cfg = Config(data_root=data, out_dir=out)
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
    return cfg


def test_forward_test_end_to_end():
    from solarflare import forward
    from solarflare.pipeline import prepare
    from solarflare.train import train

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        data, out = root / "data", root / "out"
        flares_a = tuple((3600 * (k + 1), 600, 900, amp)
                         for k, amp in enumerate((30, 80, 45, 120, 35, 90)))
        _make_solexs(data / "slx_a", n=26000, flares=flares_a, t0=T0, seed=1)

        cfg = _cfg(data, out)
        prep = prepare(cfg, verbose=False)
        (out / "reports").mkdir(parents=True, exist_ok=True)
        (out / "reports" / "data_meta.json").write_text(
            json.dumps(prep.meta, default=float), encoding="utf-8")
        res = train(cfg, prep=prep, verbose=False)
        ckpt = Path(res["checkpoint"])
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        check("checkpoints now store the training normaliser", "norm" in ck)

        fdir = forward.freeze(_cfg(data, out), ckpt, "v1", verbose=False)
        man = json.loads((fdir / "manifest.json").read_text("utf-8"))
        check("freeze records a data cutoff at the end of the data",
              abs(man["data_cutoff_unix"] - (T0 + 26000)) <= 120, str(man["data_cutoff_utc"]))
        check("freeze records thresholds and training climatology",
              0 < man["thresholds"]["in_flare"] < 1
              and 0 <= man["training_climatology"]["in_flare_rate"] <= 1)
        try:
            forward.freeze(_cfg(data, out), ckpt, "v1", verbose=False)
            check("a frozen model is never overwritten", False)
        except FileExistsError:
            check("a frozen model is never overwritten", True)

        r0 = forward.forward_predict(_cfg(data, out), "v1", verbose=False)
        check("no forecasts before any data exists after the cutoff", r0["new_forecasts"] == 0)

        # New data arrives after the freeze, contiguous with the old.
        flares_b = tuple((3600 * (k + 1), 600, 900, amp) for k, amp in enumerate((60, 110, 40, 95)))
        _make_solexs(data / "slx_b", n=30000, flares=flares_b, t0=T0 + 26000, seed=2)

        frozen = torch.load(fdir / "frozen.pt", map_location="cpu", weights_only=False)
        refit = prepare(_cfg(data, out), verbose=False).norm
        check("frozen normaliser is the training one, bit for bit",
              all(np.array_equal(frozen["norm"][k], ck["norm"][k]) for k in ck["norm"]))
        check("...and a refit on the grown data would differ (why it is frozen)",
              not np.allclose(refit.mean_soft, ck["norm"]["mean_soft"]))

        r1 = forward.forward_predict(_cfg(data, out), "v1", verbose=False)
        with (fdir / "predictions.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        origins = np.array([float(r["origin_unix"]) for r in rows])
        check("forecasts are issued for the new data", r1["new_forecasts"] > 100, str(r1))
        check("every forecast is for a moment after the cutoff",
              bool(origins.size) and origins.min() > man["data_cutoff_unix"])
        check("forecasts sit on the fixed 60 s stride",
              np.all(np.rint(origins / 60.0) * 60.0 == origins))
        check("every forecast records the model hash",
              all(r["model_sha256"] == man["checkpoint_sha256"][:16] for r in rows))

        before = (fdir / "predictions.csv").read_bytes()
        r2 = forward.forward_predict(_cfg(data, out), "v1", verbose=False)
        check("re-running predict adds nothing and rewrites nothing",
              r2["new_forecasts"] == 0 and (fdir / "predictions.csv").read_bytes() == before)

        rep = forward.forward_score(_cfg(data, out), "v1", verbose=False)
        check("forecasts are scored", rep.get("forecasts_scored", 0) > 0, str(rep.get("note")))
        check("the most recent forecasts wait for their truth to settle",
              rep.get("forecasts_pending_truth", 0) > 0)
        check("flares in the forward period are counted", rep.get("flares_in_period", 0) >= 2,
              str(rep.get("flares_in_period")))
        blk = rep.get("flare_in_progress_now", {})
        check("binary skill reported with a training-climatology reference",
              "TSS" in blk and "BSS_vs_training_climatology" in blk, str(list(blk)))
        check("flux forecasts are compared with persistence",
              all("skill_vs_persistence" in v for v in rep.get("flux_forecast", {}).values())
              and rep.get("flux_forecast"))
        check("a readable summary is written", (fdir / "FORWARD.md").exists())

        # Independent truth: a GOES-style list at the injected flare peaks.
        goes = root / "goes.csv"
        with goes.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["begin_time", "max_time", "end_time", "max_class"])
            from datetime import datetime, UTC
            for peak, rise, decay, amp in flares_b:
                p = T0 + 26000 + peak
                w.writerow([datetime.fromtimestamp(p - rise, UTC).isoformat(),
                            datetime.fromtimestamp(p, UTC).isoformat(),
                            datetime.fromtimestamp(p + 2 * decay, UTC).isoformat(),
                            "M1.2" if amp > 50 else "B5.0"])
        rep_g = forward.forward_score(_cfg(data, out), "v1", goes_events=goes,
                                      min_class="C1.0", verbose=False)
        g = rep_g.get("goes", {})
        check("GOES classes below the floor are excluded",
              g.get("goes_flares_in_period") == 3, str(g.get("goes_flares_in_period")))
        la = g.get("label_agreement", {})
        check("label agreement with the independent list is reported and high",
              la.get("fraction", 0) >= 0.66, str(la))

        # An old checkpoint without a normaliser, after the data changed.
        old = root / "old.pt"
        ck.pop("norm")
        torch.save(ck, old)
        try:
            forward.freeze(_cfg(data, out), old, "v_old", verbose=False)
            check("old checkpoint + changed data refuses to freeze", False)
        except RuntimeError as exc:
            check("old checkpoint + changed data refuses to freeze", "data changed" in str(exc),
                  str(exc))
        shutil.rmtree(fdir.parent / "v_old", ignore_errors=True)

        # The frozen config records the energy scale; a model frozen before the
        # scale was selectable must reload on the legacy scale it was trained on.
        check("frozen config records the SoLEXS energy scale",
              frozen["config"]["pre"].get("solexs_energy_scale") == "sarwade2025")
        legacy_dir = fdir.parent / "v_pre_scale"
        legacy_dir.mkdir()
        f2 = dict(frozen)
        f2["config"] = {k: dict(v) for k, v in frozen["config"].items()}
        f2["config"]["pre"].pop("solexs_energy_scale")
        torch.save(f2, legacy_dir / "frozen.pt")
        c2 = _cfg(data, out)
        forward.load_frozen(legacy_dir, c2)
        check("a model frozen before scales existed reloads on the legacy scale",
              c2.pre.solexs_energy_scale == "legacy_linear")


def test_goes_parsing():
    from solarflare.forward import goes_class_flux, load_goes_events
    check("class to flux", abs(goes_class_flux("M2.5") - 2.5e-5) < 1e-12
          and abs(goes_class_flux("x1") - 1e-4) < 1e-12 and goes_class_flux("?") != goes_class_flux("?"))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "swpc.json"
        p.write_text(json.dumps([
            {"begin_time": "2026-09-14T01:02:00Z", "max_time": "2026-09-14T01:10:00Z",
             "end_time": "2026-09-14T01:30:00Z", "max_class": "C3.4"},
            {"begin_time": "2026-09-14T05:00:00Z", "max_time": "2026-09-14T05:04:00Z",
             "end_time": None, "max_class": "M1.0"},  # still in progress
        ]), encoding="utf-8")
        ev = load_goes_events(p)
        check("SWPC JSON parsed, ongoing flare kept with end = peak",
              len(ev) == 2 and ev[1]["end"] == ev[1]["peak"], str(ev))
        h = Path(d) / "hek.json"
        h.write_text(json.dumps({"result": [
            {"event_starttime": "2026-09-14T01:02:00", "event_peaktime": "2026-09-14T01:10:00",
             "event_endtime": "2026-09-14T01:30:00", "fl_goescls": "X1.1"}]}), encoding="utf-8")
        ev = load_goes_events(h)
        check("HEK export parsed", len(ev) == 1 and abs(ev[0]["flux"] - 1.1e-4) < 1e-12)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} forward-test groups\n")
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
    print("All forward-test checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
