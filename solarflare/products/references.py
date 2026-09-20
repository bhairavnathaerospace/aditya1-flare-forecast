"""A fair "no change" reference for a GOES-labelled, Aditya-only flux forecast.

    python -m solarflare references

evaluate.py scores the flux forecast against persistence of the *GOES* flux at
the forecast origin. That reference reads the instrument the model is asked to
predict, which an Aditya-only system never sees. The operational alternative is
persistence of what SoLEXS measures, mapped to GOES units by a calibration fitted
on the training period only.

On test windows where both references exist -- GOES valid at the origin and in
the future, SoLEXS observing at the origin -- this reports RMSE for:

* the model's median forecast (frozen checkpoint, recomputed here);
* "no change", GOES at the origin;
* "no change", SoLEXS at the origin, calibrated on training data;
* training climatology;

and the same for the nowcast (horizon 0): can the network infer the GOES flux
from Aditya data better than a one-line calibration?

Writes <out-dir>/reports/fair_references.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from solarflare.config import run_config
from solarflare.pipeline import make_loaders, prepare, resolve_device
from solarflare.settings import load_settings
from solarflare.train import collect_predictions, load_model


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main(argv=None) -> int:
    S = load_settings()
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--out-dir", default=str(S.model_dir), help="a trained run")
    args = ap.parse_args(argv)

    cfg = run_config(Path(args.out_dir))
    prep = prepare(cfg, verbose=False)

    # SoLEXS GOES-long analogue and GOES target on one sorted time axis.
    t = np.concatenate([s.time_unix for s in prep.segments])
    gl = np.concatenate([s.goes_long if s.goes_long is not None else np.full(len(s), np.nan)
                         for s in prep.segments]).astype(np.float64)
    soft_ok = np.concatenate([s.soft_mask for s in prep.segments]) > 0
    goes = np.concatenate([s.log_flux for s in prep.segments]).astype(np.float64)
    goes_ok = np.concatenate([s.target_valid for s in prep.segments]) > 0
    order = np.argsort(t, kind="stable")
    t, gl, soft_ok, goes, goes_ok = t[order], gl[order], soft_ok[order], goes[order], goes_ok[order]

    # Calibration on the training period only.
    train_end = max(prep.windows[i].t_unix for i in prep.splits["train"])
    m = (t <= train_end) & soft_ok & goes_ok & np.isfinite(gl) & (gl > 0)
    slope, icpt = np.polyfit(np.log10(gl[m]), goes[m], 1)
    calib_sd = float(np.std(goes[m] - (icpt + slope * np.log10(gl[m]))))

    device = resolve_device(cfg.train.device)
    model = load_model(Path(args.out_dir) / "checkpoints" / "best.pt", cfg, device)
    _, _, te, _ = make_loaders(prep, cfg)
    test = collect_predictions(model, te, device)

    i = np.searchsorted(t, test["y_t_unix"])
    i = np.clip(i, 0, t.size - 1)
    exact = np.isclose(t[i], test["y_t_unix"])
    slx_origin = np.where(exact & soft_ok[i] & (gl[i] > 0), icpt + slope * np.log10(np.clip(gl[i], 1e-9, None)), np.nan)
    origin_ok = (test["y_nowcast_mask"] > 0) & np.isfinite(slx_origin)

    qs = list(cfg.win.quantiles)
    i50 = int(np.argmin(np.abs(np.array(qs) - 0.5)))
    tr_mean = float(np.mean(goes[(t <= train_end) & goes_ok]))
    out = {
        "calibration_train_only": f"log10 F = {icpt:.3f} + {slope:.3f} log10 rate (sd {calib_sd:.3f} dex)",
        "n_test_windows": int(test["y_t_unix"].size),
        "n_with_both_references": int(origin_ok.sum()),
        "nowcast": {
            "model_rmse": rmse(test["nowcast"][origin_ok], test["y_nowcast"][origin_ok]),
            "solexs_calibration_rmse": rmse(slx_origin[origin_ok], test["y_nowcast"][origin_ok]),
        },
        "forecast": {},
    }
    for h, hs in enumerate(cfg.win.forecast_horizons_s):
        mm = origin_ok & (test["y_forecast_mask"][:, h] > 0)
        y = test["y_forecast"][mm, h]
        out["forecast"][f"{int(hs / 60)}min"] = {
            "n": int(mm.sum()),
            "model_rmse": rmse(test["forecast"][mm, h, i50], y),
            "no_change_goes_rmse": rmse(test["y_persistence"][mm], y),
            "no_change_solexs_calibrated_rmse": rmse(slx_origin[mm], y),
            "climatology_rmse": rmse(np.full(y.size, tr_mean), y),
        }

    p = Path(args.out_dir) / "reports" / "fair_references.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out["calibration_train_only"], f"| {out['n_with_both_references']} of {out['n_test_windows']} test windows")
    print(f"nowcast RMSE: model {out['nowcast']['model_rmse']:.3f}  SoLEXS calibration {out['nowcast']['solexs_calibration_rmse']:.3f}")
    print(f"{'horizon':>8} {'n':>7} {'model':>7} {'noch GOES':>10} {'noch SoLEXS':>12} {'clim':>6}")
    for k, v in out["forecast"].items():
        print(f"{k:>8} {v['n']:7d} {v['model_rmse']:7.3f} {v['no_change_goes_rmse']:10.3f} "
              f"{v['no_change_solexs_calibrated_rmse']:12.3f} {v['climatology_rmse']:6.3f}")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
