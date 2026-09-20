"""Are the model's probabilities usable as probabilities? And can that be fixed?

    python -m solarflare calibration-report

A network trained on rare events ranks them well but its raw probabilities need
not mean what they say. This fits the same isotonic calibration the frozen model
carries (solarflare.probcal, fitted on the **validation** split only) and applies
it to the test split, reporting Brier, Brier skill and reliability before and
after, plus the TSS at each one's best validation threshold (a monotone map
cannot change AUC).

Writes <out-dir>/reports/calibration.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from solarflare import probcal
from solarflare.config import run_config
from solarflare.metrics import best_threshold, brier_score, brier_skill_score, reliability, roc_auc, skill_scores
from solarflare.pipeline import make_loaders, prepare, resolve_device
from solarflare.settings import load_settings
from solarflare.train import collect_predictions, load_model


def scores(y, p, thr):
    s = skill_scores(y, p >= thr)
    return {"TSS": s["TSS"], "POD": s["POD"], "FAR": s["FAR"],
            "Brier": brier_score(y, p), "BSS_vs_climatology": brier_skill_score(y, p),
            "AUC": roc_auc(y, p), "threshold": float(thr),
            "mean_predicted": float(p.mean()), "base_rate": float(y.mean())}


def main(argv=None) -> int:
    S = load_settings()
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--out-dir", default=str(S.model_dir), help="a trained run")
    args = ap.parse_args(argv)

    cfg = run_config(Path(args.out_dir))
    prep = prepare(cfg, verbose=False)
    device = resolve_device(cfg.train.device)
    model = load_model(Path(args.out_dir) / "checkpoints" / "best.pt", cfg, device)
    _, va, te, _ = make_loaders(prep, cfg)
    val, test = collect_predictions(model, va, device), collect_predictions(model, te, device)

    out = {"heads": {}}
    tasks = [("in_flare", val["p_inflare"], val["y_in_flare"], val["y_nowcast_mask"],
              test["p_inflare"], test["y_in_flare"], test["y_nowcast_mask"])]
    for h, hs in enumerate(cfg.win.occurrence_horizons_s):
        tasks.append((f"within_{int(hs / 60)}min", val["p_occurrence"][:, h], val["y_occurrence"][:, h],
                      val["y_occurrence_mask"][:, h], test["p_occurrence"][:, h],
                      test["y_occurrence"][:, h], test["y_occurrence_mask"][:, h]))

    for name, pv, yv, mv, pt, yt, mt in tasks:
        v, t = mv > 0, mt > 0
        pv, yv, pt, yt = pv[v], yv[v], pt[t], yt[t]
        cal = probcal.fit(pv, yv)
        pv_c, pt_c = probcal.apply(cal, pv), probcal.apply(cal, pt)
        thr_raw, _ = best_threshold(yv, pv, "TSS")
        thr_cal, _ = best_threshold(yv, pv_c, "TSS")
        out["heads"][name] = {
            "n_val": int(yv.size), "n_test": int(yt.size),
            "raw": scores(yt, pt, thr_raw),
            "calibrated": scores(yt, pt_c, thr_cal),
            "reliability_raw": reliability(yt, pt),
            "reliability_calibrated": reliability(yt, pt_c),
        }
        r, c = out["heads"][name]["raw"], out["heads"][name]["calibrated"]
        print(f"{name:>14}  base {r['base_rate']:.3f} | raw: Brier {r['Brier']:.4f} "
              f"BSS {r['BSS_vs_climatology']:+.3f} mean p {r['mean_predicted']:.3f} TSS {r['TSS']:.3f}"
              f"  ->  calibrated: Brier {c['Brier']:.4f} BSS {c['BSS_vs_climatology']:+.3f} "
              f"mean p {c['mean_predicted']:.3f} TSS {c['TSS']:.3f}", flush=True)

    dest = Path(args.out_dir) / "reports" / "calibration.json"
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest}")
    print("Isotonic regression is monotone, so AUC is unchanged by construction; "
          "what changes is whether the numbers can be read as probabilities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
