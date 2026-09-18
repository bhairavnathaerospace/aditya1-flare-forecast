"""Are the model's probabilities usable as probabilities? And can that be fixed?

    python scripts/recalibrate.py --out-dir outputs/archive_goes --cache-dir outputs/archive/cache

The GOES-labelled model has good discrimination (AUC 0.94) but poor calibration:
its Brier skill for the occurrence heads is negative, so "70 %" does not mean 70 %.
This fits isotonic regression on the **validation** split only and applies it to
the test split, reporting Brier, Brier skill and reliability before and after,
plus the TSS at the same operating point (a monotone map cannot change AUC).

Writes <out-dir>/reports/calibration.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarflare.config import Config  # noqa: E402
from solarflare.metrics import best_threshold, brier_score, brier_skill_score, reliability, roc_auc, skill_scores  # noqa: E402
from solarflare.pipeline import make_loaders, prepare, resolve_device  # noqa: E402
from solarflare.train import collect_predictions, load_model  # noqa: E402


def scores(y, p, thr):
    s = skill_scores(y, p >= thr)
    return {"TSS": s["TSS"], "POD": s["POD"], "FAR": s["FAR"],
            "Brier": brier_score(y, p), "BSS_vs_climatology": brier_skill_score(y, p),
            "AUC": roc_auc(y, p), "threshold": float(thr),
            "mean_predicted": float(p.mean()), "base_rate": float(y.mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="D:/Data")
    ap.add_argument("--out-dir", default="outputs/archive_goes")
    ap.add_argument("--cache-dir", default="outputs/archive/cache")
    ap.add_argument("--goes-dir", default="D:/Data/goes")
    args = ap.parse_args()
    from sklearn.isotonic import IsotonicRegression

    cfg = Config(data_root=Path(args.data_root), out_dir=Path(args.out_dir))
    cfg.cache_dir = Path(args.cache_dir)
    cfg.pre.label_source = "goes"
    cfg.pre.goes_dir = args.goes_dir
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
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(pv, yv)
        pv_c, pt_c = iso.predict(pv), iso.predict(pt)
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
