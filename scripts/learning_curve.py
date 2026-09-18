"""Do we need more data? A learning curve on the rise-phase peak forecast.

    python scripts/learning_curve.py --out-dir outputs/archive_goes --cache-dir outputs/archive/cache

Each fold keeps its test block fixed and trains on only the most recent share of
its training flares (25 / 50 / 75 / 100 %). If skill is still climbing at 100 %,
more history would help; if it has flattened, the limit is the information in the
data, not its amount.

Two curves are measured:

* **all flares** -- SoLEXS-driven peak forecasting;
* **the HEL1OS overlap**, with and without hard X-rays, which says whether the
  fusion gain is still growing and so whether more simultaneous days are worth
  asking for.

Writes <out-dir>/reports/learning_curve.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarflare.config import Config  # noqa: E402
from solarflare.forecast import grouped_cv, rise_hard_coverage  # noqa: E402
from solarflare.pipeline import prepare  # noqa: E402
from solarflare.preprocess.events import build_rise_dataset  # noqa: E402


def mae(res: dict, enc: str = "tcn") -> dict:
    s = res.get("summary", {}).get(enc, {})
    return {k: s.get(k) for k in ("peak_log_MAE_mean", "peak_log_MAE_std",
                                  "peak_skill_vs_current_mean", "exceeds_AUC_mean",
                                  "time_to_peak_MAE_min_mean")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="D:/Data")
    ap.add_argument("--out-dir", default="outputs/archive_goes")
    ap.add_argument("--cache-dir", default="outputs/archive/cache")
    ap.add_argument("--goes-dir", default="D:/Data/goes")
    ap.add_argument("--encoder", default="tcn")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fractions", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--min-hard-coverage", type=float, default=0.5)
    args = ap.parse_args()

    cfg = Config(data_root=Path(args.data_root), out_dir=Path(args.out_dir))
    cfg.cache_dir = Path(args.cache_dir)
    cfg.pre.label_source = "goes"
    cfg.pre.goes_dir = args.goes_dir
    prep = prepare(cfg, verbose=True)
    ds = build_rise_dataset(prep.segments, cfg, stride_s=60.0 if prep.archive else None)
    cov = rise_hard_coverage(prep.segments, ds)
    hel = {e for e, c in cov.items() if c >= args.min_hard_coverage}
    print(f"rise samples {len(ds)}; HEL1OS covers >= {args.min_hard_coverage:.0%} of the rise "
          f"for {len(hel)} events", flush=True)

    fracs = [float(x) for x in args.fractions.split(",")]
    out = {"encoder": args.encoder, "folds": args.folds, "fractions": fracs,
           "n_events_hel1os": len(hel), "all_flares": {}, "hel1os_overlap": {}}
    dest = Path(args.out_dir) / "reports" / "learning_curve.json"

    for f in fracs:
        print(f"\n===== all flares, train fraction {f:.0%} =====", flush=True)
        r = grouped_cv(prep, cfg, encoders=(args.encoder,), n_folds=args.folds, ds=ds,
                       train_fraction=f, verbose=True)
        out["all_flares"][f"{f:g}"] = mae(r, args.encoder)
        dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for f in fracs:
        arms = {}
        for name, drop in (("soft+hard", False), ("soft-only", True)):
            print(f"\n===== HEL1OS overlap, {name}, train fraction {f:.0%} =====", flush=True)
            r = grouped_cv(prep, cfg, encoders=(args.encoder,), n_folds=args.folds, ds=ds,
                           event_subset=hel, drop_hard=drop, train_fraction=f, verbose=True)
            arms[name] = mae(r, args.encoder)
        gain = None
        if arms["soft-only"]["peak_log_MAE_mean"] and arms["soft+hard"]["peak_log_MAE_mean"]:
            gain = arms["soft-only"]["peak_log_MAE_mean"] - arms["soft+hard"]["peak_log_MAE_mean"]
        out["hel1os_overlap"][f"{f:g}"] = {**arms, "gain_dex": gain}
        dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\nLEARNING CURVE (peak log-MAE, dex; lower is better)")
    print(f"{'fraction':>9} {'all flares':>12} {'±sd':>7} | {'HEL1OS soft+hard':>17} {'soft-only':>10} {'gain':>8}")
    for f in fracs:
        a = out["all_flares"][f"{f:g}"]
        h = out["hel1os_overlap"][f"{f:g}"]
        print(f"{f:9.0%} {a['peak_log_MAE_mean']:12.4f} {a['peak_log_MAE_std']:7.4f} | "
              f"{h['soft+hard']['peak_log_MAE_mean']:17.4f} {h['soft-only']['peak_log_MAE_mean']:10.4f} "
              f"{h['gain_dex']:+8.4f}")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
