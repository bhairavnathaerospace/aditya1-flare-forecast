"""Does HEL1OS help? The paired soft-only vs soft+hard ablation, across seeds.

    python -m solarflare hel1os-value

Reads outputs/ablations/hel1os/seed_*/reports/fusion_ablation.json (one paired
ablation per seed, from ``python -m solarflare fusion --seed N``) and reports the
peak-error reduction from adding HEL1OS for each seed and across seeds. A result
that holds for one seed only is a draw of the dice, not a finding.

Writes outputs/ablations/hel1os/HEL1OS_VALUE.md and hel1os_value.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from solarflare.settings import load_settings


def main(argv=None) -> int:
    s = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(s.ablations / "hel1os"))
    args = ap.parse_args(argv)
    root = Path(args.runs)
    files = sorted(root.glob("seed_*/reports/fusion_ablation.json"))
    if not files:
        print(f"no runs in {root}")
        return 1
    rows = []
    for f in files:
        d = json.loads(f.read_text("utf-8"))
        g = d["peak_MAE_reduction_from_hel1os"]
        rows.append({"seed": f.parents[1].name, "mae_soft_hard": d["peak_log_MAE_soft_hard"],
                     "mae_soft_only": d["peak_log_MAE_soft_only"], "gain_dex": g["value"],
                     "ci": [g["ci_lo"], g["ci_hi"]], "relative": d["relative_reduction"],
                     "flares_improved": d["events_improved_fraction"], "n_flares": d["n_test_events"]})
    gains = np.array([r["gain_dex"] for r in rows])
    rel = np.array([r["relative"] for r in rows])
    excl = sum(r["ci"][0] > 0 for r in rows)
    verdict = ("holds up: every seed's interval excludes zero" if excl == len(rows) and len(rows) >= 3
               else "mixed: quote the spread, not one seed" if (gains > 0).sum() > len(rows) / 2
               else "does not hold")
    summary = {"seeds": rows, "mean_gain_dex": round(float(gains.mean()), 4),
               "mean_relative": round(float(rel.mean()), 4),
               "relative_sd": round(float(rel.std(ddof=1)), 4) if len(rel) > 1 else None,
               "intervals_excluding_zero": int(excl), "verdict": verdict}
    (root / "hel1os_value.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Does HEL1OS help? (paired soft-only vs soft+hard, peak size of rising flares)", "",
             "| seed | error soft+hard (dex) | error soft only | gain [95% CI] | relative | flares improved | n |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['mae_soft_hard']:.4f} | {r['mae_soft_only']:.4f} | {r['gain_dex']:+.4f} "
                     f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {100 * r['relative']:.1f}% | "
                     f"{100 * r['flares_improved']:.0f}% | {r['n_flares']} |")
    sd = f" +- {100 * summary['relative_sd']:.1f}%" if summary["relative_sd"] is not None else ""
    lines += ["", f"Across {len(rows)} seed(s): {100 * summary['mean_relative']:.1f}%{sd} lower peak error with "
              f"HEL1OS; {excl} of {len(rows)} intervals exclude zero. Verdict: **{verdict}**.", ""]
    (root / "HEL1OS_VALUE.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
