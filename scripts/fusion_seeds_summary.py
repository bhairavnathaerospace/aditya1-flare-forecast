"""Summarise the reseeded HEL1OS fusion ablations.

    python scripts/fusion_seeds_summary.py

Reads outputs/archive_goes/reports/fusion_seeds/*.json (one paired ablation per
encoder and seed) and reports, per run and across runs, the peak-error reduction
from adding HEL1OS with its own bootstrap interval. A result that only holds for
one seed is a draw of the dice, not a finding.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SEEDS = Path("outputs/archive_goes/reports/fusion_seeds")


def main() -> int:
    files = sorted(SEEDS.glob("*.json"))
    if not files:
        print(f"no runs in {SEEDS}")
        return 1
    rows = []
    print(f"{'run':>16} {'soft+hard':>10} {'soft-only':>10} {'gain dex':>9} {'95% CI':>20} {'rel':>7} {'better':>7} {'n':>5}")
    for f in files:
        d = json.loads(f.read_text("utf-8"))
        g = d["peak_MAE_reduction_from_hel1os"]
        rows.append((f.stem, d["peak_log_MAE_soft_hard"], d["peak_log_MAE_soft_only"], g["value"],
                     g["ci_lo"], g["ci_hi"], d["relative_reduction"], d["events_improved_fraction"],
                     d["n_test_events"]))
        print(f"{f.stem:>16} {rows[-1][1]:10.4f} {rows[-1][2]:10.4f} {g['value']:+9.4f} "
              f"[{g['ci_lo']:+.4f}, {g['ci_hi']:+.4f}] {100 * d['relative_reduction']:6.1f}% "
              f"{100 * d['events_improved_fraction']:6.0f}% {d['n_test_events']:5d}")
    gains = np.array([r[3] for r in rows])
    print(f"\nacross {len(rows)} run(s): mean gain {gains.mean():+.4f} dex, "
          f"spread {gains.std(ddof=1) if len(gains) > 1 else float('nan'):.4f}, "
          f"range [{gains.min():+.4f}, {gains.max():+.4f}]")
    pos = int((gains > 0).sum())
    excl = sum(1 for r in rows if r[4] > 0)
    print(f"positive in {pos}/{len(rows)}; interval excludes zero in {excl}/{len(rows)}")
    print("verdict:", "holds up" if excl == len(rows) and len(rows) >= 3
          else "mixed - quote the spread, not one seed" if pos > len(rows) / 2 else "does not hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
