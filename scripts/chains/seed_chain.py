"""Step 1 of the post-run plan: does the HEL1OS peak-forecast gain survive reseeding?

The 2026-09-15 run measured +10.3 % [CI +0.012, +0.042 dex] from adding HEL1OS,
with one encoder (TCN) and one seed (1337) per fold. This repeats the identical
paired ablation with three more seeds, and once with the linear encoder, so the
gain can be quoted with its spread across seeds instead of a single draw.

Each run writes reports/fusion_ablation.json; it is moved to
reports/fusion_seeds/<encoder>_seed<seed>.json straight afterwards.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = "outputs/archive_goes"
REP = ROOT / OUT / "reports"
SEEDS = REP / "fusion_seeds"
LOGS = REP / "logs"
PY = sys.executable
COMMON = ["--data-root", "D:/Data", "--out-dir", OUT, "--labels", "goes",
          "--goes-dir", "D:/Data/goes", "--cache-dir", "outputs/archive/cache"]

RUNS = [("tcn", 1338), ("tcn", 1339), ("tcn", 1340), ("linear", 1337)]

for enc, seed in RUNS:
    tag = f"{enc}_seed{seed}"
    dest = SEEDS / f"{tag}.json"
    if dest.exists():
        print(f"[{time.strftime('%H:%M:%S')}] {tag} already done", flush=True)
        continue
    print(f"[{time.strftime('%H:%M:%S')}] fusion {tag} ...", flush=True)
    t0 = time.time()
    with (LOGS / f"fusion-{tag}.log").open("w", encoding="utf-8") as fh:
        rc = subprocess.run([PY, "-u", "-m", "solarflare.cli", "fusion", *COMMON,
                             "--encoder", enc, "--folds", "3", "--seed", str(seed)],
                            cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    src = REP / "fusion_ablation.json"
    if rc == 0 and src.exists():
        shutil.move(src, dest)
    print(f"[{time.strftime('%H:%M:%S')}] fusion {tag} exit={rc} ({time.time() - t0:.0f}s)", flush=True)

print("SEED CHAIN DONE", flush=True)
