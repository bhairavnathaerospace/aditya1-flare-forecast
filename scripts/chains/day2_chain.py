"""Day-2 queue: the post-run plan, run one step at a time on a free GPU.

Waits for the reseeded HEL1OS ablation (seed_chain.py) to finish, then:

1. learning curve -- 25/50/75/100 % of each fold's training flares, for all
   flares and for the HEL1OS overlap with and without hard X-rays;
2. PatchTST as a fifth rise-phase encoder, identical folds, seeds and metrics
   (written to its own out-dir so the four-encoder results are untouched);
3. probability recalibration -- isotonic on validation, scored on test;
4. anchored-flux retrain -- flux heads predict "calibrated SoLEXS flux now + a
   change" instead of an absolute level, then the same fair-reference scoring.

Every step writes its own log and JSON; a failure is recorded and the queue
carries on, so one bad step cannot cost the rest of the day.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = "outputs/archive_goes"
LOGS = ROOT / OUT / "reports" / "logs"
SEED_LOG = ROOT / OUT / "reports" / "seed_chain.log"
PY = sys.executable
DATA = ["--data-root", "D:/Data", "--goes-dir", "D:/Data/goes", "--cache-dir", "outputs/archive/cache"]
GOES = ["--labels", "goes"]

while SEED_LOG.exists() and "SEED CHAIN DONE" not in SEED_LOG.read_text("utf-8", errors="ignore"):
    print(f"[{time.strftime('%H:%M:%S')}] waiting for the seed chain", flush=True)
    time.sleep(120)

STEPS = [
    ("learning curve", [PY, "-u", "scripts/learning_curve.py", "--out-dir", OUT, *DATA], "learning-curve.log"),
    ("patchtst rise CV", [PY, "-u", "-m", "solarflare.cli", "forecast", "--encoders", "patchtst",
                          "--folds", "3", "--out-dir", "outputs/archive_goes_patchtst", *DATA, *GOES],
     "forecast-cv-patchtst.log"),
    ("recalibrate probabilities", [PY, "-u", "scripts/recalibrate.py", "--out-dir", OUT, *DATA], "recalibrate.log"),
    ("anchored-flux retrain", [PY, "-u", "-m", "solarflare.cli", "train", "--anchor-flux",
                               "--out-dir", "outputs/archive_goes_anchor", *DATA, *GOES], "train-anchor.log"),
    ("anchored fair references", [PY, "-u", "scripts/fair_references.py",
                                  "--out-dir", "outputs/archive_goes_anchor", *DATA], "fair-references-anchor.log"),
]

for name, cmd, log in STEPS:
    print(f"[{time.strftime('%H:%M:%S')}] {name} ...", flush=True)
    t0 = time.time()
    with (LOGS / log).open("w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"[{time.strftime('%H:%M:%S')}] {name} exit={rc} ({time.time() - t0:.0f}s)", flush=True)

print("DAY2 CHAIN DONE", flush=True)
