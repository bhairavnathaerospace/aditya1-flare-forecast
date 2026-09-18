"""Freeze the anchored-flux model as v3 once the PatchTST run releases the GPU.

The anchored model beats both "no change" references from 15 min out and reads
the current flux better than the calibration itself, so it is the one that
should be carried forward for the independent forward test.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "outputs/archive_goes/reports/day3_patchtst.log"
LOGS = ROOT / "outputs/archive_goes/reports/logs"
PY = sys.executable

while LOG.exists() and "PATCHTST DONE" not in LOG.read_text("utf-8", errors="ignore"):
    time.sleep(120)

cmd = [PY, "-u", "-m", "solarflare.cli", "forward-test", "freeze", "--name", "v3",
       "--anchor-flux", "--out-dir", "outputs/archive_goes_anchor", "--data-root", "D:/Data",
       "--goes-dir", "D:/Data/goes", "--cache-dir", "outputs/archive/cache", "--labels", "goes"]
print(f"[{time.strftime('%H:%M:%S')}] freeze v3 (anchored) ...", flush=True)
t0 = time.time()
with (LOGS / "forward-freeze-v3.log").open("w", encoding="utf-8") as fh:
    rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
print(f"[{time.strftime('%H:%M:%S')}] freeze v3 exit={rc} ({time.time() - t0:.0f}s)", flush=True)
print("FREEZE V3 DONE", flush=True)
