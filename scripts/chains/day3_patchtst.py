"""Re-run the PatchTST step the day-2 queue lost to a wrong command name.

`forecast-cv` is the run_all stage name; the CLI subcommand is `forecast`.
Waits for the day-2 chain to finish so the GPU is free, then runs PatchTST over
the same folds, seeds and metrics as the other encoders, into its own out-dir.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "outputs/archive_goes/reports/day2_chain.log"
LOGS = ROOT / "outputs/archive_goes/reports/logs"
PY = sys.executable

while LOG.exists() and "DAY2 CHAIN DONE" not in LOG.read_text("utf-8", errors="ignore"):
    time.sleep(120)

cmd = [PY, "-u", "-m", "solarflare.cli", "forecast", "--encoders", "patchtst", "--folds", "3",
       "--out-dir", "outputs/archive_goes_patchtst", "--data-root", "D:/Data",
       "--goes-dir", "D:/Data/goes", "--cache-dir", "outputs/archive/cache", "--labels", "goes"]
print(f"[{time.strftime('%H:%M:%S')}] patchtst rise CV ...", flush=True)
t0 = time.time()
with (LOGS / "forecast-patchtst.log").open("w", encoding="utf-8") as fh:
    rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
print(f"[{time.strftime('%H:%M:%S')}] patchtst exit={rc} ({time.time() - t0:.0f}s)", flush=True)
print("PATCHTST DONE", flush=True)
