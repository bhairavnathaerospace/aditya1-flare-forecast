"""Final steps of the GOES-labelled run. resume_chain.py stopped after the re-evaluation
(its evaluation backup was an empty file left by the 16:40 interruption); the training
metadata was restored by hand from history.json and nowcast-train.log."""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = "outputs/archive_goes"
LOGS = ROOT / OUT / "reports" / "logs"
PY = sys.executable
DATA = ["--data-root", "D:/Data", "--out-dir", OUT]
GOES = ["--labels", "goes", "--goes-dir", "D:/Data/goes", "--cache-dir", "outputs/archive/cache"]


def step(name, cmd, log):
    print(f"[{time.strftime('%H:%M:%S')}] {name} ...", flush=True)
    t0 = time.time()
    with (LOGS / log).open("w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"[{time.strftime('%H:%M:%S')}] {name} exit={rc} ({time.time() - t0:.0f}s)", flush=True)


step("post-hoc event definitions",
     [PY, "-u", "scripts/posthoc_event_definitions.py", *DATA, "--energy-scale", "sarwade2025", *GOES],
     "posthoc-event-definitions.log")
step("freeze v2", [PY, "-u", "-m", "solarflare.cli", "forward-test", "freeze", "--name", "v2", *DATA, *GOES],
     "forward-freeze-v2.log")
step("report", [PY, "-m", "solarflare.cli", "report", "--out-dir", OUT], "report-final.log")
print("CHAIN DONE", flush=True)
