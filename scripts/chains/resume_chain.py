"""Resume the GOES-labelled archive run after the first chain was killed at 15:34
(its parent chat session ended during forecast-cv fold 2).

Kept: nowcast training (checkpoints/best.pt) and its test evaluation.
Second attempt (16:40) died of a console Ctrl+C after baselines finished, so this
is launched in its own process group, which ignores Ctrl+C.
Re-run: the test evaluation (persistence reference now scored only
on real origins), forecast-cv, fusion, post-hoc re-scoring, freeze v2, report.
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = "outputs/archive_goes"
REP = ROOT / OUT / "reports"
LOGS = REP / "logs"
PY = sys.executable
DATA = ["--data-root", "D:/Data", "--out-dir", OUT]
GOES = ["--labels", "goes", "--goes-dir", "D:/Data/goes", "--cache-dir", "outputs/archive/cache"]


def step(name, cmd, log):
    print(f"[{time.strftime('%H:%M:%S')}] {name} ...", flush=True)
    t0 = time.time()
    with (LOGS / log).open("w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"[{time.strftime('%H:%M:%S')}] {name} exit={rc} ({time.time() - t0:.0f}s)", flush=True)
    return rc


step("run_all (baselines, forecast-cv, fusion, report)",
     [PY, "-u", "scripts/run_all.py", *DATA, "--encoders", "tcn,gru,transformer,linear", "--folds", "3",
      "--skip", "tests-correctness,tests-robustness,tests-scale,tests-extract,tests-hel1os,"
                "tests-forward,tests-goes,cache,inspect,nowcast-train,baselines",
      "--cli-args", " ".join(GOES)],
     "run_all-resume.log")

ev = REP / "evaluation.json"
before = REP / "evaluation_before_persistence_fix.json"
if ev.exists() and not before.exists():
    shutil.copy2(ev, before)
rc = step("re-evaluate best.pt with the persistence fix",
          [PY, "-u", "-m", "solarflare.cli", "evaluate", *DATA, *GOES], "evaluate-fixed.log")
if rc == 0 and before.exists():
    new, old = json.loads(ev.read_text("utf-8")), json.loads(before.read_text("utf-8"))
    if "training" in old and "training" not in new:
        new["training"] = old["training"]
        ev.write_text(json.dumps(new, indent=2), encoding="utf-8")

step("post-hoc event definitions",
     [PY, "-u", "scripts/posthoc_event_definitions.py", *DATA, "--energy-scale", "sarwade2025", *GOES],
     "posthoc-event-definitions.log")
step("freeze v2", [PY, "-u", "-m", "solarflare.cli", "forward-test", "freeze", "--name", "v2", *DATA, *GOES],
     "forward-freeze-v2.log")
step("report", [PY, "-m", "solarflare.cli", "report", "--out-dir", OUT], "report-final.log")
print("CHAIN DONE", flush=True)
