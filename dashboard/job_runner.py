"""Run one console job: its steps in order, detached from the console window.

    python dashboard/job_runner.py outputs/jobs/<id>.json

The console (mission_control.pyw) writes the job file and starts this with the
system Python in its own process group, so closing the console never stops a
job. The job file lists the steps; this runner:

* runs each step with the project folder as working directory, streaming its
  output into the job log (and, when a step names one, into a second log such
  as <run>/reports/train.log that the console's log panel reads);
* records state, step and exit codes back into the job file (atomically);
* stops at the first failing step;
* asks Windows not to sleep while it works (SetThreadExecutionState).
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save(path: Path, job: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(job, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def keep_awake(on: bool) -> None:
    try:
        es_continuous, es_system = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system if on else 0))
    except (AttributeError, OSError):
        pass


def run_step(step: dict, env: dict, out, extra, job: dict, path: Path) -> int:
    """Run one step, streaming its output to the job log (and ``extra``)."""
    try:
        p = subprocess.Popen(step["cmd"], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                             creationflags=0x08000000)
    except OSError as exc:
        out.write(f"could not start: {exc}\n")
        return -1
    job["child_pid"] = p.pid
    save(path, job)
    for line in p.stdout:
        out.write(line)
        out.flush()
        if extra:
            extra.write(line)
            extra.flush()
    return p.wait()


def main() -> int:
    path = Path(sys.argv[1])
    job = json.loads(path.read_text(encoding="utf-8"))
    job.update({"state": "running", "runner_pid": os.getpid(), "started_unix": time.time(), "exit_codes": []})
    save(path, job)
    log = Path(job["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    keep_awake(True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    ok = True
    with log.open("a", encoding="utf-8", errors="replace") as out:
        out.write(f"[{stamp()}] job '{job['name']}' started, {len(job['steps'])} step(s)\n")
        out.flush()
        for i, step in enumerate(job["steps"]):
            job["step"] = i
            save(path, job)
            out.write(f"\n[{stamp()}] step {i + 1}/{len(job['steps'])}: {step['label']}\n$ {' '.join(step['cmd'])}\n")
            out.flush()
            t0 = time.time()
            with contextlib.ExitStack() as stack:
                extra = None
                if step.get("log"):
                    Path(step["log"]).parent.mkdir(parents=True, exist_ok=True)
                    extra = stack.enter_context(open(step["log"], "a", encoding="utf-8", errors="replace"))
                rc = run_step(step, env, out, extra, job, path)
            job["exit_codes"].append(rc)
            out.write(f"[{stamp()}] step {i + 1} finished with exit code {rc} after {time.time() - t0:.0f} s\n")
            out.flush()
            if rc != 0:
                ok = False
                break
        job.update({"state": "done" if ok else "failed", "finished_unix": time.time()})
        out.write(f"\n[{stamp()}] job {'finished' if ok else 'FAILED'}\n")
    keep_awake(False)
    save(path, job)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
