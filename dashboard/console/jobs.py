"""Console jobs: one at a time, run detached by dashboard/job_runner.py.

A job is a JSON file under outputs/jobs/ listing its steps; the runner records
state, current step and exit codes back into it. Closing the console never
stops a job.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .common import JOBS, NEW_GROUP, NO_WINDOW, ROOT, pid_alive, read_json


def find_python() -> str | None:
    """The Python that runs jobs: this one, or for the .exe the installed one."""
    if not getattr(sys, "frozen", False):
        exe = Path(sys.executable)
        cand = exe.with_name("python.exe")
        return str(cand if cand.exists() else exe)
    for name in ("python", "py"):
        p = shutil.which(name)
        if p:
            return p
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python"
    for d in sorted(base.glob("Python3*"), reverse=True):
        if (d / "python.exe").exists():
            return str(d / "python.exe")
    return None


def current_job() -> tuple[Path | None, dict | None]:
    files = sorted(JOBS.glob("*.json"), key=lambda p: p.stat().st_mtime) if JOBS.is_dir() else []
    for f in reversed(files):
        j = read_json(f)
        if j:
            return f, j
    return None, None


def job_running() -> tuple[Path | None, dict | None, bool]:
    path, job = current_job()
    return path, job, bool(job and job.get("state") == "running" and pid_alive(job.get("runner_pid")))


def launch(name: str, steps: list[dict]) -> str | None:
    """Start a job; returns an error message, or None when it started.
    ``"PY"`` in a step's command stands for the Python that runs jobs."""
    if job_running()[2]:
        return "Another job is running. Wait for it, or stop it first."
    py = find_python()
    if py is None:
        return "Could not find the installed Python that runs the jobs."
    for s in steps:
        s["cmd"] = [py if c == "PY" else c for c in s["cmd"]]
    JOBS.mkdir(parents=True, exist_ok=True)
    jid = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    job = {"id": jid, "name": name, "steps": steps, "state": "starting",
           "log": str(JOBS / f"{jid}.log"), "created_unix": time.time()}
    path = JOBS / f"{jid}.json"
    path.write_text(json.dumps(job, indent=1), encoding="utf-8")
    subprocess.Popen([py, str(ROOT / "dashboard" / "job_runner.py"), str(path)], cwd=ROOT,
                     creationflags=NEW_GROUP | NO_WINDOW, close_fds=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return None


def stop(path: Path, job: dict) -> None:
    """Stop the job's whole process tree and record it."""
    subprocess.run(["taskkill", "/PID", str(job["runner_pid"]), "/T", "/F"], capture_output=True,
                   creationflags=NO_WINDOW)
    job.update({"state": "stopped", "finished_unix": time.time()})
    path.write_text(json.dumps(job, indent=1), encoding="utf-8")
    with open(job["log"], "a", encoding="utf-8") as fh:
        fh.write(f"\n[{datetime.now():%H:%M:%S}] stopped from the console\n")
