"""The console: two windows, refreshed every second.

Operations window
  Pipeline     every stage of ``python -m solarflare pipeline`` and its state
  Training     the run being trained: tiles, four charts, the network diagram
  Machine      GPU, CPU, memory, disk and training throughput, last 3 minutes
  Terminal and watchdog

Flare Watch window
  The frozen model's alerts replayed day by day, with the flares of the day and
  the lead times over the whole replay. It opens beside the operations window,
  so the two can sit on one screen each.

Actions start one job at a time through dashboard/job_runner.py with the
installed Python, detached: closing a window never stops a job. Paths come from
config/project.toml, as for every other command in the project.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import re
import shutil
import subprocess
import threading
import time
import tkinter as tk
from datetime import UTC, datetime
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

from . import jobs
from .common import (AMBER, DOWNLOADS, FAINT, GREEN, GROUND, LINE, MONO, MUTED, NO_WINDOW, OUTPUTS, PANEL,
                     REFRESH_MS, RED, ROOT, S, SMALL, STEEL, TEAL, TEST_SUITES, TEXT, UI, UI_B, flat_button,
                     fmt_dur, tail)
from .flarewatch import FlareWatchTab
from .pipeline_view import PipelineTab
from .system import SystemPanel
from .training import TrainingTab, best_index, run_label, runs

ERROR_RE = re.compile(r"Traceback|Error\b|error:|CUDA out of memory|MemoryError|FAILED|Killed")
#: the line worth quoting: the exception itself, not the "job FAILED" footer
EXC_RE = re.compile(r"^\s*\w*(Error|Exception|Interrupt)\b.*:|CUDA out of memory|MemoryError")
TITLE = "SoLEXHEL-Net · Mission Console"
WATCH_TITLE = "SoLEXHEL-Net · Flare Watch"


class Poller(threading.Thread):
    """Slow checks off the UI thread: nvidia-smi every second, downloads every 30 s."""

    def __init__(self):
        super().__init__(daemon=True)
        self.gpu: dict | None = None
        self.downloading = False
        self._n = 0

    def run(self):
        q = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
        while True:
            try:
                out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=4,
                                     creationflags=NO_WINDOW).stdout.strip().splitlines()
                f = [x.strip() for x in out[0].split(",")]
                self.gpu = {"name": f[0], "util": float(f[1]), "mem": float(f[2]), "mem_total": float(f[3]),
                            "temp": float(f[4]), "power": float(f[5]) if f[5] not in ("[N/A]", "") else None}
            except (OSError, subprocess.SubprocessError, IndexError, ValueError):
                self.gpu = None
            if self._n % 30 == 0:
                try:
                    now = time.time()
                    self.downloading = any(now - p.stat().st_mtime < 180 for p in DOWNLOADS.glob("**/*.part"))
                except OSError:
                    self.downloading = False
            self._n += 1
            time.sleep(max(REFRESH_MS / 1000, 1.0))


def newest_log(run: Path | None) -> Path | None:
    if run is None:
        return None
    rep = run / "reports"
    if (rep / "train.log").exists():
        return rep / "train.log"
    logs = list(rep.glob("*.log"))
    return max(logs, key=lambda p: p.stat().st_mtime) if logs else None


def dark_combobox(root) -> None:
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Run.TCombobox", fieldbackground=PANEL, background=PANEL, foreground=TEXT,
                    arrowcolor=MUTED, bordercolor=LINE, lightcolor=PANEL, darkcolor=PANEL)
    style.map("Run.TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", TEXT)],
              selectbackground=[("readonly", PANEL)], selectforeground=[("readonly", TEXT)])
    root.option_add("*TCombobox*Listbox.background", PANEL)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)


class WatchWindow(tk.Toplevel):
    """The second screen: Flare Watch on its own, so it can live on another monitor."""

    def __init__(self, master):
        super().__init__(master, bg=GROUND)
        self.title(WATCH_TITLE)
        self.configure(bg=GROUND)
        self.minsize(820, 700)
        head = tk.Frame(self, bg=GROUND)
        head.pack(fill="x", padx=16, pady=(12, 6))
        left = tk.Frame(head, bg=GROUND)
        left.pack(side="left")
        tk.Label(left, text="ADITYA-L1  ·  SOLEXS + HEL1OS  ·  ALERTS", bg=GROUND, fg=MUTED,
                 font=SMALL).pack(anchor="w")
        tk.Label(left, text="Flare Watch", bg=GROUND, fg=TEXT, font=("Segoe UI Semibold", 16)).pack(anchor="w")
        self.clock = tk.Label(head, text="", bg=GROUND, fg=MUTED, font=MONO)
        self.clock.pack(side="right")
        self.tab = FlareWatchTab(self)
        self.tab.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        self.protocol("WM_DELETE_WINDOW", self.withdraw)


class Console(tk.Tk):
    def __init__(self, run: Path | None, tab: str = "pipeline", with_watch: bool = True):
        super().__init__()
        self.title(TITLE)
        self.configure(bg=GROUND)
        self.minsize(1020, 720)
        self.fixed_run = run
        self.poller = Poller()
        self.poller.start()
        self._term_tab = "job"
        self._gpu_idle = 0
        self._pipeline_state: dict = {}
        dark_combobox(self)
        self._build()
        self.show_tab(tab)
        self.watch = WatchWindow(self)
        self._place(with_watch)
        self.bind("<r>", lambda e: self.refresh(reschedule=False))
        self.after(300, self.refresh)

    def _place(self, with_watch: bool) -> None:
        """Side by side, both fully on this screen; drag either onto a second monitor."""
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        h = max(int(sh * 0.92) - 40, 700)
        gap, edge = 8, 8
        w = max(min(int(sw * 0.55), sw - 820 - gap - 2 * edge), 1020)
        ww = max(sw - w - gap - 2 * edge, 820)
        self.geometry(f"{w}x{h}+{edge}+0")
        self.watch.geometry(f"{ww}x{h}+{w + gap + edge}+0")
        if not with_watch:
            self.watch.withdraw()

    # ---- layout -------------------------------------------------------------------
    def _build(self):
        top = tk.Frame(self, bg=GROUND)
        top.pack(fill="x", padx=16, pady=(12, 6))
        left = tk.Frame(top, bg=GROUND)
        left.pack(side="left")
        tk.Label(left, text="ADITYA-L1  ·  SOLEXS + HEL1OS  ·  FLARE NOWCAST & FORECAST", bg=GROUND, fg=MUTED,
                 font=SMALL).pack(anchor="w")
        tk.Label(left, text="Mission console", bg=GROUND, fg=TEXT, font=("Segoe UI Semibold", 16)).pack(anchor="w")
        right = tk.Frame(top, bg=GROUND)
        right.pack(side="right")
        self.clock = tk.Label(right, text="", bg=GROUND, fg=MUTED, font=MONO)
        self.clock.pack(anchor="e")
        row = tk.Frame(right, bg=GROUND)
        row.pack(anchor="e", pady=(4, 0))
        tk.Label(row, text="Training run", bg=GROUND, fg=MUTED, font=UI).pack(side="left", padx=(0, 6))
        self.run_var = tk.StringVar(value="Follow latest")
        self.run_box = ttk.Combobox(row, textvariable=self.run_var, width=28, state="readonly",
                                    style="Run.TCombobox", font=UI)
        self.run_box.pack(side="left")
        self.run_box.bind("<<ComboboxSelected>>", lambda e: self.refresh(reschedule=False))

        bar = tk.Frame(self, bg=GROUND)
        bar.pack(fill="x", padx=16, pady=(2, 8))
        self.btns = {}
        for key, text, cmd, fg in (
                ("pipeline", "Run full pipeline", self.act_pipeline, TEAL),
                ("redo", "Redo stage…", self.act_redo, TEXT),
                ("check", "Check data", self.act_check, TEXT),
                ("extract", "Extract new data", self.act_extract, TEXT),
                ("cache", "Update cache", self.act_cache, TEXT),
                ("tests", "Run tests", self.act_tests, TEXT)):
            b = flat_button(bar, text, cmd, fg)
            b.pack(side="left", padx=(0, 6))
            self.btns[key] = b
        self.btns["stop"] = flat_button(bar, "Stop job", self.act_stop, RED)
        self.btns["stop"].pack(side="left", padx=(12, 6))
        flat_button(bar, "Flare Watch ⧉", self.show_watch, STEEL).pack(side="left", padx=(0, 6))
        flat_button(bar, "Outputs", lambda: os.startfile(OUTPUTS if OUTPUTS.exists() else ROOT), MUTED).pack(
            side="left", padx=(0, 6))
        flat_button(bar, "Results", self.act_results, MUTED).pack(side="left")
        jobrow = tk.Frame(self, bg=GROUND)
        jobrow.pack(fill="x", padx=16, pady=(0, 6))
        self.job_label = tk.Label(jobrow, text="", bg=GROUND, fg=MUTED, font=UI, anchor="w")
        self.job_label.pack(side="left")

        body = tk.Frame(self, bg=GROUND)
        body.pack(fill="both", expand=True, padx=16)
        body.grid_columnconfigure(0, weight=13, uniform="b")
        body.grid_columnconfigure(1, weight=7, uniform="b")
        body.grid_rowconfigure(0, weight=1)

        main = tk.Frame(body, bg=GROUND)
        main.grid(row=0, column=0, sticky="nsew")
        strip = tk.Frame(main, bg=GROUND)
        strip.pack(fill="x", pady=(0, 6))
        self.tab_labels = {}
        for key, text in (("pipeline", "Pipeline"), ("training", "Training")):
            lab = tk.Label(strip, text=text, bg=GROUND, fg=MUTED, font=("Segoe UI Semibold", 11), cursor="hand2",
                           padx=2)
            lab.pack(side="left", padx=(0, 18))
            lab.bind("<Button-1>", lambda e, k=key: self.show_tab(k))
            self.tab_labels[key] = lab
        tk.Frame(main, bg=LINE, height=1).pack(fill="x", pady=(0, 10))
        holder = tk.Frame(main, bg=GROUND)
        holder.pack(fill="both", expand=True)
        self.tabs = {"pipeline": PipelineTab(holder), "training": TrainingTab(holder)}

        side = tk.Frame(body, bg=GROUND)
        side.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(0, weight=0)
        side.grid_rowconfigure(1, weight=3)
        side.grid_rowconfigure(2, weight=2)
        self.system = SystemPanel(side)
        self.system.grid(row=0, column=0, sticky="nsew")

        term = tk.Frame(side, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        term.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        tabs = tk.Frame(term, bg=PANEL)
        tabs.pack(fill="x", padx=10, pady=(7, 0))
        tk.Label(tabs, text="TERMINAL", bg=PANEL, fg=MUTED, font=SMALL).pack(side="left", padx=(2, 10))
        self.term_tabs = {}
        for key, text in (("job", "Job output"), ("stage", "Stage log"), ("train", "Training log")):
            lab = tk.Label(tabs, text=text, bg=PANEL, fg=MUTED, font=UI, cursor="hand2")
            lab.pack(side="left", padx=(0, 10))
            lab.bind("<Button-1>", lambda e, k=key: self._switch_term(k))
            self.term_tabs[key] = lab
        self.term_src = tk.Label(tabs, text="", bg=PANEL, fg=FAINT, font=SMALL)
        self.term_src.pack(side="right")
        self.log = tk.Text(term, bg=PANEL, fg=TEXT, font=("Consolas", 9), relief="flat",
                           highlightthickness=0, wrap="none", height=10)
        self.log.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        for tag, col in (("best", TEAL), ("muted", MUTED), ("err", RED), ("step", STEEL)):
            self.log.tag_configure(tag, foreground=col)

        watch = tk.Frame(side, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        watch.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        tk.Label(watch, text="WATCHDOG  ·  checked every second", bg=PANEL, fg=MUTED, font=SMALL).pack(
            anchor="w", padx=12, pady=(9, 0))
        self.watchdog = tk.Text(watch, bg=PANEL, fg=TEXT, font=UI, relief="flat", highlightthickness=0,
                                wrap="word", height=6)
        self.watchdog.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        for tag, col in (("ALERT", RED), ("WARN", AMBER), ("INFO", STEEL), ("OK", GREEN)):
            self.watchdog.tag_configure(tag, foreground=col, font=UI_B)

        self.foot = tk.Label(self, text="", bg=GROUND, fg=FAINT, font=SMALL, anchor="w")
        self.foot.pack(fill="x", padx=16, pady=(8, 10), anchor="w")
        self._switch_term("job", refresh=False)

    def show_tab(self, key: str):
        for k, frame in self.tabs.items():
            frame.pack_forget()
            self.tab_labels[k].config(fg=TEXT if k == key else MUTED)
        self.tabs[key].pack(fill="both", expand=True)
        self._tab = key

    def show_watch(self):
        self.watch.deiconify()
        self.watch.lift()

    def _switch_term(self, key: str, refresh: bool = True):
        self._term_tab = key
        for k, lab in self.term_tabs.items():
            lab.config(fg=TEXT if k == key else MUTED, font=UI_B if k == key else UI)
        if refresh:
            self.refresh(reschedule=False)

    # ---- actions ------------------------------------------------------------------
    def start_job(self, name: str, steps: list[dict], tab: str | None = None) -> None:
        err = jobs.launch(name, steps)
        if err:
            messagebox.showinfo("Job", err)
            return
        if tab:
            self.show_tab(tab)
        self._switch_term("job")

    def _download_guard(self) -> bool:
        if self.poller.downloading:
            return messagebox.askyesno(
                "Download in progress",
                "A PRADAN download is still writing files. Half-finished files are skipped safely, "
                "but anything that lands after this starts will not be included.\n\nStart anyway?")
        return True

    def act_pipeline(self):
        if not self._download_guard():
            return
        if not messagebox.askyesno(
                "Run full pipeline",
                "Run every stage that is not done yet, in order: cache, copied-day check, training "
                "(with and without SHARP, a few hours each on the GPU; SHARP is kept only if it wins on "
                "validation), freeze, calibration, catalogue, alerts, day-ahead forecasts, the HEL1OS ablation "
                "(3 seeds), flare physics and outputs/RESULTS.md.\n\n"
                "It resumes where it stopped, keeps going past a failed stage when later stages do not need it, "
                "and keeps the laptop awake. Closing this window does not stop it.\n\nStart?"):
            return
        self.start_job("Full pipeline", [{"label": "pipeline", "cmd": ["PY", "-u", "-m", "solarflare", "pipeline",
                                                                       "--keep-going"]}], tab="pipeline")

    def act_redo(self):
        from solarflare import runall

        st = runall.stages(S)
        names = [x.name for x in st]
        name = simpledialog.askstring("Redo stage", "Stage to run again (everything built on it runs again too):\n\n"
                                      + ", ".join(names), parent=self)
        if not name:
            return
        name = name.strip()
        if name not in names:
            messagebox.showerror("Redo stage", f"No stage called '{name}'.")
            return
        dep = sorted(runall.dependents(name, st), key=names.index)
        if not messagebox.askyesno("Redo stage", f"Run '{name}' again" + (f", then {', '.join(dep)}" if dep else "")
                                   + ".\nEarlier results are replaced; a frozen model is moved aside, never "
                                     "overwritten.\n\nStart?"):
            return
        self.start_job(f"Redo {name}", [{"label": f"redo {name}", "cmd": [
            "PY", "-u", "-m", "solarflare", "pipeline", "--redo", name, "--keep-going"]}], tab="pipeline")

    def act_check(self):
        self.start_job("Check data", [{"label": "data status", "cmd": ["PY", "-u", "scripts/data_status.py",
                                                                        "--data-root", str(S.data_root)]}])

    def act_extract(self):
        free = shutil.disk_usage(S.data_root).free / 1e9
        if not self._download_guard():
            return
        floor = f"{S.min_free_gb:g}"
        if not messagebox.askyesno(
                "Extract new data",
                "Extract every downloaded zip not yet extracted, in full (photon lists included). Zips are "
                "never deleted; already-extracted products are skipped. Then read the new files into the cache "
                f"and re-check for copied SoLEXS days.\n\nThe data drive has {free:.0f} GB free; extraction stops "
                f"by itself at {floor} GB.\n\nStart?"):
            return
        data = ["--data-root", str(S.data_root), "--min-free-gb", floor]
        self.start_job("Extract new data", [
            {"label": "HEL1OS", "cmd": ["PY", "-u", "scripts/unzip_archive.py", *data, "--instrument", "hel1os",
                                        "--members", "all"]},
            {"label": "SoLEXS", "cmd": ["PY", "-u", "scripts/unzip_archive.py", *data, "--instrument", "solexs"]},
            {"label": "cache", "cmd": ["PY", "-u", "-m", "solarflare", "cache"]},
            {"label": "copied-day check", "cmd": ["PY", "-u", "-m", "solarflare", "quality"]}])

    def act_cache(self):
        if self._download_guard() and messagebox.askyesno(
                "Update cache", "Read any new SoLEXS/HEL1OS products into the preprocessing cache. Existing "
                                "entries are reused; only new files are read.\n\nStart?"):
            self.start_job("Update cache", [{"label": "cache", "cmd": ["PY", "-u", "-m", "solarflare", "cache"]}])

    def act_tests(self):
        self.start_job("Run tests", [{"label": s, "cmd": ["PY", "-u", "-m", f"tests.{s}"]} for s in TEST_SUITES])

    def act_stop(self):
        path, job, running = jobs.job_running()
        if not running:
            messagebox.showinfo("Stop job", "No job is running.")
            return
        if messagebox.askyesno("Stop job", f"Stop '{job['name']}' now? A pipeline stopped midway resumes from the "
                                           "interrupted stage next time; a training run keeps its best checkpoint "
                                           "so far but is trained again when resumed."):
            jobs.stop(path, job)

    def act_results(self):
        p = OUTPUTS / "RESULTS.md"
        if p.exists():
            os.startfile(p)
        else:
            messagebox.showinfo("Results", "outputs/RESULTS.md is written by the last pipeline stage ('summary').")

    # ---- refresh ------------------------------------------------------------------
    def current_run(self) -> Path | None:
        all_runs = runs()
        labels = ["Follow latest"] + [run_label(r) for r in all_runs]
        if list(self.run_box["values"]) != labels:
            self.run_box["values"] = labels
        choice = self.run_var.get()
        if choice != "Follow latest":
            return next((r for r in all_runs if run_label(r) == choice), None)
        if self.fixed_run is not None:
            return self.fixed_run
        return all_runs[0] if all_runs else None

    def refresh(self, reschedule: bool = True):
        try:
            self._refresh()
        except Exception as exc:                      # keep the console alive on odd files
            self.foot.config(text=f"refresh error: {type(exc).__name__}: {exc}")
        if reschedule:
            self.after(REFRESH_MS, self.refresh)

    def _refresh(self):
        now = time.time()
        stamp = datetime.now(UTC).strftime("%Y-%m-%d  %H:%M:%S UTC")
        self.clock.config(text=stamp)
        self.watch.clock.config(text=stamp)
        _, job, job_running = jobs.job_running()
        self._update_job_bar(job, job_running)
        self._pipeline_state = self.tabs["pipeline"].refresh()
        run = self.current_run()
        live, hist, status, cfg = self.tabs["training"].refresh(run, now, self.poller.gpu, job_running)
        self.system.refresh(self.poller.gpu, live, now)
        self.watch.tab.refresh()
        self._update_terminal(run, job)
        self._update_watch(job, job_running, live, hist, status, cfg, now)
        self.foot.config(text=f"Training tab follows {run_label(run) if run else 'nothing yet'}   ·   refreshed "
                              "every second   ·   R refreshes now   ·   jobs keep running if these windows are "
                              "closed   ·   settings: config/project.toml")

    def _update_job_bar(self, job, running: bool):
        for k, b in self.btns.items():
            b.config(state="normal" if (k == "stop") == running else "disabled")
        if job is None:
            self.job_label.config(text="No jobs yet", fg=MUTED)
        elif running:
            el = time.time() - float(job.get("started_unix", time.time()))
            step = int(job.get("step", 0)) + 1
            self.job_label.config(text=f"Running: {job['name']}  ·  step {step}/{len(job['steps'])}  ·  {fmt_dur(el)}",
                                  fg=TEAL)
        else:
            st = job.get("state", "?")
            col = {"done": GREEN, "failed": RED, "stopped": AMBER}.get(st, RED)
            word = {"done": "finished", "failed": "FAILED", "stopped": "stopped",
                    "running": "ended unexpectedly", "starting": "did not start"}.get(st, st)
            when = job.get("finished_unix") or job.get("created_unix")
            ago = f"  ·  {fmt_dur(time.time() - when)} ago" if when else ""
            self.job_label.config(text=f"Last job: {job['name']}  ·  {word}{ago}", fg=col)

    def _stage_log(self) -> tuple[Path | None, str]:
        stages = self._pipeline_state.get("stages", {}) if self._pipeline_state else {}
        if not stages:
            return None, "no pipeline stage yet"
        name, d = max(stages.items(), key=lambda kv: kv[1].get("started", ""))
        return (Path(d["log"]) if d.get("log") else None), f"{name}  ·  {d.get('status', '?')}"

    def _update_terminal(self, run: Path | None, job):
        if self._term_tab == "job":
            lp = Path(job["log"]) if job and job.get("log") else None
            label = f"{job['name']}  ·  {lp.name}" if lp else "no job yet"
            empty = "(nothing yet — use the buttons above to start a job)"
        elif self._term_tab == "stage":
            lp, label = self._stage_log()
            empty = "(no pipeline stage has run yet)"
        else:
            lp = newest_log(run)
            if lp is None:
                p = S.outputs / "pipeline" / "logs"
                cands = list(p.glob("train*.log")) if p.is_dir() else []
                lp = max(cands, key=lambda x: x.stat().st_mtime) if cands else None
            label = lp.name if lp else "no training log"
            empty = "(no training log yet)"
        self.term_src.config(text=label)
        lines = tail(lp, 80) if lp and lp.exists() else [empty]
        at_end = self.log.yview()[1] > 0.98
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        for ln in lines:
            s = ln.strip()
            if ERROR_RE.search(ln):
                tag = "err"
            elif s.startswith("ep ") and s.endswith("*"):
                tag = "best"
            elif s.startswith("[") or s.startswith("$ ") or s.startswith("====="):
                tag = "step"
            else:
                tag = "" if s.startswith("ep ") else "muted"
            self.log.insert("end", ln + "\n", tag)
        if at_end:
            self.log.see("end")
        self.log.config(state="disabled")

    def _update_watch(self, job, job_running, live, hist, status, cfg, now):
        items: list[tuple[str, str]] = []
        if job:
            st = job.get("state")
            if st == "running" and not job_running:
                items.append(("ALERT", f"Job '{job['name']}' stopped unexpectedly (its process is gone). "
                                       "See the job output."))
            elif st == "failed":
                codes = job.get("exit_codes", [])
                items.append(("ALERT", f"Job '{job['name']}' failed (exit code {codes[-1] if codes else '?'})."))
            elif job_running:
                items.append(("OK", f"Job '{job['name']}' running."))
            lines = tail(Path(job["log"]), 80) if job.get("log") else []
            errs = [ln for ln in lines if EXC_RE.search(ln)] or \
                [ln for ln in lines if ERROR_RE.search(ln) and "job FAILED" not in ln and "exit code" not in ln
                 and "[FAILED]" not in ln]
            if errs and (job_running or st in ("failed", "running")):
                items.append(("ALERT", "Error in the job output: " + errs[-1].strip()[:160]))
        for name, d in (self._pipeline_state or {}).get("stages", {}).items():
            if d.get("status") == "failed":
                items.append(("ALERT", f"Pipeline stage '{name}' failed: {d.get('error', '?')}. "
                                       f"Log: outputs/pipeline/logs/{name}.log"))
            elif d.get("status") == "running" and job_running:
                items.append(("OK", f"Pipeline stage '{name}' running."))
        if live and status in ("TRAINING", "VALIDATING", "STALLED"):
            # (a finished run reports FINISHED even when its live file froze)
            lr_ = live.get("loss_running")
            if lr_ is not None and not math.isfinite(float(lr_)):
                items.append(("ALERT", "Training loss is NaN/inf: the run is diverging. Stop it and check."))
            if status == "STALLED":
                items.append(("ALERT", f"No training update for {fmt_dur(now - float(live.get('updated_unix', 0)))}."))
            elif status == "VALIDATING" and now - float(live.get("updated_unix", now)) > 1200:
                items.append(("WARN", "Validation has taken over 20 min."))
            if hist:
                be = best_index(hist)
                since = len(hist) - 1 - be
                patience = int((cfg.get("train") or {}).get("early_stop_patience", 12))
                if since >= max(patience // 2, 3):
                    items.append(("INFO", f"No better (smoothed) score for {since} epochs; early stop after "
                                          f"{patience}."))
            g = self.poller.gpu
            if g and status == "TRAINING":
                self._gpu_idle = self._gpu_idle + 1 if g["util"] < 5 else 0
                if self._gpu_idle >= 30:
                    items.append(("WARN", "GPU idle for 30 s while training: data loading is the bottleneck, "
                                          "or the run is stuck."))
        g = self.poller.gpu
        if g and g["temp"] >= 85:
            items.append(("WARN", f"GPU at {g['temp']:.0f} °C. Keep the laptop ventilated."))
        for label, path in (("data drive", S.data_root), ("outputs drive", OUTPUTS)):
            try:
                p = path if path.exists() else path.parent
                free = shutil.disk_usage(p).free / 1e9
                if free < S.min_free_gb:
                    items.append(("ALERT", f"The {label} has {free:.0f} GB free, below the {S.min_free_gb:g} GB floor: "
                                           "extraction and the pipeline stop. Move data to the big drive."))
                elif free < S.min_free_gb + 15:
                    items.append(("WARN", f"The {label} has {free:.0f} GB free (floor {S.min_free_gb:g} GB)."))
            except OSError:
                items.append(("WARN", f"The {label} ({path}) is not reachable."))
        has_sharp = S.sharp_dir.exists() and any(S.sharp_dir.glob("sharp_*.csv"))
        if not has_sharp:
            items.append(("INFO", f"No SHARP files in {S.sharp_dir}: the pipeline trains on X-rays only."))
        if self.poller.downloading:
            items.append(("INFO", "A PRADAN download is in progress."))
        if not any(k in ("ALERT", "WARN") for k, _ in items):
            items.insert(0, ("OK", "All checks pass."))
        seen, unique = set(), []
        for it in items:
            if it not in seen:
                seen.add(it)
                unique.append(it)
        alerts = sum(k == "ALERT" for k, _ in unique)
        self.title(("⚠ " if alerts else "") + TITLE)
        self.watchdog.config(state="normal")
        self.watchdog.delete("1.0", "end")
        order = {"ALERT": 0, "WARN": 1, "OK": 2, "INFO": 3}
        for kind, msg in sorted(unique, key=lambda x: order[x[0]]):
            self.watchdog.insert("end", f"{kind:<6}", kind)
            self.watchdog.insert("end", f"{msg}\n")
        self.watchdog.config(state="disabled")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="training run to follow, e.g. outputs/model")
    ap.add_argument("--tab", default="pipeline", choices=("pipeline", "training"))
    ap.add_argument("--no-watch", action="store_true", help="do not open the Flare Watch window")
    ap.add_argument("--snapshot", default=None, help="save a screenshot after the first refresh and exit")
    ap.add_argument("--selftest", default=None, metavar="JSON",
                    help="build the windows hidden, refresh once, write what they show to JSON and exit")
    args = ap.parse_args()
    app = Console(Path(args.run).resolve() if args.run else None, args.tab, not args.no_watch)
    if args.selftest:
        import json

        app.withdraw()
        app.watch.withdraw()
        app.update()
        app._refresh()
        Path(args.selftest).write_text(json.dumps({
            "root": str(ROOT), "outputs": str(OUTPUTS), "footer": app.foot.cget("text"),
            "pipeline": app.tabs["pipeline"].head.cget("text"), "watchdog": app.watchdog.get("1.0", "end").strip(),
            "training": app.tabs["training"].tiles["status"].value.cget("text"),
            "machine": app.system.disks.cget("text"),
            "flare_watch_days": len(app.watch.tab.days)}, indent=1), encoding="utf-8")
        app.destroy()
        return
    if args.snapshot:
        def snap():
            from PIL import ImageGrab
            app.update()
            x, y = app.winfo_rootx(), app.winfo_rooty()
            ImageGrab.grab(bbox=(x, y, x + app.winfo_width(), y + app.winfo_height())).save(args.snapshot)
            app.destroy()
        app.after(7000, snap)
    with contextlib.suppress(KeyboardInterrupt):
        app.mainloop()
