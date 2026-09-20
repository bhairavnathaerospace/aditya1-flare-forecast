"""Pipeline tab: every stage of ``python -m solarflare pipeline`` and where it stands.

Reads outputs/pipeline/state.json, written by solarflare/runall.py as stages
start and finish; the stage list itself comes from runall, so the two cannot
disagree.
"""

from __future__ import annotations

import time
import tkinter as tk
from datetime import UTC, datetime

from .common import (AMBER, FAINT, GREEN, GROUND, LINE, MUTED, PANEL, RED, S, SMALL, TEAL, TEXT, UI, UI_B,
                     fmt_dur, read_json)

CHIP = {"done": ("DONE", GREEN), "running": ("RUNNING", TEAL), "failed": ("FAILED", RED),
        "stale": ("REDO", AMBER), "interrupted": ("STOPPED", AMBER), "-": ("TO DO", FAINT)}


def _age(stamp: str | None) -> float | None:
    if not stamp:
        return None
    try:
        return time.time() - datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp()
    except ValueError:
        return None


class PipelineTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=GROUND)
        from solarflare import runall

        self.runall = runall
        self.head = tk.Label(self, text="", bg=GROUND, fg=TEXT, font=UI_B, anchor="w", justify="left")
        self.head.pack(fill="x", pady=(0, 4))
        self.sub = tk.Label(self, text="", bg=GROUND, fg=MUTED, font=UI, anchor="w", justify="left", wraplength=900)
        self.sub.pack(fill="x", pady=(0, 8))
        box = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        box.pack(fill="both", expand=True)
        self.table = tk.Frame(box, bg=PANEL)
        self.table.pack(fill="both", expand=True, padx=12, pady=10)
        for c, w in enumerate((0, 0, 1, 0)):
            self.table.grid_columnconfigure(c, weight=w)
        self.rows: dict[str, tuple[tk.Label, tk.Label, tk.Label, tk.Label]] = {}
        self._names: list[str] = []
        self.foot = tk.Label(self, text="", bg=GROUND, fg=FAINT, font=SMALL, anchor="w", justify="left", wraplength=900)
        self.foot.pack(fill="x", pady=(6, 0))

    def _build_rows(self, stages) -> None:
        for w in self.table.winfo_children():
            w.destroy()
        self.rows.clear()
        for c, text in enumerate(("STATE", "STAGE", "WHAT IT DOES", "TIME")):
            tk.Label(self.table, text=text, bg=PANEL, fg=MUTED, font=SMALL, anchor="w").grid(
                row=0, column=c, sticky="w", padx=(0, 14), pady=(0, 4))
        for i, st in enumerate(stages, start=1):
            cells = (tk.Label(self.table, text="", bg=PANEL, font=("Consolas", 8, "bold"), anchor="w", width=8),
                     tk.Label(self.table, text=st.name, bg=PANEL, fg=TEXT, font=("Consolas", 9), anchor="w"),
                     tk.Label(self.table, text=st.title, bg=PANEL, fg=MUTED, font=UI, anchor="w"),
                     tk.Label(self.table, text="", bg=PANEL, fg=MUTED, font=("Consolas", 9), anchor="e"))
            for c, lab in enumerate(cells):
                lab.grid(row=i, column=c, sticky="we" if c == 2 else "w", padx=(0, 14), pady=1)
            self.rows[st.name] = cells
        self._names = [st.name for st in stages]

    def refresh(self) -> dict:
        """Returns the state file (for the watchdog)."""
        stages = self.runall.stages(S)
        if [st.name for st in stages] != self._names:
            self._build_rows(stages)
        try:
            state = self.runall.State(S.outputs / "pipeline" / "state.json")
        except ValueError:                       # caught mid-write: next tick
            return {}
        done = running = 0
        failed = []
        for st in stages:
            d = state.get(st.name)
            status = "done" if self.runall.is_done(st, state) else d.get("status", "-")
            chip, col = CHIP.get(status, CHIP["-"])
            state_cell, name_cell, _, time_cell = self.rows[st.name]
            state_cell.config(text=chip, fg=col)
            name_cell.config(fg=TEXT if status != "-" else MUTED)
            if status == "running":
                running += 1
                time_cell.config(text=f"{fmt_dur(_age(d.get('started')))} so far", fg=TEAL)
            elif d.get("seconds"):
                time_cell.config(text=fmt_dur(d["seconds"]), fg=MUTED)
            else:
                time_cell.config(text="", fg=MUTED)
            done += status == "done"
            if status == "failed":
                failed.append(f"{st.name} ({d.get('error', 'see its log')})")
        mode = self.runall.use_sharp_mode(S)
        self.head.config(text=f"{done} of {len(stages)} stages done" + (f"   ·   {running} running" if running else "")
                         + (f"   ·   {len(failed)} failed" if failed else ""), fg=RED if failed else TEXT)
        dec = read_json(S.ablations / "sharp" / "decision.json")
        sharp = (f"SHARP as a network input: {'kept' if dec['use_sharp'] else 'left out'} "
                 f"(validation {dec['xray_only']['val_score']} without, {dec['with_sharp']['val_score']} with)"
                 if dec else {"auto": "SHARP as a network input: decided on validation after both trainings",
                              "on": "SHARP as a network input: on (project.toml)",
                              "off": "SHARP as a network input: off (project.toml, or no SHARP files)"}[mode])
        self.sub.config(text=f"{sharp}.   Seeds for the HEL1OS ablation: {S.pipeline.get('seeds')}.")
        self.foot.config(text=("Failed: " + "; ".join(failed) + ".  " if failed else "")
                         + "Stage logs: outputs/pipeline/logs/.  'Run full pipeline' resumes from the first stage "
                           "not done; 'Redo stage…' reruns one stage and everything built on it.")
        return state.d
