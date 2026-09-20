"""Machine and throughput panel: what the laptop is doing this second.

Six traces, each the last three minutes: GPU use, GPU memory, GPU temperature,
CPU, memory, and training throughput (batches a second, from the batch counter
in live.json). Disk space on the data and output drives sits underneath.
"""

from __future__ import annotations

import shutil
import tkinter as tk

from .common import (AMBER, FAINT, LINE, MUTED, OUTPUTS, PANEL, RED, S, SMALL, STEEL, TEAL, UI,
                     CpuLoad, Sparkline, memory)

SPAN = 180                       # seconds of history at one sample a second


class SystemPanel(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        tk.Label(self, text="MACHINE  ·  last 3 minutes", bg=PANEL, fg=MUTED, font=SMALL).pack(
            anchor="w", padx=12, pady=(9, 2))
        grid = tk.Frame(self, bg=PANEL)
        grid.pack(fill="both", expand=True, padx=12, pady=(0, 6))
        for c in (0, 1):
            grid.grid_columnconfigure(c, weight=1, uniform="s")
        self.sparks = {
            "gpu": Sparkline(grid, "GPU", "%", SPAN, 0, 100, TEAL),
            "gpu_mem": Sparkline(grid, "GPU mem", "%", SPAN, 0, 100, STEEL),
            "gpu_temp": Sparkline(grid, "GPU temp", "°C", SPAN, 30, 95, AMBER),
            "cpu": Sparkline(grid, "CPU", "%", SPAN, 0, 100, STEEL),
            "ram": Sparkline(grid, "Memory", "%", SPAN, 0, 100, TEAL),
            "rate": Sparkline(grid, "Batches/s", "", SPAN, 0, None, AMBER),
        }
        for i, sp in enumerate(self.sparks.values()):
            sp.grid(row=i // 2, column=i % 2, sticky="nsew", padx=(0 if i % 2 == 0 else 12, 0), pady=(2, 6))
        self.disks = tk.Label(self, text="", bg=PANEL, fg=MUTED, font=UI, anchor="w")
        self.disks.pack(fill="x", padx=12, pady=(0, 9))
        self.cpu = CpuLoad()
        self.cpu.read()
        self._batch = None
        self._disk_every = 0

    def refresh(self, gpu: dict | None, live: dict | None, now: float) -> None:
        if gpu:
            self.sparks["gpu"].push(gpu["util"])
            self.sparks["gpu_mem"].push(100.0 * gpu["mem"] / max(gpu["mem_total"], 1),
                                        f"{gpu['mem'] / 1024:.1f}/{gpu['mem_total'] / 1024:.0f} GB")
            self.sparks["gpu_temp"].push(gpu["temp"], f"{gpu['temp']:.0f} °C"
                                         + (f" · {gpu['power']:.0f} W" if gpu.get("power") else ""))
        else:
            for k in ("gpu", "gpu_mem", "gpu_temp"):
                self.sparks[k].push(None)
        self.sparks["cpu"].push(self.cpu.read())
        m = memory()
        self.sparks["ram"].push(100.0 * m[0] / m[1] if m else None, f"{m[0]:.1f}/{m[1]:.0f} GB" if m else "--")

        # batches a second, from the counter training writes about every 2 s
        rate = None
        if live and live.get("status") == "training":
            b, t = live.get("batch"), live.get("updated_unix")
            if b is not None and t is not None:
                if self._batch and t > self._batch[1] and b >= self._batch[0]:
                    rate = (b - self._batch[0]) / (t - self._batch[1])
                if not self._batch or t > self._batch[1]:
                    self._batch = (b, t)
        else:
            self._batch = None
        self.sparks["rate"].push(rate, f"{rate:.2f}/s" if rate else ("--" if not live else "idle"))

        self._disk_every -= 1
        if self._disk_every <= 0:                      # disks change slowly: every 30 s
            self._disk_every = 30
            parts = []
            for label, path in (("data", S.data_root), ("outputs", OUTPUTS)):
                try:
                    p = path if path.exists() else path.parent
                    free = shutil.disk_usage(p).free / 1e9
                    col = RED if free < S.min_free_gb else (AMBER if free < S.min_free_gb + 15 else MUTED)
                    parts.append((f"{label} {free:.0f} GB free", col))
                except OSError:
                    parts.append((f"{label} unreachable", RED))
            self.disks.config(text="   ·   ".join(t for t, _ in parts),
                              fg=RED if any(c == RED for _, c in parts) else
                              (AMBER if any(c == AMBER for _, c in parts) else MUTED))
        if not gpu:
            self.disks.config(fg=FAINT if self.disks.cget("text") == "" else self.disks.cget("fg"))


def epoch_seconds(live: dict | None, hist: list | None) -> float | None:
    """How long an epoch takes: the last recorded one, else the average so far.

    ``live["last_epoch_s"]`` was the elapsed time of the whole run in older runs,
    so it is only trusted when it is clearly shorter than that."""
    for h in reversed(hist or []):
        if h.get("seconds"):
            return float(h["seconds"])
    if not live:
        return None
    done = int(live.get("epochs_done", 0))
    elapsed = float(live.get("elapsed_s", 0))
    last = float(live.get("last_epoch_s") or 0)
    if last and (done <= 1 or last < 0.9 * elapsed):
        return last
    return elapsed / done if done else None


def eta_hours(live: dict | None, hist: list | None, patience: int) -> tuple[float | None, float | None]:
    """Hours left (to the last epoch, and to an early stop at the current best)."""
    if not live or live.get("status") not in ("training", "validating"):
        return None, None
    done, total = int(live.get("epochs_done", 0)), int(live.get("epochs_total", 0))
    per = epoch_seconds(live, hist)
    if not per:
        return None, None
    frac = int(live.get("batch", 0)) / max(int(live.get("batches", 1)), 1)
    best = int(live.get("best_epoch", -1))
    left_all = per * max(total - done - frac, 0) / 3600.0
    left_stop = per * max(best + patience - done - frac, 0) / 3600.0 if best >= 0 else left_all
    return left_all, min(left_stop, left_all)


