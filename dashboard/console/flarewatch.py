"""Flare Watch tab: the frozen model's alerts replayed one UTC day at a time.

Reads outputs/alerts/watch.npz (minute series written by ``python -m solarflare
alerts``) and outputs/alerts/alert_rules.json (the thresholds, fixed on the
validation period). Three panels share the time axis:

  1. SoLEXS flux in GOES units, with GOES XRS-B for comparison (never an input)
     and the GOES flare list;
  2. calibrated P(>= C1 flare within 15 min), shaded amber where the C alert is on;
  3. the M signal (higher of the network's flux forecast and the flux now),
     shaded red where the M alert is on.
"""

from __future__ import annotations

import tkinter as tk
from datetime import UTC, datetime
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .common import (AMBER, FAINT, GROUND, LINE, MUTED, PANEL, RED, S, SMALL, STEEL, TEAL, TEXT, UI, UI_B,
                     flat_button, muted_legend, read_json, style_axes)

DAY = 86400.0


def goes_class(logf: float) -> str:
    if not np.isfinite(logf):
        return "--"
    for letter, base in (("X", -4), ("M", -5), ("C", -6), ("B", -7), ("A", -8)):
        if logf >= base or letter == "A":
            return f"{letter}{10 ** (logf - base):.1f}"
    return "--"


def episodes(on: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def day_str(t0: float) -> str:
    return datetime.fromtimestamp(t0, UTC).strftime("%Y-%m-%d")


class FlareWatchTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=GROUND)
        self.path = S.alerts / "watch.npz"
        self.rules_path = S.alerts / "alert_rules.json"
        self.W: dict | None = None
        self.rules: dict | None = None
        self._stamp = None
        self.days: list[float] = []
        self.big_days: list[float] = []
        self.day: float | None = None

        bar = tk.Frame(self, bg=GROUND)
        bar.pack(fill="x", pady=(0, 8))
        flat_button(bar, "◀  Day", lambda: self.step(-1)).pack(side="left")
        self.day_var = tk.StringVar()
        self.day_box = ttk.Combobox(bar, textvariable=self.day_var, width=12, state="readonly", style="Run.TCombobox",
                                    font=UI)
        self.day_box.pack(side="left", padx=6)
        self.day_box.bind("<<ComboboxSelected>>", lambda e: self.show(self._parse(self.day_var.get())))
        flat_button(bar, "Day  ▶", lambda: self.step(1)).pack(side="left")
        flat_button(bar, "Previous M/X flare", lambda: self.jump_big(-1), AMBER).pack(side="left", padx=(14, 6))
        flat_button(bar, "Next M/X flare", lambda: self.jump_big(1), AMBER).pack(side="left")
        flat_button(bar, "Latest data", lambda: self.show(self.days[-1] if self.days else None), MUTED).pack(
            side="left", padx=(14, 0))
        self.period = tk.Label(bar, text="", bg=GROUND, fg=MUTED, font=UI)
        self.period.pack(side="right")

        self.summary = tk.Label(self, text="", bg=GROUND, fg=TEXT, font=UI_B, anchor="w", justify="left")
        self.summary.pack(fill="x", pady=(0, 6))

        frame = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        frame.pack(fill="both", expand=True)
        self.fig = Figure(figsize=(9, 6.2), dpi=100, facecolor=PANEL)
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 1], hspace=0.14)
        self.ax = [self.fig.add_subplot(gs[0])]
        self.ax += [self.fig.add_subplot(gs[i], sharex=self.ax[0]) for i in (1, 2)]
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().configure(bg=PANEL, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.mpl_connect("motion_notify_event", self._hover)

        self.readout = tk.Label(self, text="Move the pointer over the plots to read the values at a minute.",
                                bg=GROUND, fg=MUTED, font=("Consolas", 9), anchor="w")
        self.readout.pack(fill="x", pady=(6, 0))
        self.note = tk.Label(self, text="", bg=GROUND, fg=FAINT, font=SMALL, anchor="w", justify="left",
                             wraplength=900)
        self.note.pack(fill="x", pady=(2, 0))

    # ---- data ------------------------------------------------------------------------
    def refresh(self) -> None:
        """Called every refresh tick: reload only when the files changed."""
        try:
            stamp = (self.path.stat().st_mtime, self.rules_path.stat().st_mtime)
        except OSError:
            stamp = None
        if stamp == self._stamp:
            return
        self._stamp = stamp
        if stamp is None:
            self.W = None
            self._empty()
            return
        z = np.load(self.path)
        self.W = {k: z[k] for k in z.files}
        self.rules = read_json(self.rules_path)
        t = self.W["t"]
        have = np.isfinite(self.W["flux"]) | np.isfinite(self.W["p_c"])
        self.days = sorted({float(np.floor(x / DAY) * DAY) for x in t[have]})
        cls = self.W["flare_class"].astype(str)
        big = np.array([c[:1] in "MX" for c in cls], bool)
        self.big_days = sorted({float(np.floor(x / DAY) * DAY) for x in self.W["flare_peak"][big]} & set(self.days))
        self.day_box["values"] = [day_str(d) for d in reversed(self.days)]
        start = self.day if self.day in self.days else (self.big_days[-1] if self.big_days else
                                                          (self.days[-1] if self.days else None))
        self.show(start)

    def _parse(self, s: str) -> float | None:
        try:
            return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC).timestamp()
        except ValueError:
            return None

    def step(self, k: int) -> None:
        if not self.days or self.day is None:
            return
        i = min(max(self.days.index(self.day) + k, 0), len(self.days) - 1) if self.day in self.days else 0
        self.show(self.days[i])

    def jump_big(self, k: int) -> None:
        if not self.big_days:
            return
        cur = self.day or 0.0
        cand = [d for d in self.big_days if (d > cur if k > 0 else d < cur)]
        if cand:
            self.show(cand[0] if k > 0 else cand[-1])

    # ---- drawing -----------------------------------------------------------------------
    def _empty(self) -> None:
        for a in self.ax:
            a.clear()
            style_axes(a)
        self.ax[0].text(0.5, 0.5, "No replay yet.\nRun the pipeline (stage 'alerts') to create outputs/alerts/watch.npz.",
                        transform=self.ax[0].transAxes, ha="center", va="center", color=MUTED, fontsize=10)
        self.summary.config(text="")
        self.period.config(text="")
        self.canvas.draw_idle()

    def show(self, d0: float | None) -> None:
        if self.W is None or d0 is None:
            self._empty()
            return
        self.day = d0
        self.day_var.set(day_str(d0))
        W, rules = self.W, self.rules or {}
        t = W["t"]
        sel = (t >= d0) & (t < d0 + DAY)
        h = (t[sel] - d0) / 3600.0
        flux, goes = W["flux"][sel], W["goes"][sel]
        pc, mc, mn = W["p_c"][sel], W["m_combined"][sel], W["m_network"][sel]
        thr_c = rules.get("C", {}).get("threshold")
        thr_m = rules.get("M", {}).get("threshold")
        on_c = np.nan_to_num(pc, nan=-1) >= thr_c if thr_c is not None else np.zeros(h.size, bool)
        on_m = np.nan_to_num(mc, nan=-99) >= thr_m if thr_m is not None else np.zeros(h.size, bool)
        self._day = {"t": t[sel], "h": h, "flux": flux, "goes": goes, "pc": pc, "mc": mc}

        test = d0 >= float(W["test_start"])
        self.period.config(text=("TEST PERIOD  ·  thresholds were fixed before these days were scored" if test else
                                 "VALIDATION PERIOD  ·  the thresholds were chosen on these days"),
                           fg=TEAL if test else MUTED)

        fp, fc = W["flare_peak"], W["flare_class"].astype(str)
        fin = (fp >= d0) & (fp < d0 + DAY)
        counts = {k: int(sum(c[:1] == k for c in fc[fin])) for k in "CMX"}
        sc, ec = episodes(on_c)
        sm, em = episodes(on_m)
        flares = ", ".join(f"{n} {k}" for k, n in counts.items() if n) or "none"
        self.summary.config(text=f"{day_str(d0)}   ·   GOES flares: {flares}   ·   C alert raised {sc.size}× "
                                 f"({int(on_c.sum())} min)   ·   M alert raised {sm.size}× ({int(on_m.sum())} min)")

        a1, a2, a3 = self.ax
        for a in self.ax:
            a.clear()
            style_axes(a)
        # 1. flux
        a1.plot(h, goes, color=MUTED, lw=0.9, ls="--", label="GOES XRS-B (truth, not an input)")
        a1.plot(h, flux, color=TEAL, lw=1.5, label="SoLEXS, in GOES units")
        lo = np.nanmin(np.concatenate([flux, goes])) if np.isfinite(np.concatenate([flux, goes])).any() else -7
        hi = np.nanmax(np.concatenate([flux, goes])) if np.isfinite(np.concatenate([flux, goes])).any() else -5
        a1.set_ylim(min(lo - 0.2, -6.6), max(hi + 0.45, -5.3))
        for y, lab in ((-6, "C"), (-5, "M"), (-4, "X")):
            if a1.get_ylim()[0] < y < a1.get_ylim()[1]:
                a1.axhline(y, color=LINE, lw=0.9)
                a1.text(24.15, y, lab, color=MUTED, fontsize=8, va="center")
        for p, c in zip(fp[fin], fc[fin]):
            x = (p - d0) / 3600.0
            y = a1.get_ylim()[1] - 0.12
            col = RED if c[:1] in "MX" else TEXT
            a1.plot([x], [y], "v", color=col, ms=5)
            a1.text(x, y + 0.02, c, color=col, fontsize=7, ha="center", va="bottom")
        a1.set_ylabel("log10 flux, W/m²", color=MUTED, fontsize=8)
        muted_legend(a1, loc="upper left", ncol=2)
        # 2. C probability
        a2.plot(h, pc, color=STEEL, lw=1.3, label="P(≥ C1 flare within 15 min), calibrated")
        if thr_c is not None:
            a2.axhline(thr_c, color=AMBER, lw=0.9, ls="--", label=f"C alert threshold {thr_c:.2f}")
            a2.fill_between(h, 0, 1, where=on_c, color=AMBER, alpha=0.3, lw=0, transform=a2.get_xaxis_transform())
        a2.set_ylim(0, 1.25)
        a2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        a2.set_ylabel("probability", color=MUTED, fontsize=8)
        muted_legend(a2, loc="upper left", ncol=2)
        # 3. M signal
        a3.plot(h, mn, color=FAINT, lw=0.9, label="network: highest median forecast, +5/15/30 min")
        a3.plot(h, mc, color=STEEL, lw=1.3, label="M signal: higher of network and flux now")
        if thr_m is not None:
            a3.axhline(thr_m, color=RED, lw=0.9, ls="--", label=f"M alert threshold ({goes_class(thr_m)})")
            a3.fill_between(h, 0, 1, where=on_m, color=RED, alpha=0.34, lw=0, transform=a3.get_xaxis_transform())
        fin_m = np.isfinite(mc)
        top3 = max(np.nanmax(mc[fin_m]) if fin_m.any() else -5.5, thr_m if thr_m is not None else -5.0)
        a3.set_ylim(min(np.nanmin(mc[fin_m]) - 0.2 if fin_m.any() else -7, -6.2), top3 + 0.75)
        a3.set_ylabel("log10 flux, W/m²", color=MUTED, fontsize=8)
        muted_legend(a3, loc="upper left", ncol=3)
        a3.set_xlim(0, 24)
        a3.set_xticks(range(0, 25, 3))
        a3.set_xticklabels([f"{x:02d}:00" for x in range(0, 25, 3)])
        a3.set_xlabel("UTC", color=MUTED, fontsize=8)
        for a in (a1, a2):
            a.tick_params(labelbottom=False)
        self.fig.subplots_adjust(left=0.075, right=0.965, top=0.98, bottom=0.07)
        self.canvas.draw_idle()

        r = rules
        if r:
            self.note.config(text=(
                f"Alert rules (outputs/alerts/alert_rules.json, model {r.get('model_sha256', '?')}): "
                f"C when P ≥ {r['C']['threshold']} ({r['C'].get('false_alarms_per_day_validation')} false alarms/day on "
                f"validation); M when the M signal ≥ {goes_class(r['M']['threshold'])} "
                f"({r['M'].get('false_alarms_per_day_validation')}/day). Each value is drawn at the minute it became "
                "known; nothing from later minutes is used."))

    def _hover(self, ev) -> None:
        if ev.inaxes is None or not getattr(self, "_day", None) or ev.xdata is None:
            return
        d = self._day
        if d["h"].size == 0:
            return
        i = int(np.clip(np.searchsorted(d["h"], ev.xdata), 0, d["h"].size - 1))
        p = d["pc"][i]
        self.readout.config(
            text=f"{datetime.fromtimestamp(float(d['t'][i]), UTC):%H:%M} UTC   SoLEXS {goes_class(d['flux'][i]):>6}   "
                 f"GOES {goes_class(d['goes'][i]):>6}   P(≥C1, 15 min) {'--' if not np.isfinite(p) else f'{p:.2f}'}   "
                 f"M signal {goes_class(d['mc'][i])}", fg=TEXT)

