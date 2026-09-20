"""Flare Watch screen: the frozen model's alerts replayed one UTC day at a time.

Reads outputs/alerts/watch.npz (minute series written by ``python -m solarflare
alerts``) and outputs/alerts/alert_rules.json (the thresholds, fixed on the
validation period). Four panels share the time axis:

  1. SoLEXS flux in GOES units, with GOES XRS-B for comparison (never an input)
     and the GOES flare list;
  2. calibrated P(>= C1 flare within 15 min), shaded amber where the C alert is on;
  3. the M signal (higher of the network's flux forecast and the flux now),
     shaded red where the M alert is on;
  4. how much of each minute's window HEL1OS was watching.

Beside them: the alert state at the last minute of the replay, every flare of
the day with the lead its alerts gave, and the lead times over the whole replay.
"""

from __future__ import annotations

import tkinter as tk
from datetime import UTC, datetime
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .common import (AMBER, FAINT, GREEN, GROUND, LINE, MUTED, PANEL, RED, S, SMALL, STEEL, TEAL, TEXT, UI,
                     UI_B, flat_button, muted_legend, read_json, style_axes)

DAY = 86400.0
PRE_MIN = 30                 # an alert counts for a flare from this long before its GOES start


def goes_class(logf) -> str:
    if logf is None or not np.isfinite(logf):
        return "--"
    for letter, base in (("X", -4), ("M", -5), ("C", -6), ("B", -7), ("A", -8)):
        if logf >= base or letter == "A":
            return f"{letter}{10 ** (logf - base):.1f}"
    return "--"


def episodes(on: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def shade(ax, h, on, color, alpha):
    """Mark the minutes an alert is on. Short episodes get a minimum width, or a
    two-minute alert would be an invisible hairline on a 24 h axis."""
    a, b = episodes(on)
    for i, j in zip(a, b):
        x0 = h[i]
        x1 = h[min(j, h.size - 1)]
        ax.axvspan(x0, max(x1, x0 + 0.08), color=color, alpha=alpha, lw=0)


def day_str(t0: float) -> str:
    return datetime.fromtimestamp(t0, UTC).strftime("%Y-%m-%d")


def hhmm(t: float) -> str:
    return datetime.fromtimestamp(float(t), UTC).strftime("%H:%M")


def next_on(on: np.ndarray) -> np.ndarray:
    """For each minute, the index of the next minute the alert is on (n + 1 if never)."""
    n = on.size
    idx = np.where(on, np.arange(n), n + 1)
    return np.minimum.accumulate(idx[::-1])[::-1]


class FlareWatchTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=GROUND)
        self.path = S.alerts / "watch.npz"
        self.rules_path = S.alerts / "alert_rules.json"
        self.W: dict | None = None
        self.rules: dict | None = None
        self._stamp = "unset"          # distinct from None, which means "no files yet"
        self.days: list[float] = []
        self.big_days: list[float] = []
        self.day: float | None = None
        self.leads: dict | None = None

        bar = tk.Frame(self, bg=GROUND)
        bar.pack(fill="x", pady=(0, 8))
        flat_button(bar, "◀  Day", lambda: self.step(-1)).pack(side="left")
        self.day_var = tk.StringVar()
        self.day_box = ttk.Combobox(bar, textvariable=self.day_var, width=12, state="readonly",
                                    style="Run.TCombobox", font=UI)
        self.day_box.pack(side="left", padx=6)
        self.day_box.bind("<<ComboboxSelected>>", lambda e: self.show(self._parse(self.day_var.get())))
        flat_button(bar, "Day  ▶", lambda: self.step(1)).pack(side="left")
        flat_button(bar, "◀ M/X flare", lambda: self.jump_big(-1), AMBER).pack(side="left", padx=(14, 6))
        flat_button(bar, "M/X flare ▶", lambda: self.jump_big(1), AMBER).pack(side="left")
        flat_button(bar, "Latest data", lambda: self.show(self.days[-1] if self.days else None), MUTED).pack(
            side="left", padx=(14, 0))
        self.period = tk.Label(bar, text="", bg=GROUND, fg=MUTED, font=UI)
        self.period.pack(side="right")

        self.summary = tk.Label(self, text="", bg=GROUND, fg=TEXT, font=UI_B, anchor="w", justify="left")
        self.summary.pack(fill="x", pady=(0, 6))

        body = tk.Frame(self, bg=GROUND)
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=7, uniform="w")
        body.grid_columnconfigure(1, weight=3, uniform="w")
        body.grid_rowconfigure(0, weight=1)

        frame = tk.Frame(body, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        frame.grid(row=0, column=0, sticky="nsew")
        self.fig = Figure(figsize=(9, 7), dpi=100, facecolor=PANEL)
        gs = self.fig.add_gridspec(4, 1, height_ratios=[1.6, 1, 1, 0.45], hspace=0.12)
        self.ax = [self.fig.add_subplot(gs[0])]
        self.ax += [self.fig.add_subplot(gs[i], sharex=self.ax[0]) for i in (1, 2, 3)]
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().configure(bg=PANEL, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.mpl_connect("motion_notify_event", self._hover)

        side = tk.Frame(body, bg=GROUND)
        side.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(1, weight=3)
        side.grid_rowconfigure(2, weight=2)

        status = tk.Frame(side, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        status.grid(row=0, column=0, sticky="nsew")
        tk.Label(status, text="ALERT STATE", bg=PANEL, fg=MUTED, font=SMALL).pack(
            anchor="w", padx=12, pady=(9, 4))
        self.chips = {}
        for key, label in (("C", "≥ C1 flare within 15 min"), ("M", "flux reaches M1 within 30 min")):
            row = tk.Frame(status, bg=PANEL)
            row.pack(fill="x", padx=12, pady=(0, 6))
            chip = tk.Label(row, text="--", bg=PANEL, fg=FAINT, font=("Consolas", 10, "bold"), width=5)
            chip.pack(side="left")
            tk.Label(row, text=label, bg=PANEL, fg=MUTED, font=UI).pack(side="left", padx=(8, 0))
            self.chips[key] = chip
        self.asof = tk.Label(status, text="", bg=PANEL, fg=FAINT, font=SMALL, anchor="w", justify="left",
                             wraplength=200)
        self.asof.pack(fill="x", padx=12, pady=(0, 9))

        table = tk.Frame(side, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        table.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        tk.Label(table, text="FLARES THIS DAY  ·  lead before the peak", bg=PANEL, fg=MUTED,
                 font=SMALL).pack(anchor="w", padx=12, pady=(9, 2))
        self.table = tk.Text(table, bg=PANEL, fg=TEXT, font=("Consolas", 8), relief="flat",
                             highlightthickness=0, wrap="none", height=9)
        self.table.pack(fill="both", expand=True, padx=12, pady=(2, 10))
        self.table.tag_configure("head", foreground=MUTED)
        self.table.tag_configure("miss", foreground=RED)
        self.table.tag_configure("hit", foreground=TEXT)

        hist = tk.Frame(side, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        hist.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        self.fig_h = Figure(figsize=(3.4, 2.1), dpi=100, facecolor=PANEL)
        self.ax_h = self.fig_h.add_subplot(111)
        self.cv_h = FigureCanvasTkAgg(self.fig_h, master=hist)
        self.cv_h.get_tk_widget().configure(bg=PANEL, highlightthickness=0)
        self.cv_h.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self._blank_histogram()

        self.readout = tk.Label(self, text="Move the pointer over the plots to read the values at a minute.",
                                bg=GROUND, fg=MUTED, font=("Consolas", 9), anchor="w")
        self.readout.pack(fill="x", pady=(6, 0))
        self.note = tk.Label(self, text="", bg=GROUND, fg=FAINT, font=SMALL, anchor="w", justify="left",
                             wraplength=1100)
        self.note.pack(fill="x", pady=(2, 0))
        self._empty()                  # until the first refresh finds a replay

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
        self._flare_leads()
        self._status()
        self._histogram()
        start = self.day if self.day in self.days else (self.big_days[-1] if self.big_days else
                                                        (self.days[-1] if self.days else None))
        self.show(start)

    def _on(self, key: str) -> np.ndarray:
        """The alert series over the whole replay, from the saved rule."""
        W, r = self.W, (self.rules or {}).get(key) or {}
        sig = W["p_c"] if key == "C" else W["m_combined"]
        thr = r.get("threshold")
        return (np.nan_to_num(sig, nan=-99.0) >= thr) if thr is not None else np.zeros(sig.size, bool)

    def _flare_leads(self) -> None:
        """For every flare: the lead each alert gave, counting alerts from 30 min
        before the GOES start to the GOES peak (the alert study's window)."""
        W = self.W
        t = W["t"]
        nxt = {k: next_on(self._on(k)) for k in ("C", "M")}

        def idx(x):
            return np.clip(np.rint((np.asarray(x, float) - t[0]) / 60.0).astype(np.int64), 0, t.size - 1)

        k0, k1 = idx(W["flare_start"] - 60.0 * PRE_MIN), idx(W["flare_peak"])
        leads = {}
        for key in ("C", "M"):
            first = nxt[key][k0]
            lead = np.where(first <= k1, (k1 - first).astype(float), np.nan)
            lead[(W["flare_peak"] < t[0]) | (W["flare_peak"] > t[-1])] = np.nan
            leads[key] = lead
        self.leads = {**leads, "peak": W["flare_peak"], "class": W["flare_class"].astype(str),
                      "test": W["flare_peak"] >= float(W["test_start"])}

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

    # ---- side panels -------------------------------------------------------------------
    def _status(self) -> None:
        W = self.W
        t = W["t"]
        have = np.flatnonzero(np.isfinite(W["flux"]) | np.isfinite(W["p_c"]))
        if not have.size:
            return
        i = int(have[-1])
        for key in ("C", "M"):
            on = bool(self._on(key)[i])
            self.chips[key].config(text="ON" if on else "OFF", fg=(AMBER if key == "C" else RED) if on else GREEN)
        p = W["p_c"][i]
        self.asof.config(text=f"Last minute in the replay: {day_str(t[i])} {hhmm(t[i])} UTC, SoLEXS "
                              f"{goes_class(W['flux'][i])}, P(>= C1) "
                              f"{'--' if not np.isfinite(p) else format(float(p), '.2f')}. Not a live feed.")

    def _histogram(self) -> None:
        ax = self.ax_h
        ax.clear()
        style_axes(ax, "Lead before the peak, test period")
        L = self.leads
        bits = []
        if L is not None:
            big = np.array([c[:1] in "MX" for c in L["class"]])
            for key, col, keep in (("C", AMBER, L["test"]), ("M", RED, L["test"] & big)):
                v = L[key][keep & np.isfinite(L[key])]
                if v.size:
                    ax.hist(np.clip(v, 0, 45), bins=np.arange(0, 47, 3), histtype="step", lw=1.4, color=col)
                    bits.append(f"{key} median {np.median(v):.0f} min (n={v.size})")
        style_axes(ax, "Lead before the peak")
        for i, b in enumerate(bits):
            ax.text(0.97, 0.92 - 0.12 * i, b, transform=ax.transAxes, ha="right", va="top", fontsize=7,
                    color=AMBER if b.startswith("C") else RED)
        ax.set_xlabel("minutes before the GOES peak", color=MUTED, fontsize=8)
        ax.set_ylabel("flares", color=MUTED, fontsize=8)
        self.fig_h.tight_layout()
        self.cv_h.draw_idle()

    def _table(self, d0: float) -> None:
        L = self.leads
        self.table.config(state="normal")
        self.table.delete("1.0", "end")
        if L is None:
            self.table.config(state="disabled")
            return
        sel = np.flatnonzero((L["peak"] >= d0) & (L["peak"] < d0 + DAY))
        self.table.insert("end", f"{'class':<7}{'peak UTC':<10}{'C lead':>8}{'M lead':>8}\n", "head")
        if not sel.size:
            self.table.insert("end", "no GOES flare this day\n", "head")
        for i in sel:
            c, m = L["C"][i], L["M"][i]
            big = L["class"][i][:1] in "MX"
            c_txt = "missed" if not np.isfinite(c) else f"{c:.0f} min"
            m_txt = "" if not big else ("missed" if not np.isfinite(m) else f"{m:.0f} min")
            self.table.insert("end", f"{L['class'][i]:<7}{hhmm(L['peak'][i]):<10}{c_txt:>8}{m_txt:>8}\n",
                              "miss" if not np.isfinite(c) else "hit")
        self.table.config(state="disabled")

    # ---- drawing -----------------------------------------------------------------------
    def _blank_histogram(self, message: str = "after the alerts stage") -> None:
        """An axes nobody has styled draws white: blank it explicitly."""
        ax = self.ax_h
        ax.clear()
        ax.set_facecolor(PANEL)
        ax.set_axis_off()
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color=FAINT, fontsize=8)
        self.fig_h.tight_layout()
        self.cv_h.draw_idle()

    def _empty(self) -> None:
        """Nothing to replay yet: one plain message, not four empty grids."""
        for a in self.ax:
            a.clear()
            a.set_facecolor(PANEL)
            a.set_axis_off()
        self.ax[0].text(0.5, 0.0, "No replay yet.\n\nThe alerts stage of the pipeline writes "
                                  "outputs/alerts/watch.npz;\nthis screen fills in as soon as it does.",
                        transform=self.ax[0].transAxes, ha="center", va="center", color=MUTED, fontsize=10)
        self._blank_histogram()
        self.table.config(state="normal")
        self.table.delete("1.0", "end")
        self.table.insert("end", "no replay yet\n", "head")
        self.table.config(state="disabled")
        for chip in self.chips.values():
            chip.config(text="--", fg=FAINT)
        self.asof.config(text="Fills once the frozen model has run over the validation and test periods.")
        self.summary.config(text="")
        self.period.config(text="")
        self.note.config(text="")
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
        hard = W["hard_frac"][sel]
        thr_c = rules.get("C", {}).get("threshold")
        thr_m = rules.get("M", {}).get("threshold")
        on_c, on_m = self._on("C")[sel], self._on("M")[sel]
        self._day = {"t": t[sel], "h": h, "flux": flux, "goes": goes, "pc": pc, "mc": mc, "hard": hard}

        test = d0 >= float(W["test_start"])
        self.period.config(text="TEST PERIOD" if test else "VALIDATION PERIOD", fg=TEAL if test else MUTED)

        fp, fc = W["flare_peak"], W["flare_class"].astype(str)
        fin = (fp >= d0) & (fp < d0 + DAY)
        counts = {k: int(sum(c[:1] == k for c in fc[fin])) for k in "CMX"}
        sc, _ = episodes(on_c)
        sm, _ = episodes(on_m)
        flares = ", ".join(f"{n} {k}" for k, n in counts.items() if n) or "none"
        cov = float(np.mean(np.isfinite(flux))) if flux.size else 0.0
        self.summary.config(text=f"{day_str(d0)}   ·   GOES flares: {flares}   ·   C alert raised {sc.size}x "
                                 f"({int(on_c.sum())} min)   ·   M alert raised {sm.size}x ({int(on_m.sum())} min)"
                                 f"   ·   SoLEXS covered {100 * cov:.0f}% of the day")

        a1, a2, a3, a4 = self.ax
        for a in self.ax:
            a.clear()
            a.set_axis_on()
            style_axes(a)
        # 1. flux
        a1.plot(h, goes, color=MUTED, lw=0.9, ls="--", label="GOES XRS-B (truth, not an input)")
        a1.plot(h, flux, color=TEAL, lw=1.5, label="SoLEXS, in GOES units")
        both = np.concatenate([flux, goes])
        lo = np.nanmin(both) if np.isfinite(both).any() else -7
        hi = np.nanmax(both) if np.isfinite(both).any() else -5
        a1.set_ylim(min(lo - 0.2, -6.6), max(hi + 0.8, -5.0))
        for y, lab in ((-6, "C"), (-5, "M"), (-4, "X")):
            if a1.get_ylim()[0] < y < a1.get_ylim()[1]:
                a1.axhline(y, color=LINE, lw=0.9)
                a1.text(24.15, y, lab, color=MUTED, fontsize=8, va="center")
        for p, c in zip(fp[fin], fc[fin]):
            x = (p - d0) / 3600.0
            y = a1.get_ylim()[0] + 0.07
            col = RED if c[:1] in "MX" else TEXT
            a1.plot([x], [y], "^", color=col, ms=5)
            a1.text(x, y + 0.06, c, color=col, fontsize=7, ha="center", va="bottom")
        a1.set_ylabel("log10 flux, W/m2", color=MUTED, fontsize=8)
        muted_legend(a1, loc="upper left", ncol=2)
        # 2. C probability
        a2.plot(h, pc, color=STEEL, lw=1.3, label="P(>= C1 flare within 15 min), calibrated")
        if thr_c is not None:
            a2.axhline(thr_c, color=AMBER, lw=0.9, ls="--", label=f"C alert threshold {thr_c:.2f}")
            shade(a2, h, on_c, AMBER, 0.32)
        a2.set_ylim(0, 1.25)
        a2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        a2.set_ylabel("probability", color=MUTED, fontsize=8)
        muted_legend(a2, loc="upper left", ncol=2)
        # 3. M signal
        a3.plot(h, mn, color=FAINT, lw=0.9, label="network: highest median forecast, +5/15/30 min")
        a3.plot(h, mc, color=STEEL, lw=1.3, label="M signal: higher of network and flux now")
        if thr_m is not None:
            a3.axhline(thr_m, color=RED, lw=0.9, ls="--", label=f"M alert threshold ({goes_class(thr_m)})")
            shade(a3, h, on_m, RED, 0.36)
        fin_m = np.isfinite(mc)
        top3 = max(np.nanmax(mc[fin_m]) if fin_m.any() else -5.5, thr_m if thr_m is not None else -5.0)
        a3.set_ylim(min(np.nanmin(mc[fin_m]) - 0.2 if fin_m.any() else -7, -6.2), top3 + 0.75)
        a3.set_ylabel("log10 flux, W/m2", color=MUTED, fontsize=8)
        muted_legend(a3, loc="upper left", ncol=3)
        # 4. HEL1OS coverage
        a4.fill_between(h, 0, np.nan_to_num(hard), color=STEEL, alpha=0.45, lw=0)
        a4.set_ylim(0, 1.05)
        a4.set_yticks([0, 1])
        a4.set_ylabel("HEL1OS", color=MUTED, fontsize=8)
        a4.set_xlim(0, 24)
        a4.set_xticks(range(0, 25, 3))
        a4.set_xticklabels([f"{x:02d}:00" for x in range(0, 25, 3)])
        a4.set_xlabel("UTC", color=MUTED, fontsize=8)
        for a in (a1, a2, a3):
            a.tick_params(labelbottom=False)
        self.fig.subplots_adjust(left=0.08, right=0.96, top=0.985, bottom=0.07)
        self.canvas.draw_idle()
        self._table(d0)

        r = rules
        if r:
            self.note.config(text=(
                f"Alert rules (outputs/alerts/alert_rules.json, model {r.get('model_sha256', '?')}): "
                f"C when P >= {r['C']['threshold']} ({r['C'].get('false_alarms_per_day_validation')} false alarms/day "
                f"on validation); M when the M signal >= {goes_class(r['M']['threshold'])} "
                f"({r['M'].get('false_alarms_per_day_validation')}/day). The bottom band is the share of each 2 h "
                "window the HEL1OS detectors were watching. Each value is drawn at the minute it became known; "
                "nothing from later minutes is used."))

    def _hover(self, ev) -> None:
        if ev.inaxes is None or not getattr(self, "_day", None) or ev.xdata is None:
            return
        d = self._day
        if d["h"].size == 0:
            return
        i = int(np.clip(np.searchsorted(d["h"], ev.xdata), 0, d["h"].size - 1))
        p, hard = d["pc"][i], d["hard"][i]
        self.readout.config(
            text=f"{datetime.fromtimestamp(float(d['t'][i]), UTC):%H:%M} UTC   SoLEXS {goes_class(d['flux'][i]):>6}"
                 f"   GOES {goes_class(d['goes'][i]):>6}   P(>= C1, 15 min) "
                 f"{'--' if not np.isfinite(p) else format(float(p), '.2f')}   M signal {goes_class(d['mc'][i]):>6}"
                 f"   HEL1OS {'--' if not np.isfinite(hard) else format(100 * float(hard), '.0f') + '%'}", fg=TEXT)
