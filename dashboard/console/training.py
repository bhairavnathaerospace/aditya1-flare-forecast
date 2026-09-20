"""Training tab: the run being trained (or the last one), epoch by epoch.

Reads from the run folder: reports/live.json (written by solarflare/train.py
every ~2 s), reports/history.json (one row per epoch) and reports/config.json
(patience, smoothing), so the numbers shown are the ones training acts on.
"""

from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .common import (AMBER, FAINT, GREEN, GROUND, LINE, MUTED, OUTPUTS, PANEL, RED, STEEL, TEAL, TEXT, UI_B,
                     SMALL, Tile, blend, fmt_dur, muted_legend, read_json, style_axes)

HEADS = [("head_phase", "Phase", "phase"), ("head_inflare", "Flare now", "in_flare"),
         ("head_occurrence", "Flare soon", "occurrence"), ("head_nowcast", "Flux now", "nowcast"),
         ("head_forecast", "Flux forecast", "forecast"), ("head_peak", "Peak size", "peak")]


def runs() -> list[Path]:
    """Trained or training runs anywhere under outputs/ (model, ablations), newest first."""
    found = []
    for pat in ("*/reports", "*/*/reports", "*/*/*/reports"):
        for rep in OUTPUTS.glob(pat):
            files = [f for f in (rep / "live.json", rep / "history.json") if f.exists()]
            if files:
                found.append((max(f.stat().st_mtime for f in files), rep.parent))
    return [p for _, p in sorted(found, reverse=True)]


def run_label(run: Path) -> str:
    try:
        return run.relative_to(OUTPUTS).as_posix()
    except ValueError:
        return str(run)


def score_of(h: dict) -> float:
    """The score model selection uses: smoothed when the run recorded it."""
    return float(h.get("score_smoothed", h.get("score", -1e9)))


def best_index(hist: list[dict]) -> int:
    return max(range(len(hist)), key=lambda i: score_of(hist[i]))


class TrainingTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=GROUND)
        self._drawn = None
        tiles = tk.Frame(self, bg=GROUND)
        tiles.pack(fill="x")
        self.t_status = Tile(tiles, "Status")
        self.t_epoch = Tile(tiles, "Epoch", bar=True)
        self.t_batch = Tile(tiles, "Batch in this epoch", bar=True)
        self.t_time = Tile(tiles, "Elapsed")
        self.t_best = Tile(tiles, "Best validation score")
        self.t_gpu = Tile(tiles, "GPU")
        for i, t in enumerate((self.t_status, self.t_epoch, self.t_batch, self.t_time, self.t_best, self.t_gpu)):
            t.grid(row=i // 3, column=i % 3, sticky="nsew", padx=(0 if i % 3 == 0 else 8, 0), pady=(0, 8))
            tiles.grid_columnconfigure(i % 3, weight=1, uniform="t")

        charts = tk.Frame(self, bg=GROUND)
        charts.pack(fill="both", expand=True)
        for c in (0, 1):
            charts.grid_columnconfigure(c, weight=1, uniform="c")
        charts.grid_rowconfigure(0, weight=1)
        self.fig_loss, self.ax_loss, self.cv_loss = self._chart(charts, 0)
        self.fig_val, self.ax_val, self.cv_val = self._chart(charts, 1)
        self.ax_val2 = self.ax_val.twinx()

        net = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        net.pack(fill="both", expand=True, pady=(10, 0))
        tk.Label(net, text="NETWORK  ·  where it is learning", bg=PANEL, fg=MUTED, font=SMALL).pack(
            anchor="w", padx=12, pady=(9, 0))
        self.net = tk.Canvas(net, bg=PANEL, highlightthickness=0, height=230)
        self.net.pack(fill="both", expand=True, padx=8, pady=(2, 8))

    def _chart(self, parent, col):
        frame = tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        frame.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 10, 0))
        fig = Figure(figsize=(5, 2.3), dpi=100, facecolor=PANEL)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().configure(bg=PANEL, highlightthickness=0)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        return fig, ax, canvas

    # ---- refresh -------------------------------------------------------------------
    def refresh(self, run: Path | None, now: float, gpu: dict | None,
                job_running: bool = True) -> tuple[dict | None, list, str, dict]:
        """Returns (live, history, status, run config) for the watchdog."""
        if run is None:
            self.t_status.set("IDLE", "no training run under outputs/ yet", MUTED)
            return None, [], "IDLE", {}
        rep = run / "reports"
        live = read_json(rep / "live.json")
        hist = read_json(rep / "history.json") or []
        hist = hist if isinstance(hist, list) else []
        cfg = read_json(rep / "config.json") or {}
        status, running = self._tiles(live, hist, now, gpu, job_running)
        key = (str(run), len(hist), live.get("updated_unix") if live else None)
        if key != self._drawn:
            self._charts(hist, live if running else None)
            self._draw_net(live if running else None)
            self._drawn = key
        return live, hist, status, cfg

    def _tiles(self, live, hist, now, gpu, job_running: bool) -> tuple[str, bool]:
        if live:
            age = now - float(live.get("updated_unix", 0))
            st = live.get("status", "training")
            if st == "finished":
                status, color, sub = "FINISHED", GREEN, "early stop" if live.get("early_stopped") else "all epochs"
            elif age > 900 and not job_running:
                status, color, sub = "STOPPED", AMBER, f"stopped {fmt_dur(age)} ago, before it finished"
            elif age > 900:
                status, color, sub = "STALLED", RED, f"no update for {fmt_dur(age)}"
            elif st == "validating":
                status, color, sub = "VALIDATING", AMBER, f"updated {int(age)} s ago"
            else:
                status, color, sub = "TRAINING", TEAL, f"updated {int(age)} s ago"
        elif hist:
            status, color, sub = "LAST RUN", STEEL, "no live file"
        else:
            status, color, sub = "IDLE", MUTED, "nothing trained in this folder yet"
        self.t_status.set(status, sub, color)
        total = int(live.get("epochs_total", 0)) if live else len(hist)
        done = int(live.get("epochs_done", len(hist))) if live else len(hist)
        ep = int(live.get("epoch", done)) if live else max(done - 1, 0)
        running = status in ("TRAINING", "VALIDATING")
        if running:
            self.t_epoch.set(f"{ep + 1} / {total}", f"{done} finished (early stop likely sooner)",
                             frac=(done / total) if total else 0)
            b, nb = int(live.get("batch", 0)), int(live.get("batches", 0))
            self.t_batch.set(f"{b} / {nb}", f"lr {live.get('lr', 0):.2e}" if live.get("lr") else "",
                             frac=b / nb if nb else 0)
        else:
            self.t_epoch.set(f"{done} / {total}" if total else "--", "epochs run", frac=1.0 if done else 0)
            self.t_batch.set("--", "not training", frac=0)
        el = float(live.get("elapsed_s", 0)) if live else None
        per = float(live.get("elapsed_s", 0)) / done if running and done else None
        self.t_time.set(fmt_dur(el), f"~{fmt_dur(per)} per epoch" if per else ("" if el else "--"))
        if hist:
            be = best_index(hist)
            self.t_best.set(f"{score_of(hist[be]):.4f}", f"epoch {hist[be].get('epoch', be)}  (smoothed validation skill)")
        else:
            self.t_best.set("--", "after the first epoch")
        if gpu:
            self.t_gpu.set(f"{gpu['util']:.0f} %", f"{gpu['mem'] / 1024:.1f} / {gpu['mem_total'] / 1024:.1f} GB   "
                           f"{gpu['temp']:.0f} °C" + (f"   {gpu['power']:.0f} W" if gpu.get("power") else ""),
                           AMBER if gpu["temp"] >= 83 else TEXT)
        else:
            self.t_gpu.set("--", "nvidia-smi not answering", MUTED)
        return status, running

    # ---- charts --------------------------------------------------------------------
    def _charts(self, hist: list[dict], live: dict | None):
        ax = self.ax_loss
        ax.clear()
        style_axes(ax, "Training loss per epoch   (lower is better)")
        if hist:
            x = [h["epoch"] for h in hist]
            first = True
            for k in ("phase", "in_flare", "occurrence", "forecast", "peak"):
                ys = [h.get(f"train_{k}") for h in hist]
                if any(v is not None for v in ys):
                    ax.plot(x, ys, lw=0.9, color=FAINT, label="each output's loss" if first else None)
                    first = False
            ax.plot(x, [h.get("train_loss") for h in hist], color=TEAL, lw=1.8, label="total (weighted)")
            muted_legend(ax, loc="upper right")
        if live and live.get("loss_running") is not None:
            xe = live.get("epoch", 0) + live.get("batch", 0) / max(live.get("batches", 1), 1)
            ax.plot([xe], [live["loss_running"]], "o", ms=6, mfc=PANEL, mec=TEAL, mew=1.6)
            ax.text(xe, live["loss_running"], "  now", color=TEAL, fontsize=8, va="bottom")
        ax.set_xlabel("epoch", color=MUTED, fontsize=8)
        self.fig_loss.tight_layout()
        self.cv_loss.draw_idle()

        ax, ax2 = self.ax_val, self.ax_val2
        ax.clear()
        ax2.clear()
        style_axes(ax, "Validation after each epoch")
        if hist:
            x = [h["epoch"] for h in hist]
            for k, lab, c in (("val_TSS_inflare", "TSS flare now", TEAL), ("val_TSS_occurrence", "TSS flare soon", STEEL)):
                ys = [h.get(k) for h in hist]
                if any(v is not None for v in ys):
                    ax.plot(x, ys, color=c, lw=1.6, label=lab)
            sm = [h.get("score_smoothed") for h in hist]
            if any(v is not None for v in sm):
                ax.plot(x, sm, color=TEXT, lw=1.0, ls=":", label="selection score (smoothed)")
            mae = [h.get("val_forecast_MAE") for h in hist]
            if any(v is not None for v in mae):
                ax2.plot(x, mae, color=AMBER, lw=1.2, ls="--", label="forecast error (dex, right)")
                ax2.set_ylim(0, max(v for v in mae if v is not None) * 1.45)
            be = best_index(hist)
            ax.axvline(hist[be]["epoch"], color=MUTED, lw=0.8, ls=":")
            ax.text(hist[be]["epoch"], 0.02, f"best {hist[be]['epoch']} ", color=MUTED, fontsize=7, va="bottom", ha="right")
        ax.set_ylim(0, 1.18)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("epoch", color=MUTED, fontsize=8)
        ax.set_ylabel("TSS  (1 = perfect)", color=MUTED, fontsize=8)
        ax2.tick_params(colors=MUTED, labelsize=8)
        for s in ax2.spines.values():
            s.set_visible(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            leg = ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7, frameon=False, ncol=2)
            for t in leg.get_texts():
                t.set_color(MUTED)
        self.fig_val.tight_layout()
        self.cv_val.draw_idle()

    def _draw_net(self, live: dict | None):
        c = self.net
        c.delete("all")
        w = max(c.winfo_width(), 600)
        h = max(c.winfo_height(), 210)
        grads = (live or {}).get("grad_norm") or {}
        parts = (live or {}).get("parts_running") or {}
        vals = [v for v in grads.values() if v and v > 0]
        lo, hi = (math.log10(min(vals)), math.log10(max(vals))) if vals else (0.0, 1.0)

        def heat(group: str) -> tuple[str, str]:
            v = grads.get(group)
            if not v:
                return blend(PANEL, LINE, 0.9), ""
            t = 0.25 + 0.75 * ((math.log10(v) - lo) / (hi - lo) if hi > lo else 1.0)
            return blend(PANEL, TEAL, t * 0.85), f"∇ {v:.3g}"

        def box(x, y, bw, bh, title, group=None, extra="", show_grad=True):
            fill, gtxt = heat(group) if group else (blend(PANEL, LINE, 0.5), "")
            sub = "  ".join(s for s in ((gtxt if show_grad else ""), extra) if s)
            c.create_rectangle(x, y, x + bw, y + bh, fill=fill, outline=LINE)
            if bh < 40:                       # short head boxes: one line, metrics on the right
                c.create_text(x + 8, y + bh / 2, text=title, anchor="w", fill=TEXT, font=UI_B)
                if sub:
                    c.create_text(x + bw - 6, y + bh / 2, text=sub, anchor="e", fill=TEXT, font=("Consolas", 8))
            else:
                c.create_text(x + 8, y + 8, text=title, anchor="nw", fill=TEXT, font=UI_B)
                if sub:
                    c.create_text(x + 8, y + bh - 8, text=sub, anchor="sw", fill=TEXT, font=("Consolas", 8))
            return (x, y, bw, bh)

        def arrow(a, b):
            ax_, ay = a[0] + a[2], a[1] + a[3] / 2
            bx, by = b[0], b[1] + b[3] / 2
            c.create_line(ax_, ay, (ax_ + bx) / 2, ay, (ax_ + bx) / 2, by, bx, by, fill=FAINT,
                          arrow="last", arrowshape=(6, 7, 3))

        cols = [0.01, 0.155, 0.33, 0.475, 0.61, 0.765]
        bw, bh = w * 0.125, 46
        mid = h / 2 - bh / 2 - 6
        s_in = box(w * cols[0], mid - 56, bw, bh, "SoLEXS", extra="soft X-rays")
        h_in = box(w * cols[0], mid + 56, bw, bh, "HEL1OS", extra="hard X-rays")
        s_en = box(w * cols[1], mid - 56, bw * 1.2, bh, "Soft encoder", "soft_enc")
        h_en = box(w * cols[1], mid + 56, bw * 1.2, bh, "Hard encoder", "hard_enc")
        fus = box(w * cols[2], mid, bw * 1.05, bh, "Gated fusion", "fusion")
        tru = box(w * cols[3], mid, bw * 0.95, bh, "Trunk", "trunk")
        pool = box(w * cols[4], mid, bw * 1.05, bh, "Attention", "pool")
        for a, b in ((s_in, s_en), (h_in, h_en), (s_en, fus), (h_en, fus), (fus, tru), (tru, pool)):
            arrow(a, b)
        top = 8
        hh = (h - top * 2 - 14) / len(HEADS)
        for i, (grp, lab, part) in enumerate(HEADS):
            loss = parts.get(part)
            hb = box(w * cols[5], top + i * hh, w * 0.225, hh - 5, lab, grp,
                     f"{loss:.3f}" if loss is not None else "", show_grad=False)
            arrow(pool, hb)
        note = ("Brightness = size of the last gradient in each part (where the weights change most); "
                "numbers on the right = each output's loss." if grads else
                "Blocks light up while a run is training (gradient size per part).")
        c.create_text(8, h - 2, text=note, anchor="sw", fill=FAINT, font=SMALL)
