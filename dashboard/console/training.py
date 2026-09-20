"""Training screen: the run being trained (or the last one), epoch by epoch.

Reads from the run folder: reports/live.json (written by solarflare/train.py
every ~2 s), reports/history.json (one row per epoch) and reports/config.json
(patience, smoothing), so the numbers shown are the ones training acts on.

Eight tiles, then four charts: the loss and each head's share of it, validation
skill after every epoch, the learning rate against how long each epoch took,
and the gradient reaching each part of the network as it trains.
"""

from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .common import (AMBER, FAINT, GREEN, GROUND, LINE, MUTED, OUTPUTS, PANEL, RED, SMALL, STEEL, TEAL,
                     TEXT, UI_B, Sparkline, Tile, blend, fmt_dur, muted_legend, read_json, style_axes)
from .system import epoch_seconds, eta_hours

HEADS = [("head_phase", "Phase", "phase"), ("head_inflare", "Flare now", "in_flare"),
         ("head_occurrence", "Flare soon", "occurrence"), ("head_nowcast", "Flux now", "nowcast"),
         ("head_forecast", "Flux forecast", "forecast"), ("head_peak", "Peak size", "peak")]
PARTS = ["soft_enc", "hard_enc", "fusion", "trunk", "pool"] + [h[0] for h in HEADS]
PART_COLORS = [TEAL, STEEL, AMBER, GREEN, "#b06fc4", "#c4886f", "#6fc4b0", "#c46f8f", "#8f6fc4", "#c4b06f", "#6f8fc4"]
GRAD_SPAN = 600                  # gradient samples kept (about 20 min at one every 2 s)


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
        self.grads: list[tuple[float, dict]] = []
        tiles = tk.Frame(self, bg=GROUND)
        tiles.pack(fill="x")
        self.tiles = {}
        for i, (key, label, bar) in enumerate((
                ("status", "Status", False), ("epoch", "Epoch", True), ("batch", "Batch in this epoch", True),
                ("time", "Elapsed", False), ("left", "Time left", False), ("best", "Best validation score", False),
                ("since", "Epochs since best", False), ("speed", "Throughput", False))):
            t = Tile(tiles, label, bar=bar)
            t.grid(row=i // 4, column=i % 4, sticky="nsew", padx=(0 if i % 4 == 0 else 8, 0), pady=(0, 8))
            tiles.grid_columnconfigure(i % 4, weight=1, uniform="t")
            self.tiles[key] = t

        charts = tk.Frame(self, bg=GROUND)
        charts.pack(fill="x")
        for c in (0, 1):
            charts.grid_columnconfigure(c, weight=1, uniform="c")
        for r in (0, 1):
            charts.grid_rowconfigure(r, weight=1)
        self.fig_loss, self.ax_loss, self.cv_loss = self._chart(charts, 0, 0)
        self.fig_val, self.ax_val, self.cv_val = self._chart(charts, 0, 1)
        self.ax_val2 = self.ax_val.twinx()
        self.fig_lr, self.ax_lr, self.cv_lr = self._chart(charts, 1, 0)
        self.ax_lr2 = self.ax_lr.twinx()
        self.fig_grad, self.ax_grad, self.cv_grad = self._chart(charts, 1, 1)

        bottom = tk.Frame(self, bg=GROUND)
        bottom.pack(fill="both", expand=True, pady=(10, 0))
        bottom.grid_columnconfigure(0, weight=3, uniform="n")
        bottom.grid_columnconfigure(1, weight=2, uniform="n")
        bottom.grid_rowconfigure(0, weight=1)
        net = tk.Frame(bottom, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        net.grid(row=0, column=0, sticky="nsew")
        tk.Label(net, text="NETWORK  ·  where it is learning", bg=PANEL, fg=MUTED, font=SMALL).pack(
            anchor="w", padx=12, pady=(9, 0))
        self.net = tk.Canvas(net, bg=PANEL, highlightthickness=0, height=172)
        self.net.pack(fill="both", expand=True, padx=8, pady=(2, 8))

        side = tk.Frame(bottom, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        side.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        tk.Label(side, text="THIS EPOCH  ·  live", bg=PANEL, fg=MUTED, font=SMALL).pack(anchor="w", padx=12,
                                                                                         pady=(9, 2))
        self.loss_spark = Sparkline(side, "running loss", "", 300, None, None, TEAL, height=30)
        self.loss_spark.pack(fill="x", padx=12, pady=(0, 4))
        self.table = tk.Text(side, bg=PANEL, fg=TEXT, font=("Consolas", 9), relief="flat", highlightthickness=0,
                             wrap="none", height=12)
        self.table.pack(fill="both", expand=True, padx=12, pady=(2, 10))
        self.table.tag_configure("k", foreground=MUTED)
        self.table.tag_configure("v", foreground=TEXT)

    def _chart(self, parent, row, col):
        frame = tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        frame.grid(row=row, column=col, sticky="nsew", padx=(0 if col == 0 else 10, 0),
                   pady=(0 if row == 0 else 10, 0))
        fig = Figure(figsize=(4.6, 1.7), dpi=100, facecolor=PANEL)
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
            self.tiles["status"].set("IDLE", "no training run under outputs/ yet", MUTED)
            return None, [], "IDLE", {}
        rep = run / "reports"
        live = read_json(rep / "live.json")
        hist = read_json(rep / "history.json") or []
        hist = hist if isinstance(hist, list) else []
        cfg = read_json(rep / "config.json") or {}
        status, running = self._tiles(live, hist, now, gpu, job_running, cfg)
        if running and live and live.get("grad_norm"):
            t = float(live.get("updated_unix", now))
            if not self.grads or t > self.grads[-1][0]:
                self.grads.append((t, dict(live["grad_norm"])))
                del self.grads[:-GRAD_SPAN]
        self.loss_spark.push(live.get("loss_running") if running else None,
                             f"{live['loss_running']:.3f}" if running and live.get("loss_running") is not None else "--")
        self._numbers(live, hist, running)
        key = (str(run), len(hist), live.get("updated_unix") if live else None)
        if key != self._drawn:
            self._charts(hist, live if running else None)
            self._draw_net(live if running else None)
            self._drawn = key
        return live, hist, status, cfg

    def _tiles(self, live, hist, now, gpu, job_running: bool, cfg: dict) -> tuple[str, bool]:
        T = self.tiles
        patience = int((cfg.get("train") or {}).get("early_stop_patience", 12))
        # history.json is the record of what happened; live.json only says what is
        # happening now, and a stale live file must not be read as a stalled run
        last_epoch = int(hist[-1].get("epoch", len(hist) - 1)) if hist else -1
        total_planned = int(live.get("epochs_total", 0)) if live else 0
        best_i = best_index(hist) if hist else -1
        ran_out = bool(hist) and (len(hist) >= total_planned > 0
                                  or (best_i >= 0 and last_epoch >= hist[best_i].get("epoch", best_i) + patience))
        if live:
            age = now - float(live.get("updated_unix", 0))
            st = live.get("status", "training")
            if st == "finished":
                status, color, sub = "FINISHED", GREEN, "early stop" if live.get("early_stopped") else "all epochs"
            elif age > 900 and ran_out:
                status, color, sub = "FINISHED", GREEN, f"{len(hist)} epochs; live file went stale"
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
        T["status"].set(status, sub, color)
        total = total_planned or len(hist)
        done = max(int(live.get("epochs_done", 0)) if live else 0, len(hist))
        ep = int(live.get("epoch", done)) if live else max(done - 1, 0)
        running = status in ("TRAINING", "VALIDATING")
        if running:
            T["epoch"].set(f"{ep + 1} / {total}", f"{done} finished", frac=(done / total) if total else 0)
            b, nb = int(live.get("batch", 0)), int(live.get("batches", 0))
            T["batch"].set(f"{b} / {nb}", f"lr {live.get('lr', 0):.2e}" if live.get("lr") else "",
                           frac=b / nb if nb else 0)
        else:
            T["epoch"].set(f"{done} / {total}" if total else "--", "epochs run", frac=1.0 if done else 0)
            T["batch"].set("--", "not training", frac=0)
        el = float(live.get("elapsed_s", 0)) if live else None
        per = epoch_seconds(live, hist)
        T["time"].set(fmt_dur(el), f"{fmt_dur(per)} an epoch" if per else ("" if el else "--"))
        all_h, stop_h = (None, None) if status in ("FINISHED", "STOPPED") else eta_hours(live, hist, patience)
        if all_h is not None:
            T["left"].set(f"{all_h:.1f} h", f"~{stop_h:.1f} h if it stops early" if stop_h < all_h - 0.05
                          else f"all {total} epochs")
        else:
            T["left"].set("--", "not training")
        if hist:
            be = best_index(hist)
            since = len(hist) - 1 - be
            T["best"].set(f"{score_of(hist[be]):.4f}", f"epoch {hist[be].get('epoch', be)}  (smoothed)")
            T["since"].set(str(since), f"early stop at {patience}",
                           RED if since >= patience - 2 else (AMBER if since >= max(patience // 2, 3) else TEXT))
        else:
            T["best"].set("--", "after the first epoch")
            T["since"].set("--", f"early stop at {patience}")
        if running and per and live.get("batches"):
            rate = int(live["batches"]) / per
            T["speed"].set(f"{rate:.1f}/s", f"{int(live['batches'])} batches per epoch")
        elif gpu:
            T["speed"].set(f"{gpu['util']:.0f} %", f"GPU {gpu['temp']:.0f} °C, "
                           f"{gpu['mem'] / 1024:.1f}/{gpu['mem_total'] / 1024:.0f} GB")
        else:
            T["speed"].set("--", "no GPU reading")
        return status, running

    def _numbers(self, live, hist, running: bool) -> None:
        """The live numbers as text: loss parts, gradients, last validation."""
        self.table.config(state="normal")
        self.table.delete("1.0", "end")
        rows: list[tuple[str, str]] = []
        parts = (live or {}).get("parts_running") or {}
        if running and parts:
            rows += [(f"loss {k}", f"{v:.4f}") for k, v in parts.items()]
        grads = (live or {}).get("grad_norm") or {}
        if running and grads:
            top = sorted(grads.items(), key=lambda kv: -float(kv[1] or 0))[:3]
            rows.append(("largest gradients", ", ".join(f"{k} {float(v):.2g}" for k, v in top)))
        if hist:
            h = hist[-1]
            for k, lab in (("val_TSS_inflare", "val TSS flare now"), ("val_TSS_occurrence", "val TSS flare soon"),
                           ("val_nowcast_MAE", "val flux now MAE"), ("val_forecast_MAE", "val forecast MAE"),
                           ("score", "score (raw)"), ("score_smoothed", "score (smoothed)")):
                if h.get(k) is not None:
                    rows.append((f"{lab} @ep{h.get('epoch', '?')}", f"{float(h[k]):.4f}"))
        if not rows:
            rows = [("waiting", "no live run")]
        for k, v in rows:
            self.table.insert("end", f"{k:<26}", "k")
            self.table.insert("end", f"{v}\n", "v")
        self.table.config(state="disabled")

    # ---- charts --------------------------------------------------------------------
    def _waiting(self, live: dict | None) -> None:
        """Before the first epoch ends there is nothing to plot: explain the wait."""
        left = ""
        if live and live.get("batches"):
            frac = int(live.get("batch", 0)) / max(int(live["batches"]), 1)
            el = float(live.get("elapsed_s", 0))
            if frac > 0.02 and el > 0:
                left = f", about {max((el / frac - el) / 60.0, 0):.0f} min to go"
        for fig, ax, cv, title in ((self.fig_loss, self.ax_loss, self.cv_loss, "Loss per epoch"),
                                   (self.fig_val, self.ax_val, self.cv_val, "Validation after each epoch"),
                                   (self.fig_lr, self.ax_lr, self.cv_lr, "Learning rate and epoch time"),
                                   (self.fig_grad, self.ax_grad, self.cv_grad, "Gradient per part   (log)")):
            ax.clear()
            style_axes(ax, title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, f"the first epoch is still running{left}", transform=ax.transAxes, ha="center",
                    va="center", color=FAINT, fontsize=9)
            fig.tight_layout()
            cv.draw_idle()
        self.ax_val2.clear()
        self.ax_val2.set_xticks([])
        self.ax_val2.set_yticks([])
        self.ax_lr2.clear()
        self.ax_lr2.set_xticks([])
        self.ax_lr2.set_yticks([])

    def _charts(self, hist: list[dict], live: dict | None):
        if not hist:
            self._waiting(live)
            return
        x = [h["epoch"] for h in hist]
        ax = self.ax_loss
        ax.clear()
        style_axes(ax, "Loss per epoch   (total and each head)")
        if hist:
            for i, (_, lab, part) in enumerate(HEADS):
                ys = [h.get(f"train_{part}") for h in hist]
                if any(v is not None for v in ys):
                    ax.plot(x, ys, lw=0.9, color=PART_COLORS[i % len(PART_COLORS)], alpha=0.75, label=lab)
            ax.plot(x, [h.get("train_loss") for h in hist], color=TEXT, lw=1.8, label="total (weighted)")
            muted_legend(ax, loc="upper right", ncol=2, fontsize=6)
        if live and live.get("loss_running") is not None:
            xe = live.get("epoch", 0) + live.get("batch", 0) / max(live.get("batches", 1), 1)
            ax.plot([xe], [live["loss_running"]], "o", ms=6, mfc=PANEL, mec=TEXT, mew=1.6)
        self.fig_loss.tight_layout()
        self.cv_loss.draw_idle()

        ax, ax2 = self.ax_val, self.ax_val2
        ax.clear()
        ax2.clear()
        style_axes(ax, "Validation after each epoch")
        if hist:
            for k, lab, c in (("val_TSS_inflare", "TSS flare now", TEAL), ("val_TSS_occurrence", "TSS flare soon", STEEL)):
                ys = [h.get(k) for h in hist]
                if any(v is not None for v in ys):
                    ax.plot(x, ys, color=c, lw=1.6, label=lab)
            sm = [h.get("score_smoothed") for h in hist]
            if any(v is not None for v in sm):
                ax.plot(x, sm, color=TEXT, lw=1.1, ls=":", label="selection score")
            mae = [h.get("val_forecast_MAE") for h in hist]
            if any(v is not None for v in mae):
                ax2.plot(x, mae, color=AMBER, lw=1.2, ls="--", label="forecast error (dex, right)")
                ax2.set_ylim(0, max(v for v in mae if v is not None) * 1.45)
            be = best_index(hist)
            ax.axvline(hist[be]["epoch"], color=MUTED, lw=0.8, ls=":")
            ax.text(hist[be]["epoch"], 0.02, f"best {hist[be]['epoch']} ", color=MUTED, fontsize=7, va="bottom",
                    ha="right")
        ax.set_ylim(0, 1.18)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("epoch", color=MUTED, fontsize=8)
        ax2.tick_params(colors=MUTED, labelsize=8)
        for s in ax2.spines.values():
            s.set_visible(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            leg = ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=6, frameon=False, ncol=2)
            for t in leg.get_texts():
                t.set_color(MUTED)
        self.fig_val.tight_layout()
        self.cv_val.draw_idle()

        ax, ax2 = self.ax_lr, self.ax_lr2
        ax.clear()
        ax2.clear()
        style_axes(ax, "Learning rate and epoch time")
        if hist:
            lrs = [h.get("lr") for h in hist]
            if any(v is not None for v in lrs):
                ax.plot(x, lrs, color=STEEL, lw=1.5, label="learning rate")
                ax.set_yscale("log")
            secs = [h.get("seconds") or h.get("epoch_s") for h in hist]
            if any(v is not None for v in secs):
                ax2.plot(x, [s / 60 if s else None for s in secs], color=AMBER, lw=1.2, ls="--",
                         label="minutes per epoch (right)")
        ax.set_xlabel("epoch", color=MUTED, fontsize=8)
        ax2.tick_params(colors=MUTED, labelsize=8)
        for s in ax2.spines.values():
            s.set_visible(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            leg = ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=6, frameon=False)
            for t in leg.get_texts():
                t.set_color(MUTED)
        self.fig_lr.tight_layout()
        self.cv_lr.draw_idle()

        ax = self.ax_grad
        ax.clear()
        style_axes(ax, "Gradient per part   (log)")
        if len(self.grads) >= 3:
            t0 = self.grads[-1][0]
            xs = [(t - t0) / 60.0 for t, _ in self.grads]
            for i, part in enumerate(PARTS):
                ys = [g.get(part) for _, g in self.grads]
                if any(v for v in ys):
                    ax.plot(xs, [v if v and v > 0 else None for v in ys], lw=1.1,
                            color=PART_COLORS[i % len(PART_COLORS)], label=part.replace("head_", ""))
            ax.set_yscale("log")
            muted_legend(ax, loc="upper right", ncol=3, fontsize=5.5)
            ax.set_xlabel("minutes ago", color=MUTED, fontsize=8)
        else:
            ax.text(0.5, 0.5, "fills while a run is training", transform=ax.transAxes, ha="center", va="center",
                    color=FAINT, fontsize=9)
        self.fig_grad.tight_layout()
        self.cv_grad.draw_idle()

    def _draw_net(self, live: dict | None):
        c = self.net
        c.delete("all")
        w = max(c.winfo_width(), 560)
        h = max(c.winfo_height(), 110)          # draw to the canvas it actually has
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
            parts_ = [gtxt if show_grad else ""] + ([extra] if bh >= 34 else [])
            sub = "  ".join(s for s in parts_ if s)
            c.create_rectangle(x, y, x + bw, y + bh, fill=fill, outline=LINE)
            if bh < 40:                       # short boxes: one line, the number on the right
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
        bh = min(42.0, max(h * 0.22, 22.0))
        bw = w * 0.125
        off = min(52.0, max((h - bh - 16) / 2.0, bh * 0.75))
        mid = h / 2 - bh / 2 - 6
        s_in = box(w * cols[0], mid - off, bw, bh, "SoLEXS", extra="soft X-rays")
        h_in = box(w * cols[0], mid + off, bw, bh, "HEL1OS", extra="hard X-rays")
        s_en = box(w * cols[1], mid - off, bw * 1.2, bh, "Soft encoder", "soft_enc", show_grad=False)
        h_en = box(w * cols[1], mid + off, bw * 1.2, bh, "Hard encoder", "hard_enc", show_grad=False)
        fus = box(w * cols[2], mid, bw * 1.05, bh, "Gated fusion", "fusion", show_grad=False)
        tru = box(w * cols[3], mid, bw * 0.95, bh, "Trunk", "trunk", show_grad=False)
        pool = box(w * cols[4], mid, bw * 1.05, bh, "Attention", "pool", show_grad=False)
        for a, b in ((s_in, s_en), (h_in, h_en), (s_en, fus), (h_en, fus), (fus, tru), (tru, pool)):
            arrow(a, b)
        top = 4
        hh = (h - top * 2 - 12) / len(HEADS)
        for i, (grp, lab, part) in enumerate(HEADS):
            loss = parts.get(part)
            hb = box(w * cols[5], top + i * hh, w * 0.225, hh - 5, lab, grp,
                     f"{loss:.3f}" if loss is not None else "", show_grad=False)
            arrow(pool, hb)
        note = ("Brighter = larger gradient in that part right now; the numbers on the right are each output's "
                "loss." if grads else "Blocks light up while a run is training (gradient size per part).")
        c.create_text(8, h - 2, text=note, anchor="sw", fill=FAINT, font=SMALL)
