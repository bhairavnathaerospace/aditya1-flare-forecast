"""Live training status for the desktop console (dashboard/mission_control.pyw).

Training writes ``<out>/reports/live.json`` every couple of seconds and
``history.json`` after every epoch, both atomically (write a temp file, then
rename), so a reader never sees half a file. Nothing here may stop a training
run: every write is wrapped, and a failure only disables further updates.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from pathlib import Path

#: Parameter-name prefixes shown as blocks in the console's network diagram.
GROUPS = ("soft_enc", "hard_enc", "fusion", "trunk", "pool", "head_phase", "head_inflare",
          "head_nowcast", "head_forecast", "head_occurrence", "head_peak")


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    os.replace(tmp, path)


def grad_norms(model) -> dict[str, float]:
    """L2 norm of the gradient of each parameter group (before clipping)."""
    acc: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = next((g for g in GROUPS if name.startswith(g)), None)
        if g is None:
            continue
        acc[g] = acc.get(g, 0.0) + float(p.grad.detach().pow(2).sum())
    return {k: math.sqrt(v) for k, v in acc.items()}


class LiveStatus:
    def __init__(self, out_dir: Path, *, epochs: int, batches: int, params: int, device: str,
                 every_s: float = 2.0):
        self.path = Path(out_dir) / "reports" / "live.json"
        self.hist_path = Path(out_dir) / "reports" / "history.json"
        self.every_s = every_s
        self.enabled = True
        self._last = 0.0
        self.t0 = time.time()
        self.state = {"run": Path(out_dir).name, "status": "training", "device": str(device),
                      "params": int(params), "epochs_total": int(epochs), "epoch": 0,
                      "batch": 0, "batches": int(batches), "started_unix": self.t0,
                      "best_epoch": None, "best_score": None, "epochs_done": 0}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.enabled = False
        self._write(force=True)

    def due(self) -> bool:
        return self.enabled and time.time() - self._last >= self.every_s

    def _write(self, force: bool = False) -> None:
        if not self.enabled or (not force and not self.due()):
            return
        try:
            now = time.time()
            self.state["updated_unix"] = now
            self.state["elapsed_s"] = round(now - self.t0, 1)
            write_json_atomic(self.path, self.state)
            self._last = now
        except Exception:                   # never let a status file stop training
            self.enabled = False

    def batch(self, epoch: int, batch: int, lr: float, running: dict[str, float], n: int,
              grads: dict[str, float] | None = None) -> None:
        """Running means over the epoch so far; ``grads`` from grad_norms() before clipping."""
        self.state.update({"status": "training", "epoch": epoch, "batch": batch, "lr": lr,
                           "loss_running": running.get("loss", 0.0) / max(n, 1),
                           "parts_running": {k: v / max(n, 1) for k, v in running.items() if k != "loss"}})
        if grads:
            self.state["grad_norm"] = grads
        self._write(force=True)

    def validating(self, epoch: int) -> None:
        self.state.update({"status": "validating", "epoch": epoch, "batch": self.state["batches"]})
        self._write(force=True)

    def epoch_done(self, history: list[dict], best_epoch: int, best_score: float) -> None:
        self.state.update({"status": "training", "epochs_done": len(history), "best_epoch": best_epoch,
                           "best_score": best_score, "last_epoch_s": round(time.time() - self.t0, 1)})
        if self.enabled:
            with contextlib.suppress(Exception):
                write_json_atomic(self.hist_path, history)
        self._write(force=True)

    def finish(self, status: str = "finished") -> None:
        self.state["status"] = status
        self._write(force=True)
