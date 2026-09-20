"""Training loop for SoLEXHEL-Net."""

from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .live import GROUPS, LiveStatus, grad_norms
from .pipeline import prepare, make_loaders, resolve_device, phase_class_weights, Prepared
from .models.net import FluxAnchor, SolexHelNet, flux_anchor, count_parameters
from .models.losses import MultiTaskLoss
from .metrics import best_threshold, roc_auc, regression_scores


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    # Legacy global seed on purpose: third-party code (sklearn splits in
    # baselines.py, some torch paths) still reads the global NumPy RNG, and
    # a local Generator would not make those reproducible.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_threads(n: int) -> None:
    """Torch defaults to half the cores on this box; on CPU that halves speed."""
    import os
    n = n or (os.cpu_count() or 1)
    torch.set_num_threads(max(int(n), 1))


def tune_backend(device: torch.device) -> None:
    """Enable the GPU fast paths that matter for this model.

    The network is a stack of fixed-shape 1-D convolutions, which is exactly
    the case cuDNN autotuning is designed for -- shapes never change after the
    first batch, so the one-off benchmarking cost is repaid immediately.
    TF32 costs nothing here: the inputs are counts normalised to order unity,
    far from needing full float32 mantissa.
    """
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def cosine_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def clip_by_group(model: torch.nn.Module, max_norm: float) -> None:
    """Clip the gradient of each network part (live.GROUPS) to ``max_norm`` separately."""
    groups: dict[str, list[torch.nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            g = next((g for g in GROUPS if name.startswith(g)), "other")
            groups.setdefault(g, []).append(p)
    for params in groups.values():
        torch.nn.utils.clip_grad_norm_(params, max_norm)


def smoothed_score(history: list[dict], score: float, k: int) -> float:
    """Mean of this epoch's validation score and the ``k - 1`` epochs before it.

    Model selection and early stopping use this, not the raw score, so that one
    lucky epoch on a noisy head can neither end the run early nor be saved as
    "best" (v4: raw score peaked at epoch 5, then patience ran out)."""
    if k <= 1 or not history:
        return float(score)
    return float(np.mean([h["score"] for h in history[-(k - 1):]] + [score]))


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def collect_predictions(model: SolexHelNet, loader: DataLoader,
                        device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    acc: dict[str, list] = {}

    def push(k, v):
        acc.setdefault(k, []).append(np.asarray(v))

    for batch in loader:
        b = _to_device(batch, device)
        out = model(b["soft"], b["soft_mask"], b["hard"], b["hard_mask"], b["clock"])
        push("p_inflare", torch.sigmoid(out["in_flare"]).cpu().numpy())
        push("p_phase", torch.softmax(out["phase"], -1).cpu().numpy())
        push("p_occurrence", torch.sigmoid(out["occurrence"]).cpu().numpy())
        push("nowcast", out["nowcast"].cpu().numpy())
        push("forecast", out["forecast"].cpu().numpy())
        push("peak", out["peak"].cpu().numpy())
        for k in ("in_flare", "phase", "nowcast", "nowcast_mask", "forecast",
                  "forecast_mask", "occurrence", "occurrence_mask",
                  "peak", "peak_mask", "t_unix", "persistence"):
            push(f"y_{k}", batch[k].numpy())
    return {k: np.concatenate(v, axis=0) for k, v in acc.items()}


def quick_val_metrics(pred: dict[str, np.ndarray]) -> dict[str, float]:
    """Cheap per-epoch summary used for model selection.

    Selection is on validation TSS for flare occurrence rather than on the loss:
    the loss is dominated by the regression heads, and a model can improve it
    while getting worse at the thing the system exists to do.
    """
    y = pred["y_in_flare"]
    m = pred["y_nowcast_mask"] > 0
    out: dict[str, float] = {}
    if m.sum() > 0 and len(np.unique(y[m])) > 1:
        _, s = best_threshold(y[m], pred["p_inflare"][m], "TSS")
        out["val_TSS_inflare"] = s["TSS"]
        out["val_AUC_inflare"] = roc_auc(y[m], pred["p_inflare"][m])
    else:
        out["val_TSS_inflare"] = float("nan")
        out["val_AUC_inflare"] = float("nan")

    occ_t, occ_m = pred["y_occurrence"], pred["y_occurrence_mask"] > 0
    tss = []
    for h in range(occ_t.shape[1]):
        mm = occ_m[:, h]
        if mm.sum() > 0 and len(np.unique(occ_t[mm, h])) > 1:
            _, s = best_threshold(occ_t[mm, h], pred["p_occurrence"][mm, h], "TSS")
            tss.append(s["TSS"])
    out["val_TSS_occurrence"] = float(np.mean(tss)) if tss else float("nan")

    r = regression_scores(pred["y_nowcast"], pred["nowcast"],
                          pred["y_nowcast_mask"])
    out["val_nowcast_MAE"] = r["MAE"]

    q50 = pred["forecast"][:, :, pred["forecast"].shape[2] // 2]
    fm = pred["y_forecast_mask"] > 0
    if fm.any():
        out["val_forecast_MAE"] = float(
            np.abs(q50 - pred["y_forecast"])[fm].mean())
    else:
        out["val_forecast_MAE"] = float("nan")
    return out


def selection_score(m: dict[str, float]) -> float:
    """Composite: flare skill first, regression accuracy as a tie-break."""
    tss_now = m.get("val_TSS_inflare", float("nan"))
    tss_occ = m.get("val_TSS_occurrence", float("nan"))
    parts = [v for v in (tss_now, tss_occ) if not math.isnan(v)]
    if not parts:
        return -float(m.get("val_forecast_MAE", 1e9))
    return float(np.mean(parts)) - 0.05 * float(m.get("val_forecast_MAE", 0.0))


def train(cfg: Config, prep: Prepared | None = None, verbose: bool = True) -> dict:
    set_seed(cfg.train.seed)
    set_threads(getattr(cfg.train, "torch_threads", 0))
    device = resolve_device(cfg.train.device)
    tune_backend(device)

    if prep is None:
        prep = prepare(cfg, verbose=verbose)
    tr, va, te, stats = make_loaders(prep, cfg)

    anchor = flux_anchor(cfg.model, cfg.pre, prep.norm.mean_soft, prep.norm.std_soft)
    model = SolexHelNet(prep.n_soft, prep.n_hard, prep.n_clock,
                        cfg.model, cfg.win, anchor=anchor).to(device)
    crit = MultiTaskLoss(
        cfg.win.quantiles, cfg.train.focal_gamma,
        cfg.train.w_phase, cfg.train.w_occurrence,
        cfg.train.w_nowcast, cfg.train.w_forecast,
        phase_weight=phase_class_weights(stats).to(device),
    ).to(device)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(crit.parameters()),
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    total_steps = max(cfg.train.epochs * max(len(tr), 1), 1)
    warmup = int(cfg.train.warmup_frac * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_warmup(s, total_steps, warmup))

    out_dir = Path(cfg.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoints" / "best.pt"

    if verbose:
        print(f"\ndevice={device}  params={count_parameters(model):,}  "
              f"receptive field={model.receptive_field()} steps "
              f"(window {cfg.steps_per_window})")
        print(f"train batches={len(tr)}  val batches={len(va)}  test batches={len(te)}")

    # Status for the desktop console (dashboard/mission_control.pyw); never fatal.
    live = LiveStatus(out_dir, epochs=cfg.train.epochs, batches=len(tr),
                      params=count_parameters(model), device=device)
    history: list[dict] = []
    best_score = -np.inf
    best_epoch = -1
    patience = 0
    t0 = time.time()

    for epoch in range(cfg.train.epochs):
        t_epoch = time.time()
        model.train()
        run: dict[str, float] = {}
        n_batches = 0
        for batch in tr:
            b = _to_device(batch, device)
            out = model(b["soft"], b["soft_mask"], b["hard"], b["hard_mask"], b["clock"])
            loss, parts = crit(out, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grads = grad_norms(model) if live.due() else None
            if cfg.train.balance_head_gradients:
                clip_by_group(model, cfg.train.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            sched.step()
            run["loss"] = run.get("loss", 0.0) + float(loss.detach())
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v
            n_batches += 1
            if grads is not None:
                live.batch(epoch, n_batches, sched.get_last_lr()[0], run, n_batches, grads)
        for k in run:
            run[k] /= max(n_batches, 1)

        vm: dict[str, float] = {}
        live.validating(epoch)
        if len(va):
            vpred = collect_predictions(model, va, device)
            vm = quick_val_metrics(vpred)
        score = selection_score(vm) if vm else -run["loss"]

        # Selection on a trailing mean of the score: one lucky epoch on a noisy
        # head can neither end the run early nor be saved as "best".
        smoothed = smoothed_score(history, score, int(getattr(cfg.train, "select_smooth_epochs", 1)))
        rec = {"epoch": epoch, "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t_epoch, 1),
               **{f"train_{k_}": v for k_, v in run.items()}, **vm,
               "score": score, "score_smoothed": smoothed}
        history.append(rec)

        improved = smoothed > best_score + 1e-5
        if improved:
            best_score, best_epoch, patience = smoothed, epoch, 0
            torch.save({
                "model": model.state_dict(),
                "loss": crit.state_dict(),
                "epoch": epoch,
                "score": score,
                "n_soft": prep.n_soft, "n_hard": prep.n_hard,
                "n_clock": prep.n_clock,
                # The model is only defined together with the input scaling it
                # was trained on; refitting it later on grown data shifts every
                # input (see forward.py).
                "norm": {k: np.asarray(v) for k, v in vars(prep.norm).items()},
                # Stored so the anchor is restored on load: it changes what the
                # flux heads mean, and adds no weights that would flag its absence.
                "anchor": dataclasses.asdict(anchor) if anchor else None,
                "config": {"model": vars(cfg.model), "win": vars(cfg.win),
                           "pre": vars(cfg.pre)},
            }, ckpt_path)
        else:
            patience += 1
        live.epoch_done(history, best_epoch, float(best_score))

        if verbose:
            msg = (f"ep {epoch:3d} | loss {run['loss']:.4f} | "
                   f"now {run.get('nowcast', 0):.3f} fore {run.get('forecast', 0):.3f} "
                   f"occ {run.get('occurrence', 0):.3f}")
            if vm:
                msg += (f" | vTSS {vm['val_TSS_inflare']:.3f}"
                        f" vTSSocc {vm['val_TSS_occurrence']:.3f}"
                        f" vMAE {vm['val_forecast_MAE']:.3f}")
            msg += f" | score {score:.4f}" + ("  *" if improved else "")
            print(msg)

        if patience >= cfg.train.early_stop_patience:
            if verbose:
                print(f"early stop at epoch {epoch} (best {best_epoch})")
            live.state["early_stopped"] = True
            break

    elapsed = time.time() - t0
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports" / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
    live.finish()

    if verbose:
        print(f"\ntrained {len(history)} epochs in {elapsed:.1f}s; "
              f"best epoch {best_epoch} score {best_score:.4f}")
        print(f"checkpoint: {ckpt_path}")

    return {"history": history, "best_epoch": best_epoch,
            "best_score": float(best_score), "checkpoint": str(ckpt_path),
            "label_stats": stats, "seconds": elapsed}


def load_model(ckpt_path: Path, cfg: Config,
               device: torch.device | None = None) -> SolexHelNet:
    device = device or resolve_device(cfg.train.device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck.get("anchor")
    model = SolexHelNet(ck["n_soft"], ck["n_hard"], ck["n_clock"],
                        cfg.model, cfg.win,
                        anchor=FluxAnchor(**a) if a else None).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model
