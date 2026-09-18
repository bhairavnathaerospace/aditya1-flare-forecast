"""Rise-phase forecasting: training, grouped CV, and architecture comparison.

Everything here is grouped by flare event. With ~13 events, the difference
between a real result and a lucky split is large, so single-number scores are
not reported without an interval around them.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import Config
from .metrics import skill_scores, roc_auc
from .models.riseflare import RiseNet, RiseLoss
from .pipeline import Prepared, resolve_device
from .preprocess.dataset import Segment, Normalizer
from .preprocess.events import (
    RiseDataset, build_rise_dataset, format_threshold, summarise, PEAK_TIME_SCALE_S,
)
from .torch_data import normalized_inputs
from .train import set_seed, set_threads, tune_backend, cosine_warmup


class RiseWindows(Dataset):
    """Trailing window ending at each rise-phase sample."""

    def __init__(self, segments: list[Segment], ds: RiseDataset,
                 indices: list[int], cfg: Config, norm: Normalizer,
                 inputs: dict | None = None, drop_hard: bool = False):
        self.segments = segments
        self.ds = ds
        self.indices = list(indices)
        self.L = cfg.steps_per_window
        #: Present every window as if HEL1OS had not observed: the soft-only
        #: arm of the fusion ablation. Same samples, same model, same seeds.
        self.drop_hard = drop_hard
        # Reuse one normalised copy across folds and encoders; rebuilding it per
        # loader renormalises the whole archive dozens of times per sweep.
        inputs = inputs if inputs is not None else normalized_inputs(segments, norm)
        self._soft = inputs["soft"]
        self._hard = inputs["hard"]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int) -> dict[str, torch.Tensor]:
        s = self.ds.samples[self.indices[k]]
        seg = self.segments[s.seg]
        lo = max(s.end - self.L, 0)
        pad = self.L - (s.end - lo)

        def take(arr):
            w = arr[lo:s.end]
            if pad > 0:  # left-pad short windows; the mask marks them absent
                w = np.concatenate([np.zeros((pad, w.shape[1]), np.float32), w])
            return torch.from_numpy(np.ascontiguousarray(w))

        def take_mask(m):
            w = m[lo:s.end].astype(np.float32)
            if pad > 0:
                w = np.concatenate([np.zeros(pad, np.float32), w])
            return torch.from_numpy(w)

        hard = take(self._hard[s.seg])
        hard_mask = take_mask(seg.hard_mask)
        if self.drop_hard:
            hard = torch.zeros_like(hard)
            hard_mask = torch.zeros_like(hard_mask)
        return {
            "soft": take(self._soft[s.seg]),
            "hard": hard,
            "soft_mask": take_mask(seg.soft_mask),
            "hard_mask": hard_mask,
            # Elapsed time since onset, in hours: the model should know how
            # much of the rise it has actually seen.
            "lead": torch.tensor(s.lead_s / PEAK_TIME_SCALE_S, dtype=torch.float32),
            "y_log_peak": torch.tensor(s.y_log_peak, dtype=torch.float32),
            "y_time_to_peak": torch.tensor(s.y_time_to_peak, dtype=torch.float32),
            "y_exceeds": torch.tensor(s.y_exceeds, dtype=torch.float32),
            "y_lde": torch.tensor(s.y_lde, dtype=torch.float32),
            "event_id": torch.tensor(s.event_id, dtype=torch.long),
            "lead_s": torch.tensor(s.lead_s, dtype=torch.float32),
            "y_current": torch.tensor(s.y_current, dtype=torch.float32),
        }


@dataclass
class FoldResult:
    encoder: str
    fold: int
    n_train: int
    n_test: int
    metrics: dict


def _loader(dsw: RiseWindows, cfg: Config, shuffle: bool,
            generator: torch.Generator | None = None) -> DataLoader:
    pin = resolve_device(cfg.train.device).type == "cuda"
    cap = cfg.train.max_batches_per_epoch
    if shuffle and cap and len(dsw) > cap * cfg.train.batch_size:
        # Same epoch cap as nowcast training; the sampler takes the seeded
        # generator, so the order-independence of the comparison is kept.
        from torch.utils.data import RandomSampler
        sampler = RandomSampler(dsw, replacement=False,
                                num_samples=cap * cfg.train.batch_size,
                                generator=generator)
        return DataLoader(dsw, batch_size=cfg.train.batch_size, sampler=sampler,
                          num_workers=0, pin_memory=pin, drop_last=False)
    return DataLoader(dsw, batch_size=cfg.train.batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=pin, drop_last=False,
                      generator=generator)


def train_rise(segments, ds, tr_idx, va_idx, cfg, norm, encoder, verbose=False,
               seed: int | None = None, inputs: dict | None = None,
               drop_hard: bool = False):
    """Train one RiseNet.

    ``seed`` resets every random stream immediately before the model is built
    and gives the shuffling its own generator. Without it, an encoder's
    initialisation and batch order depend on how much randomness the encoders
    trained *before* it consumed: measured on this data, the linear encoder's
    fold-0 error moved by 0.069 depending on whether TCN ran first -- 38x the
    0.0018 gap separating the top two architectures. Every encoder in a fold
    now starts from the same state, so the comparison is order-independent.
    """
    if seed is not None:
        set_seed(seed)
    device = resolve_device(cfg.train.device)
    tune_backend(device)
    n_soft = segments[0].soft.shape[1]
    n_hard = segments[0].hard.shape[1]

    model = RiseNet(n_soft, n_hard, cfg.model, encoder=encoder).to(device)
    crit = RiseLoss(cfg.win.quantiles, cfg.train.focal_gamma).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(crit.parameters()),
                            lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    if inputs is None:
        inputs = normalized_inputs(segments, norm)
    tl = _loader(RiseWindows(segments, ds, tr_idx, cfg, norm, inputs, drop_hard),
                 cfg, True, generator=gen)
    vl = (_loader(RiseWindows(segments, ds, va_idx, cfg, norm, inputs, drop_hard), cfg, False)
          if va_idx else None)

    total = max(cfg.train.epochs * max(len(tl), 1), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_warmup(s, total, int(cfg.train.warmup_frac * total)))

    best_state, best_score, patience = None, np.inf, 0
    for epoch in range(cfg.train.epochs):
        model.train()
        for b in tl:
            b = {k: v.to(device) for k, v in b.items()}
            out = model(b["soft"], b["soft_mask"], b["hard"], b["hard_mask"], b["lead"])
            loss, _ = crit(out, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            sched.step()

        if vl is not None:
            p = predict_rise(model, vl, device)
            score = float(np.mean(np.abs(p["peak_q"][:, 1] - p["y_log_peak"])))
        else:
            score = float(loss.detach())

        if score < best_score - 1e-5:
            best_score, patience = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                break
        if verbose:
            print(f"    ep {epoch:3d} val peak-MAE {score:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_rise(model: RiseNet, loader: DataLoader, device) -> dict:
    model.eval()
    acc: dict[str, list] = {}
    for b in loader:
        bb = {k: v.to(device) for k, v in b.items()}
        out = model(bb["soft"], bb["soft_mask"], bb["hard"], bb["hard_mask"], bb["lead"])
        acc.setdefault("peak_q", []).append(out["peak_q"].cpu().numpy())
        acc.setdefault("time_to_peak", []).append(out["time_to_peak"].cpu().numpy())
        acc.setdefault("p_exceeds", []).append(torch.sigmoid(out["exceeds"]).cpu().numpy())
        acc.setdefault("p_lde", []).append(torch.sigmoid(out["lde"]).cpu().numpy())
        for k in ("y_log_peak", "y_time_to_peak", "y_exceeds", "y_lde",
                  "event_id", "lead_s", "y_current"):
            acc.setdefault(k, []).append(b[k].numpy())
    return {k: np.concatenate(v, axis=0) for k, v in acc.items()}


def bootstrap_ci(values: np.ndarray, groups: np.ndarray, stat, n_boot: int = 400,
                 seed: int = 0) -> tuple[float, float, float]:
    """Block bootstrap over **events**, not samples.

    Resampling individual rise samples would treat 40 near-identical points
    from one flare as 40 independent observations and produce absurdly tight
    intervals. Resampling whole events respects the real unit of replication.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    point = stat(np.arange(len(groups)))
    # Sample indices of each group, found once. Searching the whole array per
    # picked group per draw is O(groups x samples x draws): ~1e10 operations
    # per fold for an archive's thousands of flares.
    order = np.argsort(groups, kind="stable")
    uniq, starts = np.unique(groups[order], return_index=True)
    members = np.split(order, starts[1:])
    if len(uniq) < 3:
        return point, float("nan"), float("nan")
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[j] for j in pick])
        try:
            draws.append(stat(idx))
        except Exception:
            continue
    if not draws:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def score_rise(pred: dict, thr_exceed: float = 0.5, thr_lde: float = 0.5,
               clim_log_peak: float | None = None,
               clim_time_to_peak: float | None = None) -> dict:
    """Score one test fold.

    Every error is reported next to the error of two reference forecasts, and
    a raw MAE on its own should never be quoted:

    * **current level** -- predict that the flare peaks at the flux it has
      already reached. A peak cannot be lower than this, so it is the natural
      "no forecast" answer during a rise.
    * **climatology** -- predict the mean peak (and mean time-to-peak) of the
      *training* events. Fitted on train only; a climatology computed from
      the test fold would be using the answer.
    """
    g = pred["event_id"]
    y = pred["y_log_peak"]
    out: dict = {}

    def peak_mae(idx):
        return float(np.mean(np.abs(pred["peak_q"][idx, 1] - y[idx])))

    def ttp_mae(idx):
        return float(np.mean(np.abs(
            pred["time_to_peak"][idx] - pred["y_time_to_peak"][idx])) * 60.0)

    p, lo, hi = bootstrap_ci(None, g, peak_mae)
    out["peak_log_MAE"] = {"value": p, "ci_lo": lo, "ci_hi": hi}
    p, lo, hi = bootstrap_ci(None, g, ttp_mae)
    out["time_to_peak_MAE_min"] = {"value": p, "ci_lo": lo, "ci_hi": hi}

    model_mae = out["peak_log_MAE"]["value"]
    refs: dict = {}
    if "y_current" in pred:
        refs["current"] = float(np.mean(np.abs(pred["y_current"] - y)))
    if clim_log_peak is not None:
        refs["climatology"] = float(np.mean(np.abs(clim_log_peak - y)))
    for name, ref in refs.items():
        out[f"peak_log_MAE_{name}"] = ref
        out[f"peak_skill_vs_{name}"] = (1.0 - model_mae / ref) if ref > 0 else float("nan")

    if clim_time_to_peak is not None:
        ref = float(np.mean(np.abs(clim_time_to_peak - pred["y_time_to_peak"])) * 60.0)
        out["time_to_peak_MAE_min_climatology"] = ref
        out["ttp_skill_vs_climatology"] = (
            1.0 - out["time_to_peak_MAE_min"]["value"] / ref if ref > 0 else float("nan"))

    for key, prob, truth, thr in (("exceeds", pred["p_exceeds"], pred["y_exceeds"], thr_exceed),
                                  ("lde", pred["p_lde"], pred["y_lde"], thr_lde)):
        pos_events = int(len(np.unique(g[truth > 0.5])))
        neg_events = int(len(np.unique(g[truth <= 0.5])))
        if len(np.unique(truth)) > 1:
            s = skill_scores(truth, prob >= thr)

            def auc_stat(idx, _p=prob, _t=truth):
                if len(np.unique(_t[idx])) < 2:
                    raise ValueError("degenerate")
                return roc_auc(_t[idx], _p[idx])
            a, alo, ahi = bootstrap_ci(None, g, auc_stat)
            out[key] = {"TSS": s["TSS"], "HSS": s["HSS"], "POD": s["POD"],
                        "FAR": s["FAR"], "base_rate": s["base_rate"],
                        "AUC": a, "AUC_ci_lo": alo, "AUC_ci_hi": ahi,
                        "positive_events": pos_events, "negative_events": neg_events}
            # Reference: rank samples by the flux already reached. Whether a
            # flare *will* exceed a threshold is dominated by how bright it
            # already is, so a classifier that cannot out-rank the current
            # level has learned nothing beyond it. Measured on 2026-09-10 the
            # current level alone scored AUC 0.998, above every model.
            if "y_current" in pred:
                ref = roc_auc(truth, pred["y_current"])
                out[key]["AUC_current_level"] = ref
                out[key]["AUC_gain_vs_current"] = a - ref
        else:
            out[key] = {"note": "single class in this fold",
                        "positive_events": pos_events, "negative_events": neg_events}

    # Skill as a function of how much of the rise has been seen: the operative
    # question is how early a useful magnitude estimate becomes available.
    lead = pred["lead_s"]
    bins = [(0, 120), (120, 300), (300, 600), (600, 1e9)]
    by_lead = {}
    for a, b in bins:
        m = (lead >= a) & (lead < b)
        if m.sum() >= 5:
            by_lead[f"{int(a/60)}-{int(b/60) if b < 1e9 else 'inf'}min"] = {
                "n": int(m.sum()),
                "peak_log_MAE": float(np.mean(np.abs(
                    pred["peak_q"][m, 1] - pred["y_log_peak"][m]))),
            }
    out["by_lead_time"] = by_lead
    return out


def _train_climatology(ds: RiseDataset, idx: list[int]) -> tuple[float, float]:
    """Mean log-peak and mean time-to-peak over the training events.

    Weighted per **event**, not per sample: a slow two-hour rise contributes
    dozens of samples and would otherwise define "typical" on its own.
    Time-to-peak is averaged over samples within each event first, because it
    changes along the rise while the peak does not.
    """
    peaks: dict[int, float] = {}
    ttps: dict[int, list[float]] = {}
    for i in idx:
        s = ds.samples[i]
        peaks[s.event_id] = s.y_log_peak
        ttps.setdefault(s.event_id, []).append(s.y_time_to_peak)
    if not peaks:
        return float("nan"), float("nan")
    return (float(np.mean(list(peaks.values()))),
            float(np.mean([np.mean(v) for v in ttps.values()])))


def grouped_cv(prep: Prepared, cfg: Config, encoders=("tcn",), n_folds: int = 4,
               verbose: bool = True, ds: RiseDataset | None = None,
               event_subset: set[int] | None = None, drop_hard: bool = False,
               keep_predictions: bool = False, train_fraction: float = 1.0) -> dict:
    """Rolling-origin CV over events, for each encoder.

    Fold k trains on the earliest events and tests on the next block, always
    forward in time -- no fold ever trains on a flare that happens after its
    test flares.

    ``event_subset`` restricts the CV to those events; folds are a function of
    the subset only, so two calls with the same subset share identical folds
    and seeds -- which is what makes ``fusion_ablation`` a paired comparison.

    ``train_fraction`` < 1 keeps only the most recent share of each fold's
    training events, leaving the test blocks untouched: that is a learning
    curve ("would more history help?"), not a different experiment.
    """
    set_seed(cfg.train.seed)
    set_threads(getattr(cfg.train, "torch_threads", 0))
    device = resolve_device(cfg.train.device)

    # An archive holds thousands of flares; sampling every 20 s of every rise
    # gives hundreds of thousands of nearly identical samples. One sample a
    # minute still covers each rise several times.
    if ds is None:
        ds = build_rise_dataset(prep.segments, cfg,
                                stride_s=60.0 if prep.archive else None)
        if verbose:
            print(summarise(ds, cfg.pre.dt_seconds))
    if len(ds) < 20:
        return {"error": "too few rise-phase samples to cross-validate",
                "n_samples": len(ds)}

    ev_time: dict[int, float] = {}
    for s in ds.samples:
        if event_subset is None or s.event_id in event_subset:
            ev_time.setdefault(s.event_id, s.t_unix)
    events = sorted(ev_time, key=lambda e: ev_time[e])
    n_ev = len(events)
    if n_ev < n_folds + 2:
        n_folds = max(2, n_ev - 2)

    results: list[FoldResult] = []
    inputs = normalized_inputs(prep.segments, prep.norm)
    failures: list[dict] = []
    predictions: list[dict] = []
    block = max(n_ev // (n_folds + 1), 1)

    for enc in encoders:
        if verbose:
            print(f"\n--- encoder: {enc} ---")
        for fold in range(n_folds):
            train_end = block * (fold + 1)
            test_end = min(train_end + block, n_ev)
            if test_end <= train_end:
                continue
            keep = max(2, int(round(train_fraction * train_end)))
            tr_ev = set(events[max(0, train_end - keep):train_end])
            te_ev = set(events[train_end:test_end])
            # Hold out the most recent training events for early stopping: the
            # last 10%, at least one. With a dozen flares that is one event;
            # with thousands, a single event would make early stopping a coin
            # flip on whichever flare happened to come last.
            n_va = max(1, int(round(0.1 * train_end))) if train_end >= 2 else 0
            va_ev = set(events[train_end - n_va:train_end])
            tr_ev -= va_ev

            tr = [i for i, s in enumerate(ds.samples) if s.event_id in tr_ev]
            va = [i for i, s in enumerate(ds.samples) if s.event_id in va_ev]
            te = [i for i, s in enumerate(ds.samples) if s.event_id in te_ev]
            if not tr or not te:
                continue

            t0 = time.time()
            # One encoder running out of memory must not discard the folds
            # that already succeeded, nor the encoders queued behind it.
            try:
                model = train_rise(prep.segments, ds, tr, va, cfg, prep.norm, enc,
                                   seed=cfg.train.seed + 1000 * fold,
                                   inputs=inputs, drop_hard=drop_hard)
                pred = predict_rise(
                    model, _loader(RiseWindows(prep.segments, ds, te, cfg, prep.norm,
                                               inputs, drop_hard),
                                   cfg, False), device)
                clim_peak, clim_ttp = _train_climatology(ds, tr + va)
                m = score_rise(pred, clim_log_peak=clim_peak,
                               clim_time_to_peak=clim_ttp)
                if keep_predictions:
                    predictions.append({"encoder": enc, "fold": fold, "index": np.asarray(te),
                                        **{k: pred[k] for k in ("peak_q", "y_log_peak",
                                                                "event_id", "y_current",
                                                                "p_exceeds", "y_exceeds",
                                                                "time_to_peak",
                                                                "y_time_to_peak")}})
            except torch.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                failures.append({"encoder": enc, "fold": fold, "error": "CUDA OOM"})
                if verbose:
                    print(f"  fold {fold}: OUT OF MEMORY -- skipped "
                          f"(try a smaller --batch-size, or fewer ssm_layers)")
                continue
            except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
                failures.append({"encoder": enc, "fold": fold,
                                 "error": f"{type(exc).__name__}: {exc}"})
                if verbose:
                    print(f"  fold {fold}: FAILED ({type(exc).__name__}: {exc})")
                continue
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            m["seconds"] = round(time.time() - t0, 1)
            results.append(FoldResult(enc, fold, len(tr), len(te), m))
            if verbose:
                print(f"  fold {fold}: train {len(tr):4d} / test {len(te):4d} "
                      f"| peak MAE {m['peak_log_MAE']['value']:.4f} "
                      f"| ttp MAE {m['time_to_peak_MAE_min']['value']:.1f} min "
                      f"| {m['seconds']}s")

    out = {
        "n_events": n_ev,
        "n_samples": sum(1 for s in ds.samples if s.event_id in ev_time),
        "threshold_rate": ds.threshold_rate,
        "folds": [
            {"encoder": r.encoder, "fold": r.fold, "n_train": r.n_train,
             "n_test": r.n_test, **r.metrics} for r in results
        ],
        "summary": _summarise_folds(results),
        "failures": failures,
    }
    if keep_predictions:
        out["_predictions"] = predictions
    return out


def rise_hard_coverage(segments: list[Segment], ds: RiseDataset) -> dict[int, float]:
    """Per event: the fraction of its sampled rise that HEL1OS observed."""
    dt = None
    num: dict[int, float] = {}
    den: dict[int, int] = {}
    for s in ds.samples:
        seg = segments[s.seg]
        if dt is None:
            dt = float(seg.time_unix[1] - seg.time_unix[0]) if len(seg) > 1 else 1.0
        onset = max(s.end - int(round(s.lead_s / dt)), 0)
        m = seg.hard_mask[onset:s.end]
        num[s.event_id] = num.get(s.event_id, 0.0) + float(m.sum())
        den[s.event_id] = den.get(s.event_id, 0) + int(m.size)
    return {e: num[e] / max(den[e], 1) for e in num}


def fusion_ablation(prep: Prepared, cfg: Config, encoder: str = "tcn",
                    n_folds: int = 3, min_hard_coverage: float = 0.5,
                    verbose: bool = True) -> dict:
    """Does HEL1OS add forecasting skill? A paired, event-grouped test.

    Restricted to flares whose rise HEL1OS actually watched, the same encoder
    is cross-validated twice over identical folds and seeds: once with the
    hard X-ray input, once with it blanked. The answer is the per-event
    difference in peak error, bootstrapped over events -- not two MAEs quoted
    side by side, whose intervals would overlap even when one arm is
    consistently better on the same flares.
    """
    ds = build_rise_dataset(prep.segments, cfg,
                            stride_s=60.0 if prep.archive else None)
    cov = rise_hard_coverage(prep.segments, ds)
    subset = {e for e, c in cov.items() if c >= min_hard_coverage}
    if verbose:
        print(f"fusion ablation: {len(subset)} of {len(cov)} flares have HEL1OS "
              f"covering >= {min_hard_coverage:.0%} of the rise")
    if len(subset) < n_folds + 2:
        return {"error": "too few flares with simultaneous HEL1OS coverage",
                "n_events": len(subset)}

    arms = {}
    for name, drop in (("soft+hard", False), ("soft-only", True)):
        if verbose:
            print(f"\n=== arm: {name} ===")
        arms[name] = grouped_cv(prep, cfg, encoders=(encoder,), n_folds=n_folds,
                                verbose=verbose, ds=ds, event_subset=subset,
                                drop_hard=drop, keep_predictions=True)

    def by_key(res):
        return {(p["fold"], int(i)): (float(q), float(y), int(g), float(pe), float(ye))
                for p in res.get("_predictions", [])
                for i, q, y, g, pe, ye in zip(p["index"], p["peak_q"][:, 1], p["y_log_peak"],
                                              p["event_id"], p["p_exceeds"], p["y_exceeds"])}

    a, b = by_key(arms["soft+hard"]), by_key(arms["soft-only"])
    common = sorted(set(a) & set(b))
    out: dict = {"encoder": encoder, "n_folds": n_folds,
                 "min_hard_coverage": min_hard_coverage,
                 "n_events_eligible": len(subset), "n_paired_samples": len(common)}
    if common:
        err_f = np.array([abs(a[k][0] - a[k][1]) for k in common])
        err_s = np.array([abs(b[k][0] - b[k][1]) for k in common])
        groups = np.array([a[k][2] for k in common])
        diff = err_s - err_f  # > 0: hard X-rays reduce the peak error

        def mean_diff(idx):
            return float(np.mean(diff[idx]))

        p, lo, hi = bootstrap_ci(None, groups, mean_diff, n_boot=1000)
        per_event = {}
        for g, d in zip(groups, diff):
            per_event.setdefault(int(g), []).append(d)
        ev_means = np.array([np.mean(v) for v in per_event.values()])
        out.update({
            "peak_log_MAE_soft_hard": float(err_f.mean()),
            "peak_log_MAE_soft_only": float(err_s.mean()),
            "peak_MAE_reduction_from_hel1os": {"value": p, "ci_lo": lo, "ci_hi": hi},
            "relative_reduction": float(p / err_s.mean()) if err_s.mean() > 0 else float("nan"),
            "events_improved_fraction": float(np.mean(ev_means > 0)),
            "n_test_events": int(len(per_event)),
        })
        ye = np.array([a[k][4] for k in common])
        if len(np.unique(ye)) > 1:
            out["exceeds_AUC_soft_hard"] = roc_auc(ye, np.array([a[k][3] for k in common]))
            out["exceeds_AUC_soft_only"] = roc_auc(ye, np.array([b[k][3] for k in common]))
    for name, res in arms.items():
        res.pop("_predictions", None)
        out[name] = res
    if verbose:
        print_fusion(out)
    return out


def print_fusion(res: dict) -> None:
    print("\n" + "=" * 74)
    print("FUSION ABLATION - does HEL1OS add skill? (paired, grouped by flare)")
    print("=" * 74)
    if "error" in res:
        print(f"{res['error']} (n={res.get('n_events')})")
        return
    r = res.get("peak_MAE_reduction_from_hel1os")
    print(f"{res['n_events_eligible']} flares with HEL1OS on the rise; "
          f"{res.get('n_test_events', 0)} tested out of sample, "
          f"{res['n_paired_samples']} paired samples, encoder {res['encoder']}")
    if r:
        print(f"peak log-MAE  soft+hard {res['peak_log_MAE_soft_hard']:.4f}   "
              f"soft-only {res['peak_log_MAE_soft_only']:.4f}")
        print(f"reduction from adding HEL1OS: {r['value']:+.4f} "
              f"[95% CI {r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
              f"({100 * res['relative_reduction']:+.1f}%), "
              f"better on {100 * res['events_improved_fraction']:.0f}% of flares")
        if r["ci_lo"] == r["ci_lo"]:
            verdict = ("HEL1OS measurably helps" if r["ci_lo"] > 0 else
                       "HEL1OS measurably hurts" if r["ci_hi"] < 0 else
                       "no detectable effect at this sample size")
            print(f"  -> {verdict}")
    if "exceeds_AUC_soft_hard" in res:
        print(f"exceedance AUC  soft+hard {res['exceeds_AUC_soft_hard']:.3f}   "
              f"soft-only {res['exceeds_AUC_soft_only']:.3f}")


def _summarise_folds(results: list[FoldResult]) -> dict:
    out: dict = {}
    for enc in sorted({r.encoder for r in results}):
        rows = [r for r in results if r.encoder == enc]
        peak = [r.metrics["peak_log_MAE"]["value"] for r in rows]
        ttp = [r.metrics["time_to_peak_MAE_min"]["value"] for r in rows]
        exc = [r.metrics["exceeds"].get("AUC") for r in rows
               if isinstance(r.metrics.get("exceeds"), dict)
               and r.metrics["exceeds"].get("AUC") is not None]
        exc = [v for v in exc if v == v]

        def mean_of(key, rows=rows):
            vals = [r.metrics.get(key) for r in rows]
            vals = [v for v in vals if v is not None and v == v]
            return float(np.mean(vals)) if vals else float("nan")

        out[enc] = {
            "n_folds": len(rows),
            "peak_log_MAE_mean": float(np.mean(peak)) if peak else float("nan"),
            "peak_log_MAE_std": float(np.std(peak)) if peak else float("nan"),
            "peak_log_MAE_current_mean": mean_of("peak_log_MAE_current"),
            "peak_log_MAE_climatology_mean": mean_of("peak_log_MAE_climatology"),
            "peak_skill_vs_current_mean": mean_of("peak_skill_vs_current"),
            "peak_skill_vs_climatology_mean": mean_of("peak_skill_vs_climatology"),
            "time_to_peak_MAE_min_mean": float(np.mean(ttp)) if ttp else float("nan"),
            "ttp_skill_vs_climatology_mean": mean_of("ttp_skill_vs_climatology"),
            "exceeds_AUC_mean": float(np.mean(exc)) if exc else float("nan"),
            **_class_summary(rows, "exceeds"),
            **_class_summary(rows, "lde"),
            "total_seconds": float(sum(r.metrics["seconds"] for r in rows)),
        }
    return out


def _class_summary(rows: list[FoldResult], key: str) -> dict:
    """How much evidence a classification AUC actually rests on.

    An AUC is only as good as the number of *events* in its minority class.
    With one positive flare in one fold it is a statement about one flare, so
    the count travels with the score, along with the current-level reference.
    """
    defined = [r.metrics[key] for r in rows
               if isinstance(r.metrics.get(key), dict) and "AUC" in r.metrics[key]]
    pos = sum(r.metrics[key].get("positive_events", 0) for r in rows
              if isinstance(r.metrics.get(key), dict))

    def m(field):
        vals = [d.get(field) for d in defined]
        vals = [v for v in vals if v is not None and v == v]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        f"{key}_folds_with_both_classes": len(defined),
        f"{key}_positive_events_total": int(pos),
        f"{key}_AUC_current_level_mean": m("AUC_current_level"),
        f"{key}_AUC_gain_vs_current_mean": m("AUC_gain_vs_current"),
    }


def print_comparison(res: dict) -> None:
    if "error" in res:
        print(f"\n{res['error']} (n={res.get('n_samples')})")
        return
    print("\n" + "=" * 74)
    print("RISE-PHASE FORECASTING - grouped rolling-origin CV over events")
    print("=" * 74)
    print(f"{res['n_events']} events, {res['n_samples']} rise samples, "
          f"large-flare threshold {format_threshold(res['threshold_rate'])}")
    print(f"\n{'encoder':>12}  {'folds':>5}  {'peak logMAE':>16}  {'skill/cur':>9}  "
          f"{'skill/clim':>10}  {'ttp min':>7}  {'ttp skill':>9}  {'exc AUC':>7}  {'sec':>5}")
    for enc, s in sorted(res["summary"].items(),
                         key=lambda kv: kv[1]["peak_log_MAE_mean"]):
        print(f"{enc:>12}  {s['n_folds']:5d}  "
              f"{s['peak_log_MAE_mean']:7.4f} +/-{s['peak_log_MAE_std']:6.4f}  "
              f"{s['peak_skill_vs_current_mean']:9.3f}  "
              f"{s['peak_skill_vs_climatology_mean']:10.3f}  "
              f"{s['time_to_peak_MAE_min_mean']:7.1f}  "
              f"{s['ttp_skill_vs_climatology_mean']:9.3f}  "
              f"{s['exceeds_AUC_mean']:7.3f}  {s['total_seconds']:5.0f}")
    any_s = next(iter(res["summary"].values()), {})
    print(f"\nreference MAE: current level {any_s.get('peak_log_MAE_current_mean', float('nan')):.4f}, "
          f"training climatology {any_s.get('peak_log_MAE_climatology_mean', float('nan')):.4f}")
    print("skill > 0 beats that reference; skill <= 0 means the model adds nothing over it.")
    exc_folds = any_s.get("exceeds_folds_with_both_classes", 0)
    exc_pos = any_s.get("exceeds_positive_events_total", 0)
    print(f"\nexceedance AUC rests on {exc_folds} fold(s) with both classes and "
          f"{exc_pos} positive event(s) in total; current flux alone scores AUC "
          f"{any_s.get('exceeds_AUC_current_level_mean', float('nan')):.3f}.")
    if exc_pos < 3:
        print("  -> too few large flares for this AUC to mean anything yet.")
    print("\nSpread across folds is the number that matters: with a handful of")
    print("events, a mean without its fold-to-fold scatter is not a result.")


def save(res: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
