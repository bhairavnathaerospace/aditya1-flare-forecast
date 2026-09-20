"""Forward (prospective) testing: freeze a model, forecast new data, score later.

A held-out test period is still chosen after the data exist, and every design
decision made while looking at test scores leaks into it. The only evaluation
nothing can have been tuned on is data that did not exist when the model was
fixed. Three steps:

``freeze``   Snapshot everything the forecast depends on into
             ``<out>/forward/<name>/``: weights, the input normaliser, the
             decision thresholds chosen on validation, and training-set base
             rates for the reference forecasts. Record the data cutoff (the last
             observed bin at freeze time) and a hash of the weights.

             The normaliser is the subtle one. ``prepare`` refits it from the
             training split of whatever data is on disk; once new days arrive
             the split moves, the statistics change, and a "frozen" model would
             silently be fed differently scaled inputs.

``predict``  Forecast every origin after the cutoff not yet in the append-only
             ``predictions.csv``, using only the trailing window -- the same
             path a live feed would take. Re-run whenever new data is
             downloaded; existing rows are never rewritten.

``score``    Once enough later data exists to know what happened, compare each
             forecast with the truth, next to the training-climatology and
             persistence references, at the thresholds frozen in step 1, with
             day-block bootstrap intervals. Optionally also against an
             independent flare list (GOES), because the default truth comes from
             this project's own SoLEXS flare detector.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from dataclasses import asdict
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .metrics import brier_score, reliability, roc_auc, skill_scores
from .pipeline import cache_dir_for, prepare, resolve_device
from .preprocess.cache import build_cache, index_sources, load_cached, mask_intervals
from .preprocess.dataset import Normalizer, Segment, build_segments_from_raw, build_targets

FROZEN = "frozen.pt"
MANIFEST = "manifest.json"
PREDICTIONS = "predictions.csv"


def _utc(u: float) -> str:
    return datetime.fromtimestamp(float(u), UTC).strftime("%Y-%m-%d %H:%M:%S")


def forward_dir(cfg: Config, name: str) -> Path:
    return Path(cfg.out_dir) / "forward" / name


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------

def freeze(cfg: Config, checkpoint: Path, name: str, verbose: bool = True) -> Path:
    from .pipeline import make_loaders
    from .train import collect_predictions, load_model
    from .metrics import best_threshold

    out = forward_dir(cfg, name)
    if (out / FROZEN).exists():
        raise FileExistsError(f"{out / FROZEN} exists; a frozen model is never overwritten. "
                              "Choose a new --name.")
    checkpoint = Path(checkpoint)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)

    prep = prepare(cfg, verbose=verbose)

    # The normaliser must be the one the model was trained with.
    if "norm" in ck:
        norm = Normalizer(*(np.asarray(ck["norm"][k]) for k in
                            ("mean_soft", "std_soft", "mean_hard", "std_hard")))
        norm_source = "checkpoint"
    else:
        # Older checkpoints did not store it. Refitting is exact only if the
        # data -- hence the split -- is unchanged since training; check that
        # against the split sizes training recorded.
        meta_path = Path(cfg.out_dir) / "reports" / "data_meta.json"
        if not meta_path.exists():
            raise RuntimeError("checkpoint has no normaliser and no data_meta.json to "
                               "verify a refit against; retrain before freezing")
        trained = json.loads(meta_path.read_text("utf-8"))
        for key in ("n_windows", "n_train", "n_val", "n_test"):
            if trained.get(key) != prep.meta.get(key):
                raise RuntimeError(
                    f"data changed since training ({key}: {trained.get(key)} at training, "
                    f"{prep.meta.get(key)} now), so the normaliser cannot be reproduced. "
                    "Freeze with the data as it was at training, or retrain.")
        norm = prep.norm
        norm_source = "refit (split verified identical to training)"

    device = resolve_device(cfg.train.device)
    model = load_model(checkpoint, cfg, device)
    _, va, _, _ = make_loaders(prep, cfg)
    val = collect_predictions(model, va, device)

    # Calibration and operating points, both from VALIDATION, frozen now. The
    # thresholds apply to the calibrated probabilities the forecasts will carry.
    from . import probcal

    n_occ = len(cfg.win.occurrence_horizons_s)
    m = val["y_nowcast_mask"] > 0
    calibration = {"in_flare": probcal.fit(val["p_inflare"][m], val["y_in_flare"][m]),
                   "occurrence": []}
    p_in = probcal.apply(calibration["in_flare"], val["p_inflare"])
    p_occ = np.zeros_like(val["p_occurrence"], dtype=np.float64)
    for h in range(n_occ):
        mm = val["y_occurrence_mask"][:, h] > 0
        calibration["occurrence"].append(probcal.fit(val["p_occurrence"][mm, h], val["y_occurrence"][mm, h]))
        p_occ[:, h] = probcal.apply(calibration["occurrence"][h], val["p_occurrence"][:, h])
    thresholds = {"in_flare": 0.5, "occurrence": [0.5] * n_occ}
    if m.sum() and len(np.unique(val["y_in_flare"][m])) > 1:
        thresholds["in_flare"] = float(best_threshold(val["y_in_flare"][m], p_in[m], "TSS")[0])
    for h in range(n_occ):
        mm = val["y_occurrence_mask"][:, h] > 0
        if mm.sum() and len(np.unique(val["y_occurrence"][mm, h])) > 1:
            thresholds["occurrence"][h] = float(best_threshold(
                val["y_occurrence"][mm, h], p_occ[mm, h], "TSS")[0])

    climatology = _train_climatology(prep, cfg)
    cutoff = max(float(s.time_unix[np.flatnonzero(np.maximum(s.soft_mask, s.hard_mask))[-1]])
                 for s in prep.segments if np.maximum(s.soft_mask, s.hard_mask).any())

    out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    frozen = {
        "model": ck["model"], "anchor": ck.get("anchor"),
        "n_soft": ck["n_soft"], "n_hard": ck["n_hard"],
        "n_clock": ck["n_clock"],
        "norm": {k: np.asarray(v) for k, v in asdict(norm).items()},
        "config": _cfg_dict(cfg),
        "thresholds": thresholds, "climatology": climatology, "calibration": calibration,
        "data_cutoff_unix": cutoff,
    }
    torch.save(frozen, out / FROZEN)
    manifest = {
        "name": name,
        "frozen_at_utc": _utc(time.time()),
        "frozen_at_unix": time.time(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha,
        "checkpoint_epoch": ck.get("epoch"),
        "data_cutoff_utc": _utc(cutoff),
        "data_cutoff_unix": cutoff,
        "normaliser": norm_source,
        "thresholds": thresholds,
        "calibrated_probabilities": True,
        "training_climatology": climatology,
        "train_end_utc": _utc(prep.meta["split_dates"]["train_end"])
        if prep.meta.get("split_dates") else None,
        "n_sources_at_freeze": len(prep.sources),
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if verbose:
        print(f"\nfrozen '{name}': weights sha256 {sha[:16]}..., data cutoff {_utc(cutoff)} UTC")
        print(f"  thresholds (from validation): in-flare {thresholds['in_flare']:.3f}, "
              f"occurrence {[round(t, 3) for t in thresholds['occurrence']]}")
        print(f"  normaliser: {norm_source}")
        print(f"  -> {out}")
    return out


def _cfg_dict(cfg: Config) -> dict:
    return {"pre": vars(cfg.pre).copy(), "win": vars(cfg.win).copy(),
            "model": vars(cfg.model).copy()}


def _train_climatology(prep, cfg: Config) -> dict:
    """Base rates over the training period: the climatology a forecaster could
    have known at freeze time. Never estimated from the forward period.

    Uses the training windows *before* quiet-window thinning. Thinning keeps
    every flare window and drops quiet ones, so the thinned set overstates how
    often flares happen -- a reference built on it would be a worse forecast
    than true climatology and flatter every Brier skill score.
    """
    from .preprocess.dataset import chronological_split

    tr = chronological_split(prep.windows, cfg, mode=prep.meta["split_mode"])[0]
    by_seg: dict[int, list[int]] = {}
    for i in tr:
        w = prep.windows[i]
        by_seg.setdefault(w.seg, []).append(w.end - 1)
    in_flare, occ, flux = [], [], []
    for si, ends in by_seg.items():
        seg = prep.segments[si]
        t = build_targets(seg, cfg)
        e = np.asarray(ends)
        ok = t["nowcast_mask"][e] > 0
        in_flare.append(t["in_flare"][e][ok])
        flux.append(t["nowcast"][e][ok])
        om = t["occurrence_mask"][e] > 0
        occ.append(np.where(om, t["occurrence"][e], np.nan))
    occ_all = np.concatenate(occ)
    return {
        "in_flare_rate": float(np.mean(np.concatenate(in_flare))),
        "occurrence_rate": [float(np.nanmean(occ_all[:, h])) for h in range(occ_all.shape[1])],
        "mean_log_flux": float(np.mean(np.concatenate(flux))),
    }


def load_frozen(path: Path, cfg: Config):
    from .models.net import FluxAnchor, SolexHelNet

    f = torch.load(Path(path) / FROZEN, map_location="cpu", weights_only=False)
    # Models frozen before the energy scale was selectable were trained on the
    # legacy scale; feeding them published-scale features would be wrong.
    f["config"]["pre"].setdefault("solexs_energy_scale", "legacy_linear")
    f["config"]["pre"].setdefault("label_source", "solexs")
    for part in ("pre", "win", "model"):
        for k, v in f["config"][part].items():
            setattr(getattr(cfg, part), k, v)
    device = resolve_device(cfg.train.device)
    a = f.get("anchor")
    model = SolexHelNet(f["n_soft"], f["n_hard"], f["n_clock"], cfg.model, cfg.win,
                        anchor=FluxAnchor(**a) if a else None).to(device)
    model.load_state_dict(f["model"])
    model.eval()
    norm = Normalizer(*(np.asarray(f["norm"][k]) for k in
                        ("mean_soft", "std_soft", "mean_hard", "std_hard")))
    return f, model, norm, device


def _segments(cfg: Config, verbose: bool) -> tuple[list[Segment], list]:
    """Current data -> segments, without any split or refit."""
    sources = index_sources([Path(cfg.data_root)])
    entries = build_cache(sources, cfg.pre, cache_dir_for(cfg),
                          workers=cfg.pre.cache_workers if len(sources) >= 8 else 1,
                          verbose=verbose)
    soft = load_cached(entries, cache_dir_for(cfg), "solexs")
    hard = load_cached(entries, cache_dir_for(cfg), "hel1os")
    if cfg.pre.exclude_intervals and Path(cfg.pre.exclude_intervals).exists():
        mask_intervals(soft, cfg.pre.exclude_intervals)
    segments, _ = build_segments_from_raw(soft, hard, cfg)
    return segments, sources


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_predict(cfg: Config, name: str, stride_s: float = 60.0,
                    batch_size: int = 256, verbose: bool = True) -> dict:
    out = forward_dir(cfg, name)
    frozen, model, norm, device = load_frozen(out, cfg)
    manifest = json.loads((out / MANIFEST).read_text("utf-8"))
    cutoff = float(frozen["data_cutoff_unix"])
    L = cfg.steps_per_window
    dt = cfg.pre.dt_seconds
    stride = max(int(stride_s / dt), 1)

    segments, sources = _segments(cfg, verbose)
    pred_path = out / PREDICTIONS
    done: set[float] = set()
    if pred_path.exists():
        with pred_path.open(encoding="utf-8") as fh:
            done = {round(float(r["origin_unix"]), 3) for r in csv.DictReader(fh)}

    issued = _utc(time.time())
    fields = _prediction_fields(cfg)
    new_rows: list[dict] = []
    for seg in segments:
        if len(seg) < L or seg.time_unix[-1] <= cutoff:
            continue
        avail = np.maximum(seg.soft_mask, seg.hard_mask)
        csum = np.concatenate([[0.0], np.cumsum(avail)])
        # Origins on a fixed UTC stride, so re-runs land on the same instants.
        ends = np.arange(L, len(seg) + 1)
        origin = seg.time_unix[ends - 1]
        keep = ((origin > cutoff)
                & (np.rint(origin / dt).astype(np.int64) % stride == 0)
                & (avail[ends - 1] > 0)
                & ((csum[ends] - csum[ends - L]) / L >= cfg.win.min_observed_fraction))
        ends = [int(e) for e, o in zip(ends[keep], origin[keep]) if round(float(o), 3) not in done]
        if not ends:
            continue
        soft_n = np.nan_to_num(norm.apply_soft(seg.soft), nan=0.0, posinf=0.0,
                               neginf=0.0).astype(np.float32)
        hard_n = np.nan_to_num(norm.apply_hard(seg.hard), nan=0.0, posinf=0.0,
                               neginf=0.0).astype(np.float32)
        for b0 in range(0, len(ends), batch_size):
            batch = ends[b0:b0 + batch_size]
            sl = [slice(e - L, e) for e in batch]

            def stack(a, _sl=sl):
                return torch.from_numpy(np.stack([a[s] for s in _sl]).astype(np.float32)).to(device)

            o = model(stack(soft_n), stack(seg.soft_mask), stack(hard_n),
                      stack(seg.hard_mask), stack(seg.clock))
            p_now = torch.sigmoid(o["in_flare"]).cpu().numpy()
            p_occ = torch.sigmoid(o["occurrence"]).cpu().numpy()
            cal = frozen.get("calibration")
            if cal:                     # frozen since calibration existed
                from . import probcal
                p_now = probcal.apply(cal["in_flare"], p_now)
                p_occ = np.column_stack([probcal.apply(cal["occurrence"][h], p_occ[:, h])
                                         for h in range(p_occ.shape[1])])
            now = o["nowcast"].cpu().numpy()
            fc = o["forecast"].cpu().numpy()
            for k, e in enumerate(batch):
                row = {
                    "origin_utc": _utc(seg.time_unix[e - 1]),
                    "origin_unix": f"{seg.time_unix[e - 1]:.3f}",
                    "issued_utc": issued,
                    "model_sha256": manifest["checkpoint_sha256"][:16],
                    "soft_available": int(seg.soft_mask[e - 1] > 0),
                    "hard_available": int(seg.hard_mask[e - 1] > 0),
                    # Known at the origin: the persistence reference.
                    "current_log_flux": f"{seg.log_flux[e - 1]:.5f}" if seg.soft_mask[e - 1] > 0 else "",
                    "p_flare_now": f"{p_now[k]:.5f}",
                    "nowcast_log_flux": f"{now[k]:.5f}",
                }
                for h, hs in enumerate(cfg.win.occurrence_horizons_s):
                    row[f"p_flare_within_{int(hs / 60)}min"] = f"{p_occ[k, h]:.5f}"
                for h, hs in enumerate(cfg.win.forecast_horizons_s):
                    for j, q in enumerate(cfg.win.quantiles):
                        row[f"fcst_{int(hs / 60)}min_q{int(round(q * 100))}"] = f"{fc[k, h, j]:.5f}"
                new_rows.append(row)

    new_rows.sort(key=lambda r: float(r["origin_unix"]))
    write_header = not pred_path.exists()
    if new_rows:
        with pred_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if write_header:
                w.writeheader()
            w.writerows(new_rows)

    frozen_at = float(manifest["frozen_at_unix"])
    arrived_after = sum(1 for s in sources if Path(s.path).exists()
                        and _newest_mtime(Path(s.path)) > frozen_at)
    run = {
        "issued_utc": issued,
        "new_forecasts": len(new_rows),
        "total_forecasts": len(done) + len(new_rows),
        "first_new_origin_utc": new_rows[0]["origin_utc"] if new_rows else None,
        "last_new_origin_utc": new_rows[-1]["origin_utc"] if new_rows else None,
        "source_files_modified_after_freeze": arrived_after,
    }
    with (out / "predict_runs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(run) + "\n")
    if verbose:
        print(f"\n{len(new_rows)} new forecasts "
              + (f"({run['first_new_origin_utc']} -> {run['last_new_origin_utc']} UTC)"
                 if new_rows else f"- no data after the cutoff {_utc(cutoff)} UTC yet")
              + f"; {run['total_forecasts']} in {pred_path}")
        print(f"source files modified after the freeze: {arrived_after}")
    return run


def _prediction_fields(cfg: Config) -> list[str]:
    f = ["origin_utc", "origin_unix", "issued_utc", "model_sha256", "soft_available",
         "hard_available", "current_log_flux", "p_flare_now", "nowcast_log_flux"]
    f += [f"p_flare_within_{int(hs / 60)}min" for hs in cfg.win.occurrence_horizons_s]
    f += [f"fcst_{int(hs / 60)}min_q{int(round(q * 100))}"
          for hs in cfg.win.forecast_horizons_s for q in cfg.win.quantiles]
    return f


def _newest_mtime(p: Path) -> float:
    if p.is_file():
        return p.stat().st_mtime
    files = [f for f in p.rglob("*") if f.is_file()]
    return max((f.stat().st_mtime for f in files), default=p.stat().st_mtime)


# ---------------------------------------------------------------------------
# independent truth: a GOES flare list
# ---------------------------------------------------------------------------

_CLASS_SCALE = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}


def goes_class_flux(cls: str) -> float:
    """"M2.5" -> 2.5e-5 W/m^2; unparseable -> nan."""
    m = re.match(r"\s*([ABCMX])\s*([\d.]+)?", str(cls or "").upper())
    if not m:
        return float("nan")
    return _CLASS_SCALE[m.group(1)] * float(m.group(2) or 1.0)


def _parse_time(v) -> float | None:
    if v in (None, "", "null"):
        return None
    s = str(v).strip().replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M"):
        try:
            d = datetime.fromisoformat(s) if fmt is None else datetime.strptime(s, fmt)
            if d.tzinfo is None:
                d = d.replace(tzinfo=UTC)
            return d.timestamp()
        except ValueError:
            continue
    return None


def load_goes_events(path: Path) -> list[dict]:
    """Read a flare list: NOAA SWPC JSON (begin_time/max_time/end_time/
    max_class), a HEK export (event_starttime/event_peaktime/event_endtime/
    fl_goescls), or a CSV with equivalent columns."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text("utf-8"))
        if isinstance(rows, dict):
            rows = rows.get("result") or rows.get("events") or []
    else:
        with path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

    def pick(r, *keys):
        low = {k.lower(): v for k, v in r.items()}
        return next((low[k] for k in keys if low.get(k) not in (None, "")), None)

    events = []
    for r in rows:
        start = _parse_time(pick(r, "begin_time", "start_time", "start", "event_starttime"))
        peak = _parse_time(pick(r, "max_time", "peak_time", "peak", "event_peaktime"))
        end = _parse_time(pick(r, "end_time", "end", "event_endtime"))
        cls = pick(r, "max_class", "class", "goes_class", "fl_goescls")
        if start is None and peak is None:
            continue
        start = start if start is not None else peak
        peak = peak if peak is not None else start
        events.append({"start": start, "peak": peak,
                       "end": end if end is not None and end >= peak else peak,
                       "class": str(cls or ""), "flux": goes_class_flux(cls)})
    events.sort(key=lambda e: e["start"])
    return events


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def _day_bootstrap(stat, days: np.ndarray, n_boot: int = 500, seed: int = 0):
    """Resample whole UTC days: forecasts a minute apart are not independent."""
    from .forecast import bootstrap_ci
    return bootstrap_ci(None, days, stat, n_boot=n_boot, seed=seed)


def _binary_block(y: np.ndarray, p: np.ndarray, thr: float, clim: float,
                  days: np.ndarray) -> dict:
    out = {"n": int(y.size), "positives": int(y.sum()), "threshold": thr,
           "base_rate": float(y.mean()) if y.size else float("nan")}
    if y.size == 0:
        return out
    s = skill_scores(y, p >= thr)
    out.update({k: s[k] for k in ("TSS", "HSS", "POD", "FAR", "TP", "FP", "FN", "TN")})
    bs = brier_score(y, p)
    bs_ref = brier_score(y, np.full_like(p, clim))
    out["Brier"] = bs
    out["Brier_training_climatology"] = bs_ref
    out["BSS_vs_training_climatology"] = 1.0 - bs / bs_ref if bs_ref > 0 else float("nan")
    if len(np.unique(y)) > 1:
        out["AUC"] = roc_auc(y, p)

        def tss(idx):
            if len(np.unique(y[idx])) < 2:
                raise ValueError("degenerate")
            return skill_scores(y[idx], p[idx] >= thr)["TSS"]
        _, lo, hi = _day_bootstrap(tss, days)
        out["TSS_ci"] = [lo, hi]
        out["reliability"] = reliability(y, p)
    return out


def forward_score(cfg: Config, name: str, goes_events: Path | None = None,
                  min_class: str = "C1.0", verbose: bool = True) -> dict:
    out = forward_dir(cfg, name)
    frozen, _, _, _ = load_frozen(out, cfg)
    manifest = json.loads((out / MANIFEST).read_text("utf-8"))
    pred_path = out / PREDICTIONS
    if not pred_path.exists():
        return {"error": "no forecasts yet; run `forward-test predict` first"}
    with pred_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    dt = cfg.pre.dt_seconds
    cutoff = float(frozen["data_cutoff_unix"])
    thr = frozen["thresholds"]
    clim = frozen["climatology"]
    occ_h = list(cfg.win.occurrence_horizons_s)
    fc_h = list(cfg.win.forecast_horizons_s)
    qs = list(cfg.win.quantiles)
    i50 = int(np.argmin(np.abs(np.array(qs) - 0.5)))

    segments, _ = _segments(cfg, verbose)
    # Labels need the centred background (half a window) and the longest
    # horizon to be observed after the origin before they are final.
    settle_s = cfg.pre.background_window_s / 2 + max(occ_h + fc_h)

    index: dict[int, tuple[int, int]] = {}
    for si, seg in enumerate(segments):
        if seg.time_unix[-1] <= cutoff:
            continue
        for i in np.flatnonzero(seg.time_unix > cutoff):
            index[int(np.rint(seg.time_unix[i] / dt))] = (si, int(i))
    targets: dict[int, dict] = {}

    rec = {k: [] for k in ("origin", "y_now", "m_now", "p_now")}
    rec.update({f"occ{h}": [] for h in range(len(occ_h))})
    rec.update({f"mocc{h}": [] for h in range(len(occ_h))})
    rec.update({f"pocc{h}": [] for h in range(len(occ_h))})
    rec.update({f"yfc{h}": [] for h in range(len(fc_h))})
    rec.update({f"mfc{h}": [] for h in range(len(fc_h))})
    rec.update({f"qfc{h}": [] for h in range(len(fc_h))})
    rec.update({f"lofc{h}": [] for h in range(len(fc_h))})
    rec.update({f"hifc{h}": [] for h in range(len(fc_h))})
    rec["current"] = []
    pending = 0
    for r in rows:
        o = float(r["origin_unix"])
        hit = index.get(int(np.rint(o / dt)))
        if hit is None:
            pending += 1
            continue
        si, i = hit
        seg = segments[si]
        if o + settle_s > seg.time_unix[-1]:
            pending += 1
            continue
        if si not in targets:
            targets[si] = build_targets(seg, cfg)
        t = targets[si]
        rec["origin"].append(o)
        rec["y_now"].append(t["in_flare"][i])
        rec["m_now"].append(t["nowcast_mask"][i] > 0)
        rec["p_now"].append(float(r["p_flare_now"]))
        rec["current"].append(float(r["current_log_flux"]) if r["current_log_flux"] else np.nan)
        for h, hs in enumerate(occ_h):
            rec[f"occ{h}"].append(t["occurrence"][i, h])
            rec[f"mocc{h}"].append(t["occurrence_mask"][i, h] > 0)
            rec[f"pocc{h}"].append(float(r[f"p_flare_within_{int(hs / 60)}min"]))
        for h, hs in enumerate(fc_h):
            rec[f"yfc{h}"].append(t["forecast"][i, h])
            rec[f"mfc{h}"].append(t["forecast_mask"][i, h] > 0)
            key = f"fcst_{int(hs / 60)}min_q"
            rec[f"qfc{h}"].append(float(r[f"{key}{int(round(qs[i50] * 100))}"]))
            rec[f"lofc{h}"].append(float(r[f"{key}{int(round(qs[0] * 100))}"]))
            rec[f"hifc{h}"].append(float(r[f"{key}{int(round(qs[-1] * 100))}"]))
    a = {k: np.asarray(v) for k, v in rec.items()}
    n = int(a["origin"].size) if "origin" in a else 0

    report: dict = {
        "name": name,
        "scored_utc": _utc(time.time()),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "frozen_at_utc": manifest["frozen_at_utc"],
        "data_cutoff_utc": manifest["data_cutoff_utc"],
        "forecasts_total": len(rows),
        "forecasts_scored": n,
        "forecasts_pending_truth": pending,
    }
    if n == 0:
        report["note"] = ("nothing to score yet: forecasts need "
                          f"{settle_s / 3600:.1f} h of later data before their labels are final")
        _write_score(out, report, verbose)
        return report

    days = np.floor(a["origin"] / 86400).astype(np.int64)
    report["period_utc"] = [_utc(a["origin"].min()), _utc(a["origin"].max())]
    report["days_scored"] = int(np.unique(days).size)
    ev = [e for s in segments for e in s.events
          if a["origin"].min() <= e.peak_unix <= a["origin"].max()]
    report["flares_in_period"] = len(ev)

    m = a["m_now"]
    report["flare_in_progress_now"] = _binary_block(
        a["y_now"][m].astype(int), a["p_now"][m], thr["in_flare"], clim["in_flare_rate"], days[m])
    report["flare_within_horizon"] = {}
    for h, hs in enumerate(occ_h):
        mm = a[f"mocc{h}"]
        report["flare_within_horizon"][f"{int(hs / 60)}min"] = _binary_block(
            a[f"occ{h}"][mm].astype(int), a[f"pocc{h}"][mm], thr["occurrence"][h],
            clim["occurrence_rate"][h], days[mm])

    report["flux_forecast"] = {}
    for h, hs in enumerate(fc_h):
        mm = a[f"mfc{h}"] & np.isfinite(a["current"])
        if not mm.any():
            continue
        y, q = a[f"yfc{h}"][mm], a[f"qfc{h}"][mm]
        mae = float(np.mean(np.abs(q - y)))
        mae_p = float(np.mean(np.abs(a["current"][mm] - y)))
        mae_c = float(np.mean(np.abs(clim["mean_log_flux"] - y)))
        report["flux_forecast"][f"{int(hs / 60)}min"] = {
            "n": int(mm.sum()), "MAE": mae,
            "MAE_persistence": mae_p, "skill_vs_persistence": 1 - mae / mae_p if mae_p else float("nan"),
            "MAE_training_climatology": mae_c,
            "skill_vs_training_climatology": 1 - mae / mae_c if mae_c else float("nan"),
            "interval_coverage": float(np.mean((y >= a[f"lofc{h}"][mm]) & (y <= a[f"hifc{h}"][mm]))),
            "nominal_coverage": float(qs[-1] - qs[0]),
        }

    if goes_events:
        report["goes"] = _score_against_goes(load_goes_events(goes_events), min_class, a,
                                             occ_h, thr, days, ev, cfg)
    _write_score(out, report, verbose)
    return report


def _score_against_goes(events: list[dict], min_class: str, a: dict, occ_h: list[float],
                        thr: dict, days: np.ndarray, own_events: list, cfg: Config) -> dict:
    """Independent truth. Two questions: does the forecast predict GOES flares,
    and does our SoLEXS detector find the same flares GOES lists?"""
    floor = goes_class_flux(min_class)
    lo, hi = a["origin"].min(), a["origin"].max()
    big = [e for e in events if e["flux"] >= floor and lo - 7200 <= e["start"] <= hi + 7200]
    out: dict = {"min_class": min_class, "goes_flares_in_period": len(big)}
    if not big:
        out["note"] = "the flare list has no events of this class in the scored period"
        return out
    starts = np.array([e["start"] for e in big])
    ends = np.array([e["end"] for e in big])
    o = a["origin"]
    # "In progress at the origin" and "a flare starts within H after it".
    inside = np.array([np.any((starts <= t) & (ends >= t)) for t in o]).astype(int)
    m = np.ones(o.size, bool)
    out["flare_in_progress_now"] = _binary_block(inside, a["p_now"], thr["in_flare"],
                                                 float(inside.mean()), days)
    out["flare_within_horizon"] = {}
    for h, hs in enumerate(occ_h):
        y = np.array([np.any(((starts > t) & (starts <= t + hs)) | ((starts <= t) & (ends >= t)))
                      for t in o]).astype(int)
        out["flare_within_horizon"][f"{int(hs / 60)}min"] = _binary_block(
            y[m], a[f"pocc{h}"][m], thr["occurrence"][h], float(y.mean()), days[m])
        out["flare_within_horizon"][f"{int(hs / 60)}min"]["note"] = (
            "reference climatology here is the GOES base rate in this period, "
            "so BSS measures resolution only")

    # Label agreement: GOES peak within +/-10 min of one of our detected peaks.
    ours = np.array([e.peak_unix for e in own_events])
    tol = 600.0
    matched = sum(1 for e in big if ours.size and np.min(np.abs(ours - e["peak"])) <= tol)
    out["label_agreement"] = {
        "goes_flares_found_by_solexs_detector": matched,
        "fraction": matched / len(big),
        "solexs_detections_in_period": int(ours.size),
        "peak_tolerance_s": tol,
    }
    return out


def _write_score(out: Path, report: dict, verbose: bool) -> None:
    (out / "score.json").write_text(json.dumps(report, indent=2, default=float),
                                    encoding="utf-8")
    md = _markdown(report)
    (out / "FORWARD.md").write_text(md, encoding="utf-8")
    if verbose:
        print("\n" + md)
        print(f"wrote {out / 'score.json'} and {out / 'FORWARD.md'}")


def _f(v, nd=3) -> str:
    try:
        return "--" if v is None or v != v else f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _markdown(r: dict) -> str:
    L = [f"# Forward test `{r['name']}`", "",
         f"Model sha256 `{r['checkpoint_sha256'][:16]}`, frozen {r['frozen_at_utc']} UTC. "
         f"Data cutoff at freeze: **{r['data_cutoff_utc']} UTC** - every forecast below "
         f"is for a moment after it.", "",
         f"Forecasts issued: {r['forecasts_total']}; scored: {r['forecasts_scored']}; "
         f"waiting for later data before their truth is final: {r['forecasts_pending_truth']}.", ""]
    if r.get("note"):
        L += [f"> {r['note']}", ""]
        return "\n".join(L)
    L += [f"Scored period {r['period_utc'][0]} -> {r['period_utc'][1]} UTC, "
          f"{r['days_scored']} day(s), **{r['flares_in_period']} flare(s)** detected in it.", ""]
    if r["flares_in_period"] < 10:
        L += ["> Fewer than 10 flares: every skill score below describes a handful of "
              "events and its interval is wide. Keep forecasting before drawing conclusions.", ""]

    def table(title, blocks):
        out = [f"## {title}", "",
               "| target | n | positives | TSS [95% CI, by day] | HSS | AUC | BSS vs training climatology |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        for k, b in blocks.items():
            ci = b.get("TSS_ci", [None, None])
            out.append(f"| {k} | {b['n']} | {b['positives']} | {_f(b.get('TSS'))} "
                       f"[{_f(ci[0])}, {_f(ci[1])}] | {_f(b.get('HSS'))} | {_f(b.get('AUC'))} | "
                       f"{_f(b.get('BSS_vs_training_climatology'))} |")
        return out + [""]

    L += table("Against flares detected in SoLEXS",
               {"in progress now": r["flare_in_progress_now"],
                **{f"within {k}": v for k, v in r["flare_within_horizon"].items()}})
    if r.get("flux_forecast"):
        L += ["## Flux forecast (log count rate)", "",
              "| horizon | n | MAE | skill vs persistence | skill vs training climatology | q10-q90 coverage |",
              "|---|---:|---:|---:|---:|---:|"]
        for k, b in r["flux_forecast"].items():
            L.append(f"| {k} | {b['n']} | {_f(b['MAE'], 4)} | {_f(b['skill_vs_persistence'])} | "
                     f"{_f(b['skill_vs_training_climatology'])} | {_f(b['interval_coverage'], 2)} "
                     f"(nominal {_f(b['nominal_coverage'], 2)}) |")
        L += ["", "Skill > 0 beats the reference; <= 0 adds nothing over it.", ""]
    g = r.get("goes")
    if g:
        L += [f"## Against the GOES flare list (>= {g['min_class']})", "",
              f"{g['goes_flares_in_period']} GOES flare(s) of that class in the period."]
        if g.get("label_agreement"):
            la = g["label_agreement"]
            L += [f"Our SoLEXS detector found {la['goes_flares_found_by_solexs_detector']} of them "
                  f"({_f(100 * la['fraction'], 0)}%, peak within {int(la['peak_tolerance_s'] / 60)} min).", ""]
        if g.get("flare_within_horizon"):
            L += table("GOES truth",
                       {"in progress now": g["flare_in_progress_now"],
                        **{f"within {k}": v for k, v in g["flare_within_horizon"].items()}})
        elif g.get("note"):
            L += ["", f"> {g['note']}", ""]
    return "\n".join(L)
