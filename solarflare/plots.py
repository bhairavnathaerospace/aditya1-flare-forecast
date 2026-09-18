"""Diagnostic figures.  Matplotlib only, no seaborn, one chart per figure."""

from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Config
from .pipeline import Prepared


def _save(fig, path: Path, dpi: int = 140) -> Path | None:
    """Write a figure, but never let a failed write kill the caller.

    Figures are a side product; training and evaluation results are not.  On
    Windows a PNG write can fail with EINVAL simply because a viewer, indexer
    or antivirus is holding the file, and losing a completed training run to
    that would be absurd.  One retry, then warn and carry on.
    """
    path = Path(path)
    for attempt in (1, 2):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            return path
        except OSError as exc:
            if attempt == 1:
                continue
            print(f"  WARNING: could not write {path.name} ({exc}); "
                  f"continuing without it")
            plt.close(fig)
            return None
    return None


def _hours(t_unix: np.ndarray) -> np.ndarray:
    return (t_unix - t_unix[0]) / 3600.0


def _utc_label(t0: float) -> str:
    return datetime.fromtimestamp(t0, UTC).strftime("%Y-%m-%d %H:%M UTC")


def plot_overview(prep: Prepared, cfg: Config, out_dir: Path) -> list[Path]:
    """Data overview built from stitched segments.

    Short samples get one light-curve figure per segment. Archives get a
    single mission overview instead: a 20 s light curve over ~1000 days is
    four million points, unreadable as one line and slow to render.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if prep.archive:
        p = _plot_mission(prep, cfg, out_dir)
        return [p] if p else []

    paths: list[Path] = []
    for seg in prep.segments:
        t = seg.time_unix
        h = _hours(t)
        fig, ax = plt.subplots(figsize=(13, 5))
        if seg.goes_long is not None:
            ax.plot(h, seg.goes_long, lw=0.7, color="#1f77b4",
                    label="SoLEXS 1.55-12.4 keV (GOES-long analogue)")
            ax.plot(h, seg.background, lw=1.2, color="#d62728", ls="--",
                    label=f"background (P{cfg.pre.background_percentile:g})")
            for i, e in enumerate(seg.events):
                ax.axvspan((e.start_unix - t[0]) / 3600, (e.end_unix - t[0]) / 3600,
                           color="orange", alpha=0.18,
                           label="detected flare" if i == 0 else None)
                y_pk = (seg.goes_long[min(e.peak_idx, len(seg) - 1)]
                        if e.goes_class else e.peak_rate)
                ax.plot((e.peak_unix - t[0]) / 3600, y_pk, "v", color="k", ms=5)
            ax.set_yscale("log")
            ax.set_title(f"SoLEXS {seg.name} - {len(seg.events)} flares detected")
            ax.legend(loc="upper left", fontsize=8)
        else:
            from .config import HEL1OS_DETECTORS
            from .preprocess.features import hel1os_feature_names
            names = hel1os_feature_names(cfg.pre)
            # The wide band of every detector; detectors are never combined.
            for det, (_, bands) in HEL1OS_DETECTORS.items():
                lo, hi = max(bands, key=lambda b: b[1] - b[0])
                col = names.index(f"log_hls_{det}_{lo:g}_{hi:g}keV")
                if col >= seg.hard.shape[1] or not seg.hard[:, col].any():
                    continue
                y = np.where(seg.hard[:, names.index(f"hls_{det}_coverage")] > 0,
                             np.expm1(seg.hard[:, col]), np.nan)
                ax.plot(h, y, lw=0.7, label=f"{det.upper()} {lo:g}-{hi:g} keV")
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_title(f"HEL1OS {seg.name} - fill rows masked, "
                         f"{100 * seg.hard_mask.mean():.0f}% of bins observed, "
                         f"{cfg.pre.dt_seconds:g} s grid")
            ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.set_xlabel(f"hours since {_utc_label(float(t[0]))}")
        ax.set_ylabel("count rate (s$^{-1}$)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = _save(fig, out_dir / f"lightcurve_{seg.name}.png")
        if p:
            paths.append(p)
    return paths


def _plot_mission(prep: Prepared, cfg: Config, out_dir: Path) -> Path | None:
    """10-minute maxima with flare peaks, flares per month, daily coverage."""
    import warnings
    from collections import Counter

    soft = [s for s in prep.segments if s.goes_long is not None]
    fig, axes = plt.subplots(3, 1, figsize=(15, 10),
                             gridspec_kw={"height_ratios": [3, 1.3, 1]})

    ax = axes[0]
    k = max(int(600 / cfg.pre.dt_seconds), 1)
    for s in soft:
        n = (len(s) // k) * k
        if n == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-gap blocks -> NaN
            mx = np.nanmax(s.goes_long[:n].reshape(-1, k), axis=1)
            bg = np.nanmedian(s.background[:n].reshape(-1, k), axis=1)
        tt = [datetime.fromtimestamp(float(x), UTC) for x in s.time_unix[:n:k]]
        ax.plot(tt, mx, lw=0.4, color="#1f77b4")
        ax.plot(tt, bg, lw=0.8, color="#d62728", alpha=0.8)
    # Markers sit on the SoLEXS curve. GOES-labelled events carry W/m^2, so they
    # are placed at the SoLEXS rate at their peak and coloured by GOES flux
    # relative to A1 (1e-8 W/m^2).
    peaks = [(e.peak_unix,
              float(s.goes_long[min(e.peak_idx, len(s) - 1)]) if e.goes_class else e.peak_rate,
              (e.peak_rate / 1e-8) if e.goes_class else e.magnitude)
             for s in soft for e in s.events]
    if peaks:
        pt, pr, pm = zip(*peaks)
        sc = ax.scatter([datetime.fromtimestamp(float(x), UTC) for x in pt], pr,
                        c=np.log10(np.maximum(pm, 1e-3)), s=6, cmap="viridis", zorder=3)
        goes_lbl = any(e.goes_class for s in soft for e in s.events)
        fig.colorbar(sc, ax=ax, pad=0.01,
                     label="log10 GOES peak flux / 1e-8 W m$^{-2}$" if goes_lbl
                     else "log10 peak excess / background")
    ax.set_yscale("log")
    ax.set_ylabel("cts s$^{-1}$ (10-min max)")
    ax.set_title(f"SoLEXS GOES-long analogue - {len(peaks)} flares over "
                 f"{prep.meta.get('observed_days', 0):.0f} observed days "
                 f"(blue: rate, red: background)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    months = Counter(datetime.fromtimestamp(float(p[0]), UTC).strftime("%Y-%m") for p in peaks)
    if months:
        keys = sorted(months)
        ax.bar(range(len(keys)), [months[m] for m in keys], color="#ff7f0e")
        step = max(len(keys) // 16, 1)
        ax.set_xticks(range(0, len(keys), step))
        ax.set_xticklabels(keys[::step], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("flares / month")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    dt = cfg.pre.dt_seconds
    per_day: dict[str, Counter] = {"SoLEXS": Counter(), "HEL1OS": Counter()}
    for s in prep.segments:
        days = (s.time_unix // 86400).astype(np.int64)
        for key, mask in (("SoLEXS", s.soft_mask), ("HEL1OS", s.hard_mask)):
            d_, v = np.unique(days[mask > 0], return_counts=True)
            for di, vi in zip(d_, v):
                per_day[key][int(di)] += vi * dt / 86400
    for (label, counter), color in zip(per_day.items(), ("#1f77b4", "#2ca02c")):
        if counter:
            ds = sorted(counter)
            ax.plot([datetime.fromtimestamp(d_ * 86400, UTC) for d_ in ds],
                    [counter[d_] for d_ in ds], ".", ms=2, label=label, color=color)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("daily coverage")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, out_dir / "mission_overview.png")


def plot_spectrogram(prep: Prepared, cfg: Config, out_dir: Path) -> list[Path]:
    """Time-energy spectrogram of the day holding the largest detected flare.

    Needs raw 340-channel spectra, which the cache deliberately does not keep,
    so that one day is re-read from its source file on demand.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    from .io.solexs import channel_energies, read_solexs, read_solexs_zip
    from .config import SOLEXS_CH_LO

    events = [e for s in prep.segments for e in s.events]
    if not events:
        return []
    big = max(events, key=lambda e: e.peak_excess)
    day = datetime.fromtimestamp(float(big.peak_unix), UTC).strftime("%Y%m%d")
    src = next((s for s in prep.sources if s.kind == "solexs" and s.date == day), None)
    if src is None:
        return []
    obs = (read_solexs_zip(Path(src.path), src.detector) if src.fmt == "zip"
           else read_solexs(Path(src.path)))
    if obs is None:
        return []

    dt_bin = 60
    n = obs.n // dt_bin
    spec = obs.spectra[: n * dt_bin, SOLEXS_CH_LO:].reshape(n, dt_bin, -1).mean(axis=1)
    e = channel_energies(cfg.pre.solexs_energy_scale, obs.detector)[SOLEXS_CH_LO:]
    hours = np.arange(n) * dt_bin / 3600.0

    fig, ax = plt.subplots(figsize=(13, 5))
    with np.errstate(divide="ignore"):
        z = np.log10(np.clip(spec.T, 1e-4, None))
    im = ax.pcolormesh(hours, e, z, shading="auto", cmap="magma")
    ax.axvline((big.peak_unix - obs.time_unix[0]) / 3600, color="cyan", lw=0.8, ls=":")
    ax.set_ylim(e[0], 15)
    ax.set_xlabel(f"hours since {_utc_label(float(obs.time_unix[0]))}")
    ax.set_ylabel(f"energy (keV, {cfg.pre.solexs_energy_scale} scale)")
    ax.set_title(f"SoLEXS {obs.detector} {day} - day of the largest detected flare "
                 f"(dotted line: its peak)")
    fig.colorbar(im, ax=ax, label="log$_{10}$ counts s$^{-1}$ channel$^{-1}$")
    fig.tight_layout()
    p = _save(fig, out_dir / f"spectrogram_{day}.png")
    return [p] if p else []


def plot_history(history: list[dict], out_dir: Path) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ep, [h.get("train_loss") for h in history], label="train loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("training loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    for key, lbl in (("val_TSS_inflare", "TSS (flare now)"),
                     ("val_TSS_occurrence", "TSS (occurrence)")):
        vals = [h.get(key, np.nan) for h in history]
        if np.isfinite(vals).any():
            axes[1].plot(ep, vals, label=lbl)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("TSS")
    axes[1].set_title("validation skill")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir / "training_history.png")


def plot_test_timeline(pred: dict, cfg: Config, out_dir: Path) -> Path | None:
    """Predicted vs observed over the test period, with forecast bands."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Only windows with a valid soft-X-ray target are scored; hard-only
    # segments would otherwise appear as a long flat stretch of unscored
    # predictions and dominate the axis.
    keep = pred["y_nowcast_mask"] > 0
    if not keep.any():
        keep = np.ones_like(pred["y_nowcast_mask"], dtype=bool)
    pred = {k: v[keep] for k, v in pred.items() if hasattr(v, "__len__")
            and len(v) == keep.size}

    order = np.argsort(pred["y_t_unix"])
    t = pred["y_t_unix"][order]
    h = _hours(t)
    m = pred["y_nowcast_mask"][order] > 0

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    axes[0].plot(h[m], pred["y_nowcast"][order][m], lw=0.9, color="k",
                 label="observed log flux")
    axes[0].plot(h[m], pred["nowcast"][order][m], lw=0.9, color="#1f77b4",
                 label="nowcast")
    axes[0].set_ylabel("log1p rate")
    axes[0].set_title("nowcast")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    hi_idx = len(cfg.win.forecast_horizons_s) - 1
    q = pred["forecast"][order]
    fm = pred["y_forecast_mask"][order][:, hi_idx] > 0
    axes[1].plot(h[fm], pred["y_forecast"][order][fm, hi_idx], lw=0.9, color="k",
                 label=f"observed at +{cfg.win.forecast_horizons_s[hi_idx]/60:.0f} min")
    axes[1].plot(h[fm], q[fm, hi_idx, q.shape[2] // 2], lw=0.9, color="#2ca02c",
                 label="forecast q50")
    axes[1].fill_between(h[fm], q[fm, hi_idx, 0], q[fm, hi_idx, -1],
                         color="#2ca02c", alpha=0.25, label="q10-q90")
    axes[1].set_ylabel("log1p rate")
    axes[1].set_title(f"forecast +{cfg.win.forecast_horizons_s[hi_idx]/60:.0f} min")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(h, pred["p_inflare"][order], lw=0.9, color="#d62728",
                 label="P(flare now)")
    for i, hs in enumerate(cfg.win.occurrence_horizons_s):
        axes[2].plot(h, pred["p_occurrence"][order][:, i], lw=0.8,
                     label=f"P(flare within {hs/60:.0f} min)")
    axes[2].fill_between(h, 0, pred["y_in_flare"][order], color="orange",
                         alpha=0.25, step="mid", label="observed flare")
    axes[2].set_ylim(0, 1.02)
    axes[2].set_ylabel("probability")
    axes[2].set_xlabel(f"hours since {_utc_label(float(t[0]))}")
    axes[2].legend(fontsize=8, ncol=3)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, out_dir / "test_timeline.png")


def plot_reliability(report: dict, out_dir: Path) -> Path | None:
    rel = report.get("nowcast_in_flare", {}).get("reliability", {})
    if not rel or not rel.get("bin_centre"):
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="perfect")
    ax.plot(rel["predicted"], rel["observed"], "o-", color="#1f77b4",
            label="model")
    for x, y, c in zip(rel["predicted"], rel["observed"], rel["count"]):
        ax.annotate(str(c), (x, y), fontsize=7, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("reliability - flare in progress")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_dir / "reliability.png")
