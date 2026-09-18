"""Command line entry points.

    python -m solarflare.cli inspect     # what is in the data
    python -m solarflare.cli train       # preprocess, train, evaluate, plot
    python -m solarflare.cli evaluate    # re-score a saved checkpoint
    python -m solarflare.cli predict     # run the model over an observation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import Config
from datetime import UTC


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", default=".", help="folder holding the mission products")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--dt", type=float, default=None, help="grid cadence, seconds")
    p.add_argument("--window", type=float, default=None, help="input window, seconds")
    p.add_argument("--device", default=None, help="cpu / cuda / auto")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel processes for the preprocessing cache "
                        "(memory: ~1.5 GB each)")
    p.add_argument("--cache-dir", default=None, dest="cache_dir",
                   help="share one preprocessing cache between output folders")
    p.add_argument("--energy-scale", default=None, dest="energy_scale",
                   choices=["sarwade2025", "legacy_linear"],
                   help="SoLEXS channel-to-energy scale (default: published "
                        "sarwade2025). Use legacy_linear only to reproduce models "
                        "trained before the calibration fix, e.g. outputs/archive.")
    p.add_argument("--labels", default=None, dest="label_source", choices=["solexs", "goes"],
                   help="flare truth: GOES flare list + XRS-B flux (needs --goes-dir), "
                        "or this project's SoLEXS detector (default)")
    p.add_argument("--goes-dir", default=None, dest="goes_dir",
                   help="folder with GOES-R sci_xrsf-l2-flsum and -avg1m netCDF files")
    p.add_argument("--anchor-flux", action="store_true", dest="anchor_flux",
                   help="predict flux as calibrated SoLEXS flux now + a learned change")
    p.add_argument("--goes-min-class", default=None, dest="goes_min_class",
                   help="smallest GOES class counted as a flare (default C1.0)")


def _build_cfg(args) -> Config:
    cfg = Config(data_root=Path(args.data_root), out_dir=Path(args.out_dir))
    if getattr(args, "cache_dir", None):
        cfg.cache_dir = Path(args.cache_dir)
    if getattr(args, "energy_scale", None):
        cfg.pre.solexs_energy_scale = args.energy_scale
    for name in ("label_source", "goes_dir", "goes_min_class"):
        if getattr(args, name, None):
            setattr(cfg.pre, name, getattr(args, name))
    if getattr(args, "anchor_flux", False):
        cfg.model.anchor_flux = True
    if getattr(args, "workers", None):
        cfg.pre.cache_workers = args.workers
    if getattr(args, "dt", None):
        cfg.pre.dt_seconds = args.dt
    if getattr(args, "window", None):
        cfg.win.input_seconds = args.window
    if getattr(args, "device", None):
        cfg.train.device = args.device
    for name in ("epochs", "batch_size", "lr", "seed"):
        v = getattr(args, name, None)
        if v is not None:
            setattr(cfg.train, name, v)
    return cfg


def cmd_cache(args) -> None:
    """Build or refresh the per-file cache only. Safe to re-run while a
    download is still going: finished files are skipped, partial ones ignored."""
    from collections import Counter
    from .preprocess.cache import index_sources, build_cache
    from .pipeline import cache_dir_for

    cfg = _build_cfg(args)
    sources = index_sources([Path(cfg.data_root)])
    kinds = Counter((s.kind, s.fmt) for s in sources)
    print(f"indexed {len(sources)} sources under {cfg.data_root}: "
          + ", ".join(f"{k[0]}/{k[1]} {v}" for k, v in sorted(kinds.items())))
    entries = build_cache(sources, cfg.pre, cache_dir_for(cfg),
                          workers=cfg.pre.cache_workers)
    months = Counter(e["source"]["date"][:6] for e in entries
                     if e.get("status") == "ok" and e["source"]["kind"] == "solexs"
                     and e["source"]["date"])
    if months:
        print("SoLEXS days cached per month: "
              + " ".join(f"{m[:4]}-{m[4:]}:{n}" for m, n in sorted(months.items())))
    failed = [e for e in entries if e.get("status") == "failed"]
    for e in failed[:20]:
        print(f"  failed: {Path(e['source']['path']).name}: {e.get('error')}")
    if len(failed) > 20:
        print(f"  ... and {len(failed) - 20} more (see cache/manifest.json)")


def cmd_inspect(args) -> None:
    """Survey the data from the cache (works at archive scale).

    --lines adds instrumental-line energy-scale diagnostics on a handful of
    days spread across the mission, which also reveals any gain drift.
    """
    from collections import Counter
    from datetime import datetime
    from .pipeline import prepare

    cfg = _build_cfg(args)
    prep = prepare(cfg)

    def _t(u: float, fmt: str = "%Y-%m-%d %H:%M") -> str:
        return datetime.fromtimestamp(float(u), UTC).strftime(fmt)

    entries = prep.cache_entries
    lc = [e for e in entries if e.get("lc_check", {}).get("checked")]
    bad = [e for e in lc if not e["lc_check"].get("exact")]
    print(f"\nChannel-threshold cross-check (pipeline .lc == sum of channels 41+): "
          f"{len(lc) - len(bad)}/{len(lc)} days exact")
    for e in bad[:10]:
        print(f"  MISMATCH {Path(e['source']['path']).name}: "
              f"max diff {e['lc_check'].get('max_abs_diff')}")
    failed = [e for e in entries if e.get("status") == "failed"]
    if failed:
        print(f"Unreadable files: {len(failed)} (first: "
              f"{Path(failed[0]['source']['path']).name}: {failed[0].get('error')})")

    goes = cfg.pre.label_source == "goes"
    events = sorted(((s.name, e) for s in prep.segments for e in s.events),
                    key=lambda x: -(x[1].peak_rate if goes else x[1].peak_excess))
    title = (f"GOES flares (>= {cfg.pre.goes_min_class}) on the data grid" if goes
             else "Flares detected")
    print(f"\n{title}: {len(events)}")
    if events:
        mags = np.array([e.magnitude for _, e in events])
        durs = np.array([e.duration_s for _, e in events]) / 60
        if goes:
            print("  classes: " + " ".join(f"{k}:{v}" for k, v in sorted(
                Counter(e.goes_class[:1] for _, e in events).items())))
        else:
            print(f"  peak/background: median {np.median(mags):.1f}x, 90th pct "
                  f"{np.percentile(mags, 90):.1f}x, max {mags.max():.1f}x")
        print(f"  duration: median {np.median(durs):.0f} min, "
              f"{(durs >= 60).mean() * 100:.0f}% longer than 1 h")
        if prep.archive:
            per_month = Counter(_t(e.peak_unix, "%Y-%m") for _, e in events)
            print("  per month: " + " ".join(f"{m}:{n}" for m, n in sorted(per_month.items())))
        show = events if len(events) <= 30 else events[:15]
        print(f"  {'largest ' + str(len(show)) if len(events) > 30 else 'all'}:")
        for _, e in show:
            size = (f"{e.goes_class:>6s}" if goes else
                    f"peak {e.peak_rate:7.2f} c/s  {e.magnitude:6.1f}x bg")
            print(f"    {_t(e.start_unix)} -> peak {_t(e.peak_unix, '%H:%M')} -> "
                  f"{_t(e.end_unix, '%H:%M')}  {size}  rise {e.rise_time_s / 60:5.1f} min  "
                  f"dur {e.duration_s / 60:6.1f} min")

    dt = cfg.pre.dt_seconds
    both = sum(float((s.soft_mask * s.hard_mask).sum()) for s in prep.segments) * dt / 86400
    hard = sum(float(s.hard_mask.sum()) for s in prep.segments) * dt / 86400
    print(f"\nSimultaneous SoLEXS + HEL1OS coverage: {both:.2f} days "
          f"(HEL1OS total {hard:.2f} days)")
    if both > 0:
        per_month: Counter = Counter()
        for s in prep.segments:
            m = (s.soft_mask * s.hard_mask) > 0
            if m.any():
                for month, n in Counter(_t(u, "%Y-%m") for u in s.time_unix[m]).items():
                    per_month[month] += n * dt / 86400
        print("  per month (days): " + " ".join(f"{k}:{v:.1f}" for k, v in sorted(per_month.items())))
        # A flare is usable for fusion when HEL1OS watched its rise.
        seen = []
        for s in prep.segments:
            for e in s.events:
                a = int(np.searchsorted(s.time_unix, e.start_unix))
                b = max(int(np.searchsorted(s.time_unix, e.peak_unix)), a + 1)
                if s.hard_mask[a:b].mean() >= 0.5:
                    seen.append(e)
        print(f"  flares with HEL1OS covering >= 50% of the rise: {len(seen)} of {len(events)}"
              + ((f"; largest {max(seen, key=lambda x: x.peak_rate).goes_class}" if goes else
                  f"; largest {max(x.magnitude for x in seen):.1f}x background") if seen else ""))
    if both < 1.0:
        print("  Less than a day of simultaneous data: the soft+hard fusion pathway "
              "cannot be trained or tested on real overlap. "
              + ("No HEL1OS data was found at all." if hard == 0 else
                 "Download HEL1OS for the same dates as SoLEXS."))

    if getattr(args, "lines", False):
        _inspect_lines(prep, cfg.pre.solexs_energy_scale)


def _inspect_lines(prep, prep_scale: str = "sarwade2025") -> None:
    """Energy-scale line diagnostics on up to six days across the mission."""
    from .calibration import report as line_report, quiet_spectrum, find_lines
    from .io.solexs import read_solexs, read_solexs_zip

    days = [s for s in prep.sources if s.kind == "solexs"]
    if not days:
        return
    pick = [days[int(i)] for i in np.linspace(0, len(days) - 1, min(6, len(days)))]
    drift = []
    for i, src in enumerate(pick):
        obs = (read_solexs_zip(Path(src.path), src.detector) if src.fmt == "zip"
               else read_solexs(Path(src.path)))
        if obs is None:
            continue
        if i == 0:
            print(f"\nEnergy-scale diagnostics ({src.date}):")
            print(line_report(obs, prep_scale))
        spec, n = quiet_spectrum(obs)
        lines = find_lines(spec, energy_scale=prep_scale, detector=obs.detector) if n else []
        drift.append((src.date, lines[0].channel if lines else float("nan"),
                      lines[0].significance if lines else float("nan")))
    if len(drift) > 1:
        print("\nStrongest instrumental line across the mission (gain stability):")
        for d, ch, sig in drift:
            print(f"  {d}: channel {ch:7.2f}  ({sig:5.1f} sigma)")
        chans = np.array([c for _, c, _ in drift if c == c])
        if chans.size > 1:
            print(f"  spread {chans.max() - chans.min():.2f} channels -- "
                  + ("stable gain" if chans.max() - chans.min() < 1.0 else
                     "the line moves: detector gain drifts over the mission, so a "
                     "single fixed gain is not correct for every day"))


def cmd_train(args) -> None:
    from .pipeline import prepare
    from .train import train, load_model
    from .evaluate import evaluate, save_report
    from . import plots

    cfg = _build_cfg(args)
    out = Path(cfg.out_dir)
    cfg.to_json(out / "reports" / "config.json")

    prep = prepare(cfg)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "reports" / "data_meta.json").write_text(
        json.dumps(prep.meta, indent=2, default=float), encoding="utf-8")

    if not args.no_plots:
        figs = plots.plot_overview(prep, cfg, out / "figures")
        figs += plots.plot_spectrogram(prep, cfg, out / "figures")
        print(f"\nwrote {len(figs)} data figures to {out / 'figures'}")

    res = train(cfg, prep=prep)

    model = load_model(Path(res["checkpoint"]), cfg)
    report = evaluate(model, prep, cfg)
    report["training"] = {k: v for k, v in res.items() if k != "history"}
    save_report(report, out / "reports" / "evaluation.json")

    if not args.no_plots:
        from .train import collect_predictions
        from .pipeline import make_loaders, resolve_device
        _, _, te, _ = make_loaders(prep, cfg)
        pred = collect_predictions(model, te, resolve_device(cfg.train.device))
        plots.plot_history(res["history"], out / "figures")
        plots.plot_test_timeline(pred, cfg, out / "figures")
        plots.plot_reliability(report, out / "figures")
        print(f"\nfigures: {out / 'figures'}")
    print(f"report:  {out / 'reports' / 'evaluation.json'}")


def cmd_baselines(args) -> None:
    from .pipeline import prepare
    from .baselines import run_baselines
    import json as _json

    cfg = _build_cfg(args)
    prep = prepare(cfg)
    rep = run_baselines(prep, cfg)
    out = Path(cfg.out_dir) / "reports" / "baselines.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")


def cmd_forecast(args) -> None:
    """Rise-phase forecasting with grouped rolling-origin CV."""
    from .pipeline import prepare
    from .forecast import grouped_cv, print_comparison, save
    from .models.zoo import ENCODERS

    cfg = _build_cfg(args)
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    bad = [e for e in encoders if e not in ENCODERS]
    if bad:
        raise SystemExit(f"unknown encoder(s) {bad}; choose from {list(ENCODERS)}")

    prep = prepare(cfg)
    res = grouped_cv(prep, cfg, encoders=encoders, n_folds=args.folds)
    print_comparison(res)
    out = Path(cfg.out_dir) / "reports" / "forecast_cv.json"
    save(res, out)
    print(f"\nwrote {out}")


def cmd_fusion(args) -> None:
    """Paired soft-only vs soft+hard comparison on flares HEL1OS observed."""
    from .pipeline import prepare
    from .forecast import fusion_ablation, save

    cfg = _build_cfg(args)
    prep = prepare(cfg)
    res = fusion_ablation(prep, cfg, encoder=args.encoder, n_folds=args.folds,
                          min_hard_coverage=args.min_hard_coverage)
    out = Path(cfg.out_dir) / "reports" / "fusion_ablation.json"
    save(res, out)
    print(f"\nwrote {out}")


def cmd_forward(args) -> None:
    """Prospective test: freeze -> predict (re-run as data arrives) -> score."""
    from . import forward

    cfg = _build_cfg(args)
    if args.action == "freeze":
        ckpt = Path(args.checkpoint or (Path(cfg.out_dir) / "checkpoints" / "best.pt"))
        forward.freeze(cfg, ckpt, args.name)
    elif args.action == "predict":
        forward.forward_predict(cfg, args.name, stride_s=args.stride)
    else:
        forward.forward_score(cfg, args.name,
                              goes_events=Path(args.goes_events) if args.goes_events else None,
                              min_class=args.min_class)


def cmd_report(args) -> None:
    from .report import build
    cfg = _build_cfg(args)
    print(f"wrote {build(Path(cfg.out_dir))}")


def cmd_evaluate(args) -> None:
    from .pipeline import prepare
    from .train import load_model
    from .evaluate import evaluate, save_report

    cfg = _build_cfg(args)
    prep = prepare(cfg)
    ckpt = Path(args.checkpoint or (Path(cfg.out_dir) / "checkpoints" / "best.pt"))
    model = load_model(ckpt, cfg)
    report = evaluate(model, prep, cfg)
    save_report(report, Path(cfg.out_dir) / "reports" / "evaluation.json")


def cmd_predict(args) -> None:
    from .predict import run_inference

    cfg = _build_cfg(args)
    ckpt = Path(args.checkpoint or (Path(cfg.out_dir) / "checkpoints" / "best.pt"))
    run_inference(cfg, ckpt, Path(args.output) if args.output else None,
                  limit=args.limit)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="solarflare",
        description="Deep learning for solar flare nowcasting and forecasting "
                    "from Aditya-L1 SoLEXS (soft X-ray) and HEL1OS (hard X-ray).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("cache", help="build/refresh the preprocessing cache only "
                                     "(re-run freely while a download continues)")
    _add_common(p)
    p.set_defaults(func=cmd_cache)

    p = sub.add_parser("inspect", help="summarise the available data")
    _add_common(p)
    p.add_argument("--lines", action="store_true",
                   help="search the quiet spectrum for instrumental calibration "
                        "lines and report what they imply for the energy scale")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("train", help="preprocess, train, evaluate")
    _add_common(p)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("baselines", help="classical baselines on the same splits")
    _add_common(p)
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser(
        "forecast",
        help="rise-phase forecasting: predict peak magnitude/timing from the "
             "early rise, with grouped rolling-origin CV over events")
    _add_common(p)
    p.add_argument("--encoders", default="tcn,ssm,gru,linear",
                   help="comma-separated: tcn,ssm,gru,transformer,linear")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_forecast)

    p = sub.add_parser(
        "fusion",
        help="does HEL1OS add skill? paired soft-only vs soft+hard CV on flares "
             "whose rise HEL1OS observed")
    _add_common(p)
    p.add_argument("--encoder", default="tcn")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--min-hard-coverage", type=float, default=0.5, dest="min_hard_coverage")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_fusion)

    p = sub.add_parser(
        "forward-test",
        help="prospective test on data after the model was frozen: "
             "freeze, then predict as new data arrives, then score")
    _add_common(p)
    p.add_argument("action", choices=["freeze", "predict", "score"])
    p.add_argument("--name", required=True, help="name of this frozen model, e.g. v1")
    p.add_argument("--checkpoint", default=None, help="freeze: default <out-dir>/checkpoints/best.pt")
    p.add_argument("--stride", type=float, default=60.0, help="predict: seconds between forecasts")
    p.add_argument("--goes-events", default=None, dest="goes_events",
                   help="score: independent flare list (NOAA SWPC JSON, HEK JSON, or CSV)")
    p.add_argument("--min-class", default="C1.0", dest="min_class",
                   help="score: smallest GOES class counted as a flare")
    p.set_defaults(func=cmd_forward)

    p = sub.add_parser("report", help="render RESULTS.md from the JSON reports")
    _add_common(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("evaluate", help="score a saved checkpoint")
    _add_common(p)
    p.add_argument("--checkpoint", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="stream predictions over the data")
    _add_common(p)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output", default=None, help="CSV path")
    p.add_argument("--limit", type=int, default=None,
                   help="max prediction steps PER SEGMENT (not in total)")
    p.set_defaults(func=cmd_predict)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
