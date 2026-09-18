"""Export one replay day for the flare dashboard: light curves + frozen-model predictions.

    python scripts/dashboard_export.py --forward-dir outputs/archive/forward/v1 --day 2026-07-04
    python scripts/dashboard_export.py --forward-dir outputs/archive_goes/forward/v2 --day 2026-07-04

Runs the *frozen* forward-test model causally along the day at a 1-minute stride,
exactly as a live system would (each prediction sees only the trailing window),
and writes a compact JSON the dashboard page reads:

* light curves at 1 min: GOES-18 XRS-B (independent reference), SoLEXS GOES-band
  rate and its GOES-calibrated flux, HEL1OS CZT 20-40 keV and CdTe 5-20 keV;
* per-minute predictions: flare-now probability, phase probabilities, flare within
  15/30/60 min, flux quantiles at 1-60 min, predicted peak size and time;
* GOES flares of the day and the model's validation thresholds.

Every flux is expressed in GOES XRS-B W/m^2. A GOES-labelled model predicts that
directly; a SoLEXS-labelled model predicts log(1 + count rate), which is mapped with
the archive SoLEXS->GOES fit from scripts/goes_crosscheck.py (0.065 dex scatter).
Runs on CPU so it can share the machine with a training job.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, UTC
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarflare.config import Config  # noqa: E402
from solarflare.forward import MANIFEST, load_frozen  # noqa: E402
from solarflare.io.goes import load_goes  # noqa: E402
from solarflare.preprocess.cache import Source, build_cache, load_cached  # noqa: E402
from solarflare.preprocess.dataset import PEAK_TIME_SCALE_S, build_segments_from_raw  # noqa: E402
from solarflare.preprocess.labels import PHASE_NAMES  # noqa: E402
from solarflare.preprocess.timeline import stitch  # noqa: E402


def _r(a, nd=4):
    """Array -> JSON list with NaN as null, rounded to keep the file small."""
    return [None if not np.isfinite(v) else round(float(v), nd) for v in np.asarray(a, float)]


def _minute_mean(t, x, ok, t_min):
    """Average a 20 s series into the 1-min bins starting at t_min (NaN if unobserved)."""
    idx = np.floor((t - t_min[0]) / 60.0).astype(np.int64)
    good = ok & np.isfinite(x) & (idx >= 0) & (idx < t_min.size)
    s = np.bincount(idx[good], weights=x[good], minlength=t_min.size)
    n = np.bincount(idx[good], minlength=t_min.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward-dir", required=True, help="frozen model dir, e.g. outputs/archive/forward/v1")
    ap.add_argument("--day", required=True, help="UTC day to replay, YYYY-MM-DD")
    ap.add_argument("--data-root", default="D:/Data")
    ap.add_argument("--goes-dir", default="D:/Data/goes")
    ap.add_argument("--manifest", default="outputs/archive/cache/manifest.json",
                    help="an archive cache manifest, used only to locate the day's products")
    ap.add_argument("--cache-dir", default="outputs/dashboard/cache")
    ap.add_argument("--crosscheck", default="outputs/archive_goes/reports/goes_crosscheck.json")
    ap.add_argument("--reports", default=None, help="reports dir of the model's run (for validated skill)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    fdir = Path(args.forward_dir)
    cfg = Config(data_root=Path(args.data_root), out_dir=Path("outputs/dashboard"))
    cfg.cache_dir = Path(args.cache_dir)
    cfg.train.device = "cpu"
    frozen, model, norm, device = load_frozen(fdir, cfg)
    manifest = json.loads((fdir / MANIFEST).read_text("utf-8"))
    label_source = cfg.pre.label_source
    if label_source == "goes":
        cfg.pre.goes_dir = args.goes_dir
    dt = cfg.pre.dt_seconds
    L = cfg.steps_per_window

    day0 = datetime.strptime(args.day, "%Y-%m-%d").replace(tzinfo=UTC)
    t0, t1 = day0.timestamp(), (day0 + timedelta(days=1)).timestamp()
    ctx0 = (day0 - timedelta(days=1)).timestamp()  # background + window context

    # --- the day's products, located through an existing cache manifest ----
    entries = json.loads(Path(args.manifest).read_text("utf-8"))
    want_dates = {(day0 - timedelta(days=1)).strftime("%Y%m%d"), day0.strftime("%Y%m%d")}
    sources = []
    for e in entries:
        s = e["source"]
        if s["kind"] == "solexs" and s["date"] in want_dates:
            sources.append(Source(**s))
        elif s["kind"] == "hel1os" and e.get("t_stop", 0) > ctx0 and e.get("t_start", 1e20) < t1:
            sources.append(Source(**s))
    if not any(s.kind == "solexs" for s in sources):
        raise SystemExit(f"no SoLEXS products for {args.day} in {args.manifest}")
    cache = build_cache(sources, cfg.pre, Path(args.cache_dir), workers=2, verbose=False)
    soft_raw = load_cached(cache, Path(args.cache_dir), "solexs")
    hard_raw = load_cached(cache, Path(args.cache_dir), "hel1os")
    segments, _ = build_segments_from_raw(soft_raw, hard_raw, cfg)
    seg = max((s for s in segments if s.goes_long is not None),
              key=lambda s: np.sum((s.time_unix >= t0) & (s.time_unix < t1)))

    # --- causal predictions at a 1-min stride ----------------------------------
    avail = np.maximum(seg.soft_mask, seg.hard_mask)
    csum = np.concatenate([[0.0], np.cumsum(avail)])
    ends = np.arange(L, len(seg) + 1)
    origin = seg.time_unix[ends - 1]
    keep = ((origin >= t0) & (origin < t1)
            & (np.rint(origin / dt).astype(np.int64) % max(int(60 / dt), 1) == 0)
            & (avail[ends - 1] > 0)
            & ((csum[ends] - csum[ends - L]) / L >= cfg.win.min_observed_fraction))
    ends = ends[keep]
    soft_n = np.nan_to_num(norm.apply_soft(seg.soft), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    hard_n = np.nan_to_num(norm.apply_hard(seg.hard), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    outs = {k: [] for k in ("p_now", "phase", "occ", "now", "fc", "peak")}
    with torch.no_grad():
        for b0 in range(0, len(ends), 128):
            sl = [slice(int(e) - L, int(e)) for e in ends[b0:b0 + 128]]

            def stack(a, _sl=sl):
                return torch.from_numpy(np.stack([a[s] for s in _sl]).astype(np.float32))

            o = model(stack(soft_n), stack(seg.soft_mask), stack(hard_n),
                      stack(seg.hard_mask), stack(seg.clock))
            outs["p_now"].append(torch.sigmoid(o["in_flare"]).numpy())
            outs["phase"].append(torch.softmax(o["phase"], -1).numpy())
            outs["occ"].append(torch.sigmoid(o["occurrence"]).numpy())
            outs["now"].append(o["nowcast"].numpy())
            outs["fc"].append(o["forecast"].numpy())
            outs["peak"].append(o["peak"].numpy())
    P = {k: np.concatenate(v) for k, v in outs.items()}

    # --- everything to log10 W/m^2 -------------------------------------------
    goes = load_goes(Path(args.goes_dir))
    cc = json.loads(Path(args.crosscheck).read_text("utf-8"))["flux_calibration"]
    a, b = cc["intercept"], cc["slope"]
    calib_note = f"archive fit ({cc['fit']}, {cc['residual_dex_sd']:.3f} dex)"
    if cfg.pre.solexs_energy_scale != json.loads(Path(args.crosscheck).read_text("utf-8"))["energy_scale"]:
        # The archive fit belongs to another energy scale's band; refit on the
        # previous day only, so the replay day itself never informs its calibration.
        ctx = (seg.time_unix >= ctx0) & (seg.time_unix < t0) & (seg.soft_mask > 0)
        g = goes.flux_on_grid(seg.time_unix[ctx])
        r = seg.goes_long[ctx].astype(float)
        ok = np.isfinite(g) & (g > 0) & (r > 0)
        b, a = np.polyfit(np.log10(r[ok]), np.log10(g[ok]), 1)
        sd = float(np.std(np.log10(g[ok]) - (a + b * np.log10(r[ok]))))
        calib_note = (f"refit on the previous day for the {cfg.pre.solexs_energy_scale} band: "
                      f"log10 F = {a:.3f} + {b:.3f} log10 rate ({sd:.3f} dex)")

    def rate_to_logflux(rate):
        with np.errstate(divide="ignore", invalid="ignore"):
            return a + b * np.log10(np.clip(rate, 1e-3, None))

    def model_to_logflux(v):
        if label_source == "goes":
            return np.asarray(v, float)
        return rate_to_logflux(np.expm1(np.clip(np.asarray(v, float), 0, 30)))

    # --- light curves on the day's 1-min grid ----------------------------------
    t_min = t0 + 60.0 * np.arange(1440)
    xrsb = goes.flux_on_grid(t_min)
    slx_rate = _minute_mean(seg.time_unix, seg.goes_long.astype(float), seg.soft_mask > 0, t_min)
    hard_tl = stitch(hard_raw, dt, cfg.pre.max_stitch_gap_s) if hard_raw else []
    hls = {}
    for key, col in (("czt1_20_40", "hls_czt1_20_40keV"), ("cdte1_5_20", "hls_cdte1_5_20keV")):
        acc = np.full(1440, np.nan)
        for h in hard_tl:
            j = list(h.names).index(col)
            m = _minute_mean(h.time_unix, h.values[:, j], h.coverage > 0, t_min)
            acc = np.where(np.isfinite(m), m, acc)
        hls[key] = acc

    pred_t = seg.time_unix[ends - 1]
    pi = np.rint((pred_t - t0) / 60.0).astype(int)
    in_flare_phase = P["phase"].argmax(-1) > 0
    peak_log = np.where(in_flare_phase, model_to_logflux(P["peak"][:, 1]), np.nan)
    peak_min = np.where(in_flare_phase, P["peak"][:, 0] * PEAK_TIME_SCALE_S / 60.0, np.nan)

    flares = [
        {"class": str(f.goes_class), "start": f.start_unix, "peak": f.peak_unix, "end": f.end_unix,
         "peak_flux": f.peak_flux}
        for f in goes.flares if f.end_unix > t0 and f.start_unix < t1
    ]

    skill = None
    rep = Path(args.reports) if args.reports else fdir.parents[1] / "reports"
    ev = rep / "evaluation.json"
    if ev.exists():
        e = json.loads(ev.read_text("utf-8"))
        nf = e["nowcast_in_flare"]
        skill = {"nowcast": {k: nf[k] for k in ("TSS", "POD", "FAR", "POFD", "AUC", "base_rate")},
                 "occurrence": {h: {k: v[k] for k in ("TSS", "POD", "FAR", "AUC", "base_rate")}
                                for h, v in e["forecast_occurrence"].items()}}

    out = {
        "day": args.day,
        "t0": t0,
        "model": {
            "name": manifest["name"], "sha256": manifest["checkpoint_sha256"][:12],
            "label_source": label_source, "energy_scale": cfg.pre.solexs_energy_scale,
            "flare_definition": ("GOES flare list, >= " + cfg.pre.goes_min_class) if label_source == "goes"
            else "SoLEXS detector (>= 1.4x background, loose)",
            "thresholds": manifest["thresholds"], "train_end_utc": manifest.get("train_end_utc"),
            "data_cutoff_utc": manifest.get("data_cutoff_utc"),
            "validated_skill": skill,
        },
        "calibration": calib_note,
        "horizons_min": [int(h / 60) for h in cfg.win.forecast_horizons_s],
        "occ_horizons_min": [int(h / 60) for h in cfg.win.occurrence_horizons_s],
        "phase_names": PHASE_NAMES,
        "curves": {
            "goes_xrsb_log": _r(np.log10(np.where(xrsb > 0, xrsb, np.nan)), 3),
            "solexs_rate": _r(slx_rate, 1),
            "solexs_flux_log": _r(rate_to_logflux(slx_rate), 3),
            "hel1os_czt1_20_40": _r(hls["czt1_20_40"], 2),
            "hel1os_cdte1_5_20": _r(hls["cdte1_5_20"], 2),
        },
        "pred": {
            "i": pi.tolist(),
            "p_now": _r(P["p_now"], 3),
            "phase": [_r(p, 3) for p in P["phase"]],
            "p_occ": [_r(p, 3) for p in P["occ"]],
            "now_log": _r(model_to_logflux(P["now"]), 3),
            "fc_log": [[_r(model_to_logflux(P["fc"][k, h]), 3) for h in range(P["fc"].shape[1])]
                       for k in range(len(pi))],
            "peak_log": _r(peak_log, 3),
            "peak_min": _r(peak_min, 1),
        },
        "flares": flares,
        "coverage": {"solexs_frac": float(np.isfinite(slx_rate).mean()),
                     "hel1os_frac": float(np.isfinite(hls["czt1_20_40"]).mean()),
                     "n_predictions": int(len(pi))},
    }
    dest = Path(args.out or f"outputs/dashboard/replay_{args.day.replace('-', '')}_{manifest['name']}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"{len(pi)} predictions, SoLEXS {100 * out['coverage']['solexs_frac']:.0f}% / HEL1OS "
          f"{100 * out['coverage']['hel1os_frac']:.0f}% of the day; {len(flares)} GOES flares -> {dest} "
          f"({dest.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
