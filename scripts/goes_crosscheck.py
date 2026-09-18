"""SoLEXS against GOES-18: flux cross-calibration and flare-label agreement.

    python scripts/goes_crosscheck.py --data-root D:/Data --goes-dir D:/Data/goes --cache-dir outputs/archive/cache

Answers, on the real archive and only where SoLEXS was observing:

1. How does the SoLEXS GOES-long analogue count rate map onto GOES XRS-B flux
   (W/m^2)? A log-log fit with its scatter, and the ratio per GOES decade.
2. How many GOES flares (per class) does the project's SoLEXS detector find?
3. How different are the two flare definitions in time budget and base rate?

Writes <out-dir>/reports/goes_crosscheck.json and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarflare.config import Config  # noqa: E402
from solarflare.io.goes import class_flux, load_goes  # noqa: E402
from solarflare.pipeline import cache_dir_for  # noqa: E402
from solarflare.preprocess.cache import build_cache, index_sources, load_cached  # noqa: E402
from solarflare.preprocess.dataset import build_segments_from_raw  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--goes-dir", required=True)
    ap.add_argument("--out-dir", default="outputs/archive_goes")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--energy-scale", default="sarwade2025")
    args = ap.parse_args()

    cfg = Config(data_root=Path(args.data_root), out_dir=Path(args.out_dir))
    if args.cache_dir:
        cfg.cache_dir = Path(args.cache_dir)
    cfg.pre.solexs_energy_scale = args.energy_scale
    dt = cfg.pre.dt_seconds

    sources = [s for s in index_sources([cfg.data_root]) if s.kind == "solexs"]
    entries = build_cache(sources, cfg.pre, cache_dir_for(cfg), workers=cfg.pre.cache_workers,
                          verbose=False)
    soft = load_cached(entries, cache_dir_for(cfg), "solexs")
    cfg.pre.label_source = "solexs"
    segs, _ = build_segments_from_raw(soft, [], cfg)
    truth = load_goes(Path(args.goes_dir))

    # --- 1. flux cross-calibration ---------------------------------------
    rate, flux, observed_days = [], [], 0.0
    for s in segs:
        obs = s.soft_mask > 0
        observed_days += float(obs.sum()) * dt / 86400
        g = truth.flux_on_grid(s.time_unix)
        ok = obs & np.isfinite(g) & np.isfinite(s.goes_long) & (s.goes_long > 0)
        rate.append(s.goes_long[ok])
        flux.append(g[ok])
    rate = np.concatenate(rate).astype(np.float64)
    flux = np.concatenate(flux)
    lx, ly = np.log10(rate), np.log10(flux)
    b, a = np.polyfit(lx, ly, 1)
    resid = ly - (a + b * lx)
    decades = {}
    for lo_c in ("A1.0", "B1.0", "C1.0", "M1.0", "X1.0"):
        lo = class_flux(lo_c)
        m = (flux >= lo) & (flux < 10 * lo)
        if m.sum() > 50:
            decades[lo_c[0]] = {
                "n_bins": int(m.sum()),
                "median_rate_per_Wm2": float(np.median(rate[m] / flux[m])),
                "fit_residual_dex_sd": float(np.std(resid[m])),
            }
    calib = {
        "n_bins": int(lx.size),
        "fit": f"log10(F_GOES) = {a:.3f} + {b:.3f} * log10(rate_SoLEXS)",
        "slope": float(b), "intercept": float(a),
        "residual_dex_sd": float(np.std(resid)),
        "residual_dex_p05_p95": [float(np.percentile(resid, 5)), float(np.percentile(resid, 95))],
        "pearson_log": float(np.corrcoef(lx, ly)[0, 1]),
        "by_goes_decade": decades,
    }

    # --- 2. label agreement where SoLEXS observed the GOES peak ---------
    ours = np.array(sorted(e.peak_unix for s in segs for e in s.events))
    obs_at = []
    for f in truth.flares:
        hit = False
        for s in segs:
            if s.time_unix[0] <= f.peak_unix <= s.time_unix[-1]:
                i = int(np.clip(np.floor((f.peak_unix - s.time_unix[0]) / dt), 0, len(s) - 1))
                hit = bool(s.soft_mask[i] > 0)
                break
        obs_at.append(hit)
    obs_at = np.array(obs_at)
    peaks = np.array([f.peak_unix for f in truth.flares])
    classes = np.array([f.goes_class[:1] for f in truth.flares])

    def nearest(a_, b_):
        if b_.size == 0:
            return np.full(a_.size, np.inf)
        i = np.clip(np.searchsorted(b_, a_), 1, b_.size - 1)
        return np.minimum(np.abs(a_ - b_[i - 1]), np.abs(a_ - b_[i]))

    d = nearest(peaks, ours)
    agree = {}
    for letter in ("B", "C", "M", "X"):
        m = obs_at & (classes == letter)
        if m.any():
            agree[letter] = {"goes_flares_while_solexs_observed": int(m.sum()),
                             "found_within_10_min": float(np.mean(d[m] <= 600))}
    in_obs = np.array([any(s.time_unix[0] <= p <= s.time_unix[-1] for s in segs) for p in peaks])
    goes_peaks_in_period = np.sort(peaks[in_obs])
    d_ours = nearest(ours, goes_peaks_in_period)

    # --- 3. time budget on SoLEXS-observed bins --------------------------
    in_solexs, in_goes_c, in_goes_m, n_obs = 0.0, 0.0, 0.0, 0.0
    for s in segs:
        obs = s.soft_mask > 0
        n_obs += obs.sum()
        in_solexs += (s.in_flare[obs] > 0).sum()
        for floor, acc in (("C1.0", "c"), ("M1.0", "m")):
            mask = np.zeros(len(s), bool)
            for f in truth.flares:
                if f.peak_flux >= class_flux(floor) and f.end_unix >= s.time_unix[0] and f.start_unix <= s.time_unix[-1]:
                    i0 = int(np.searchsorted(s.time_unix, f.start_unix))
                    i1 = int(np.searchsorted(s.time_unix, f.end_unix, side="right"))
                    mask[i0:i1] = True
            if acc == "c":
                in_goes_c += (mask & obs).sum()
            else:
                in_goes_m += (mask & obs).sum()

    out = {
        "energy_scale": args.energy_scale,
        "solexs_observed_days": observed_days,
        "flux_calibration": calib,
        "label_agreement": {
            "goes_found_by_solexs_detector": agree,
            "solexs_detections": int(ours.size),
            "solexs_detections_with_goes_peak_within_10_min": float(np.mean(d_ours <= 600)),
        },
        "time_in_flare_on_solexs_observed_bins": {
            "solexs_detector": float(in_solexs / n_obs),
            "goes_ge_C": float(in_goes_c / n_obs),
            "goes_ge_M": float(in_goes_m / n_obs),
        },
    }
    p = Path(args.out_dir) / "reports" / "goes_crosscheck.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"SoLEXS observed {observed_days:.1f} d; {calib['n_bins']} bins with valid GOES flux")
    print(f"flux: {calib['fit']}   residual sd {calib['residual_dex_sd']:.3f} dex, "
          f"r(log) = {calib['pearson_log']:.3f}")
    for k, v in decades.items():
        print(f"  GOES {k}-level: n={v['n_bins']}, rate per W/m^2 median {v['median_rate_per_Wm2']:.3g}, "
              f"residual sd {v['fit_residual_dex_sd']:.3f} dex")
    for k, v in agree.items():
        print(f"GOES {k}: {v['goes_flares_while_solexs_observed']} flares while SoLEXS observed, "
              f"{100 * v['found_within_10_min']:.0f}% found by the SoLEXS detector")
    print(f"SoLEXS detections matching a GOES peak within 10 min: "
          f"{100 * out['label_agreement']['solexs_detections_with_goes_peak_within_10_min']:.0f}% of {ours.size}")
    tb = out["time_in_flare_on_solexs_observed_bins"]
    print(f"time in flare: SoLEXS detector {100 * tb['solexs_detector']:.1f}%, GOES >=C "
          f"{100 * tb['goes_ge_C']:.1f}%, GOES >=M {100 * tb['goes_ge_M']:.1f}%")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
