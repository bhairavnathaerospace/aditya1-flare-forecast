"""Figure for the master catalogue: detection by class, SoLEXS class vs GOES, one example day.

    python -m solarflare catalog-figure --day 2026-07-04

Reads <catalog>/catalog_summary.json and master_catalog.csv (from
python -m solarflare catalog) and the cache for the light curves; writes
<catalog>/catalog_overview.png.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from solarflare.settings import load_settings

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


from solarflare.catalog.detect import PiecewiseCalibration, to_minutes
from solarflare.io.goes import class_flux, load_goes
from solarflare.preprocess.cache import load_cached
from solarflare.preprocess.timeline import stitch

INK, SOFT, HARD, BOTH, GOES, MUTED = "#1d2733", "#d9822b", "#7a4fd1", "#2a8c7c", "#4a6fa5", "#98a2ad"


def ts(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp()


def dt64(x):
    return np.array([datetime.fromtimestamp(v, UTC) for v in np.atleast_1d(x)])


def main(argv=None) -> int:
    S = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=str(S.catalog))
    ap.add_argument("--cache-dir", default=str(S.cache))
    ap.add_argument("--goes-dir", default=str(S.goes_dir))
    ap.add_argument("--day", default="2026-07-04")
    args = ap.parse_args(argv)
    cat = Path(args.catalog)
    summ = json.loads((cat / "catalog_summary.json").read_text("utf-8"))
    rows = list(csv.DictReader((cat / "master_catalog.csv").open(encoding="utf-8")))

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig = plt.figure(figsize=(12, 8.2), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.55, wspace=0.25,
                          left=0.07, right=0.93, top=0.86, bottom=0.07)
    fig.text(0.07, 0.96, "Aditya-L1 master flare catalogue", fontsize=15, weight="bold", color=INK)
    c = summ["counts"]
    fig.text(0.07, 0.93, f"{c['master']:,} flares from SoLEXS and HEL1OS detected independently, "
             f"{summ['period'][0][:10]} to {summ['period'][1][:10]}; scored against the GOES-18 list "
             f"(test period from {summ['test_period_start'][:10]})", fontsize=9.5, color=MUTED)

    # --- (a) recall by class, test period ------------------------------
    ax = fig.add_subplot(gs[0, 0])
    rec = summ["test"]["recall"]
    ceil = summ["ceiling_rule_on_goes_flux"]["recall"]
    cls = ["B", "C", "M", "X"]
    x = np.arange(len(cls))
    w = 0.26
    soft = [rec[k]["both_soft_only_recall"] or 0 for k in cls]
    hard = [rec[k]["hard_recall"] or 0 for k in cls]
    comb = [rec[k]["both_combined_recall"] or 0 for k in cls]
    comb_ch = [rec[k].get("both_combined_recall_chance") or 0 for k in cls]
    ax.bar(x - w, hard, w, color=HARD, label="HEL1OS alone")
    ax.bar(x, soft, w, color=SOFT, label="SoLEXS alone")
    ax.bar(x + w, comb, w, color=BOTH, label="combined")
    ax.scatter(x + w, comb_ch, marker="_", s=260, color=INK, linewidths=1.6, zorder=3,
               label="combined, HEL1OS shifted ±2 h (chance)")
    ax.scatter(x, [ceil[k] or 0 for k in cls], marker="D", s=16, color=GOES, zorder=3,
               label="same rule on GOES's own flux")
    for i, k in enumerate(cls):
        ax.text(x[i] + w, comb[i] + 0.02, f"{100 * comb[i]:.0f}%", ha="center", fontsize=8, color=INK)
        ax.text(x[i], -0.1, f"n={rec[k]['both_observed_n']}", ha="center", fontsize=7.5, color=MUTED,
                transform=ax.get_xaxis_transform())
    ax.set_xticks(x, [f"GOES {k}" for k in cls])
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Flares found, test period, both instruments observing", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=7.2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)

    # --- (b) class from SoLEXS vs GOES, test period --------------------
    ax = fig.add_subplot(gs[0, 1])
    pts = [(float(r["peak_flux_solexs_Wm2"]), class_flux(r["goes_class"]))
           for r in rows if r["test_period"] == "1" and r["goes_class"] and r["peak_flux_solexs_Wm2"]
           and r["class_error_dex"] != ""]
    a = np.array(pts)
    ax.scatter(a[:, 1], a[:, 0], s=6, color=SOFT, alpha=0.55, linewidths=0)
    lo, hi = 1e-7, 3e-3
    ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8)
    for b in (1e-6, 1e-5, 1e-4):
        ax.axvline(b, color=MUTED, lw=0.5, ls=":")
        ax.axhline(b, color=MUTED, lw=0.5, ls=":")
    for b, lab in ((3e-7, "B"), (3e-6, "C"), (3e-5, "M"), (3e-4, "X")):
        ax.text(b, 1.3e-7, lab, ha="center", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("GOES-18 peak flux (W/m²)")
    ax.set_ylabel("SoLEXS-derived peak flux (W/m²)")
    k = summ["test"]["class"]
    ax.set_title(f"Class from SoLEXS alone: {100 * k['letter_agreement']:.0f}% same letter, "
                 f"median error {k['median_abs_dex']} dex", loc="left", fontsize=10, color=INK)

    # --- (c) example day -------------------------------------------------
    ax = fig.add_subplot(gs[1, :])
    d0 = datetime.strptime(args.day, "%Y-%m-%d").replace(tzinfo=UTC).timestamp()
    d1 = d0 + 86400.0
    cache = Path(args.cache_dir)
    entries = json.loads((cache / "manifest.json").read_text("utf-8"))
    ts_, rs_, ok_ = [], [], []
    for e in entries:
        if e["source"]["kind"] != "solexs" or e.get("status") != "ok" or e["t_stop"] < d0 or e["t_start"] > d1:
            continue
        with np.load(cache / f"{e['key']}.npz") as z:
            names = [str(n) for n in z["names"]]
            ts_.append(z["time_unix"])
            rs_.append(z["values"][:, names.index("slx_goes_long")])
            ok_.append(z["coverage"] > 0)
    t = np.concatenate(ts_)
    order = np.argsort(t)
    tm, rm, _, vm = to_minutes(t[order], np.concatenate(rs_)[order], np.concatenate(ok_)[order], 20.0)
    ck = summ["calibration_knots"]
    flux = PiecewiseCalibration(ck["log10_rate"], ck["log10_flux"])(rm)
    m = (tm >= d0) & (tm < d1) & vm
    truth = load_goes(Path(args.goes_dir))
    g = (truth.time_unix >= d0) & (truth.time_unix < d1)
    ax.plot(dt64(truth.time_unix[g]), truth.xrsb[g], color=GOES, lw=1.0, label="GOES-18 XRS-B (reference)")
    ax.plot(dt64(tm[m]), flux[m], color=SOFT, lw=1.0, label="SoLEXS, calibrated to GOES units")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 3e-4)
    ax.set_ylabel("flux (W/m²)")
    for b, lab in ((1e-6, "C"), (1e-5, "M"), (1e-4, "X")):
        ax.axhline(b, color=MUTED, lw=0.5, ls=":")
        ax.text(0.004, b, f" {lab}1", transform=ax.get_yaxis_transform(), fontsize=7.5, color=MUTED, va="bottom")

    ax2 = ax.twinx()
    hl = stitch([s for s in load_cached(entries, cache, "hel1os")
                 if s.time_unix[-1] >= d0 and s.time_unix[0] < d1], 20.0, 21600.0)
    for h in hl:
        names = list(h.names)
        v = h.values[:, names.index("hls_cdte1_5_20keV")] + h.values[:, names.index("hls_cdte2_5_20keV")]
        on = (h.values[:, names.index("hls_cdte1_cov")] > 0) & (h.values[:, names.index("hls_cdte2_cov")] > 0)
        k_ = (h.time_unix >= d0) & (h.time_unix < d1) & on
        ax2.plot(dt64(h.time_unix[k_]), v[k_], color=HARD, lw=0.7, alpha=0.8, label="HEL1OS CdTe 5-20 keV")
    ax2.set_yscale("log")
    ax2.set_ylim(0.5, 3e5)
    ax2.set_ylabel("HEL1OS counts/s", color=HARD)
    ax2.spines["right"].set_visible(True)
    ax2.tick_params(axis="y", colors=HARD)

    day_rows = [r for r in rows if d0 <= ts(r["peak_utc"]) < d1]
    for r in day_rows:
        col = {"soft+hard": BOTH, "soft": SOFT, "hard": HARD}[r["origin"]]
        ta = ts(r["alert_utc"])
        ax.axvline(datetime.fromtimestamp(ta, UTC), color=col, lw=0.8, alpha=0.55, ymax=0.06)
        if r["class_solexs"] and class_flux(r["class_solexs"]) >= 1e-5:
            tp = ts(r["peak_utc"])
            ax.annotate(r["class_solexs"], (datetime.fromtimestamp(tp, UTC), class_flux(r["class_solexs"])),
                        xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.5, color=INK)
    n_by = {o: sum(1 for r in day_rows if r["origin"] == o) for o in ("soft+hard", "soft", "hard")}
    ax.set_title(f"{args.day}: {len(day_rows)} catalogue flares ({n_by['soft+hard']} both instruments, "
                 f"{n_by['soft']} SoLEXS only, {n_by['hard']} HEL1OS only)\n"
                 f"ticks along the bottom mark alert times; labels give the SoLEXS-derived class of M and X flares",
                 loc="left", fontsize=10, color=INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlim(datetime.fromtimestamp(d0, UTC), datetime.fromtimestamp(d1, UTC))
    ax.set_xlabel("UTC")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2[:1], l1 + l2[:1], fontsize=7.5, frameon=False, loc="upper left", ncol=3)

    out = cat / "catalog_overview.png"
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
