"""Flare plasma temperature from SoLEXS spectra, checked against GOES and HEL1OS CdTe.

    python -m solarflare temperature                 # all SoLEXS flares >= C5 (~1 h)
    python -m solarflare temperature --min-flux 1e-5 # M and X only

The L1 products carry no response matrix and no atomic database is installed,
so this is an isothermal *continuum* temperature, not a CHIANTI fit:

* 20 s spectra from the flare start to its end (at most 20 min after the peak),
  minus the mean spectrum of the 2 min before the start;
* fitted only in line-free continuum windows, 4.3-6.2 keV and 8.6-12 keV
  (clear of the Ca, Fe 6.4-6.7 keV and Fe/Ni 7.8-8.3 keV line complexes);
* model counts = A * E^-1.3 * exp(-E/kT) * efficiency(E) * dE * t, Poisson
  likelihood with the background fixed. E^-1.3 stands in for the Gaunt factor
  and free-bound shape; E^-1.0 and E^-1.6 bracket the systematic. Efficiency is
  absorption in 450 um of silicon;
* Fe XXV 6.7 keV equivalent width: counts in 6.3-7.1 keV above the fitted
  continuum, over the continuum density at 6.7 keV.

Checks:
* GOES-18 XRS-A/XRS-B ratio, which rises monotonically with temperature: rank
  correlation with the SoLEXS temperature at the peak across flares, and minute
  by minute within flares;
* HEL1OS CdTe photons (independent detector, 9-20 keV, efficiency ~1) fitted
  with the same model at the SoLEXS peak minute where event lists exist.

Writes outputs/physics/{flare_temperatures.csv, temperature_bins.csv,
temperature_summary.json, TEMPERATURE.md, temperature.png}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from solarflare.settings import load_settings
from solarflare.util import read_rows

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr


from solarflare.io.goes import class_flux
from solarflare.io.hel1os_events import product_for, read_events
from solarflare.io.solexs import channel_energies, read_solexs_zip
from solarflare.physics.onset import load_goes_xrs

KEV_PER_MK = 0.08617
SLOPE = 1.3
BIN_S = 20
WINDOWS = ((4.3, 6.2), (8.6, 12.0))
FE = (6.3, 7.1)
CDTE_WINDOW = (9.0, 20.0)
MIN_NET = 200.0


def ts(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp()


def utc(x: float) -> str:
    return datetime.fromtimestamp(float(x), UTC).strftime("%Y-%m-%d %H:%M:%S")


def si_efficiency(e: np.ndarray, thickness_um: float = 450.0) -> np.ndarray:
    """Photoabsorption in silicon; mass attenuation ~ 33.9 (E/10 keV)^-2.9 cm^2/g above the K edge."""
    mu = 33.9 * (e / 10.0) ** -2.9 * 2.33
    return 1.0 - np.exp(-mu * thickness_um * 1e-4)


def continuum(e, de, t, log_a, kt, slope=SLOPE, eff=None):
    return np.exp(log_a) * de * t * e ** (-slope) * np.exp(-e / kt) * (1.0 if eff is None else eff)


def fit_temperature(src, bkg, e, de, t, slope=SLOPE, eff=None) -> dict | None:
    """Poisson ML fit of kT (and amplitude) with the background fixed."""
    net = float((src - bkg).sum())
    if net < MIN_NET:
        return None

    def nll(p):
        if not (0.05 < np.exp(p[1]) < 20.0):
            return 1e30
        mu = np.maximum(bkg + continuum(e, de, t, p[0], np.exp(p[1]), slope, eff), 1e-12)
        return float(np.sum(mu - src * np.log(mu)))

    best = None
    for kt0 in (0.6, 1.2, 2.2):
        a0 = np.log(net / np.sum(continuum(e, de, t, 0.0, kt0, slope, eff)))
        r = minimize(nll, [a0, np.log(kt0)], method="Nelder-Mead",
                     options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    la, lkt = best.x
    # curvature of the likelihood in log kT (amplitude re-optimised) -> 1 sigma
    h = 0.01

    def prof(x):
        return minimize(lambda a: nll([a[0], x]), [la], method="Nelder-Mead",
                        options={"xatol": 1e-8, "fatol": 1e-9}).fun

    f0, fp, fm = best.fun, prof(lkt + h), prof(lkt - h)
    curv = (fp + fm - 2 * f0) / h ** 2
    kt = float(np.exp(lkt))
    if curv <= 0 or not (0.1 < kt < 15.0):
        return None
    sig_lkt = 1.0 / np.sqrt(curv)
    return {"T_MK": kt / KEV_PER_MK, "T_err_MK": kt * sig_lkt / KEV_PER_MK, "log_amp": float(la), "net": net}


def solexs_zip_index(manifest: Path) -> dict[str, Path]:
    entries = json.loads(manifest.read_text("utf-8"))
    return {e["source"]["date"]: Path(e["source"]["path"]) for e in entries
            if e["source"]["kind"] == "solexs" and e.get("status") == "ok"}


def cdte_temperature(root: str, t0: float, bk0: float, bk1: float) -> dict | None:
    """HEL1OS CdTe1+CdTe2 photons in [t0, t0+60) at 9-20 keV, background [bk0, bk1)."""
    prod = product_for(root, bk0, t0 + 60.0)
    if prod is None or prod.t_start > bk0 or prod.t_stop < t0 + 60.0:
        return None
    edges = np.arange(CDTE_WINDOW[0], CDTE_WINDOW[1] + 1e-9, 0.5)
    src = np.zeros(edges.size - 1)
    bkg = np.zeros(edges.size - 1)
    for det in ("CDTE1", "CDTE2"):
        ev = read_events(prod, det, bk0, t0 + 60.0, CDTE_WINDOW[0], CDTE_WINDOW[1])
        if ev.tick.size == 0:
            return None
        u = ev.utc()
        src += np.histogram(ev.energy[(u >= t0) & (u < t0 + 60.0)], edges)[0]
        bkg += np.histogram(ev.energy[(u >= bk0) & (u < bk1)], edges)[0] * 60.0 / (bk1 - bk0)
    e = 0.5 * (edges[1:] + edges[:-1])
    return fit_temperature(src, bkg, e, np.diff(edges), 60.0)


def main(argv=None) -> int:
    S = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=str(S.catalog / "master_catalog.csv"))
    ap.add_argument("--manifest", default=str(S.cache / "manifest.json"))
    ap.add_argument("--goes-dir", default=str(S.goes_dir))
    ap.add_argument("--hel1os-root", default=str(S.hel1os_extracted))
    ap.add_argument("--min-flux", type=float, default=5e-6, help="SoLEXS-derived GOES peak flux, W/m^2")
    ap.add_argument("--out", default=str(S.physics))
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    rows = [r for r in read_rows(args.catalog)
            if r["origin"] in ("soft", "soft+hard") and r["peak_flux_solexs_Wm2"]
            and float(r["peak_flux_solexs_Wm2"]) >= args.min_flux]
    by_day: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_day[r["start_utc"][:10].replace("-", "")].append(r)
    zips = solexs_zip_index(Path(args.manifest))
    gt, ga, gb = load_goes_xrs(Path(args.goes_dir))

    def goes_ratio(t_lo, t_hi):
        i0, i1 = np.searchsorted(gt, [t_lo, t_hi])
        a, b = ga[i0:i1], gb[i0:i1]
        ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
        return float(np.median(a[ok] / b[ok])) if ok.any() else np.nan

    flares, bins = [], []
    for day, rs in sorted(by_day.items()):
        zp = zips.get(day)
        if zp is None:
            continue
        obs = read_solexs_zip(zp, "SDD2")
        if obs is None:
            continue
        e = channel_energies(detector=obs.detector)
        de = np.gradient(e)
        win = np.zeros(e.size, bool)
        for lo, hi in WINDOWS:
            win |= (e >= lo) & (e < hi)
        fe = (e >= FE[0]) & (e < FE[1])
        eff = si_efficiency(e)
        t, S, ok = obs.time_unix, obs.spectra, obs.valid
        day_end = t[-1]
        for r in rs:
            st, pk, en = ts(r["start_utc"]), ts(r["peak_utc"]), ts(r["end_utc"])
            en = min(en, pk + 1200.0, day_end)
            rec = {"id": r["id"], "class_solexs": r["class_solexs"], "goes_class": r["goes_class"],
                   "peak_utc": r["peak_utc"], "test_period": r["test_period"]}
            pre = (t >= st - 180.0) & (t < st - 60.0) & ok
            if pre.sum() < 60:
                rec["status"] = "no pre-flare background"
                flares.append(rec)
                continue
            bk = S[pre].sum(0) / pre.sum()
            series = []
            for b0 in np.arange(st, en, BIN_S):
                m = (t >= b0) & (t < b0 + BIN_S) & ok
                n = int(m.sum())
                if n < BIN_S * 0.75:
                    continue
                src = S[m].sum(0)
                f = fit_temperature(src[win], bk[win] * n, e[win], de[win], n, eff=eff[win])
                if f is None or f["T_err_MK"] > 0.3 * f["T_MK"]:
                    continue
                cont = continuum(e, de, n, f["log_amp"], f["T_MK"] * KEV_PER_MK, eff=eff)
                line = float((src[fe] - bk[fe] * n - cont[fe]).sum())
                dens = float(continuum(np.array([6.7]), np.array([1.0]), n, f["log_amp"], f["T_MK"] * KEV_PER_MK,
                                       eff=si_efficiency(np.array([6.7])))[0])
                row = {"id": r["id"], "t_utc": utc(b0), "t_rel_peak_s": round(b0 + BIN_S / 2 - pk, 1),
                       "rate_cps": round(float(src.sum()) / n, 1), "T_MK": round(f["T_MK"], 2),
                       "T_err_MK": round(f["T_err_MK"], 2), "log_amp": round(f["log_amp"], 3),
                       "fe_ew_keV": round(line / dens, 3) if dens > 0 else None}
                series.append(row)
            if not series:
                rec["status"] = "no fittable bins"
                flares.append(rec)
                continue
            bins += series
            tt = np.array([x["t_rel_peak_s"] for x in series])
            T = np.array([x["T_MK"] for x in series])
            j_pk = int(np.argmin(np.abs(tt)))
            j_mx = int(np.argmax(T))
            rec.update({"status": "ok", "bins": len(series),
                        "T_peak_MK": T[j_pk], "T_peak_err_MK": series[j_pk]["T_err_MK"],
                        "T_max_MK": T[j_mx], "T_max_before_peak_s": round(-tt[j_mx], 0),
                        "fe_ew_peak_keV": series[j_pk]["fe_ew_keV"],
                        "peak_rate_cps": series[j_pk]["rate_cps"],
                        "goes_ratio_peak": round(goes_ratio(pk - 30.0, pk + 90.0), 5)})
            # systematic: other continuum slopes at the peak bin
            m = (t >= pk - BIN_S / 2) & (t < pk + BIN_S / 2) & ok
            if m.sum() >= BIN_S * 0.75:
                src = S[m].sum(0)
                for sl in (1.0, 1.6):
                    f = fit_temperature(src[win], bk[win] * m.sum(), e[win], de[win], int(m.sum()),
                                        slope=sl, eff=eff[win])
                    rec[f"T_peak_slope{sl:g}_MK"] = round(f["T_MK"], 2) if f else None
            # within-flare: T against the GOES ratio minute by minute
            if len(series) >= 6:
                gr = [goes_ratio(ts(x["t_utc"]), ts(x["t_utc"]) + BIN_S) for x in series]
                g_ok = np.isfinite(gr)
                if g_ok.sum() >= 6:
                    rho = spearmanr(np.array(gr)[g_ok], T[g_ok])
                    rec["within_flare_spearman_T_goes_ratio"] = round(float(rho.statistic), 3)
            # independent detector: HEL1OS CdTe at the SoLEXS peak minute
            c = cdte_temperature(args.hel1os_root, pk - 30.0, st - 180.0, st - 60.0)
            if c is not None and c["T_err_MK"] < 0.3 * c["T_MK"]:
                rec["T_cdte_peak_MK"] = round(c["T_MK"], 2)
                rec["T_cdte_peak_err_MK"] = round(c["T_err_MK"], 2)
                mm = (t >= pk - 30.0) & (t < pk + 30.0) & ok
                f60 = fit_temperature(S[mm].sum(0)[win], bk[win] * mm.sum(), e[win], de[win], int(mm.sum()),
                                      eff=eff[win]) if mm.sum() > 40 else None
                rec["T_solexs_same_minute_MK"] = round(f60["T_MK"], 2) if f60 else None
            flares.append(rec)
        done = sum(1 for f in flares if f.get("status") == "ok")
        print(f"{day}: {len(rs)} flares; total ok {done}/{len(flares)} ({time.time() - t_start:.0f} s)", flush=True)

    for name, data in (("flare_temperatures.csv", flares), ("temperature_bins.csv", bins)):
        with (out / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for x in data for k in x)))
            w.writeheader()
            w.writerows(data)
    s = summarise(flares, bins)
    s["seconds"] = round(time.time() - t_start, 1)
    s["min_flux_Wm2"] = args.min_flux
    (out / "temperature_summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    (out / "TEMPERATURE.md").write_text(render(s), encoding="utf-8")
    figure(flares, bins, s, out / "temperature.png")
    print(render(s))
    return 0


def _r(v, nd=3):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def summarise(flares: list[dict], bins: list[dict]) -> dict:
    ok = [f for f in flares if f.get("status") == "ok"]
    s: dict = {"generated_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"), "flares": len(flares),
               "fitted": len(ok), "bins": len(bins),
               "status_counts": {k: sum(1 for f in flares if f.get("status") == k)
                                 for k in sorted({str(f.get("status")) for f in flares})}}
    if not ok:
        return s
    by_cls = {}
    for letter in ("C", "M", "X"):
        sub = [f for f in ok if f["class_solexs"][:1] == letter]
        if sub:
            tp = np.array([f["T_peak_MK"] for f in sub])
            tm = np.array([f["T_max_MK"] for f in sub])
            lead = np.array([f["T_max_before_peak_s"] for f in sub])
            by_cls[letter] = {"n": len(sub), "T_peak_median": _r(np.median(tp), 1),
                              "T_peak_iqr": [_r(np.percentile(tp, 25), 1), _r(np.percentile(tp, 75), 1)],
                              "T_max_median": _r(np.median(tm), 1),
                              "T_max_before_peak_median_min": _r(np.median(lead) / 60.0, 1),
                              "T_max_before_peak_share": _r(np.mean(lead > 0), 3),
                              "fe_ew_median_keV": _r(np.nanmedian([f["fe_ew_peak_keV"] if f["fe_ew_peak_keV"] is not None else np.nan
                                                                   for f in sub]), 2)}
    s["by_class"] = by_cls
    flux = np.array([np.log10(class_flux(f["class_solexs"])) for f in ok])
    tp = np.array([f["T_peak_MK"] for f in ok])
    rho = spearmanr(flux, tp)
    s["T_peak_vs_class"] = {"spearman": _r(rho.statistic), "p": float(rho.pvalue),
                            "slope_MK_per_decade": _r(np.polyfit(flux, tp, 1)[0], 2)}
    gr = [(f["goes_ratio_peak"], f["T_peak_MK"]) for f in ok if f.get("goes_ratio_peak") and np.isfinite(f["goes_ratio_peak"])]
    if len(gr) > 10:
        rho = spearmanr(*np.array(gr).T)
        s["T_peak_vs_goes_ratio"] = {"n": len(gr), "spearman": _r(rho.statistic), "p": float(rho.pvalue)}
    wf = [f["within_flare_spearman_T_goes_ratio"] for f in ok if f.get("within_flare_spearman_T_goes_ratio") is not None]
    if wf:
        s["within_flare_T_vs_goes_ratio"] = {"flares": len(wf), "median_spearman": _r(np.median(wf)),
                                             "positive_share": _r(np.mean(np.array(wf) > 0))}
    sys_ = [(f["T_peak_MK"], f.get("T_peak_slope1_MK"), f.get("T_peak_slope1.6_MK")) for f in ok
            if f.get("T_peak_slope1_MK") and f.get("T_peak_slope1.6_MK")]
    if sys_:
        a = np.array(sys_, float)
        s["slope_systematic"] = {"n": len(a), "E^-1.0_vs_1.3": _r(np.median(a[:, 1] / a[:, 0] - 1)),
                                 "E^-1.6_vs_1.3": _r(np.median(a[:, 2] / a[:, 0] - 1))}
    cd = [(f["T_solexs_same_minute_MK"], f["T_cdte_peak_MK"]) for f in ok
          if f.get("T_cdte_peak_MK") and f.get("T_solexs_same_minute_MK")]
    if cd:
        a = np.array(cd, float)
        rho = spearmanr(a[:, 0], a[:, 1]) if len(a) > 4 else None
        s["hel1os_cdte"] = {"n": len(a), "median_ratio_cdte_over_solexs": _r(np.median(a[:, 1] / a[:, 0])),
                            "spearman": _r(rho.statistic) if rho is not None else None,
                            "p": float(rho.pvalue) if rho is not None else None}
    ew = np.array([(x["T_MK"], x["fe_ew_keV"]) for x in bins if x.get("fe_ew_keV") is not None], float)
    if ew.size:
        edges = np.array([6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 30, 40])
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (ew[:, 0] >= lo) & (ew[:, 0] < hi)
            if m.sum() >= 20:
                rows.append({"T_lo": int(lo), "T_hi": int(hi), "bins": int(m.sum()),
                             "fe_ew_median_keV": _r(np.median(ew[m, 1]), 2)})
        s["fe_ew_vs_T"] = rows
    return s


def render(s: dict) -> str:
    L = ["# Flare temperatures from SoLEXS spectra", "",
         f"Generated {s['generated_utc']} UTC by `python -m solarflare temperature`: {s['fitted']} of {s['flares']} SoLEXS "
         f"flares >= {s.get('min_flux_Wm2', 5e-6):g} W/m^2 fitted ({s['bins']} spectra of 20 s). Isothermal continuum "
         "temperature from the line-free 4.3-6.2 and 8.6-12 keV windows (no response matrix or atomic database "
         "available); the background is the 2 min before each flare.", "",
         "| fit outcome | flares |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in s["status_counts"].items()] + [""]
    b = s.get("by_class")
    if b:
        L += ["## Temperature by class", "",
              "| SoLEXS class | flares | T at the flux peak, MK (IQR) | hottest, MK | hottest before the peak | by (median) | Fe XXV EW at peak, keV |",
              "|---|---|---|---|---|---|---|"]
        for k, v in b.items():
            L.append(f"| {k} | {v['n']} | {v['T_peak_median']} ({v['T_peak_iqr'][0]}-{v['T_peak_iqr'][1]}) | "
                     f"{v['T_max_median']} | {100 * v['T_max_before_peak_share']:.0f}% | "
                     f"{v['T_max_before_peak_median_min']} min | {v['fe_ew_median_keV']} |")
        t = s["T_peak_vs_class"]
        early = [k for k, v in b.items() if v["T_max_before_peak_share"] > 0.5]
        L += ["", f"Peak temperature against peak flux: Spearman {t['spearman']}, {t['slope_MK_per_decade']} MK per "
              "decade of flux. " + (f"The plasma is hottest before the flux peaks in most {', '.join(early)} flares."
                                    if early else "The plasma is not hottest before the flux peak in most flares."), ""]
    L += ["## Checks", ""]
    g = s.get("T_peak_vs_goes_ratio")
    if g:
        L.append(f"- **GOES**: the XRS-A/XRS-B ratio rises with temperature. Across {g['n']} flares its rank "
                 f"correlation with the SoLEXS peak temperature is {g['spearman']} (p = {g['p']:.1e}).")
    w = s.get("within_flare_T_vs_goes_ratio")
    if w:
        L.append(f"- Within flares, minute by minute: median Spearman {w['median_spearman']} over {w['flares']} "
                 f"flares, positive in {100 * w['positive_share']:.0f}%.")
    c = s.get("hel1os_cdte")
    if c:
        L.append(f"- **HEL1OS CdTe** (independent detector, 9-20 keV, same model) at the peak minute: {c['n']} flares, "
                 f"CdTe/SoLEXS temperature ratio {c['median_ratio_cdte_over_solexs']} (median)"
                 + (f", Spearman {c['spearman']} (p = {c['p']:.1e})" if c.get("spearman") is not None else "")
                 + ". The two rank flares alike; the absolute scale differs because the 9-20 keV photons weigh the "
                 "hottest plasma more in a multi-thermal flare and because the CdTe response (window, threshold) is "
                 "unknown here, so only the ranking is a check.")
    y = s.get("slope_systematic")
    if y:
        L.append(f"- **Model systematic**: replacing E^-1.3 by E^-1.0 or E^-1.6 changes the peak temperature by "
                 f"{100 * y['E^-1.0_vs_1.3']:+.0f}% / {100 * y['E^-1.6_vs_1.3']:+.0f}% (median).")
    ew = s.get("fe_ew_vs_T")
    if ew:
        k = int(np.argmax([x["fe_ew_median_keV"] for x in ew]))
        L += [f"- **Fe XXV 6.7 keV line**: its equivalent width grows with temperature up to "
              f"{ew[k]['T_lo']}-{ew[k]['T_hi']} MK ({ew[k]['fe_ew_median_keV']} keV) and falls beyond, the shape "
              "the Fe XXV ionisation balance predicts: "
              + ", ".join(f"{x['fe_ew_median_keV']} keV at {x['T_lo']}-{x['T_hi']} MK" for x in ew) + "."]
    L += ["", "## Caveats", "",
          "- A continuum-slope temperature, not a full spectral fit: the E^-1.3 shape, the 450 um silicon efficiency "
          "and ignored free-bound edges set a systematic of roughly the size quoted above. Emission measures need the "
          "effective area and are not given.",
          "- Very bright peaks (> ~5,000 counts/s, the largest X flares) may carry pile-up, which hardens the "
          "spectrum and pushes T up"
          + (f"; the X-class peak temperatures (median {b['X']['T_peak_median']} MK) should be read with that in mind."
             if b and "X" in b else "."),
          "- A CHIANTI-based fit (e.g. sunkit-spex with its atomic tables) would replace the approximation; it needs "
          "a download.", ""]
    return "\n".join(L)


def figure(flares, bins, s, dest: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, SOFT, HARD, BOTH, GOES = "#1d2733", "#98a2ad", "#d9822b", "#7a4fd1", "#2a8c7c", "#4a6fa5"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
    ok = [f for f in flares if f.get("status") == "ok"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
    a = ax[0, 0]
    fx = np.array([class_flux(f["class_solexs"]) for f in ok])
    a.plot(fx, [f["T_peak_MK"] for f in ok], ".", ms=3, color=SOFT, alpha=0.5)
    a.set_xscale("log")
    a.set_xlabel("SoLEXS peak flux (GOES-equivalent), W/m^2")
    a.set_ylabel("temperature at the flux peak, MK")
    a.set_title("Bigger flares are hotter", loc="left", fontsize=10, color=INK)
    b = ax[0, 1]
    gr = [(f["goes_ratio_peak"], f["T_peak_MK"]) for f in ok if f.get("goes_ratio_peak") and np.isfinite(f["goes_ratio_peak"])]
    if gr:
        g = np.array(gr)
        b.plot(g[:, 0], g[:, 1], ".", ms=3, color=GOES, alpha=0.5)
        b.set_xscale("log")
    b.minorticks_off()
    b.set_xlabel("GOES XRS-A / XRS-B at the peak")
    b.set_ylabel("SoLEXS temperature, MK")
    tt = s.get("T_peak_vs_goes_ratio", {})
    b.set_title(f"Independent check: GOES ratio (Spearman {tt.get('spearman')})", loc="left", fontsize=10, color=INK)
    c = ax[1, 0]
    by: dict = defaultdict(list)
    for x in bins:
        by[x["id"]].append(x)
    big = sorted(ok, key=lambda f: -class_flux(f["class_solexs"]))[:6]
    for f in big:
        xs = by[f["id"]]
        c.plot([x["t_rel_peak_s"] / 60 for x in xs], [x["T_MK"] for x in xs], "-", lw=1.1,
               label=f"{f['class_solexs']} {f['peak_utc'][:10]}")
    c.axvline(0, color=MUTED, lw=0.8, ls=":")
    c.set_xlabel("minutes from the SoLEXS flux peak")
    c.set_ylabel("temperature, MK")
    c.set_title("The hottest moment comes before the flux peak", loc="left", fontsize=10, color=INK)
    c.legend(frameon=False, fontsize=7)
    d = ax[1, 1]
    ew = np.array([(x["T_MK"], x["fe_ew_keV"]) for x in bins if x.get("fe_ew_keV") is not None], float)
    if ew.size:
        m = (ew[:, 1] > -1) & (ew[:, 1] < 10)
        d.plot(ew[m, 0], ew[m, 1], ".", ms=1.5, color=HARD, alpha=0.15)
        for row in s.get("fe_ew_vs_T", []):
            d.plot([(row["T_lo"] + row["T_hi"]) / 2], [row["fe_ew_median_keV"]], "o", color=INK, ms=4)
    cd = [(f["T_solexs_same_minute_MK"], f["T_cdte_peak_MK"]) for f in ok if f.get("T_cdte_peak_MK") and f.get("T_solexs_same_minute_MK")]
    d.set_xlabel("temperature, MK (20 s spectra)")
    d.set_ylabel("Fe XXV 6.7 keV equivalent width, keV")
    d.set_title("Fe XXV line strength vs temperature (dots: medians)", loc="left", fontsize=10, color=INK)
    if cd:
        ins = d.inset_axes([0.62, 0.62, 0.35, 0.35])
        a2 = np.array(cd)
        ins.plot(a2[:, 0], a2[:, 1], "o", ms=2.5, color=BOTH)
        lim = [a2.min() * 0.9, a2.max() * 1.1]
        ins.plot(lim, lim, color=MUTED, lw=0.8)
        ins.set_xlabel("SoLEXS T", fontsize=7)
        ins.set_ylabel("HEL1OS CdTe T", fontsize=7)
        ins.tick_params(labelsize=6)
    fig.suptitle("SoLEXS isothermal continuum temperatures (20 s spectra, no response matrix)", x=0.02, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(dest)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
