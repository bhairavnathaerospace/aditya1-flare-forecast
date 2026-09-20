"""Hard X-ray spectral index of every flare HEL1OS saw above 20 keV, from the photon lists.

    python -m solarflare hxr-spectra

For each flare with a CZT 20-40 keV burst in the master catalogue:

1. CZT1 and CZT2 events (calibrated energies, disabled pixels removed) from
   20 min before to 10 min after the CZT peak.
2. Background: the 10 min ending 2 min before the flare start (SoLEXS start
   if known, else the HEL1OS start), scaled by exposure.
3. Peak spectrum: the 16 s with the most 30-60 keV counts near the CZT peak.
   A power law in counts, N(E) ~ E^-gamma, is fitted with the Poisson
   likelihood (background fixed) from 30 keV up to the last energy bin with a
   3-sigma excess, per detector and for both together. Errors: profile
   likelihood (delta ln L = 0.5). The fit from 25 keV is kept as a thermal-
   contamination check.
4. Time-resolved: adaptive bins through the impulsive phase (>= 300 net counts
   at 30-100 keV, >= 4 s), gamma in each, to test the soft-hard-soft pattern
   (spectrum hardest at flux peaks, Grigis & Benz 2004).

There is no response matrix in the L1 products, so gamma is the *count*
spectral index. It approximates the photon index where the detector
efficiency is flat; CZT redistribution (K-escape, hole tailing) is not
modelled. The Am-241 59.5 keV calibration line is located in every flare's
background as an energy-scale check.

Writes outputs/physics/{hxr_spectra.csv, hxr_timeresolved.csv, hxr_summary.json, HXR.md}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from solarflare.settings import load_settings
from solarflare.util import read_rows

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr


from solarflare.io.hel1os_events import product_for, read_events

EDGES = np.geomspace(20.0, 150.0, 15)
E_REF = 35.0


def ts(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp()


def utc(x: float) -> str:
    return datetime.fromtimestamp(float(x), UTC).strftime("%Y-%m-%d %H:%M:%S")


def shape(gamma: float, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Integral of (E/E_REF)^-gamma over each bin (keV)."""
    if abs(gamma - 1.0) < 1e-6:
        return E_REF * np.log(hi / lo)
    g1 = 1.0 - gamma
    return E_REF * ((hi / E_REF) ** g1 - (lo / E_REF) ** g1) / g1


def fit_power_law(src: np.ndarray, bkg: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict | None:
    """Poisson ML fit of src = bkg + A * shape(gamma). ``bkg`` already scaled.
    Returns gamma, its profile-likelihood interval, amplitude (counts/keV at E_REF)."""
    if src.size < 3 or (src - bkg).sum() <= 0:
        return None

    def nll(p):
        mu = bkg + np.exp(p[0]) * shape(p[1], lo, hi)
        mu = np.maximum(mu, 1e-12)
        return float(np.sum(mu - src * np.log(mu)))

    net = np.maximum(src - bkg, 1.0)
    g0 = 4.0
    a0 = np.log(max(net.sum() / shape(g0, lo, hi).sum(), 1e-6))
    best = None
    for gs in (2.5, 4.0, 6.0):
        r = minimize(nll, [a0 + (gs - g0) * 0.0, gs], method="Nelder-Mead",
                     options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    la, g = best.x
    if not (0.5 < g < 12.0):
        return None
    f0 = best.fun

    def prof(gg):
        r = minimize(lambda a: nll([a[0], gg]), [la], method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-7})
        return r.fun - f0

    def bound(sign):
        step, x_in, x = 0.02, g, g
        for _ in range(200):
            x = x_in + sign * step
            if prof(x) > 0.5:
                break
            x_in, step = x, min(step * 1.5, 1.0)
        else:
            return np.nan
        a, b = x_in, x                      # bisect between inside and outside
        for _ in range(25):
            m = 0.5 * (a + b)
            a, b = (m, b) if prof(m) <= 0.5 else (a, m)
        return 0.5 * (a + b)

    lo_g, hi_g = bound(-1), bound(+1)
    mu = bkg + np.exp(la) * shape(g, lo, hi)
    chi2 = float(np.sum((src - mu) ** 2 / np.maximum(mu, 1.0)))
    return {"gamma": float(g), "gamma_lo": float(lo_g), "gamma_hi": float(hi_g),
            "err": float((hi_g - lo_g) / 2.0), "amp": float(np.exp(la)),
            "chi2": chi2, "dof": int(src.size - 2)}


def spectrum(evs, t_lo: float, t_hi: float) -> np.ndarray:
    """Counts per energy bin in [t_lo, t_hi) (UTC), summed over detectors."""
    out = np.zeros(EDGES.size - 1)
    for ev in evs:
        u = ev.utc()
        m = (u >= t_lo) & (u < t_hi)
        out += np.histogram(ev.energy[m], EDGES)[0]
    return out


def fit_range(src, bkg, e_lo, e_hi_max=150.0):
    """Bins from e_lo up to the last contiguous bin with a 3-sigma excess."""
    lo, hi = EDGES[:-1], EDGES[1:]
    k = np.flatnonzero(lo >= e_lo - 1e-9)
    use = []
    for i in k:
        if hi[i] > e_hi_max + 1e-9:
            break
        sig = (src[i] - bkg[i]) / np.sqrt(max(src[i], 1.0))
        if sig < 3.0:
            break
        use.append(i)
    return np.array(use, int)


def am241_line(evs, t_lo, t_hi) -> float | None:
    """Centroid of the 59.5 keV calibration line in a background interval."""
    e = np.concatenate([ev.energy[(ev.utc() >= t_lo) & (ev.utc() < t_hi)] for ev in evs])
    e = e[(e > 50) & (e < 70)]
    if e.size < 200:
        return None
    h, ed = np.histogram(e, np.arange(50, 70.01, 0.5))
    c = 0.5 * (ed[1:] + ed[:-1])
    base = np.median(np.concatenate([h[:6], h[-6:]]))
    near = (c > 56) & (c < 63)
    w = np.maximum(h[near] - base, 0)
    return float(np.sum(w * c[near]) / w.sum()) if w.sum() > 50 else None


def main(argv=None) -> int:
    S = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=str(S.catalog / "master_catalog.csv"))
    ap.add_argument("--hel1os-root", default=str(S.hel1os_extracted))
    ap.add_argument("--out", default=str(S.physics))
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_rows(args.catalog) if r["hard_peak_czt_20_40_utc"]]
    rows.sort(key=lambda r: r["hard_peak_czt_20_40_utc"])
    t_start = time.time()
    lo, hi = EDGES[:-1], EDGES[1:]
    tr_use = np.flatnonzero((lo >= 30.0 - 1e-9) & (hi <= 100.0 + 1e-9))
    res, tres, peak_spectra = [], [], {}
    for r in rows:
        tp = ts(r["hard_peak_czt_20_40_utc"])
        start = min(ts(r["start_utc"]), tp - 60.0)
        prod = product_for(args.hel1os_root, tp - 1200.0, tp + 600.0)
        rec = {"id": r["id"], "czt_peak_utc": r["hard_peak_czt_20_40_utc"], "goes_class": r["goes_class"],
               "class_solexs": r["class_solexs"], "czt_20_40_peak_cps": float(r["hard_peak_czt_20_40_cps"]),
               "max_energy_keV": r["max_energy_keV"], "test_period": r["test_period"]}
        res.append(rec)
        if prod is None:
            rec["status"] = "no event list"
            continue
        b0, b1 = max(start - 720.0, prod.t_start + 30.0), start - 120.0
        if b1 - b0 < 300.0:
            rec["status"] = "no pre-flare background"
            continue
        evs = [read_events(prod, d, b0 - 5.0, tp + 600.0, 15.0, 250.0) for d in ("CZT1", "CZT2")]
        evs = [ev for ev in evs if ev.tick.size]
        if not evs:
            rec["status"] = "empty event list"
            continue
        rec["product"] = prod.path.name
        rec["detectors"] = "+".join(ev.det for ev in evs)
        rec["am241_keV"] = _r(am241_line(evs, b0, b1), 2)

        # peak: the 16 s with most 30-60 keV counts within +-2 min of the CZT peak (1 s steps)
        u = np.concatenate([ev.utc()[(ev.energy >= 30) & (ev.energy < 60)] for ev in evs])
        grid = np.arange(tp - 120.0, tp + 120.0, 1.0)
        c16 = np.convolve(np.histogram(u, np.append(grid, grid[-1] + 1.0))[0], np.ones(16), "valid")
        p0 = float(grid[int(np.argmax(c16))])
        p1 = p0 + 16.0
        rec["peak_interval_utc"] = utc(p0)
        tb = b1 - b0
        fits_ = {}
        for tag, dets in (("both", evs), *((ev.det.lower(), [ev]) for ev in evs)):
            src = spectrum(dets, p0, p1)
            bkg = spectrum(dets, b0, b1) * (16.0 / tb)
            for e_lo in (30.0, 25.0):
                use = fit_range(src, bkg, e_lo)
                f = fit_power_law(src[use], bkg[use], lo[use], hi[use]) if use.size >= 3 else None
                if f:
                    f["e_hi"] = float(hi[use[-1]])
                    f["net_counts"] = float((src[use] - bkg[use]).sum())
                    # statistical errors understate a poor fit: scale by sqrt(chi2/dof)
                    f["err_scaled"] = f["err"] * np.sqrt(max(1.0, f["chi2"] / max(f["dof"], 1)))
                fits_[(tag, e_lo)] = f
                if tag == "both" and e_lo == 30.0:
                    peak_spectra[r["id"]] = {"src": src, "bkg": bkg, "use": use,
                                             "fit": None if f is None else (f["amp"], f["gamma"])}
        f = fits_[("both", 30.0)]
        if f is None:
            rec["status"] = "too few counts above 30 keV"
            continue
        reliable = f["net_counts"] >= 300 and 1.5 <= f["gamma"] <= 10.0 and f["err_scaled"] < 0.5
        rec.update({"status": "ok" if reliable else "ok, unreliable",
                    "gamma": round(f["gamma"], 3), "gamma_err": round(f["err"], 3),
                    "gamma_err_scaled": round(f["err_scaled"], 3),
                    "fit_e_hi_keV": round(f["e_hi"], 1), "net_counts": int(f["net_counts"]),
                    "chi2_dof": f"{f['chi2']:.1f}/{f['dof']}",
                    "delta_thick_target": round(f["gamma"] + 1.0, 3),
                    "flux_35keV_cts_s_keV": round(f["amp"] / 16.0, 3)})
        for tag in ("czt1", "czt2"):
            g = fits_.get((tag, 30.0))
            rec[f"gamma_{tag}"] = round(g["gamma"], 3) if g else ""
            rec[f"gamma_{tag}_err"] = round(g["err_scaled"], 3) if g else ""
        g25 = fits_[("both", 25.0)]
        rec["gamma_from_25keV"] = round(g25["gamma"], 3) if g25 else ""

        # time-resolved through the impulsive phase: adaptive bins of >= 300 net counts at 30-100 keV
        uu = np.concatenate([ev.utc() for ev in evs])
        ee = np.concatenate([ev.energy for ev in evs])
        hard = (ee >= 30) & (ee < 100)
        bk_rate = float(((uu >= b0) & (uu < b1) & hard).sum()) / tb
        bk_spec = {ev.det: spectrum([ev], b0, b1) / tb for ev in evs}
        bk_all = sum(bk_spec.values())
        uh = np.sort(uu[hard])

        def net(a_, b_, _uh=uh, _bk=bk_rate):
            return float(np.searchsorted(_uh, b_) - np.searchsorted(_uh, a_)) - _bk * (b_ - a_)

        edges_t, t = [tp - 180.0], tp - 180.0
        while t < tp + 180.0:
            t2 = t + 4.0
            while t2 < tp + 180.0 and net(t, t2) < 300.0:
                t2 += 2.0
            if net(t, t2) < 300.0:
                break
            edges_t.append(t2)
            t = t2
        two = len(evs) == 2
        for a_, b_ in zip(edges_t[:-1], edges_t[1:]):
            dt_ = b_ - a_
            src = spectrum(evs, a_, b_)
            f = fit_power_law(src[tr_use], bk_all[tr_use] * dt_, lo[tr_use], hi[tr_use])
            if not (f and np.isfinite(f["err"]) and f["err"] < 1.0):
                continue
            row = {"id": r["id"], "t_mid_utc": utc(0.5 * (a_ + b_)), "t_rel_s": round(0.5 * (a_ + b_) - tp, 1),
                   "dt_s": dt_, "gamma": round(f["gamma"], 3), "gamma_err": round(f["err"], 3),
                   "flux_35keV_cts_s_keV": round(f["amp"] / dt_, 3)}
            if two:
                # gamma from CZT1 photons, flux from CZT2 photons: independent counting noise
                s1 = spectrum(evs[:1], a_, b_)
                f1 = fit_power_law(s1[tr_use], bk_spec[evs[0].det][tr_use] * dt_, lo[tr_use], hi[tr_use])
                s2 = spectrum(evs[1:], a_, b_)
                n2 = float((s2[tr_use] - bk_spec[evs[1].det][tr_use] * dt_).sum())
                if f1 and np.isfinite(f1["err"]) and f1["err"] < 1.5:
                    row["gamma_czt1"] = round(f1["gamma"], 3)
                    row["rate_czt2_30_100"] = round(n2 / dt_, 2)
            tres.append(row)
        print(f"{r['id']} {r['goes_class'] or r['class_solexs'] or '-':>5}  gamma {rec['gamma']:.2f} +- "
              f"{rec['gamma_err_scaled']:.2f} (to {rec['fit_e_hi_keV']:.0f} keV, {rec['net_counts']} cts; "
              f"czt1 {rec['gamma_czt1']} czt2 {rec['gamma_czt2']}; from 25 keV {rec['gamma_from_25keV']}) "
              f"Am241 {rec['am241_keV']} {rec['status']}", flush=True)

    s = summarise(res, tres)
    s["seconds"] = round(time.time() - t_start, 1)
    with (out / "hxr_spectra.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for x in res for k in x)))
        w.writeheader()
        w.writerows(res)
    with (out / "hxr_timeresolved.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for x in tres for k in x)))
        w.writeheader()
        w.writerows(tres)
    (out / "hxr_summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    (out / "HXR.md").write_text(render(s), encoding="utf-8")
    figure(res, tres, peak_spectra, s, out / "hxr_spectra.png")
    print(render(s))
    return 0


def _r(v, nd=3):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def shs(tres: list[dict], gkey: str, fkey: str) -> dict | None:
    """Per flare: Spearman correlation of gamma with log flux over its time bins."""
    by: dict = {}
    for x in tres:
        if x.get(gkey, "") != "" and x.get(fkey, "") != "" and float(x[fkey]) > 0:
            by.setdefault(x["id"], []).append((float(x[gkey]), np.log10(float(x[fkey]))))
    per = []
    for fid, xs in by.items():
        if len(xs) < 5:
            continue
        gg, ff = np.array(xs).T
        rho = spearmanr(gg, ff)
        per.append({"id": fid, "n_bins": len(xs), "spearman": round(float(rho.statistic), 3),
                    "p": float(rho.pvalue), "slope": round(float(np.polyfit(ff, gg, 1)[0]), 3)})
    if not per:
        return None
    rs = np.array([p["spearman"] for p in per])
    return {"flares": len(per), "negative": _r(np.mean(rs < 0)),
            "significant_negative_p05": _r(np.mean([(p["spearman"] < 0) and (p["p"] < 0.05) for p in per])),
            "significant_positive_p05": _r(np.mean([(p["spearman"] > 0) and (p["p"] < 0.05) for p in per])),
            "median_spearman": _r(np.median(rs)),
            "median_slope_dgamma_dlog10F": _r(np.median([p["slope"] for p in per])), "per_flare": per}


def summarise(res: list[dict], tres: list[dict]) -> dict:
    from solarflare.io.goes import class_flux

    ok = [x for x in res if x.get("status") == "ok"]
    s: dict = {"generated_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
               "flares_with_czt_burst": len(res), "reliable_fits": len(ok),
               "status_counts": {k: sum(1 for x in res if x.get("status") == k)
                                 for k in sorted({str(x.get("status")) for x in res})}}
    if not ok:
        return s
    g = np.array([x["gamma"] for x in ok])
    s["gamma_peak"] = {"median": _r(np.median(g), 2), "q25": _r(np.percentile(g, 25), 2),
                       "q75": _r(np.percentile(g, 75), 2), "min": _r(g.min(), 2), "max": _r(g.max(), 2),
                       "median_err": _r(np.median([x["gamma_err_scaled"] for x in ok]), 2)}
    pair = [(x["gamma_czt1"], x["gamma_czt2"], x["gamma_czt1_err"], x["gamma_czt2_err"]) for x in ok
            if x["gamma_czt1"] != "" and x["gamma_czt2"] != ""]
    if pair:
        d = np.array([(a - b) / np.hypot(ea, eb) for a, b, ea, eb in pair])
        s["czt1_vs_czt2"] = {"n": len(pair), "median_diff": _r(np.median([a - b for a, b, _, _ in pair])),
                             "rms_pull": _r(np.sqrt(np.mean(d ** 2)), 2), "within_2sigma": _r(np.mean(np.abs(d) <= 2))}
    t25 = [(x["gamma"], x["gamma_from_25keV"]) for x in ok if x["gamma_from_25keV"] != ""]
    if t25:
        s["from_25_vs_30keV"] = {"n": len(t25), "median_diff": _r(np.median([b - a for a, b in t25])),
                                 "steeper_from_25": _r(np.mean([b > a for a, b in t25]))}
    am = [x["am241_keV"] for x in res if x.get("am241_keV")]
    if am:
        s["am241_line_keV"] = {"n": len(am), "median": _r(np.median(am), 2), "sd": _r(np.std(am), 2),
                               "expected": 59.54}
    gf = [(x["gamma"], x["goes_class"]) for x in ok if x["goes_class"]]
    if len(gf) >= 5:
        rho = spearmanr([a for a, _ in gf], [np.log10(class_flux(c)) for _, c in gf])
        s["gamma_vs_goes_flux"] = {"n": len(gf), "spearman": _r(rho.statistic), "p": _r(rho.pvalue, 4)}
    rho = spearmanr(g, [x["flux_35keV_cts_s_keV"] for x in ok])
    s["gamma_vs_hxr_flux"] = {"n": len(ok), "spearman": _r(rho.statistic), "p": _r(rho.pvalue, 4)}
    s["soft_hard_soft"] = shs(tres, "gamma", "flux_35keV_cts_s_keV")
    s["soft_hard_soft_independent"] = shs(tres, "gamma_czt1", "rate_czt2_30_100")
    return s


def render(s: dict) -> str:
    g = s.get("gamma_peak", {})
    L = ["# Hard X-ray spectral index from HEL1OS photon lists", "",
         f"Generated {s['generated_utc']} UTC by `python -m solarflare hxr-spectra`. {s['flares_with_czt_burst']} flares have a "
         f"CZT 20-40 keV burst in the master catalogue; {s['reliable_fits']} give a reliable power-law fit at the "
         "hard X-ray peak (>= 300 net counts above 30 keV, 1.5 <= gamma <= 10, error < 0.5).", "",
         "| fit outcome | flares |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in s["status_counts"].items()] + [""]
    if not g:
        return "\n".join(L)
    L += ["## Spectral index at the hard X-ray peak", "",
          f"Count spectral index gamma (N(E) ~ E^-gamma, 30 keV up to the last 3-sigma bin, 16 s at the 30-60 keV peak): "
          f"median **{g['median']}** (IQR {g['q25']}-{g['q75']}, range {g['min']}-{g['max']}), typical error {g['median_err']} "
          "(statistical, scaled by sqrt(chi2/dof) where the fit is poor). Thick-target electron index delta = gamma + 1.", ""]
    c = s.get("czt1_vs_czt2")
    if c:
        L.append(f"- **Two independent detectors agree**: CZT1 vs CZT2 on {c['n']} flares, median difference "
                 f"{c['median_diff']:+.3f}, {100 * c['within_2sigma']:.0f}% within 2 sigma (rms pull {c['rms_pull']}).")
    a = s.get("am241_line_keV")
    if a:
        L.append(f"- **Energy scale**: the onboard Am-241 line sits at {a['median']} +- {a['sd']} keV across "
                 f"{a['n']} flares (true 59.54 keV).")
    t = s.get("from_25_vs_30keV")
    if t:
        L.append(f"- **Thermal contamination**: starting the fit at 25 keV instead of 30 keV changes gamma by "
                 f"{t['median_diff']:+.3f} (median); {100 * t['steeper_from_25']:.0f}% of flares come out steeper, as "
                 "expected where hot thermal emission still contributes at 25-30 keV.")
    gg = s.get("gamma_vs_goes_flux")
    if gg:
        hx = s["gamma_vs_hxr_flux"]

        def trend(d):
            return ("no significant trend" if d["p"] >= 0.05
                    else f"a weak trend (Spearman {d['spearman']:+.2f}, p = {d['p']})")

        L.append(f"- Across flares, gamma vs GOES peak flux shows {trend(gg)} (n = {gg['n']}); gamma vs the 35 keV "
                 f"peak flux shows {trend(hx)}. A positive Spearman means bigger flares have softer spectra.")
    L += [""]
    for key, title in (("soft_hard_soft", "photons of both detectors"),
                       ("soft_hard_soft_independent", "gamma from CZT1, flux from CZT2 (independent noise)")):
        h = s.get(key)
        if not h:
            continue
        L += [f"## Soft-hard-soft ({title})", "",
              f"{h['flares']} flares with >= 5 time bins through the impulsive phase: gamma falls as the flux rises in "
              f"**{100 * h['negative']:.0f}%** of them ({100 * h['significant_negative_p05']:.0f}% significant at p < 0.05, "
              f"against {100 * h['significant_positive_p05']:.0f}% significantly the other way). Median Spearman "
              f"{h['median_spearman']}, median slope d(gamma)/d(log10 F) = {h['median_slope_dgamma_dlog10F']}.", ""]
    L += ["## Caveats", "",
          "- No response matrix ships with the L1 products, so gamma is the *count* index. It tracks the photon "
          "index where CZT efficiency is flat (30-150 keV); K-escape and hole tailing are not modelled.",
          "- Errors are statistical; for the brightest flares a single power law is a poor fit (chi2/dof up to ~70), "
          "and the quoted error is scaled up accordingly.",
          "- HEL1OS reads events out in batches (python -m solarflare hxr-timing): below ~2,000 events/s photons are displaced "
          "in time by up to the batch spacing (seconds). Spectral shapes do not care; the time-resolved bins (>= 4 s, "
          "impulsive-phase rates) are affected only slightly.",
          "- 'ok, unreliable' fits (flat gamma < 1.5 or few counts) are mostly HEL1OS-only events with no GOES "
          "flare: likely background or particle fluctuations, not flares.", ""]
    return "\n".join(L)


def figure(res, tres, peak_spectra, s, dest: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, HARD, SOFT, BOTH = "#1d2733", "#98a2ad", "#7a4fd1", "#d9822b", "#2a8c7c"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
    ok = [x for x in res if x.get("status") == "ok"]
    lo, hi = EDGES[:-1], EDGES[1:]
    mid, wid = np.sqrt(lo * hi), hi - lo

    # (a) the brightest flare's peak spectrum
    a = ax[0, 0]
    best = max(ok, key=lambda x: x["net_counts"])
    ps = peak_spectra[best["id"]]
    net = (ps["src"] - ps["bkg"]) / wid / 16.0
    err = np.sqrt(ps["src"] + ps["bkg"]) / wid / 16.0
    m = net > 0
    a.errorbar(mid[m], net[m], yerr=err[m], fmt="o", ms=3.5, color=HARD, label="net counts (both CZT)")
    a.step(mid, ps["bkg"] / wid / 16.0, where="mid", color=MUTED, lw=1, label="pre-flare background")
    if ps["fit"]:
        amp, gam = ps["fit"]
        e = np.geomspace(30, hi[ps["use"]].max(), 50)
        a.plot(e, amp * (e / E_REF) ** -gam / 16.0, color=INK, lw=1.2, label=f"power law, gamma = {gam:.2f}")
    a.axvspan(20, 30, color=MUTED, alpha=0.12, lw=0)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("energy, keV")
    a.set_ylabel("counts / s / keV")
    a.set_title(f"Peak spectrum, {best['goes_class'] or best['class_solexs']} flare {best['czt_peak_utc'][:16]} UTC",
                loc="left", fontsize=10, color=INK)
    a.legend(frameon=False, fontsize=7.5)

    # (b) distribution + CZT1 vs CZT2
    b = ax[0, 1]
    g = np.array([x["gamma"] for x in ok])
    b.hist(g, bins=np.arange(1.5, 9.01, 0.5), color=HARD, alpha=0.8)
    b.axvline(np.median(g), color=INK, lw=1, ls="--")
    b.set_xlabel("count spectral index gamma at the HXR peak")
    b.set_ylabel("flares")
    b.set_title(f"{len(g)} flares: median gamma {np.median(g):.2f}", loc="left", fontsize=10, color=INK)
    ins = b.inset_axes([0.62, 0.5, 0.34, 0.42])
    pr = [(x["gamma_czt1"], x["gamma_czt2"]) for x in ok if x["gamma_czt1"] != "" and x["gamma_czt2"] != ""]
    if pr:
        p1, p2 = np.array(pr, float).T
        ins.plot(p1, p2, "o", ms=2.5, color=BOTH)
        lim = [min(p1.min(), p2.min()) - 0.3, max(p1.max(), p2.max()) + 0.3]
        ins.plot(lim, lim, color=MUTED, lw=0.8)
        ins.set_xlabel("gamma, CZT1", fontsize=7)
        ins.set_ylabel("gamma, CZT2", fontsize=7)
        ins.tick_params(labelsize=6)

    # (c) soft-hard-soft pooled: gamma vs flux relative to each flare's max
    c = ax[1, 0]
    by: dict = {}
    for x in tres:
        by.setdefault(x["id"], []).append(x)
    for xs in by.values():
        if len(xs) < 5:
            continue
        fx = np.array([x["flux_35keV_cts_s_keV"] for x in xs], float)
        gx = np.array([x["gamma"] for x in xs], float)
        c.plot(fx / fx.max(), gx - np.median(gx), ".", ms=2.5, color=HARD, alpha=0.35)
    h = s.get("soft_hard_soft")
    if h:
        xx = np.geomspace(0.05, 1, 20)
        c.plot(xx, h["median_slope_dgamma_dlog10F"] * np.log10(xx / 0.3), color=INK, lw=1.3,
               label=f"median slope {h['median_slope_dgamma_dlog10F']:+.2f} per decade")
        c.legend(frameon=False, fontsize=7.5)
        c.set_title(f"Soft-hard-soft: harder when brighter in {100 * h['negative']:.0f}% of {h['flares']} flares",
                    loc="left", fontsize=10, color=INK)
    c.set_xscale("log")
    c.set_xlabel("35 keV flux / flare maximum")
    c.set_ylabel("gamma - flare median")

    # (d) time evolution of the brightest flare
    d = ax[1, 1]
    xs = by.get(best["id"], [])
    if xs:
        tt = np.array([x["t_rel_s"] for x in xs])
        d.plot(tt, [x["flux_35keV_cts_s_keV"] for x in xs], "-", color=HARD, lw=1.3)
        d.set_yscale("log")
        d.set_ylabel("35 keV flux, counts/s/keV", color=HARD)
        d2 = d.twinx()
        d2.errorbar(tt, [x["gamma"] for x in xs], yerr=[x["gamma_err"] for x in xs], fmt="o", ms=2.5,
                    color=SOFT, lw=0.8)
        d2.set_ylabel("gamma (axis inverted: up = harder)", color=SOFT)
        d2.invert_yaxis()
        d2.spines["right"].set_visible(True)
        d.set_xlabel("seconds from the CZT 20-40 keV peak")
        d.set_title("Same flare: the spectrum hardens at each burst", loc="left", fontsize=10, color=INK)
    fig.suptitle("HEL1OS CZT hard X-ray spectra from the photon lists (count index, no response matrix)",
                 x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(dest)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
