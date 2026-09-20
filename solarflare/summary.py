"""outputs/RESULTS.md: one page over every product, generated from their JSON.

    python -m solarflare pipeline --only summary

Nothing here is typed by hand: each number is read from the report that
measured it, and a section whose stage has not run says so instead of quoting
an old value.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from .settings import Settings, load_settings


def _j(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def _p(v, nd: int = 0) -> str:
    return "--" if v is None else f"{100 * float(v):.{nd}f}%"


def _f(v, nd: int = 3) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "--"
    return "--" if x != x else f"{x:.{nd}f}"


def _rel(path: Path, s: Settings) -> str:
    try:
        return path.relative_to(s.outputs).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------------------
# sections: each returns markdown lines or raises (then shown as "not run yet")
# ---------------------------------------------------------------------------

def sec_data(s: Settings) -> list[str]:
    meta = _j(s.model_dir / "reports" / "data_meta.json")
    cfg = _j(s.model_dir / "reports" / "config.json")
    sp = meta["split_dates"]

    def d(x):
        return datetime.fromtimestamp(float(x), UTC).strftime("%Y-%m-%d")

    pre = cfg.get("pre", {})
    L = [f"Chronological split: training to {d(sp['train_end'])}, test from {d(sp['test_start'])} "
         f"(1 h embargo between splits). Flare truth: GOES-18 XRS flare list and flux, >= "
         f"{pre.get('goes_min_class', 'C1.0')}. Windows: {meta.get('n_train', '?')} train, "
         f"{meta.get('n_val', '?')} validation, {meta.get('n_test', '?')} test."]
    if meta.get("excluded_solexs_samples"):
        L.append(f"SoLEXS samples masked as copies of the previous day: {meta['excluded_solexs_samples']}.")
    fa = meta.get("flux_anchor_fit")
    if fa:
        L.append(f"Flux anchor fitted on training data: log10 F = {fa['intercept']:.3f} + {fa['slope']:.3f} "
                 f"log10 rate, scatter {fa['scatter_dex']:.3f} dex ({fa['n']} samples).")
    L.append(f"SHARP magnetic inputs in the network: {'yes' if pre.get('sharp_dir') else 'no'}.")
    return L


def sec_catalog(s: Settings) -> list[str]:
    c = _j(s.catalog / "catalog_summary.json")
    n, t = c["counts"], c["test"]
    r = t["recall"]
    L = [f"{n['solexs_flares']} SoLEXS flares and {n['hel1os_events']} HEL1OS events over "
         f"{c['observed_days']['solexs']:.0f} and {c['observed_days']['hel1os']:.0f} observed days, merged into "
         f"{n['master']} catalogue entries ({n['by_origin'].get('soft+hard', 0)} seen by both).", "",
         "Test period, against the GOES flare list:", "",
         "| GOES class | flares | SoLEXS recall | HEL1OS recall | both observing: combined recall (chance) |",
         "|---|---|---|---|---|"]
    for k in ("B", "C", "M", "X"):
        x = r.get(k)
        if x:
            L.append(f"| {k} | {x['soft_n']} | {_p(x['soft_recall'])} | {_p(x['hard_recall'])} (n={x['hard_n']}) | "
                     f"{_p(x['both_combined_recall'])} ({_p(x['both_combined_recall_chance'])}) |")
    sp = t.get("soft_precision", {})
    if sp:
        L += ["", "SoLEXS detections matching a GOES flare: " + ", ".join(
            f"{k} {_p(v['precision'])} ({v['unmatched_per_day']:.2f} unmatched/day)" for k, v in sp.items()) + "."]
    cl = t.get("class")
    if cl:
        L.append(f"GOES class letter from SoLEXS alone: {_p(cl['letter_agreement'])} agree, median error "
                 f"{cl['median_abs_dex']:.3f} dex.")
    return L


def sec_network(s: Settings) -> list[str]:
    ev = _j(s.model_dir / "reports" / "evaluation.json")
    nf = ev["nowcast_in_flare"]
    L = [f"Best epoch {ev['training']['best_epoch']} (validation score {ev['training']['best_score']:.3f}, "
         "smoothed over epochs).", "",
         "| task (test split) | TSS | POD | FAR | AUC | Brier skill |", "|---|---|---|---|---|---|",
         f"| flare in progress now | {_f(nf['TSS'])} | {_f(nf['POD'])} | {_f(nf['FAR'])} | {_f(nf['AUC'])} | "
         f"{_f(nf.get('BSS_vs_climatology'))} |"]
    for h, x in ev.get("forecast_occurrence", {}).items():
        if isinstance(x, dict) and "TSS" in x:
            L.append(f"| flare within {h} | {_f(x['TSS'])} | {_f(x['POD'])} | {_f(x['FAR'])} | {_f(x['AUC'])} | "
                     f"{_f(x.get('BSS_vs_climatology'))} |")
    nr = ev.get("nowcast_regression", {})
    fr = ev.get("forecast_regression", {})
    L += ["", f"GOES flux now from Aditya data: MAE {_f(nr.get('MAE'))} dex. Flux forecast MAE: " + ", ".join(
        f"{h} {_f(x['MAE'])} dex" for h, x in fr.items() if isinstance(x, dict) and "MAE" in x) + "."]
    pk = ev.get("peak")
    if pk:
        L.append(f"Peak of flares under way: size MAE {_f(pk['log_peak_flux_MAE'])} dex, time MAE "
                 f"{_f(pk['time_to_peak_MAE_min'], 1)} min ({pk['n']} windows).")
    return L


def sec_calibration(s: Settings) -> list[str]:
    c = _j(s.model_dir / "reports" / "calibration.json")
    L = ["| head | Brier skill raw | calibrated | TSS raw | calibrated |", "|---|---|---|---|---|"]
    for k, v in c["heads"].items():
        L.append(f"| {k} | {_f(v['raw']['BSS_vs_climatology'])} | {_f(v['calibrated']['BSS_vs_climatology'])} | "
                 f"{_f(v['raw']['TSS'])} | {_f(v['calibrated']['TSS'])} |")
    L += ["", "The frozen model carries this calibration (fitted on validation); its alert thresholds refer to "
          "calibrated probabilities."]
    return L


def sec_alerts(s: Settings) -> list[str]:
    a = _j(s.alerts / "leadtime_summary.json")
    rules = _j(s.alerts / "alert_rules.json")
    L = [f"Test period {a['test_period'][0]} to {a['test_period'][1]} UTC, {a['test_days_with_data']} days with data. "
         "Thresholds fixed on validation.", "",
         "| alert | rule | flares | warned | chance | event TSS | median lead before GOES peak | false alarms/day |",
         "|---|---|---|---|---|---|---|---|"]
    for c, title in (("C", ">= C1 within 15 min"), ("M", "flux reaches M1 within 30 min")):
        R = a["results"][c]
        mth = "model" if c == "C" else "combined"
        d = R[mth][f"fa_{a['primary_false_alarms_per_day'][c]:g}"]
        L.append(f"| {title} | {rules[c]['signal']} >= {rules[c]['threshold']} | {R['flares']} | {_p(d['TPR'])} | "
                 f"{_p(d['chance'])} | {_f(d['event_TSS'], 2)} | {_f(d['median_lead_min'], 0)} min | "
                 f"{_f(d['false_per_day'], 2)} |")
        if c == "M" and "before_M1" in d:
            b = d["before_M1"]
            L.append(f"| (M, before GOES first reaches M1) | | | {_p(b['alert_before_M1'])} | | | "
                     f"{_f(b['median_lead_min'], 0)} min | |")
    return L


def sec_hel1os(s: Settings) -> list[str]:
    L = []
    ev = s.model_dir / "reports" / "evaluation.json"
    if ev.exists():
        m = _j(ev).get("modality_ablation_hel1os_observed")
        if m:
            L.append(f"Network, flare-in-progress TSS where HEL1OS observed: soft+hard {_f(m['both']['TSS'])}, "
                     f"soft only {_f(m['soft_only']['TSS'])}, hard only {_f(m['hard_only']['TSS'])} "
                     f"(n={m['both']['n']}).")
    hv = s.ablations / "hel1os" / "hel1os_value.json"
    if hv.exists():
        v = _j(hv)
        sd = f" +- {100 * v['relative_sd']:.1f}%" if v.get("relative_sd") is not None else ""
        L.append(f"Peak size of rising flares, paired soft-only vs soft+hard over {len(v['seeds'])} seeds: "
                 f"{100 * v['mean_relative']:.1f}%{sd} lower error with HEL1OS; "
                 f"{v['intervals_excluding_zero']} of {len(v['seeds'])} intervals exclude zero. "
                 f"Verdict: {v['verdict']}.")
    al = s.alerts / "leadtime_summary.json"
    if al.exists():
        h = _j(al)["results"]["C"].get("hel1os_ablation")
        if h and h.get("matched_false_alarms"):
            mf = h["matched_false_alarms"]
            L.append(f">= C1 alerts on {h['flares']} flares HEL1OS watched, at equal false alarms: warned "
                     f"{_p(h['warned_with'])} with HEL1OS vs {_p(mf['warned_without'])} without "
                     f"{mf['warned_gain_ci']}; mean lead gain {mf['mean_lead_gain_min']:+.2f} min "
                     f"{mf['mean_lead_gain_ci']}.")
    if not L:
        raise FileNotFoundError
    return L


def sec_sharp(s: Settings) -> list[str]:
    L = []
    dec = s.ablations / "sharp" / "decision.json"
    if dec.exists():
        d = _j(dec)
        L.append(f"As a network input: validation score {d['xray_only']['val_score']} without, "
                 f"{d['with_sharp']['val_score']} with (needed +{d['criterion'].split('>=')[-1].strip()}): "
                 f"**{'kept' if d['use_sharp'] else 'left out'}**.")
    da = s.dayahead / "dayahead_summary.json"
    if da.exists():
        d = _j(da)
        sets = [k for k in ("xray", "sharp", "both") if k in d["feature_sets"]]
        L += ([""] if L else []) + ["Day-ahead forecasts (test AUC; persistence = flares in the last 24 h):", "",
              "| target | " + " | ".join({"xray": "SoLEXS", "sharp": "SHARP", "both": "both"}[k] for k in sets)
              + " | persistence | SHARP added [95% CI] |", "|---|" + "---|" * (len(sets) + 2)]
        for k, r in d["results"].items():
            g = r["gains"].get("sharp_added")
            L.append(f"| {k} | " + " | ".join(str(r["sets"][x]["AUC"]) for x in sets) + f" | {r['persistence']['AUC']} | "
                     + (f"{g['AUC_gain']:+.3f} {g['ci']}" if g else "--") + " |")
    if not L:
        raise FileNotFoundError
    return L


def sec_physics(s: Settings) -> list[str]:
    L = []
    t = s.physics / "temperature_summary.json"
    if t.exists():
        d = _j(t)
        by = d.get("by_class", {})
        L.append(f"Temperatures of {d['fitted']} flares from SoLEXS spectra: median at peak " + ", ".join(
            f"{k} {v['T_peak_median']} MK" for k, v in by.items()) + "; temperature peaks before flux in "
            + ", ".join(f"{k} {_p(v['T_max_before_peak_share'])}" for k, v in by.items()) + ".")
    h = s.physics / "hxr_summary.json"
    if h.exists():
        d = _j(h)
        g = d["gamma_peak"]
        L.append(f"HEL1OS CZT spectral index at the hard X-ray peak: median {g['median']} (IQR {g['q25']}-{g['q75']}) "
                 f"over {d['reliable_fits']} reliable fits; Am-241 line at {d['am241_line_keV']['median']} keV "
                 f"(expected {d['am241_line_keV']['expected']}).")
        shs = d.get("soft_hard_soft")
        if shs:
            L.append(f"Soft-hard-soft evolution in {_p(shs['negative'])} of {shs['flares']} flares "
                     f"({_p(shs['significant_negative_p05'])} significant).")
    ph = sorted(s.physics.glob("physics_summary_*.json"))
    if ph:
        d = _j(ph[-1])
        ne = d.get("neupert", {})
        if ne:
            L.append(f"Neupert effect: median correlation {_f(ne.get('median_r_best'), 2)} between hard X-rays and "
                     f"the soft X-ray derivative over {ne.get('n_significant_hxr')} flares with clear hard X-rays.")
    if not L:
        raise FileNotFoundError
    return L


SECTIONS: list[tuple[str, str, Callable[[Settings], list[str]]]] = [
    ("Data", "train", sec_data),
    ("Detection: the master catalogue (SoLEXS + HEL1OS)", "catalog", sec_catalog),
    ("The network: nowcasting and 1-60 min forecasting", "train", sec_network),
    ("Are the probabilities probabilities?", "calibration", sec_calibration),
    ("Alerts: lead time and false alarms", "alerts", sec_alerts),
    ("What HEL1OS adds", "hel1os-value", sec_hel1os),
    ("What SHARP magnetic data add", "dayahead", sec_sharp),
    ("Flare physics", "temperature, hxr-spectra, onset-study", sec_physics),
]


def build(s: Settings | None = None) -> Path:
    s = s or load_settings()
    L = ["# Aditya-L1 flare nowcasting and forecasting: results", "",
         f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC from the reports under outputs/ "
         "(`python -m solarflare pipeline`). Every threshold was fixed before the test period was scored.", ""]
    for title, stage, fn in SECTIONS:
        L += [f"## {title}", ""]
        try:
            L += fn(s)
        except (FileNotFoundError, KeyError, IndexError, TypeError, ValueError) as e:
            detail = f" ({type(e).__name__}: {e})" if not isinstance(e, FileNotFoundError) else ""
            L.append(f"*Not available yet: run stage `{stage}`.*{detail}")
        L.append("")
    L += ["## Where the details are", ""]
    for name, path in (("Model report", s.model_dir / "reports" / "RESULTS.md"),
                       ("Alerts", s.alerts / "LEADTIME.md"),
                       ("Alert rules used by the console", s.alerts / "alert_rules.json"),
                       ("Master catalogue", s.catalog / "master_catalog.csv"),
                       ("Day-ahead forecasts", s.dayahead / "DAYAHEAD.md"),
                       ("HEL1OS ablation", s.ablations / "hel1os" / "HEL1OS_VALUE.md"),
                       ("SHARP decision", s.ablations / "sharp" / "decision.json"),
                       ("Temperatures", s.physics / "TEMPERATURE.md"),
                       ("Hard X-ray spectra", s.physics / "HXR.md"),
                       ("Frozen model", s.frozen_dir)):
        if path.exists():
            L.append(f"- {name}: `{_rel(path, s)}`")
    L.append("")
    out = s.outputs / "RESULTS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build())
