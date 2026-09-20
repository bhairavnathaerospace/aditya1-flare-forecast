"""Render outputs/reports/RESULTS.md from the measured JSON reports.

Generated, never hand-written, so the documented numbers cannot drift away from
the numbers the code actually produced.
"""

from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path

from .preprocess.events import format_threshold


def _f(v, nd: int = 3, dash: str = "--") -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return dash
    if x != x:  # NaN
        return dash
    return f"{x:.{nd}f}"


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def build(out_dir: Path) -> Path:
    reports = Path(out_dir) / "reports"
    ev = _load(reports / "evaluation.json")
    bl = _load(reports / "baselines.json")
    meta = _load(reports / "data_meta.json")
    cfg = _load(reports / "config.json")

    L: list[str] = []
    a = L.append

    a("# Results")
    a("")
    a(f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M UTC} by "
      "`python -m solarflare report`.")
    a("All numbers are on the held-out **test** split. Operating thresholds were")
    a("fitted on validation and held fixed.")
    if cfg:
        pre = cfg.get("pre", {}) or {}
        win = cfg.get("win", {}) or {}
        mdl = cfg.get("model", {}) or {}
        a("")
        a(f"Run config: grid {pre.get('dt_seconds', '?')} s, window "
          f"{win.get('input_seconds', '?')} s, background "
          f"{pre.get('background_window_s', '?')} s, "
          f"clock features {'ON' if mdl.get('use_clock') else 'OFF'}.")
    a("")

    # --- data ---------------------------------------------------------
    a("## Data")
    a("")
    segs = meta.get("segments", [])
    dt_meta = float(meta.get("dt", 20.0) or 20.0)
    if segs and len(segs) > 12:
        # An archive has dozens of segments; a row each buries the report.
        soft_d = sum(s["steps"] * s["soft_observed"] for s in segs) * dt_meta / 86400
        hard_d = sum(s["steps"] * s["hard_observed"] for s in segs) * dt_meta / 86400
        first, last = segs[0]["name"].split("_")[1], segs[-1]["name"].split("_")[-1]
        a(f"{len(segs)} stitched segments, {first} -> {last}: SoLEXS observed "
          f"{soft_d:.1f} d, HEL1OS {hard_d:.1f} d, {meta.get('n_events', '?')} flares "
          f"detected. Split: {meta.get('split_mode', '?')}"
          + (f" (archive profile {meta.get('archive_profile')})" if meta.get("archive_mode") else "")
          + ". Per-segment detail is in `data_meta.json`.")
        a("")
    elif segs:
        a("| segment | steps | soft observed | hard observed | flares |")
        a("|---|---:|---:|---:|---:|")
        for s in segs:
            a(f"| `{s['name']}` | {s['steps']} | "
              f"{100 * s['soft_observed']:.0f}% | {100 * s['hard_observed']:.0f}% | "
              f"{s['n_events']} |")
        a("")
    if meta:
        a(f"Windows: {meta.get('n_windows', '?')} "
          f"(train {meta.get('n_train', '?')} / val {meta.get('n_val', '?')} / "
          f"test {meta.get('n_test', '?')}), "
          f"grid {meta.get('dt', '?')} s.")
        a("")

    stats = ev.get("label_stats", {})
    if stats:
        a("Class balance at the prediction origin:")
        a("")
        a("| split | n | in-flare rate | quiet/rise/peak/decay |")
        a("|---|---:|---:|---|")
        for k in ("train", "val", "test"):
            s = stats.get(k)
            if s:
                a(f"| {k} | {s['n']} | {_f(s['in_flare_rate'])} | "
                  f"{'/'.join(str(c) for c in s['phase_counts'])} |")
        a("")

    # --- nowcast ------------------------------------------------------
    n = ev.get("nowcast_in_flare")
    if n:
        a("## 1. Nowcast - is a flare in progress")
        a("")
        a(f"Threshold {_f(n.get('threshold'), 2)} (fitted on validation), "
          f"test base rate {_f(n.get('base_rate'))}.")
        a("")
        a("| metric | value |")
        a("|---|---:|")
        for k in ("TSS", "HSS", "AUC", "POD", "FAR", "CSI", "F1", "precision",
                  "accuracy", "Brier", "BSS_vs_climatology"):
            if k in n:
                a(f"| {k} | {_f(n[k])} |")
        a(f"| contingency TP/FP/FN/TN | "
          f"{int(n.get('TP', 0))}/{int(n.get('FP', 0))}/"
          f"{int(n.get('FN', 0))}/{int(n.get('TN', 0))} |")
        a("")

    p = ev.get("nowcast_phase")
    if p:
        a("## 2. Nowcast - flare phase")
        a("")
        a(f"Accuracy {_f(p.get('accuracy'))}, macro-F1 {_f(p.get('macro_f1'))}.")
        a("")
        a("| phase | precision | recall | F1 | support |")
        a("|---|---:|---:|---:|---:|")
        for i, name in enumerate(["quiet", "rise", "peak", "decay"]):
            pc = p.get("per_class", {}).get(str(i)) or p.get("per_class", {}).get(i)
            if pc:
                a(f"| {name} | {_f(pc['precision'])} | {_f(pc['recall'])} | "
                  f"{_f(pc['f1'])} | {pc['support']} |")
        a("")

    # --- occurrence ---------------------------------------------------
    occ = ev.get("forecast_occurrence")
    if occ:
        a("## 3. Forecast - flare within horizon")
        a("")
        a("| horizon | base rate | TSS | HSS | POD | FAR | AUC | BSS |")
        a("|---|---:|---:|---:|---:|---:|---:|---:|")
        for k, s in occ.items():
            a(f"| {k} | {_f(s.get('climatology_rate'))} | {_f(s.get('TSS'))} | "
              f"{_f(s.get('HSS'))} | {_f(s.get('POD'))} | {_f(s.get('FAR'))} | "
              f"{_f(s.get('AUC'))} | {_f(s.get('BSS_vs_climatology'))} |")
        a("")
        degen = [k for k, s in occ.items()
                 if (s.get("climatology_rate") or 0) > 0.9]
        if degen:
            a(f"> **Note.** At {', '.join(degen)} the base rate exceeds 0.9 -- with "
              "flares every couple of hours in a single day, almost every window "
              "has a flare within that horizon. BSS against climatology is "
              "meaningless there. This is a property of a one-day dataset, not "
              "of the model.")
            a("")

    # --- regression ---------------------------------------------------
    nr = ev.get("nowcast_regression")
    if nr:
        a("## 4. Nowcast regression (log flux)")
        a("")
        a(f"MAE {_f(nr.get('MAE'), 4)} | RMSE {_f(nr.get('RMSE'), 4)} | "
          f"R2 {_f(nr.get('R2'))}")
        a("")

    fr = ev.get("forecast_regression")
    if fr:
        a("## 5. Forecast regression vs persistence AND climatology")
        a("")
        a("| horizon | model RMSE | persistence | climatology | skill vs pers | "
          "skill vs clim | corr | coverage |")
        a("|---|---:|---:|---:|---:|---:|---:|---:|")
        for k, s in fr.items():
            a(f"| {k} | {_f(s.get('RMSE'), 4)} | {_f(s.get('persistence_RMSE'), 4)} | "
              f"{_f(s.get('climatology_RMSE'), 4)} | "
              f"{_f(s.get('skill_vs_persistence'))} | "
              f"{_f(s.get('skill_vs_climatology'))} | "
              f"{_f(s.get('correlation'))} | "
              f"{_f(s.get('interval_coverage'))} |")
        a("")
        nom = next((s.get("nominal_coverage") for s in fr.values()
                    if s.get("nominal_coverage")), None)
        a(f"Nominal interval coverage {_f(nom, 2)}.")
        a("")
        a("> **Read both reference columns.** Persistence is a weak baseline at long")
        a("> horizons because it extrapolates a decaying flare, so beating it there is")
        a("> easy and means little. Climatology is the honest reference: a horizon")
        a("> where the model beats persistence but not climatology, with correlation")
        a("> near zero, is a horizon where it has simply learned to predict the mean.")
        a("")
        useful = [k for k, s in fr.items()
                  if (s.get("correlation") or 0) > 0.3]
        if useful:
            a(f"On this dataset the forecast carries real information at "
              f"**{', '.join(useful)}** and degenerates to climatology beyond that.")
            a("")

    pk = ev.get("peak")
    if pk:
        a("## 6. Ongoing-event peak")
        a("")
        a(f"Time-to-peak MAE {_f(pk.get('time_to_peak_MAE_min'), 1)} min | "
          f"log peak-flux MAE {_f(pk.get('log_peak_flux_MAE'))} | "
          f"n = {pk.get('n')}")
        a("")

    # --- ablation -----------------------------------------------------
    ab = ev.get("modality_ablation")
    if ab:
        a("## 7. Modality ablation")
        a("")
        a("Flare-in-progress skill with each instrument masked off at inference, "
          "same thresholds. `clock_only` blanks both instruments: any skill left "
          "there would be memorisation of time, not physics.")
        a("")
        abh = ev.get("modality_ablation_hel1os_observed") or {}
        a("| input | TSS (all test windows) | AUC (all) | TSS (HEL1OS observing) | "
          "AUC (HEL1OS observing) |")
        a("|---|---:|---:|---:|---:|")
        for k, s in ab.items():
            h = abh.get(k, {})
            a(f"| {k} | {_f(s.get('TSS'))} | {_f(s.get('AUC'))} | "
              f"{_f(h.get('TSS'))} | {_f(h.get('AUC'))} |")
        a("")
        n_all = ab.get("both", {}).get("n")
        n_h = abh.get("both", {}).get("n")
        if n_h is not None:
            of_all = f" (of {n_all})" if n_all is not None else ""
            a(f"The right-hand columns use only the {n_h} test windows whose "
              f"prediction origin HEL1OS observed{of_all}. Over the whole test "
              f"split a hard X-ray effect is diluted by windows where blanking "
              f"HEL1OS changes nothing because it was not observing.")
            a("")
        if n_h == 0:
            a("> No test window has HEL1OS data, so this table cannot show whether "
              "hard X-rays help. See the paired fusion ablation instead.")
            a("")

    # --- baselines ----------------------------------------------------
    if bl and "error" not in bl:
        a("## 8. Baselines (identical splits and windows)")
        a("")
        a("### Flare in progress")
        a("")
        a("| model | TSS | HSS | AUC | BSS |")
        a("|---|---:|---:|---:|---:|")
        if n:
            a(f"| **SoLEXHEL-Net** | **{_f(n.get('TSS'))}** | {_f(n.get('HSS'))} | "
              f"{_f(n.get('AUC'))} | {_f(n.get('BSS_vs_climatology'))} |")
        for name in ("gbdt", "logistic"):
            s = bl.get(name, {}).get("in_flare")
            if s:
                a(f"| {name} | {_f(s.get('TSS'))} | {_f(s.get('HSS'))} | "
                  f"{_f(s.get('AUC'))} | {_f(s.get('BSS_vs_climatology'))} |")
        a("| climatology | 0.000 | 0.000 | 0.500 | 0.000 |")
        a("")

        occ_keys = list((occ or {}).keys())
        if occ_keys:
            a("### Flare occurrence within horizon (TSS)")
            a("")
            a("| model | " + " | ".join(occ_keys) + " |")
            a("|---" * (len(occ_keys) + 1) + "|")
            if occ:
                a("| **SoLEXHEL-Net** | " + " | ".join(
                    _f(occ[k].get("TSS")) for k in occ_keys) + " |")
            for name in ("gbdt", "logistic"):
                d = bl.get(name, {}).get("occurrence", {})
                if d:
                    a(f"| {name} | " + " | ".join(
                        _f(d.get(k, {}).get("TSS")) for k in occ_keys) + " |")
            a("")

        fk = list((fr or {}).keys())
        if fk:
            a("### Forecast RMSE (log flux; lower is better)")
            a("")
            a("| model | " + " | ".join(fk) + " |")
            a("|---" * (len(fk) + 1) + "|")
            if fr:
                a("| **SoLEXHEL-Net** | " + " | ".join(
                    _f(fr[k].get("RMSE"), 4) for k in fk) + " |")
            for name in ("gbdt", "persistence"):
                d = bl.get(name, {}).get("forecast", {})
                if d:
                    a(f"| {name} | " + " | ".join(
                        _f(d.get(k, {}).get("RMSE"), 4) for k in fk) + " |")
            a("")

    # --- rise-phase forecasting --------------------------------------
    fc = _load(reports / "forecast_cv.json")
    if fc and "summary" in fc:
        a("## 9. Rise-phase forecasting - architecture comparison")
        a("")
        a(f"Given the early rise of a flare, predict its peak magnitude, time to "
          f"peak, whether it exceeds {format_threshold(float(fc.get('threshold_rate', float('nan'))))}, and "
          f"whether it is long-duration. {fc.get('n_events')} events, "
          f"{fc.get('n_samples')} rise samples, rolling-origin CV **grouped by "
          f"event** (no flare ever appears on both sides of a split).")
        a("")
        a("| encoder | folds | peak log-MAE (mean +/- sd) | skill vs current | "
          "skill vs climatology | time-to-peak MAE (min) | ttp skill vs clim | "
          "exceeds AUC | runtime (s) |")
        a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        ranked = sorted(fc["summary"].items(),
                        key=lambda kv: kv[1].get("peak_log_MAE_mean", float("inf")))
        for enc, s in ranked:
            a(f"| {enc} | {s.get('n_folds')} | "
              f"{_f(s.get('peak_log_MAE_mean'), 4)} +/- {_f(s.get('peak_log_MAE_std'), 4)} | "
              f"{_f(s.get('peak_skill_vs_current_mean'))} | "
              f"{_f(s.get('peak_skill_vs_climatology_mean'))} | "
              f"{_f(s.get('time_to_peak_MAE_min_mean'), 1)} | "
              f"{_f(s.get('ttp_skill_vs_climatology_mean'))} | "
              f"{_f(s.get('exceeds_AUC_mean'))} | {_f(s.get('total_seconds'), 0)} |")
        a("")
        ref = next(iter(fc["summary"].values()), {})
        a(f"Reference forecasts (same folds): **current level** peak log-MAE "
          f"{_f(ref.get('peak_log_MAE_current_mean'), 4)} - assume the flare has "
          f"already peaked; **training climatology** "
          f"{_f(ref.get('peak_log_MAE_climatology_mean'), 4)} - predict the average "
          f"training-event peak. Skill = 1 - MAE_model / MAE_reference; at or below "
          f"zero the model adds nothing over that reference, however small its "
          f"raw MAE looks.")
        a("")
        beaten = [enc for enc, s in ranked
                  if (s.get("peak_skill_vs_current_mean") or -1) > 0
                  and (s.get("peak_skill_vs_climatology_mean") or -1) > 0]
        a(f"Encoders beating **both** references on peak magnitude: "
          f"{', '.join(beaten) if beaten else '**none**'}.")
        a("")

        ttp_pos = [enc for enc, s in ranked
                   if (s.get("ttp_skill_vs_climatology_mean") or -1) > 0]
        a(f"Encoders beating climatology on **time to peak**: "
          f"{', '.join(ttp_pos) if ttp_pos else '**none** - no model predicts peak timing better than the average training rise'}.")
        a("")

        exc_folds = ref.get("exceeds_folds_with_both_classes")
        exc_pos = ref.get("exceeds_positive_events_total")
        exc_ref = ref.get("exceeds_AUC_current_level_mean")
        if exc_folds is not None:
            a("**Large-flare exceedance.** "
              f"The AUC column rests on {exc_folds} fold(s) containing both classes "
              f"and **{exc_pos} large flare(s) in total**. Ranking samples by the "
              f"flux already reached scores AUC **{_f(exc_ref)}** on the same "
              f"fold(s):")
            a("")
            a("| encoder | exceeds AUC | AUC gain vs current level |")
            a("|---|---:|---:|")
            for enc, s in ranked:
                a(f"| {enc} | {_f(s.get('exceeds_AUC_mean'))} | "
                  f"{_f(s.get('exceeds_AUC_gain_vs_current_mean'))} |")
            a("")
            if (exc_pos or 0) < 3:
                a("> With fewer than three large flares, this AUC describes individual "
                  "events rather than a skill, and a negative gain means the model has "
                  "learned nothing beyond how bright the flare already is. Do not quote "
                  "it as a result.")
                a("")

        folds = fc.get("folds", [])
        if folds:
            a("Per fold (peak log-MAE) - the scatter between folds is the headline:")
            a("")
            encs = [e for e, _ in ranked]
            fold_ids = sorted({f["fold"] for f in folds})
            a("| fold | n test | " + " | ".join(encs) + " |")
            a("|---|---:" + "|---:" * len(encs) + "|")
            for fi in fold_ids:
                row = {f["encoder"]: f for f in folds if f["fold"] == fi}
                n_te = next(iter(row.values()))["n_test"] if row else "?"
                cells = [_f(row[e]["peak_log_MAE"]["value"], 4) if e in row else "--"
                         for e in encs]
                a(f"| {fi} | {n_te} | " + " | ".join(cells) + " |")
            a("")

            # Is the winner distinguishable from the rest, or inside the noise?
            means = [s.get("peak_log_MAE_mean") for _, s in ranked]
            sds = [s.get("peak_log_MAE_std") for _, s in ranked]
            if len(means) >= 2 and all(m == m for m in means[:2]):
                gap = means[1] - means[0]
                noise = max(sds[0], sds[1])
                verdict = ("smaller than the fold-to-fold scatter, so the ranking is "
                           "**not** statistically meaningful on this data"
                           if gap < noise else
                           "larger than the fold-to-fold scatter")
                a(f"> The gap between the top two encoders ({_f(gap, 4)}) is "
                  f"{verdict}. With this few events, treat the ordering as "
                  f"provisional until more flares are added.")
                a("")

        if fc.get("failures"):
            a("Folds that did not complete:")
            a("")
            for f in fc["failures"]:
                a(f"* `{f['encoder']}` fold {f['fold']}: {f['error']}")
            a("")

    fu = _load(reports / "fusion_ablation.json")
    if fu:
        a("## 10. Does HEL1OS add skill? (paired fusion ablation)")
        a("")
        if "error" in fu:
            a(f"Not run: {fu['error']} (n = {fu.get('n_events')}).")
            a("")
        else:
            r = fu.get("peak_MAE_reduction_from_hel1os", {})
            a(f"Only flares whose rise HEL1OS observed (>= "
              f"{_f(100 * fu.get('min_hard_coverage', 0), 0)}% of rise bins): "
              f"**{fu.get('n_events_eligible')}** flares, "
              f"{fu.get('n_test_events')} scored out of sample. The `{fu.get('encoder')}` "
              f"encoder is cross-validated twice over **identical folds and seeds** - "
              f"with the hard X-ray input and with it blanked - and the per-sample "
              f"difference in peak error is bootstrapped over flares.")
            a("")
            a("| arm | peak log-MAE | exceeds AUC |")
            a("|---|---:|---:|")
            a(f"| soft + hard | {_f(fu.get('peak_log_MAE_soft_hard'), 4)} | "
              f"{_f(fu.get('exceeds_AUC_soft_hard'))} |")
            a(f"| soft only | {_f(fu.get('peak_log_MAE_soft_only'), 4)} | "
              f"{_f(fu.get('exceeds_AUC_soft_only'))} |")
            a("")
            if r:
                lo, hi = r.get("ci_lo"), r.get("ci_hi")
                verdict = ("HEL1OS **measurably reduces** the peak error" if lo == lo and lo > 0
                           else "HEL1OS **measurably increases** the peak error" if hi == hi and hi < 0
                           else "**no detectable effect** of HEL1OS at this sample size")
                a(f"Error reduction from adding HEL1OS: **{_f(r.get('value'), 4)}** "
                  f"(95% CI {_f(lo, 4)} to {_f(hi, 4)}; "
                  f"{_f(100 * fu.get('relative_reduction', float('nan')), 1)}% of the "
                  f"soft-only error); lower error on "
                  f"{_f(100 * fu.get('events_improved_fraction', float('nan')), 0)}% of "
                  f"flares. {verdict}.")
                a("")

    ed = _load(reports / "event_definitions.json")
    if ed.get("definitions"):
        a("## 11. Skill on significant flares (post-hoc)")
        a("")
        a("Same trained model and outputs; only the truth changes to flares at least "
          "3x / 10x / 30x above background. Thresholds re-chosen on validation, applied "
          "to test; TSS intervals resample whole days. *Oracle persistence* uses the "
          "in-progress label at the origin (hindsight, not available in real time) - an "
          "upper-bound reference. *AUC current flux* ranks by the flux already reached, "
          "which is available in real time.")
        a("")
        a("| truth | target | base rate | TSS [95% CI] | oracle persistence TSS | AUC | "
          "AUC current flux |")
        a("|---|---|---:|---:|---:|---:|---:|")
        for name, block in ed["definitions"].items():
            for tgt, b in block.items():
                if not isinstance(b, dict) or "TSS" not in b:
                    continue
                ci = b.get("TSS_ci", [None, None])
                a(f"| {name} | {tgt.replace('_', ' ')} | {_f(b.get('base_rate_test'))} | "
                  f"{_f(b.get('TSS'))} [{_f(ci[0])}, {_f(ci[1])}] | "
                  f"{_f(b.get('TSS_persistence'))} | {_f(b.get('AUC'))} | "
                  f"{_f(b.get('AUC_current_flux'))} |")
        a("")

    tr = ev.get("training")
    if tr:
        a("## Training")
        a("")
        a(f"Nowcast model: best epoch {tr.get('best_epoch')} "
          f"(selection score {_f(tr.get('best_score'))}), "
          f"{_f(tr.get('seconds'), 1)} s wall-clock.")
        a("")

    a("## Figures")
    a("")
    # Figure names depend on the data (segment dates, the day of the largest
    # flare), so describe whatever the run actually produced.
    described = (
        ("mission_overview.png", "mission light curve, flares per month, coverage"),
        ("lightcurve_solexs_", "SoLEXS light curve with detected flares"),
        ("lightcurve_hel1os_", "HEL1OS bands after fill-row masking"),
        ("spectrogram_", "time-energy spectrogram of the day of the largest flare"),
        ("training_history.png", "loss and validation skill"),
        ("test_timeline.png", "predictions over the test period"),
        ("reliability.png", "probability calibration"),
    )
    fig_dir = Path(out_dir) / "figures"
    if fig_dir.exists():
        for prefix, desc in described:
            for f in sorted(fig_dir.glob(prefix + ("*" if not prefix.endswith(".png") else ""))):
                a(f"* `figures/{f.name}` - {desc}")
    a("")

    path = reports / "RESULTS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path
