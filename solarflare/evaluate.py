"""Evaluation against baselines, with operating points fitted on validation.

Nothing here touches the test set until the thresholds are already fixed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .pipeline import Prepared, make_loaders, resolve_device
from .train import collect_predictions
from .metrics import (
    skill_scores, best_threshold, roc_auc, brier_score, brier_skill_score,
    reliability, regression_scores, skill_vs_reference, multiclass_scores,
    forecast_score_mask,
)
from .models.net import SolexHelNet


def climatology_rate(y: np.ndarray) -> float:
    return float(np.mean(y)) if y.size else float("nan")


def evaluate(model: SolexHelNet, prep: Prepared, cfg: Config,
             verbose: bool = True) -> dict:
    device = resolve_device(cfg.train.device)
    _, va, te, stats = make_loaders(prep, cfg)

    val = collect_predictions(model, va, device) if len(va) else None
    test = collect_predictions(model, te, device) if len(te) else None

    # Climatology reference: the mean log flux over the TRAINING split.
    # Estimated on train, never on test -- a "climatology" fitted to the test
    # set would be using the answer.
    tr_loader, _, _, _ = make_loaders(prep, cfg)
    clim_mean = 0.0
    n_seen = 0
    for b in tr_loader:
        m = b["nowcast_mask"].numpy() > 0
        if m.any():
            v = b["nowcast"].numpy()[m]
            clim_mean += float(v.sum())
            n_seen += int(m.sum())
    clim_mean = clim_mean / max(n_seen, 1)

    report: dict = {"label_stats": stats}
    if test is None:
        report["error"] = "empty test split"
        return report

    # --- 1. Flare in progress now (nowcasting) -------------------------
    m = test["y_nowcast_mask"] > 0
    y = test["y_in_flare"][m]
    p = test["p_inflare"][m]

    thr = 0.5
    if val is not None:
        vm = val["y_nowcast_mask"] > 0
        if vm.sum() and len(np.unique(val["y_in_flare"][vm])) > 1:
            thr, _ = best_threshold(val["y_in_flare"][vm], val["p_inflare"][vm], "TSS")

    now = skill_scores(y, p >= thr) if y.size else {}
    now.update({
        "threshold": float(thr),
        "AUC": roc_auc(y, p) if y.size else float("nan"),
        "Brier": brier_score(y, p) if y.size else float("nan"),
        "BSS_vs_climatology": brier_skill_score(y, p) if y.size else float("nan"),
        "reliability": reliability(y, p) if y.size else {},
    })
    report["nowcast_in_flare"] = now

    # --- 2. Flare phase ------------------------------------------------
    ph_pred = test["p_phase"].argmax(axis=1)[m]
    report["nowcast_phase"] = multiclass_scores(test["y_phase"][m], ph_pred, 4)

    # --- 3. Flare occurrence within H (forecasting) --------------------
    occ = {}
    for h, hs in enumerate(cfg.win.occurrence_horizons_s):
        mm = test["y_occurrence_mask"][:, h] > 0
        yt = test["y_occurrence"][mm, h]
        pt = test["p_occurrence"][mm, h]
        t_h = 0.5
        if val is not None:
            vmm = val["y_occurrence_mask"][:, h] > 0
            if vmm.sum() and len(np.unique(val["y_occurrence"][vmm, h])) > 1:
                t_h, _ = best_threshold(val["y_occurrence"][vmm, h],
                                        val["p_occurrence"][vmm, h], "TSS")
        s = skill_scores(yt, pt >= t_h) if yt.size else {}
        s.update({
            "threshold": float(t_h),
            "AUC": roc_auc(yt, pt) if yt.size else float("nan"),
            "Brier": brier_score(yt, pt) if yt.size else float("nan"),
            "BSS_vs_climatology": brier_skill_score(yt, pt) if yt.size else float("nan"),
            "climatology_rate": climatology_rate(yt),
        })
        occ[f"{int(hs / 60)}min"] = s
    report["forecast_occurrence"] = occ

    # --- 4. Nowcast regression ----------------------------------------
    report["nowcast_regression"] = regression_scores(
        test["y_nowcast"], test["nowcast"], test["y_nowcast_mask"])

    # --- 5. Forecast regression vs persistence -------------------------
    qs = list(cfg.win.quantiles)
    i50 = int(np.argmin(np.abs(np.array(qs) - 0.5)))
    fore = {}
    for h, hs in enumerate(cfg.win.forecast_horizons_s):
        mm = forecast_score_mask(test["y_forecast_mask"][:, h], test["y_nowcast_mask"])
        yt = test["y_forecast"][:, h]
        q50 = test["forecast"][:, h, i50]
        ref = test["y_persistence"]
        r = regression_scores(yt, q50, mm)
        r["n"] = int(mm.sum())
        r["skill_vs_persistence"] = skill_vs_reference(yt, q50, ref, mm)
        r["persistence_RMSE"] = regression_scores(yt, ref, mm)["RMSE"]

        # Climatology is the reference that matters at long horizons: beating
        # persistence there is easy, because persistence extrapolates a
        # decaying flare. A model that merely predicts the mean beats
        # persistence and is still useless, so both references are reported.
        clim = np.full_like(yt, clim_mean)
        r["climatology_RMSE"] = regression_scores(yt, clim, mm)["RMSE"]
        r["skill_vs_climatology"] = skill_vs_reference(yt, q50, clim, mm)
        if mm.sum() > 2 and np.std(q50[mm]) > 0 and np.std(yt[mm]) > 0:
            r["correlation"] = float(np.corrcoef(q50[mm], yt[mm])[0, 1])
            r["spread_ratio"] = float(np.std(q50[mm]) / np.std(yt[mm]))
        else:
            r["correlation"] = float("nan")
            r["spread_ratio"] = float("nan")
        # Empirical coverage of the predicted interval.
        lo = test["forecast"][:, h, 0]
        hi = test["forecast"][:, h, -1]
        if mm.any():
            inside = (yt[mm] >= lo[mm]) & (yt[mm] <= hi[mm])
            r["interval_coverage"] = float(inside.mean())
            r["nominal_coverage"] = float(qs[-1] - qs[0])
        fore[f"{int(hs / 60)}min"] = r
    report["forecast_regression"] = fore

    # --- 6. Peak prediction -------------------------------------------
    pm = test["y_peak_mask"] > 0
    if pm.any():
        report["peak"] = {
            "time_to_peak_MAE_min": float(
                np.abs(test["peak"][:, 0] - test["y_peak"][:, 0])[pm[:, 0]].mean() * 60),
            "log_peak_flux_MAE": float(
                np.abs(test["peak"][:, 1] - test["y_peak"][:, 1])[pm[:, 1]].mean()),
            "n": int(pm[:, 0].sum()),
        }

    # --- 7. Ablation: how much does each instrument contribute? --------
    report["modality_ablation"] = modality_ablation(model, prep, cfg, thr)
    report["modality_ablation_hel1os_observed"] = modality_ablation(
        model, prep, cfg, thr, require_hard=True)

    if verbose:
        _print_report(report, cfg)
    return report


@torch.no_grad()
def modality_ablation(model: SolexHelNet, prep: Prepared, cfg: Config,
                      thr: float, require_hard: bool = False) -> dict:
    """Re-score the test split with each modality masked off.

    This is the honest way to answer "does the hard X-ray channel help?" --
    and, with non-overlapping input files, to show plainly that it currently
    cannot.

    The ``clock_only`` row is the guard rail: it blanks **both** instruments,
    so whatever skill remains comes from non-instrument inputs alone.  Anything
    meaningfully above AUC 0.5 there means the model is memorising time rather
    than reading the Sun, which is exactly the failure that time-of-day
    features caused on this single-day dataset.

    ``require_hard`` scores only windows where HEL1OS observed the prediction
    origin. On the archive HEL1OS covers ~72 of ~170 test days, so over the
    whole test split any hard X-ray effect is diluted by windows where blanking
    HEL1OS changes nothing because it was never there.
    """
    device = resolve_device(cfg.train.device)
    _, _, te, _ = make_loaders(prep, cfg)
    model.eval()
    out: dict = {}

    for name, zero_soft, zero_hard in (("both", False, False),
                                       ("soft_only", False, True),
                                       ("hard_only", True, False),
                                       ("clock_only", True, True)):
        ys, ps, ms = [], [], []
        for batch in te:
            b = {k: v.to(device) for k, v in batch.items()}
            sm = torch.zeros_like(b["soft_mask"]) if zero_soft else b["soft_mask"]
            hm = torch.zeros_like(b["hard_mask"]) if zero_hard else b["hard_mask"]
            o = model(b["soft"], sm, b["hard"], hm, b["clock"])
            ps.append(torch.sigmoid(o["in_flare"]).cpu().numpy())
            ys.append(batch["in_flare"].numpy())
            m = batch["nowcast_mask"].numpy() > 0
            if require_hard:
                m &= batch["hard_mask"].numpy()[:, -1] > 0
            ms.append(m)
        y = np.concatenate(ys)
        p = np.concatenate(ps)
        mk = np.concatenate(ms)
        if mk.sum() and len(np.unique(y[mk])) > 1:
            s = skill_scores(y[mk], p[mk] >= thr)
            out[name] = {"TSS": s["TSS"], "HSS": s["HSS"], "POD": s["POD"],
                         "FAR": s["FAR"], "AUC": roc_auc(y[mk], p[mk]),
                         "n": int(mk.sum())}
        else:
            out[name] = {"TSS": float("nan"), "n": int(mk.sum())}
    return out


def _fmt(v) -> str:
    if isinstance(v, float):
        return "nan" if np.isnan(v) else f"{v:.4f}"
    return str(v)


def _print_report(r: dict, cfg: Config) -> None:
    print("\n" + "=" * 74)
    print("TEST-SET EVALUATION")
    print("=" * 74)

    n = r.get("nowcast_in_flare", {})
    if n:
        print(f"\n[1] Nowcast: flare in progress   (threshold {n['threshold']:.2f} "
              f"fitted on validation)")
        print(f"    base rate {n.get('base_rate', float('nan')):.3f}   "
              f"TSS {n['TSS']:.3f}   HSS {n['HSS']:.3f}   AUC {_fmt(n['AUC'])}")
        print(f"    POD {n['POD']:.3f}  FAR {n['FAR']:.3f}  CSI {n['CSI']:.3f}  "
              f"F1 {n['F1']:.3f}")
        print(f"    Brier {_fmt(n['Brier'])}   BSS vs climatology "
              f"{_fmt(n['BSS_vs_climatology'])}")
        print(f"    TP {n['TP']} FP {n['FP']} FN {n['FN']} TN {n['TN']}")

    p = r.get("nowcast_phase", {})
    if p:
        names = ["quiet", "rise", "peak", "decay"]
        print(f"\n[2] Nowcast: flare phase   accuracy {p['accuracy']:.3f}   "
              f"macro-F1 {p['macro_f1']:.3f}")
        for c, name in enumerate(names):
            pc = p["per_class"][c] if c in p["per_class"] else p["per_class"].get(str(c), {})
            if pc:
                print(f"    {name:6s} P {pc['precision']:.3f} R {pc['recall']:.3f} "
                      f"F1 {pc['f1']:.3f}  n={pc['support']}")

    occ = r.get("forecast_occurrence", {})
    if occ:
        print("\n[3] Forecast: flare within horizon")
        print(f"    {'horizon':>8}  {'base':>6}  {'TSS':>6}  {'HSS':>6}  "
              f"{'POD':>6}  {'FAR':>6}  {'AUC':>6}  {'BSS':>7}")
        for k, s in occ.items():
            print(f"    {k:>8}  {s.get('climatology_rate', float('nan')):6.3f}  "
                  f"{s.get('TSS', float('nan')):6.3f}  {s.get('HSS', float('nan')):6.3f}  "
                  f"{s.get('POD', float('nan')):6.3f}  {s.get('FAR', float('nan')):6.3f}  "
                  f"{s.get('AUC', float('nan')):6.3f}  "
                  f"{s.get('BSS_vs_climatology', float('nan')):7.3f}")

    nr = r.get("nowcast_regression", {})
    if nr:
        print(f"\n[4] Nowcast regression (log flux)   MAE {nr['MAE']:.4f}  "
              f"RMSE {nr['RMSE']:.4f}  R2 {_fmt(nr['R2'])}")

    fr = r.get("forecast_regression", {})
    if fr:
        print("\n[5] Forecast regression (log flux), median quantile")
        print(f"    {'horizon':>8}  {'RMSE':>7}  {'persist':>7}  {'clim':>7}  "
              f"{'vs_pers':>7}  {'vs_clim':>7}  {'corr':>6}  {'cover':>6}")
        for k, s in fr.items():
            print(f"    {k:>8}  {s['RMSE']:7.4f}  "
                  f"{s.get('persistence_RMSE', float('nan')):7.4f}  "
                  f"{s.get('climatology_RMSE', float('nan')):7.4f}  "
                  f"{s.get('skill_vs_persistence', float('nan')):7.3f}  "
                  f"{s.get('skill_vs_climatology', float('nan')):7.3f}  "
                  f"{s.get('correlation', float('nan')):6.3f}  "
                  f"{s.get('interval_coverage', float('nan')):6.3f}")
        print(f"    (skill > 0 beats that reference. Nominal interval coverage "
              f"{fr[list(fr)[0]].get('nominal_coverage', float('nan')):.2f}.)")
        print("    Beating persistence but NOT climatology, with corr ~ 0, means")
        print("    the model has fallen back to predicting the mean at that horizon.")

    pk = r.get("peak", {})
    if pk:
        print(f"\n[6] Ongoing-event peak   time-to-peak MAE "
              f"{pk['time_to_peak_MAE_min']:.1f} min   "
              f"log-peak-flux MAE {pk['log_peak_flux_MAE']:.3f}  (n={pk['n']})")

    ab = r.get("modality_ablation", {})
    if ab:
        print("\n[7] Modality ablation (flare-in-progress skill on test)")
        for k, s in ab.items():
            print(f"    {k:>10}  TSS {_fmt(s.get('TSS'))}  AUC {_fmt(s.get('AUC'))}")
        co = ab.get("clock_only", {}).get("AUC")
        if co is not None and not (isinstance(co, float) and np.isnan(co)) and co > 0.6:
            print(f"    WARNING: with both instruments blanked the model still "
                  f"reaches AUC {co:.3f}.")
            print("    It is memorising non-instrument inputs (time of day) rather "
                  "than reading")
            print("    the Sun. Set ModelConfig.use_clock = False, or add more days "
                  "of data.")


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
