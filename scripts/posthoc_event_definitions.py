"""Is the nowcast model skilful on *significant* flares, or only on the loose
event definition it was trained with?

    python scripts/posthoc_event_definitions.py --data-root D:/Data --out-dir outputs/archive

The training labels come from this project's own detector, which on the
2024-2026 archive marks 66% of observed soft X-ray time as "in a flare"
(11.8 events per day, 39% of them below 2x background). With a base rate that
high, "a flare within 60 min" is true ~94% of the time and skill scores are
not comparable with GOES >=C / >=M studies.

This script keeps the trained model and its outputs fixed and changes only
the truth: flares at least 3x, 10x or 30x above background. Operating
thresholds are re-chosen on VALIDATION for each definition, then applied to
TEST. Two references accompany every row:

* oracle persistence -- "a flare of this size is in progress now" as the
  forecast, using the *label* at the origin. That label comes from the
  detector's centred background, so it is not available in real time: treat
  it as an upper-bound reference showing how much of "flare within H" is just
  "a flare is already happening", not as an operational baseline;
* current flux -- ranking windows by the flux already reached (AUC only). This
  one IS available in real time, and is the fair ranking reference.

A model that cannot beat those on significant flares has learned the
detector's loose labels, not flare onset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarflare.config import Config  # noqa: E402
from solarflare.forecast import bootstrap_ci  # noqa: E402
from solarflare.metrics import best_threshold, roc_auc, skill_scores  # noqa: E402
from solarflare.pipeline import make_loaders, prepare, resolve_device  # noqa: E402
from solarflare.preprocess.dataset import build_targets  # noqa: E402
from solarflare.preprocess.labels import occurrence_labels  # noqa: E402
from solarflare.train import collect_predictions, load_model  # noqa: E402

TIERS = (None, 3.0, 10.0, 30.0)          # x background, SoLEXS-detector labels
GOES_TIERS = (None, "M1.0", "X1.0")      # GOES classes, GOES labels


def _keep(e, tier) -> bool:
    if tier is None:
        return True
    if isinstance(tier, str):
        from solarflare.io.goes import class_flux
        return e.peak_rate >= class_flux(tier)
    return e.magnitude >= tier


def tier_labels(prep, cfg, split: str, tier: float | None):
    """Per-window labels under a magnitude floor, in loader order."""
    dt = cfg.pre.dt_seconds
    occ_steps = [max(int(h / dt), 1) for h in cfg.win.occurrence_horizons_s]
    cache: dict[int, tuple] = {}
    now, occ, m_now, m_occ, flux, day = [], [], [], [], [], []
    for i in prep.splits[split]:
        w = prep.windows[i]
        if w.seg not in cache:
            seg = prep.segments[w.seg]
            infl = np.zeros(len(seg), np.float32)
            for e in seg.events:
                if _keep(e, tier):
                    infl[e.start_idx:e.end_idx] = 1.0
            t = build_targets(seg, cfg)
            cache[w.seg] = (infl, occurrence_labels(infl, occ_steps), t)
        infl, o, t = cache[w.seg]
        j = w.end - 1
        now.append(infl[j])
        occ.append(o[j])
        m_now.append(t["nowcast_mask"][j] > 0)
        m_occ.append(t["occurrence_mask"][j] > 0)
        flux.append(t["nowcast"][j])
        day.append(int(w.t_unix // 86400))
    return (np.asarray(now), np.asarray(occ), np.asarray(m_now), np.asarray(m_occ),
            np.asarray(flux), np.asarray(day))


def score_block(y_va, p_va, y_te, p_te, persist_te, flux_te, day_te) -> dict:
    out = {"n_test": int(y_te.size), "base_rate_test": float(y_te.mean()) if y_te.size else float("nan"),
           "positives_test": int(y_te.sum())}
    if y_te.size == 0 or len(np.unique(y_te)) < 2 or len(np.unique(y_va)) < 2:
        out["note"] = "single class"
        return out
    thr, _ = best_threshold(y_va, p_va, "TSS")
    s = skill_scores(y_te, p_te >= thr)
    sp = skill_scores(y_te, persist_te > 0.5)

    def tss(idx):
        if len(np.unique(y_te[idx])) < 2:
            raise ValueError("degenerate")
        return skill_scores(y_te[idx], p_te[idx] >= thr)["TSS"]

    def tss_gain(idx):
        if len(np.unique(y_te[idx])) < 2:
            raise ValueError("degenerate")
        return (skill_scores(y_te[idx], p_te[idx] >= thr)["TSS"]
                - skill_scores(y_te[idx], persist_te[idx] > 0.5)["TSS"])

    _, lo, hi = bootstrap_ci(None, day_te, tss, n_boot=300)
    g, glo, ghi = bootstrap_ci(None, day_te, tss_gain, n_boot=300)
    out.update({
        "threshold_from_val": float(thr),
        "TSS": s["TSS"], "TSS_ci": [lo, hi], "HSS": s["HSS"], "POD": s["POD"], "FAR": s["FAR"],
        "AUC": roc_auc(y_te, p_te),
        "TSS_persistence": sp["TSS"], "HSS_persistence": sp["HSS"],
        "TSS_gain_vs_persistence": g, "TSS_gain_ci": [glo, ghi],
        "AUC_current_flux": roc_auc(y_te, flux_te),
    })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default="outputs/archive")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--energy-scale", default="legacy_linear",
                    help="must match the scale the checkpoint was trained on")
    ap.add_argument("--labels", default="solexs", choices=["solexs", "goes"],
                    help="must match the labels the checkpoint was trained with")
    ap.add_argument("--goes-dir", default="")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    cfg = Config(data_root=Path(args.data_root), out_dir=Path(args.out_dir))
    cfg.pre.solexs_energy_scale = args.energy_scale
    cfg.pre.label_source = args.labels
    cfg.pre.goes_dir = args.goes_dir
    if args.cache_dir:
        cfg.cache_dir = Path(args.cache_dir)
    prep = prepare(cfg, verbose=False)
    trained = json.loads((Path(args.out_dir) / "reports" / "data_meta.json").read_text("utf-8"))
    for k in ("n_windows", "n_train", "n_val", "n_test"):
        if trained.get(k) != prep.meta.get(k):
            raise SystemExit(f"split differs from training ({k}); results would not "
                             "describe the trained model")

    device = resolve_device(cfg.train.device)
    ckpt = Path(args.checkpoint or Path(args.out_dir) / "checkpoints" / "best.pt")
    model = load_model(ckpt, cfg, device)
    _, va, te, _ = make_loaders(prep, cfg)
    pv = collect_predictions(model, va, device)
    pt = collect_predictions(model, te, device)

    results: dict = {"definitions": {}, "horizons_min": [int(h / 60) for h in cfg.win.occurrence_horizons_s]}
    tiers = GOES_TIERS if cfg.pre.label_source == "goes" else TIERS
    for tier in tiers:
        name = ("all detected (training labels)" if tier is None else
                f">= GOES {tier}" if isinstance(tier, str) else f">= {tier:g}x background")
        nv, ov, mnv, mov, _, _ = tier_labels(prep, cfg, "val", tier)
        nt, ot, mnt, mot, ft, dt_ = tier_labels(prep, cfg, "test", tier)
        block = {"in_progress_now": score_block(
            nv[mnv], pv["p_inflare"][mnv], nt[mnt], pt["p_inflare"][mnt],
            nt[mnt], ft[mnt], dt_[mnt])}
        for h, hm in enumerate(results["horizons_min"]):
            mv, mt = mov[:, h], mot[:, h]
            block[f"within_{hm}min"] = score_block(
                ov[mv, h], pv["p_occurrence"][mv, h], ot[mt, h], pt["p_occurrence"][mt, h],
                nt[mt], ft[mt], dt_[mt])
        results["definitions"][name] = block
        n_ev = sum(1 for s in prep.segments for e in s.events if _keep(e, tier))
        block["n_events_archive"] = n_ev

    # The run's modality ablation was scored over all test windows, where
    # HEL1OS is absent on most days. Add the version restricted to windows
    # HEL1OS observed, at the run's own validation threshold.
    from solarflare.evaluate import modality_ablation
    ev_path = Path(args.out_dir) / "reports" / "evaluation.json"
    ev = json.loads(ev_path.read_text("utf-8"))
    if "modality_ablation_hel1os_observed" not in ev:
        thr = float(ev["nowcast_in_flare"]["threshold"])
        ev["modality_ablation_hel1os_observed"] = modality_ablation(
            model, prep, cfg, thr, require_hard=True)
        ev_path.write_text(json.dumps(ev, indent=2, default=float), encoding="utf-8")
    results["modality_ablation_hel1os_observed"] = ev["modality_ablation_hel1os_observed"]
    print("modality ablation, HEL1OS-observed test windows:")
    for k, s in ev["modality_ablation_hel1os_observed"].items():
        print(f"  {k:10s} n={s.get('n')}  TSS {s.get('TSS', float('nan')):.4f}  "
              f"AUC {s.get('AUC', float('nan')):.4f}")

    out = Path(args.out_dir) / "reports" / "event_definitions.json"
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")

    print(f"{'definition':32s} {'target':>14s} {'base':>6s} {'TSS':>6s} {'[95% CI]':>15s} "
          f"{'oracle':>7s} {'gain [CI]':>22s} {'AUC':>6s} {'AUCflux':>7s}")
    for name, block in results["definitions"].items():
        for tgt, b in block.items():
            if not isinstance(b, dict) or "TSS" not in b:
                continue
            print(f"{name:32s} {tgt:>14s} {b['base_rate_test']:6.3f} {b['TSS']:6.3f} "
                  f"[{b['TSS_ci'][0]:.3f},{b['TSS_ci'][1]:.3f}] {b['TSS_persistence']:7.3f} "
                  f"{b['TSS_gain_vs_persistence']:+.3f} [{b['TSS_gain_ci'][0]:+.3f},{b['TSS_gain_ci'][1]:+.3f}] "
                  f"{b['AUC']:6.3f} {b['AUC_current_flux']:7.3f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
