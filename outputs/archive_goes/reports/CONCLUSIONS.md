# Conclusions — GOES-labelled retrain, compared with the first archive run

Run of 2026-09-15. It uses the same archive, windows, split, architectures, folds and seeds as
`outputs/archive`. Three things changed:

1. **Flare truth.** The GOES-18 flare list at ≥ C1.0 replaces the loose SoLEXS detector.
2. **Targets.** They are now log10 GOES XRS-B flux in W/m² instead of log(1 + SoLEXS count rate).
3. **Energy scale.** SoLEXS inputs use the published scale (Sarwade et al. 2025).

Every number is from `outputs/archive_goes/reports/*.json`, and the old run's from
`outputs/archive/reports/*.json`. The test period is 2026-03-24 → 2026-09-12 and was never used for
training or threshold choice. Frozen forward-test model: **v2**, sha256 `a351b54d…`, data cutoff
2026-09-13 23:59 UTC.

## Bottom line

1. **Detecting a GOES flare in progress works well.** Nowcast TSS is **0.77** [0.75, 0.78] for ≥ C,
   0.79 for ≥ M and 0.75 for ≥ X. It catches 82 % of ≥ C flare minutes. Because flares occupy only 8 %
   of the time, 40 % of alarms are false. The network edges out gradient-boosted trees (0.745) and
   logistic regression (0.711).
2. **Warning that a flare is coming is weak, and the loose labels had hidden that.**
   - "≥ C flare within 60 min": TSS 0.26 now, versus 0.56 on the old labels.
   - For **≥ M and ≥ X flares, ranking by the X-ray flux already reached beats the model** at every
     horizon: AUC 0.85–0.95 versus 0.74–0.86.
   - The occurrence probabilities are worse calibrated than the base rate (Brier skill < 0).
3. **The minutes-ahead flux forecast barely beats "no change".** The first report said the model was
   66 % better at 60 min. That was an evaluation bug, fixed in this run and described in §4.
   - The fair Aditya-only reference is "no change" in calibrated SoLEXS flux. Against it the model is
     worse at 0–5 min and 6–11 % better at 30–60 min.
   - The network reads the current GOES level less accurately than a one-line calibration does
     (0.092 vs 0.069 dex).
4. **The final size of a flare that is already rising is forecastable, but only modestly.** Typical
   error is 0.20 dex (×1.6), 20 % better than the level already reached (old labels: 36 %). Time to peak
   is no better than climatology. Transformer, linear, GRU and TCN are again tied within fold scatter.
5. **HEL1OS hard X-rays measurably help, confirmed over four seeds.** On the 826 flares HEL1OS watched
   rise, adding it cuts peak error by **15.4 % ± 4.4 % (0.0425 ± 0.0138 dex)**, positive in every run and
   with an interval excluding zero in each. It also lifts the ≥ M exceedance AUC from 0.695 to 0.778. The
   gain is **architecture-dependent**: with a linear encoder it falls to 3.9 % with an interval spanning
   zero, so it needs a model that can use the *shape* of the hard X-ray rise. On HEL1OS-observed windows
   it raises nowcast TSS from 0.71 to 0.77, and HEL1OS alone nearly matches SoLEXS alone (0.71 vs 0.71).
   The same test on the old labels showed nothing detectable.
6. **Physics signatures are real, but they do not add forecast skill beyond the flux so far**
   (`outputs/physics/PHYSICS.md`):
   - hot onsets appear in C, M and X flares alike;
   - the Neupert effect shows in HEL1OS (r = 0.77, lag 0 s);
   - neither improves peak forecasts once the flux so far is known.

## 1. Side by side

### 1.1 "Is a flare in progress?" (test period)

| | old labels (SoLEXS detector) | GOES ≥ C1.0 | GOES ≥ M1.0 | GOES ≥ X1.0 |
|---|---:|---:|---:|---:|
| base rate | 56 % | 8.3 % | 1.3 % | 0.1 % |
| TSS, deep model [95 % CI] | 0.706 [0.690, 0.721] | **0.768 [0.746, 0.782]** | 0.785 [0.759, 0.809] | 0.750 [0.624, 0.827] |
| caught / false alarms | 78 % / 7 % | 82 % / 40 % | — | — |
| AUC, model | 0.931 | 0.944 | 0.951 | 0.959 |
| AUC, flux already reached | 0.832 | 0.836 | 0.948 | **0.978** |
| TSS, trees / logistic | 0.700 / 0.643 | 0.745 / 0.711 | — | — |
| Brier skill, model / trees | 0.53 / 0.57 | 0.27 / **0.52** | — | — |

The ≥ M and ≥ X rows re-score the same ≥ C model against stricter truth, with thresholds re-chosen on
validation. For X flares the flux level alone ranks moments better than the model does. The trees give
better-calibrated probabilities.

### 1.2 "Will a flare be in progress within H minutes?"

| truth | H | base rate | TSS model | TSS trees / logistic | AUC model | AUC flux reached |
|---|---:|---:|---:|---:|---:|---:|
| old labels | 15 / 30 / 60 | 68 / 77 / 87 % | 0.61 / 0.57 / 0.56 | 0.60 / 0.56 / 0.46 (trees) | 0.88 / 0.86 / 0.85 | 0.81 / 0.81 / 0.82 |
| GOES ≥ C | 15 / 30 / 60 | 14 / 19 / 28 % | **0.47 / 0.35 / 0.26** | 0.43 / 0.32 / 0.19 · 0.44 / 0.35 / 0.22 | 0.81 / 0.75 / 0.71 | 0.77 / 0.74 / 0.73 |
| GOES ≥ M | 15 / 30 / 60 | 2.0 / 2.7 / 4.0 % | 0.54 / 0.42 / 0.30 | — | 0.85 / 0.79 / 0.74 | **0.90 / 0.88 / 0.85** |
| GOES ≥ X | 15 / 30 / 60 | 0.1–0.2 % | 0.51 / 0.37 / 0.27 | — | 0.86 / 0.79 / 0.75 | **0.95 / 0.94 / 0.91** |

A hindsight reference ("a flare of that size is already in progress") reaches TSS 0.59 / 0.43 / 0.30
for ≥ C. The model gets most of the way there, so most of what it "forecasts" is flares already
under way. Its Brier skill for occurrence is −0.02 to −0.06, so its probabilities should not be quoted
as calibrated chances.

### 1.3 Flux forecast, minutes ahead

GOES log10 W/m², test windows where the GOES value at the forecast origin is real (§4).

| ahead | model RMSE | "no change" (GOES) RMSE | climatology | 80 % interval coverage |
|---|---:|---:|---:|---:|
| 1 min | 0.134 | **0.021** | 0.573 | 0.93 |
| 5 min | 0.149 | **0.073** | 0.573 | 0.92 |
| 15 min | 0.172 | **0.133** | 0.573 | 0.90 |
| 30 min | 0.202 | **0.174** | 0.573 | 0.86 |
| 60 min | 0.220 | 0.211 | 0.573 | 0.85 |

**The fair reference for an Aditya-only system.** GOES persistence reads the very instrument being
predicted. The operational alternative is "no change" in what SoLEXS measures, mapped to GOES units by
a calibration fitted on the training period only: log10 F = −6.714 + 0.637 log10 rate, 0.062 dex. It is
scored on the 106 693 test windows where SoLEXS observed the origin (`scripts/fair_references.py`).

| ahead | model RMSE | "no change", SoLEXS calibrated | "no change", GOES |
|---|---:|---:|---:|
| 0 (nowcast) | 0.092 | **0.069** | — |
| 1 min | 0.101 | **0.069** | 0.022 |
| 5 min | 0.118 | **0.095** | 0.074 |
| 15 min | 0.147 | 0.150 | 0.134 |
| 30 min | **0.180** | 0.191 | 0.176 |
| 60 min | **0.201** | 0.227 | 0.213 |

- **Beats the fair reference:** only at 30–60 min, by 6–11 %.
- **Loses at 0–5 min:** the network infers the *current* GOES flux less accurately than a one-line
  calibration of SoLEXS (0.092 vs 0.069 dex).
- **Obvious fix:** anchor the output to the calibrated SoLEXS flux and let the network predict only
  the change.

### 1.4 Peak of a flare that is already rising (rolling-origin CV, 7 172 flares, 3 folds)

| encoder | peak MAE, dex (± fold sd) | skill vs level reached | skill vs climatology | time-to-peak MAE | skill | ≥ M AUC (flux reached 0.851) |
|---|---:|---:|---:|---:|---:|---:|
| transformer | 0.203 ± 0.016 | 0.20 | 0.43 | 4.7 min | −0.03 | 0.832 |
| linear | 0.204 ± 0.015 | 0.20 | 0.43 | 4.4 min | +0.03 | 0.839 |
| GRU | 0.211 ± 0.015 | 0.17 | 0.40 | 4.5 min | +0.00 | 0.840 |
| TCN | 0.213 ± 0.012 | 0.16 | 0.40 | 4.6 min | −0.01 | 0.837 |
| *old labels, best encoder* | *0.37 dex of SoLEXS rate* | *0.36* | *0.45* | *hours (merged events)* | *0.13* | *0.89 (flux 0.86)* |

- All four encoders sit within one fold standard deviation of each other, and a linear encoder
  matches the transformer.
- The model beats the level already reached by 16–20 %.
- It does not beat that level at ranking which rises end above M1, and it does not beat climatology
  at timing the peak.
- The median lead time is 5.7 min, so this is a short-range estimate.

### 1.5 Does HEL1OS add skill?

| test | old labels | GOES labels |
|---|---|---|
| paired peak ablation (TCN, same folds and seeds) | +9.2 %, CI [−0.030, +0.243] — **not detectable**; 43 % of flares improved | **+10.3 %, CI [+0.012, +0.042 dex] — detectable**; 51 % of flares improved; AUC 0.733 vs 0.718 |
| the same ablation reseeded, 4 TCN seeds (2026-09-16) | — | gains +0.027 / +0.036 / +0.048 / +0.059 dex — **mean 15.4 % ± 4.4 %**, every interval excluding zero |
| the same ablation with a linear encoder | — | +3.9 %, CI [−0.004, +0.024] — **not detectable**: the gain needs a temporal model |
| nowcast TSS on HEL1OS-observed windows: both / SoLEXS only / HEL1OS only | 0.696 / **0.715** / 0.459 | **0.769** / 0.708 / 0.706 |

The old labels were SoLEXS's own detections, so SoLEXS defined the truth and HEL1OS had little room to
add anything. Against an independent truth, hard X-rays carry information SoLEXS does not. The reseeding asked for in the first draft of this
section has now been done (`reports/fusion_seeds/`, `scripts/fusion_seeds_summary.py`): the gain
survives it, with a seed-to-seed spread of ±4.4 % that must be quoted with it. The HEL1OS overlap is
still 76 days in two periods, April–June 2024 and June–September 2026, and the linear-encoder result
says the benefit lives in the time structure of the hard X-rays.

### 1.6 Physics features (CPU study, `outputs/physics/PHYSICS.md`)

- **Hot onsets.** Across 5 925 GOES flares, the hardness at onset is already 1.20× (C), 1.00× (M) and
  0.87× (X) that at peak. GOES independently gives 1.05×, 0.87× and 0.70×.
- **Neupert effect.** For the 45 flares with significant hard X-rays: r = 0.77, lag 0 s.
- **Forecast value.** Adding onset hardness, the GOES ratio or early hard X-ray counts to the flux
  seen so far changes peak error by less than 0.003 dex, and the ≥ M AUC stays at 0.78.

The network's HEL1OS gain in §1.5 therefore comes from the hard X-ray time series over the rise, not
from a single early number.

## 2. Comparison with the literature, now on GOES classes

Events are now GOES classes, but horizons and data still differ from most studies. Only
Telikicherla et al. and Yi et al. are close enough to compare numbers directly.

| study | task | their number | ours | reading |
|---|---|---|---|---|
| Nishizuka et al. 2018 / 2021 (DeFN) | ≥ C / ≥ M in the next 24 h, magnetograms | TSS 0.63 / 0.80 (research), 0.70 ≥ C (operational) | ≥ C within 60 min: TSS 0.26 | not the same question: 24 h from magnetic fields vs 60 min from X-rays. X-ray light curves carry little warning before onset |
| Leka et al. 2019; Barnes et al. 2016 | operational comparisons | no single winner; skill falls with horizon | architectures tied; occurrence skill falls 0.47 → 0.26 from 15 to 60 min | **same pattern** |
| Sinha et al. 2022; Cinto et al. 2020 | classical ML competitive | logistic / XGBoost strong | trees TSS 0.745 vs network 0.768; linear encoder = transformer | **same pattern** |
| Telikicherla et al. 2025 (HOPE) | DAXSS + GOES, alert before the peak | alerts 5–15 min before peak | hot onsets confirmed on 5 925 flares; median onset → peak 5.7 / 9.3 / 12.7 min (C / M / X); nowcast TSS 0.77 | **consistent**. The hot onset is universal, but adds no size information beyond the flux so far |
| Yi et al. 2026 | GOES-XRS, peak of an ongoing flare | RMSE 0.26 / 0.45 / 0.87 dex (≥ C / ≥ M / X), no persistence baseline | MAE 0.20 dex, 16–20 % better than the level reached | same order for mostly C-class rises; not like for like (MAE vs RMSE, lead times, folds) |
| Ahmadzadeh et al. 2021 | random splits inflate skill | — | chronological split, embargo, flare-grouped folds; one scoring bug still found and fixed (§4) | evaluation design dominates the result |

## 3. What can be said publicly

- **Supported:**
  - "Detects GOES C-, M- and X-class flares in progress from Aditya-L1 X-ray data, TSS 0.77–0.79 on
    unseen 2026 data."
  - "Estimates how large a rising flare will get, about 20 % better than the level already reached."
- **Supported with a caveat:** "HEL1OS hard X-rays measurably improve peak estimates (10 %)". Say it is
  preliminary until confirmed with more seeds.
- **Not supported:**
  - "Predicts solar flares before they start";
  - any flux forecast "better than persistence" (true only at 30–60 min against the Aditya-only
    reference, by 6–11 %);
  - probabilities presented as calibrated chances;
  - anything implying real-time operation, because PRADAN data arrive days to weeks late.

## 4. Problems found and fixed in this run

1. **Persistence scored on missing origins.**
   - **What went wrong:** forecasts were scored wherever the future target was real, including windows
     where the target at the origin was a stored 0 (GOES gaps, stretches with no soft X-ray truth).
     There "no change" predicted log flux 0, six decades off, inflating the model's skill over it.
   - **Fix:** `metrics.forecast_score_mask` now requires both values to be real, with a test in
     `tests/test_goes.py`. The re-scored numbers are in §1.3.
   - **Effect on the old run:** it scored flux in log(1 + counts), where a missing origin is 0 against
     typical values of ~3, so it was affected less. Re-score it before citing its forecast skill.
2. **Pipeline interruptions.**
   - **What went wrong:** the run was killed twice, once when a chat session ended and once by a
     console Ctrl+C.
   - **Fix:** it now runs detached in its own process group.
   - **Losses:** only the forecast-cv work in progress. The nowcast model and its test evaluation
     were kept. The evaluation's `training` block was restored from `history.json` and
     `nowcast-train.log`.

## 5. Next, in order

1. ~~Confirm the HEL1OS gain with 3 more seeds and a second encoder.~~ **Done 2026-09-16: it holds
   (15.4 % ± 4.4 % over four TCN seeds) and is architecture-dependent.**
2. **Learning curves** at 25–100 % of the data, SoLEXS-only and on the overlap: does more data help?
3. **Anchor the flux heads to calibrated SoLEXS flux**, with the network predicting only the change.
   **Recalibrate probabilities** (isotonic regression on validation), or report the trees' calibrated
   probabilities.
4. **PatchTST** under identical folds, seeds, labels and references (roadmap Phase 3).
5. **Cross-attention fusion only if step 1 holds** (Phase 4). Compare SoLEXS only / HEL1OS only / both /
   both + physics.
6. **Forward test v2** as PRADAN releases days after 2026-09-13.
