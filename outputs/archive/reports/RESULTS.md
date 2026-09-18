# Results

Generated 2026-09-15 00:53 UTC by `python -m solarflare.cli report`.
All numbers are on the held-out **test** split. Operating thresholds were
fitted on validation and held fixed.

Run config: grid 20.0 s, window 7200.0 s, background 21600.0 s, clock features OFF.

## Data

72 stitched segments, 20240201 -> 20260913: SoLEXS observed 760.5 d, HEL1OS 111.9 d, 8944 flares detected. Split: global (archive profile {'stride_seconds': [40.0, 120.0], 'max_batches_per_epoch': [0, 400]}). Per-segment detail is in `data_meta.json`.

Windows: 563472 (train 319756 / val 112664 / test 112665), grid 20.0 s.

Class balance at the prediction origin:

| split | n | in-flare rate | quiet/rise/peak/decay |
|---|---:|---:|---|
| train | 315659 | 0.675 | 102658/76749/10553/125699 |
| val | 112664 | 0.622 | 42562/24502/3819/41781 |
| test | 108037 | 0.563 | 47259/21067/3844/35867 |

## 1. Nowcast - is a flare in progress

Threshold 0.54 (fitted on validation), test base rate 0.563.

| metric | value |
|---|---:|
| TSS | 0.706 |
| HSS | 0.689 |
| AUC | 0.931 |
| POD | 0.776 |
| FAR | 0.066 |
| CSI | 0.736 |
| F1 | 0.848 |
| precision | 0.934 |
| accuracy | 0.843 |
| Brier | 0.117 |
| BSS_vs_climatology | 0.526 |
| contingency TP/FP/FN/TN | 47162/3326/13616/43933 |

## 2. Nowcast - flare phase

Accuracy 0.661, macro-F1 0.545.

| phase | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| quiet | 0.772 | 0.917 | 0.839 | 47259 |
| rise | 0.414 | 0.284 | 0.337 | 21067 |
| peak | 0.241 | 0.852 | 0.376 | 3844 |
| decay | 0.788 | 0.526 | 0.631 | 35867 |

## 3. Forecast - flare within horizon

| horizon | base rate | TSS | HSS | POD | FAR | AUC | BSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15min | 0.678 | 0.606 | 0.516 | 0.661 | 0.037 | 0.882 | 0.334 |
| 30min | 0.765 | 0.569 | 0.406 | 0.628 | 0.028 | 0.861 | 0.228 |
| 60min | 0.870 | 0.561 | 0.273 | 0.622 | 0.014 | 0.853 | 0.049 |

## 4. Nowcast regression (log flux)

MAE 0.0845 | RMSE 0.1146 | R2 0.990

## 5. Forecast regression vs persistence AND climatology

| horizon | model RMSE | persistence | climatology | skill vs pers | skill vs clim | corr | coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1min | 0.1614 | 0.1576 | 1.7611 | -0.024 | 0.908 | 0.991 | 0.762 |
| 5min | 0.3232 | 0.3410 | 1.7610 | 0.052 | 0.816 | 0.961 | 0.742 |
| 15min | 0.5214 | 0.5764 | 1.7609 | 0.095 | 0.704 | 0.895 | 0.713 |
| 30min | 0.6389 | 0.7360 | 1.7610 | 0.132 | 0.637 | 0.838 | 0.701 |
| 60min | 0.7384 | 0.8741 | 1.7608 | 0.155 | 0.581 | 0.780 | 0.682 |

Nominal interval coverage 0.80.

> **Read both reference columns.** Persistence is a weak baseline at long
> horizons because it extrapolates a decaying flare, so beating it there is
> easy and means little. Climatology is the honest reference: a horizon
> where the model beats persistence but not climatology, with correlation
> near zero, is a horizon where it has simply learned to predict the mean.

On this dataset the forecast carries real information at **1min, 5min, 15min, 30min, 60min** and degenerates to climatology beyond that.

## 6. Ongoing-event peak

Time-to-peak MAE 55.9 min | log peak-flux MAE 0.627 | n = 60749

## 7. Modality ablation

Flare-in-progress skill with each instrument masked off at inference, same thresholds. `clock_only` blanks both instruments: any skill left there would be memorisation of time, not physics.

| input | TSS (all test windows) | AUC (all) | TSS (HEL1OS observing) | AUC (HEL1OS observing) |
|---|---:|---:|---:|---:|
| both | 0.706 | 0.931 | 0.696 | 0.931 |
| soft_only | 0.714 | 0.934 | 0.715 | 0.935 |
| hard_only | 0.232 | 0.690 | 0.459 | 0.798 |
| clock_only | 0.000 | 0.500 | 0.000 | 0.500 |

The right-hand columns use only the 51688 test windows whose prediction origin HEL1OS observed. Over the whole test split a hard X-ray effect is diluted by windows where blanking HEL1OS changes nothing because it was not observing.

## 8. Baselines (identical splits and windows)

### Flare in progress

| model | TSS | HSS | AUC | BSS |
|---|---:|---:|---:|---:|
| **SoLEXHEL-Net** | **0.706** | 0.689 | 0.931 | 0.526 |
| gbdt | 0.700 | 0.683 | 0.928 | 0.570 |
| logistic | 0.643 | 0.627 | 0.885 | 0.394 |
| climatology | 0.000 | 0.000 | 0.500 | 0.000 |

### Flare occurrence within horizon (TSS)

| model | 15min | 30min | 60min |
|---|---|---|---|
| **SoLEXHEL-Net** | 0.606 | 0.569 | 0.561 |
| gbdt | 0.605 | 0.557 | 0.459 |
| logistic | 0.534 | 0.252 | 0.202 |

### Forecast RMSE (log flux; lower is better)

| model | 1min | 5min | 15min | 30min | 60min |
|---|---|---|---|---|---|
| **SoLEXHEL-Net** | 0.1614 | 0.3232 | 0.5214 | 0.6389 | 0.7384 |
| gbdt | 0.1376 | 0.3191 | 0.5441 | 0.7116 | 0.7612 |
| persistence | 0.1576 | 0.3410 | 0.5764 | 0.7360 | 0.8741 |

## 9. Rise-phase forecasting - architecture comparison

Given the early rise of a flare, predict its peak magnitude, time to peak, whether it exceeds 83.04 cts/s, and whether it is long-duration. 8910 events, 272997 rise samples, rolling-origin CV **grouped by event** (no flare ever appears on both sides of a split).

| encoder | folds | peak log-MAE (mean +/- sd) | skill vs current | skill vs climatology | time-to-peak MAE (min) | ttp skill vs clim | exceeds AUC | runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| transformer | 3 | 0.8428 +/- 0.0357 | 0.364 | 0.446 | 140.7 | 0.128 | 0.888 | 4295 |
| tcn | 3 | 0.8480 +/- 0.0428 | 0.360 | 0.442 | 159.8 | 0.050 | 0.881 | 5039 |
| linear | 3 | 0.8535 +/- 0.0298 | 0.356 | 0.439 | 157.6 | 0.059 | 0.885 | 998 |
| gru | 3 | 0.8606 +/- 0.0150 | 0.351 | 0.434 | 162.2 | 0.027 | 0.868 | 1467 |

Reference forecasts (same folds): **current level** peak log-MAE 1.3256 - assume the flare has already peaked; **training climatology** 1.5264 - predict the average training-event peak. Skill = 1 - MAE_model / MAE_reference; at or below zero the model adds nothing over that reference, however small its raw MAE looks.

Encoders beating **both** references on peak magnitude: transformer, tcn, linear, gru.

Encoders beating climatology on **time to peak**: transformer, tcn, linear, gru.

**Large-flare exceedance.** The AUC column rests on 3 fold(s) containing both classes and **1267 large flare(s) in total**. Ranking samples by the flux already reached scores AUC **0.858** on the same fold(s):

| encoder | exceeds AUC | AUC gain vs current level |
|---|---:|---:|
| transformer | 0.888 | 0.030 |
| tcn | 0.881 | 0.023 |
| linear | 0.885 | 0.027 |
| gru | 0.868 | 0.010 |

Per fold (peak log-MAE) - the scatter between folds is the headline:

| fold | n test | transformer | tcn | linear | gru |
|---|---:|---:|---:|---:|---:|
| 0 | 70444 | 0.8091 | 0.8407 | 0.8319 | 0.8660 |
| 1 | 68049 | 0.8923 | 0.9037 | 0.8956 | 0.8756 |
| 2 | 58618 | 0.8271 | 0.7997 | 0.8331 | 0.8401 |

> The gap between the top two encoders (0.0052) is smaller than the fold-to-fold scatter, so the ranking is **not** statistically meaningful on this data. With this few events, treat the ordering as provisional until more flares are added.

## 10. Does HEL1OS add skill? (paired fusion ablation)

Only flares whose rise HEL1OS observed (>= 50% of rise bins): **871** flares, 651 scored out of sample. The `tcn` encoder is cross-validated twice over **identical folds and seeds** - with the hard X-ray input and with it blanked - and the per-sample difference in peak error is bootstrapped over flares.

| arm | peak log-MAE | exceeds AUC |
|---|---:|---:|
| soft + hard | 0.8081 | 0.789 |
| soft only | 0.8895 | 0.842 |

Error reduction from adding HEL1OS: **0.0815** (95% CI -0.0298 to 0.2434; 9.2% of the soft-only error); lower error on 43% of flares. **no detectable effect** of HEL1OS at this sample size.

## 11. Skill on significant flares (post-hoc)

Same trained model and outputs; only the truth changes to flares at least 3x / 10x / 30x above background. Thresholds re-chosen on validation, applied to test; TSS intervals resample whole days. *Oracle persistence* uses the in-progress label at the origin (hindsight, not available in real time) - an upper-bound reference. *AUC current flux* ranks by the flux already reached, which is available in real time.

| truth | target | base rate | TSS [95% CI] | oracle persistence TSS | AUC | AUC current flux |
|---|---|---:|---:|---:|---:|---:|
| all detected (training labels) | in progress now | 0.562 | 0.706 [0.690, 0.721] | 1.000 | 0.931 | 0.832 |
| all detected (training labels) | within 15min | 0.678 | 0.606 [0.589, 0.623] | 0.829 | 0.882 | 0.809 |
| all detected (training labels) | within 30min | 0.765 | 0.568 [0.547, 0.591] | 0.735 | 0.861 | 0.805 |
| all detected (training labels) | within 60min | 0.870 | 0.561 [0.534, 0.589] | 0.646 | 0.853 | 0.816 |
| >= 3x background | in progress now | 0.443 | 0.624 [0.603, 0.645] | 1.000 | 0.895 | 0.821 |
| >= 3x background | within 15min | 0.505 | 0.537 [0.516, 0.554] | 0.877 | 0.846 | 0.788 |
| >= 3x background | within 30min | 0.560 | 0.488 [0.466, 0.511] | 0.791 | 0.814 | 0.765 |
| >= 3x background | within 60min | 0.650 | 0.430 [0.405, 0.456] | 0.682 | 0.779 | 0.740 |
| >= 10x background | in progress now | 0.261 | 0.517 [0.489, 0.546] | 1.000 | 0.835 | 0.818 |
| >= 10x background | within 15min | 0.289 | 0.437 [0.411, 0.463] | 0.905 | 0.794 | 0.787 |
| >= 10x background | within 30min | 0.315 | 0.394 [0.369, 0.419] | 0.830 | 0.767 | 0.764 |
| >= 10x background | within 60min | 0.362 | 0.348 [0.318, 0.376] | 0.721 | 0.729 | 0.733 |
| >= 30x background | in progress now | 0.153 | 0.461 [0.425, 0.498] | 1.000 | 0.802 | 0.824 |
| >= 30x background | within 15min | 0.166 | 0.407 [0.370, 0.438] | 0.926 | 0.771 | 0.799 |
| >= 30x background | within 30min | 0.178 | 0.363 [0.329, 0.389] | 0.863 | 0.749 | 0.778 |
| >= 30x background | within 60min | 0.201 | 0.335 [0.298, 0.367] | 0.763 | 0.719 | 0.749 |

## Training

Nowcast model: best epoch 26 (selection score 0.616), 11566.6 s wall-clock.

## Figures

* `figures/mission_overview.png` - mission light curve, flares per month, coverage
* `figures/spectrogram_20240514.png` - time-energy spectrogram of the day of the largest flare
* `figures/training_history.png` - loss and validation skill
* `figures/test_timeline.png` - predictions over the test period
* `figures/reliability.png` - probability calibration
