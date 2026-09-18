# Results

Generated 2026-09-14 08:58 UTC by `python -m solarflare.cli report`.
All numbers are on the held-out **test** split. Operating thresholds were
fitted on validation and held fixed.

Run config: grid 20.0 s, window 7200.0 s, background 21600.0 s, clock features OFF.

## Data

| segment | steps | soft observed | hard observed | flares |
|---|---:|---:|---:|---:|
| `solexs_SDD2_20260910` | 4320 | 100% | 0% | 13 |
| `hel1os_CZT1_2026-09-11` | 2160 | 0% | 100% | 0 |

Windows: 2702 (train 1620 / val 360 / test 362), grid 20.0 s.

Class balance at the prediction origin:

| split | n | in-flare rate | quiet/rise/peak/decay |
|---|---:|---:|---|
| train | 1620 | 0.159 | 1363/75/32/150 |
| val | 360 | 0.531 | 169/37/12/142 |
| test | 362 | 0.249 | 272/9/19/62 |

## 1. Nowcast - is a flare in progress

Threshold 0.50 (fitted on validation), test base rate 0.311.

| metric | value |
|---|---:|
| TSS | 0.244 |
| HSS | 0.308 |
| AUC | 0.740 |
| POD | 0.244 |
| FAR | 0.000 |
| CSI | 0.244 |
| F1 | 0.393 |
| precision | 1.000 |
| accuracy | 0.765 |
| Brier | 0.172 |
| BSS_vs_climatology | 0.199 |
| contingency TP/FP/FN/TN | 22/0/68/199 |

## 2. Nowcast - flare phase

Accuracy 0.751, macro-F1 0.386.

| phase | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| quiet | 0.802 | 0.980 | 0.882 | 199 |
| rise | 0.000 | 0.000 | 0.000 | 9 |
| peak | 0.217 | 0.263 | 0.238 | 19 |
| decay | 0.944 | 0.274 | 0.425 | 62 |

## 3. Forecast - flare within horizon

| horizon | base rate | TSS | HSS | POD | FAR | AUC | BSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15min | 0.547 | 0.082 | 0.075 | 0.082 | 0.000 | 0.438 | -0.184 |
| 30min | 0.754 | 0.064 | 0.033 | 0.064 | 0.000 | 0.360 | -0.975 |
| 60min | 0.972 | 0.146 | 0.009 | 0.146 | 0.000 | 0.211 | -12.171 |

> **Note.** At 60min the base rate exceeds 0.9 -- with flares every couple of hours in a single day, almost every window has a flare within that horizon. BSS against climatology is meaningless there. This is a property of a one-day dataset, not of the model.

## 4. Nowcast regression (log flux)

MAE 0.1559 | RMSE 0.2287 | R2 0.536

## 5. Forecast regression vs persistence AND climatology

| horizon | model RMSE | persistence | climatology | skill vs pers | skill vs clim | corr | coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1min | 0.2525 | 0.1732 | 0.3378 | -0.457 | 0.253 | 0.822 | 0.879 |
| 5min | 0.3079 | 0.3065 | 0.3387 | -0.005 | 0.091 | 0.537 | 0.896 |
| 15min | 0.3301 | 0.4136 | 0.3418 | 0.202 | 0.034 | 0.218 | 0.837 |
| 30min | 0.2214 | 0.3822 | 0.2252 | 0.421 | 0.017 | -0.078 | 0.938 |
| 60min | 0.2628 | 0.4393 | 0.2684 | 0.402 | 0.021 | -0.027 | 0.858 |

Nominal interval coverage 0.80.

> **Read both reference columns.** Persistence is a weak baseline at long
> horizons because it extrapolates a decaying flare, so beating it there is
> easy and means little. Climatology is the honest reference: a horizon
> where the model beats persistence but not climatology, with correlation
> near zero, is a horizon where it has simply learned to predict the mean.

On this dataset the forecast carries real information at **1min, 5min** and degenerates to climatology beyond that.

## 6. Ongoing-event peak

Time-to-peak MAE 106.1 min | log peak-flux MAE 1.266 | n = 90

## 7. Modality ablation

Flare-in-progress skill with each instrument masked off at inference.

| input | TSS | AUC |
|---|---:|---:|
| both | 0.244 | 0.740 |
| soft_only | 0.244 | 0.740 |
| hard_only | 0.000 | 0.500 |
| clock_only | 0.000 | 0.500 |

> `both` and `soft_only` are identical because the supplied HEL1OS data never overlaps the SoLEXS data -- there is no hard X-ray signal available at the times being scored. This is the expected result for non-overlapping inputs, and it is the number to watch once overlapping days are added.

## 8. Baselines (identical splits and windows)

### Flare in progress

| model | TSS | HSS | AUC | BSS |
|---|---:|---:|---:|---:|
| **SoLEXHEL-Net** | **0.244** | 0.308 | 0.740 | 0.199 |
| gbdt | 0.345 | 0.376 | 0.636 | 0.020 |
| logistic | 0.262 | 0.326 | 0.594 | -0.174 |
| climatology | 0.000 | 0.000 | 0.500 | 0.000 |

### Flare occurrence within horizon (TSS)

| model | 15min | 30min | 60min |
|---|---|---|---|
| **SoLEXHEL-Net** | 0.082 | 0.064 | 0.146 |
| gbdt | 0.114 | 0.101 | -0.500 |
| logistic | 0.146 | -0.306 | 0.068 |

### Forecast RMSE (log flux; lower is better)

| model | 1min | 5min | 15min | 30min | 60min |
|---|---|---|---|---|---|
| **SoLEXHEL-Net** | 0.2525 | 0.3079 | 0.3301 | 0.2214 | 0.2628 |
| gbdt | 0.2303 | 0.3707 | 0.3584 | 0.2451 | 0.2535 |
| persistence | 0.1732 | 0.3065 | 0.4136 | 0.3822 | 0.4393 |

## 9. Rise-phase forecasting - architecture comparison

Given the early rise of a flare, predict its peak magnitude, time to peak, whether it exceeds 4.47 cts/s, and whether it is long-duration. 13 events, 344 rise samples, rolling-origin CV **grouped by event** (no flare ever appears on both sides of a split).

| encoder | folds | peak log-MAE (mean +/- sd) | skill vs current | skill vs climatology | time-to-peak MAE (min) | ttp skill vs clim | exceeds AUC | runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tcn | 3 | 0.4166 +/- 0.3781 | -0.017 | 0.516 | 7.7 | -0.461 | 0.796 | 46 |
| ssm | 3 | 0.4276 +/- 0.3035 | -0.061 | 0.492 | 5.6 | -0.070 | 0.972 | 824 |
| gru | 3 | 0.4582 +/- 0.3297 | -0.136 | 0.457 | 7.4 | -0.331 | 0.868 | 9 |
| linear | 3 | 0.4760 +/- 0.3364 | -0.193 | 0.427 | 8.7 | -0.793 | 0.822 | 8 |
| transformer | 3 | 0.6440 +/- 0.2793 | -0.726 | 0.154 | 7.7 | -0.411 | 0.981 | 88 |

Reference forecasts (same folds): **current level** peak log-MAE 0.3731 - assume the flare has already peaked; **training climatology** 0.7673 - predict the average training-event peak. Skill = 1 - MAE_model / MAE_reference; at or below zero the model adds nothing over that reference, however small its raw MAE looks.

Encoders beating **both** references on peak magnitude: **none**.

Encoders beating climatology on **time to peak**: **none** - no model predicts peak timing better than the average training rise.

**Large-flare exceedance.** The AUC column rests on 1 fold(s) containing both classes and **1 large flare(s) in total**. Ranking samples by the flux already reached scores AUC **0.998** on the same fold(s):

| encoder | exceeds AUC | AUC gain vs current level |
|---|---:|---:|
| tcn | 0.796 | -0.202 |
| ssm | 0.972 | -0.026 |
| gru | 0.868 | -0.130 |
| linear | 0.822 | -0.176 |
| transformer | 0.981 | -0.017 |

> With fewer than three large flares, this AUC describes individual events rather than a skill, and a negative gain means the model has learned nothing beyond how bright the flare already is. Do not quote it as a result.

Per fold (peak log-MAE) - the scatter between folds is the headline:

| fold | n test | tcn | ssm | gru | linear | transformer |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 145 | 0.0423 | 0.2082 | 0.2135 | 0.1399 | 0.9666 |
| 1 | 102 | 0.9345 | 0.8568 | 0.9244 | 0.9357 | 0.6800 |
| 2 | 22 | 0.2731 | 0.2177 | 0.2368 | 0.3525 | 0.2853 |

> The gap between the top two encoders (0.0109) is smaller than the fold-to-fold scatter, so the ranking is **not** statistically meaningful on this data. With this few events, treat the ordering as provisional until more flares are added.

## Training

Nowcast model: best epoch 1 (selection score 0.732), 120.3 s wall-clock.

## Figures

* `figures/lightcurve_solexs_SDD2_20260910.png` - SoLEXS light curve with detected flares
* `figures/spectrogram_SDD2_20260910.png` - SoLEXS time-energy spectrogram
* `figures/lightcurve_hel1os_CZT1_2026-09-11.png` - HEL1OS bands after fill-row masking
* `figures/training_history.png` - loss and validation skill
* `figures/test_timeline.png` - predictions over the test period
* `figures/reliability.png` - probability calibration
