# Results

Generated 2026-09-15 17:36 UTC by `python -m solarflare.cli report`.
All numbers are on the held-out **test** split. Operating thresholds were
fitted on validation and held fixed.

Run config: grid 20.0 s, window 7200.0 s, background 21600.0 s, clock features OFF.

## Data

72 stitched segments, 20240201 -> 20260913: SoLEXS observed 760.5 d, HEL1OS 111.9 d, 7187 flares detected. Split: global (archive profile {'stride_seconds': [40.0, 120.0], 'max_batches_per_epoch': [0, 400]}). Per-segment detail is in `data_meta.json`.

Windows: 563472 (train 250088 / val 112664 / test 112665), grid 20.0 s.

Class balance at the prediction origin:

| split | n | in-flare rate | quiet/rise/peak/decay |
|---|---:|---:|---|
| train | 245453 | 0.159 | 206354/16943/8686/13470 |
| val | 110450 | 0.099 | 99526/4749/2380/3795 |
| test | 111283 | 0.083 | 102047/4165/2029/3042 |

## 1. Nowcast - is a flare in progress

Threshold 0.33 (fitted on validation), test base rate 0.083.

| metric | value |
|---|---:|
| TSS | 0.768 |
| HSS | 0.662 |
| AUC | 0.944 |
| POD | 0.816 |
| FAR | 0.395 |
| CSI | 0.532 |
| F1 | 0.695 |
| precision | 0.605 |
| accuracy | 0.940 |
| Brier | 0.056 |
| BSS_vs_climatology | 0.266 |
| contingency TP/FP/FN/TN | 7535/4923/1701/97124 |

## 2. Nowcast - flare phase

Accuracy 0.905, macro-F1 0.647.

| phase | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| quiet | 0.985 | 0.926 | 0.954 | 102047 |
| rise | 0.286 | 0.547 | 0.376 | 4165 |
| peak | 0.534 | 0.650 | 0.586 | 2029 |
| decay | 0.544 | 0.880 | 0.672 | 3042 |

## 3. Forecast - flare within horizon

| horizon | base rate | TSS | HSS | POD | FAR | AUC | BSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15min | 0.141 | 0.466 | 0.467 | 0.540 | 0.456 | 0.808 | -0.045 |
| 30min | 0.193 | 0.348 | 0.368 | 0.447 | 0.481 | 0.752 | -0.055 |
| 60min | 0.282 | 0.255 | 0.261 | 0.444 | 0.521 | 0.709 | -0.023 |

## 4. Nowcast regression (log flux)

MAE 0.0831 | RMSE 0.1286 | R2 0.851

## 5. Forecast regression vs persistence AND climatology

| horizon | model RMSE | persistence | climatology | skill vs pers | skill vs clim | corr | coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1min | 0.1338 | 0.0214 | 0.5734 | -5.251 | 0.767 | 0.946 | 0.933 |
| 5min | 0.1489 | 0.0727 | 0.5733 | -1.048 | 0.740 | 0.928 | 0.921 |
| 15min | 0.1722 | 0.1326 | 0.5731 | -0.299 | 0.699 | 0.880 | 0.900 |
| 30min | 0.2024 | 0.1742 | 0.5729 | -0.162 | 0.647 | 0.834 | 0.865 |
| 60min | 0.2201 | 0.2112 | 0.5726 | -0.042 | 0.616 | 0.784 | 0.849 |

Nominal interval coverage 0.80.

> **Read both reference columns.** Persistence is a weak baseline at long
> horizons because it extrapolates a decaying flare, so beating it there is
> easy and means little. Climatology is the honest reference: a horizon
> where the model beats persistence but not climatology, with correlation
> near zero, is a horizon where it has simply learned to predict the mean.

On this dataset the forecast carries real information at **1min, 5min, 15min, 30min, 60min** and degenerates to climatology beyond that.

## 6. Ongoing-event peak

Time-to-peak MAE 4.2 min | log peak-flux MAE 0.136 | n = 9236

## 7. Modality ablation

Flare-in-progress skill with each instrument masked off at inference, same thresholds. `clock_only` blanks both instruments: any skill left there would be memorisation of time, not physics.

| input | TSS (all test windows) | AUC (all) | TSS (HEL1OS observing) | AUC (HEL1OS observing) |
|---|---:|---:|---:|---:|
| both | 0.768 | 0.944 | 0.769 | 0.946 |
| soft_only | 0.736 | 0.941 | 0.708 | 0.938 |
| hard_only | 0.360 | 0.774 | 0.706 | 0.908 |
| clock_only | 0.000 | 0.500 | 0.000 | 0.500 |

The right-hand columns use only the 55732 test windows whose prediction origin HEL1OS observed (of 111283). Over the whole test split a hard X-ray effect is diluted by windows where blanking HEL1OS changes nothing because it was not observing.

## 8. Baselines (identical splits and windows)

### Flare in progress

| model | TSS | HSS | AUC | BSS |
|---|---:|---:|---:|---:|
| **SoLEXHEL-Net** | **0.768** | 0.662 | 0.944 | 0.266 |
| gbdt | 0.745 | 0.538 | 0.934 | 0.521 |
| logistic | 0.711 | 0.487 | 0.917 | -0.053 |
| climatology | 0.000 | 0.000 | 0.500 | 0.000 |

### Flare occurrence within horizon (TSS)

| model | 15min | 30min | 60min |
|---|---|---|---|
| **SoLEXHEL-Net** | 0.466 | 0.348 | 0.255 |
| gbdt | 0.425 | 0.320 | 0.192 |
| logistic | 0.443 | 0.345 | 0.219 |

### Forecast RMSE (log flux; lower is better)

| model | 1min | 5min | 15min | 30min | 60min |
|---|---|---|---|---|---|
| **SoLEXHEL-Net** | 0.1338 | 0.1489 | 0.1722 | 0.2024 | 0.2201 |
| gbdt | 0.1337 | 0.1506 | 0.1903 | 0.2253 | 0.2514 |
| persistence | 0.0214 | 0.0727 | 0.1326 | 0.1742 | 0.2112 |

## 9. Rise-phase forecasting - architecture comparison

Given the early rise of a flare, predict its peak magnitude, time to peak, whether it exceeds 1.0e-05 W/m^2 (GOES M1.0), and whether it is long-duration. 7172 events, 70633 rise samples, rolling-origin CV **grouped by event** (no flare ever appears on both sides of a split).

| encoder | folds | peak log-MAE (mean +/- sd) | skill vs current | skill vs climatology | time-to-peak MAE (min) | ttp skill vs clim | exceeds AUC | runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| transformer | 3 | 0.2025 +/- 0.0163 | 0.202 | 0.433 | 4.7 | -0.032 | 0.832 | 7254 |
| linear | 3 | 0.2039 +/- 0.0146 | 0.196 | 0.428 | 4.4 | 0.029 | 0.839 | 1147 |
| gru | 3 | 0.2111 +/- 0.0149 | 0.167 | 0.404 | 4.5 | 0.002 | 0.840 | 1493 |
| tcn | 3 | 0.2129 +/- 0.0121 | 0.160 | 0.401 | 4.6 | -0.005 | 0.837 | 5457 |

Reference forecasts (same folds): **current level** peak log-MAE 0.2534 - assume the flare has already peaked; **training climatology** 0.3591 - predict the average training-event peak. Skill = 1 - MAE_model / MAE_reference; at or below zero the model adds nothing over that reference, however small its raw MAE looks.

Encoders beating **both** references on peak magnitude: transformer, linear, gru, tcn.

Encoders beating climatology on **time to peak**: linear, gru.

**Large-flare exceedance.** The AUC column rests on 3 fold(s) containing both classes and **875 large flare(s) in total**. Ranking samples by the flux already reached scores AUC **0.851** on the same fold(s):

| encoder | exceeds AUC | AUC gain vs current level |
|---|---:|---:|
| transformer | 0.832 | -0.019 |
| linear | 0.839 | -0.013 |
| gru | 0.840 | -0.011 |
| tcn | 0.837 | -0.014 |

Per fold (peak log-MAE) - the scatter between folds is the headline:

| fold | n test | transformer | linear | gru | tcn |
|---|---:|---:|---:|---:|---:|
| 0 | 17062 | 0.1861 | 0.1907 | 0.2130 | 0.2058 |
| 1 | 17671 | 0.1967 | 0.1967 | 0.1920 | 0.2030 |
| 2 | 18188 | 0.2246 | 0.2243 | 0.2283 | 0.2299 |

> The gap between the top two encoders (0.0014) is smaller than the fold-to-fold scatter, so the ranking is **not** statistically meaningful on this data. With this few events, treat the ordering as provisional until more flares are added.

## 10. Does HEL1OS add skill? (paired fusion ablation)

Only flares whose rise HEL1OS observed (>= 50% of rise bins): **826** flares, 618 scored out of sample. The `tcn` encoder is cross-validated twice over **identical folds and seeds** - with the hard X-ray input and with it blanked - and the per-sample difference in peak error is bootstrapped over flares.

| arm | peak log-MAE | exceeds AUC |
|---|---:|---:|
| soft + hard | 0.2351 | 0.733 |
| soft only | 0.2620 | 0.718 |

Error reduction from adding HEL1OS: **0.0269** (95% CI 0.0124 to 0.0423; 10.3% of the soft-only error); lower error on 51% of flares. HEL1OS **measurably reduces** the peak error.

## 11. Skill on significant flares (post-hoc)

Same trained model and outputs; only the truth changes to flares at least 3x / 10x / 30x above background. Thresholds re-chosen on validation, applied to test; TSS intervals resample whole days. *Oracle persistence* uses the in-progress label at the origin (hindsight, not available in real time) - an upper-bound reference. *AUC current flux* ranks by the flux already reached, which is available in real time.

| truth | target | base rate | TSS [95% CI] | oracle persistence TSS | AUC | AUC current flux |
|---|---|---:|---:|---:|---:|---:|
| all detected (training labels) | in progress now | 0.083 | 0.768 [0.746, 0.782] | 1.000 | 0.944 | 0.836 |
| all detected (training labels) | within 15min | 0.141 | 0.466 [0.447, 0.482] | 0.588 | 0.808 | 0.767 |
| all detected (training labels) | within 30min | 0.193 | 0.348 [0.330, 0.366] | 0.430 | 0.752 | 0.742 |
| all detected (training labels) | within 60min | 0.282 | 0.255 [0.227, 0.280] | 0.295 | 0.709 | 0.729 |
| >= GOES M1.0 | in progress now | 0.013 | 0.785 [0.759, 0.809] | 1.000 | 0.951 | 0.948 |
| >= GOES M1.0 | within 15min | 0.020 | 0.540 [0.501, 0.581] | 0.631 | 0.845 | 0.899 |
| >= GOES M1.0 | within 30min | 0.027 | 0.417 [0.377, 0.459] | 0.463 | 0.789 | 0.875 |
| >= GOES M1.0 | within 60min | 0.040 | 0.304 [0.266, 0.340] | 0.313 | 0.740 | 0.847 |
| >= GOES X1.0 | in progress now | 0.001 | 0.750 [0.624, 0.827] | 1.000 | 0.959 | 0.978 |
| >= GOES X1.0 | within 15min | 0.001 | 0.506 [0.335, 0.649] | 0.654 | 0.856 | 0.953 |
| >= GOES X1.0 | within 30min | 0.001 | 0.368 [0.196, 0.509] | 0.476 | 0.786 | 0.935 |
| >= GOES X1.0 | within 60min | 0.002 | 0.274 [0.196, 0.337] | 0.312 | 0.749 | 0.907 |

## Training

Nowcast model: best epoch 35 (selection score 0.576), 15669.2 s wall-clock.

## Figures

* `figures/mission_overview.png` - mission light curve, flares per month, coverage
* `figures/spectrogram_20241003.png` - time-energy spectrogram of the day of the largest flare
* `figures/training_history.png` - loss and validation skill
* `figures/test_timeline.png` - predictions over the test period
* `figures/reliability.png` - probability calibration
