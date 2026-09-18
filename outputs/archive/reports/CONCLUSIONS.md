# Conclusions — Aditya-L1 SoLEXS + HEL1OS flare nowcasting and forecasting

Archive run of 2026-09-14/15. SoLEXS 2024-02-01 → 2026-09-12 (824 days, 760.5 observed),
HEL1OS 343 products (111.9 observed days), 76.3 days simultaneous. 8 944 flares detected in
SoLEXS; HEL1OS watched at least half the rise of 871 of them.
Every number below is from `outputs/archive/reports/*.json`; literature numbers are quoted from the
papers listed at the end, which were checked at source on 2026-09-15.

## Bottom line

1. **Nowcasting whether a flare is in progress works**: TSS 0.71, AUC 0.93 on an unseen
   Mar–Sep 2026 test period. Gradient-boosted trees do as well (TSS 0.70) and are better calibrated.
2. **Much of the apparent forecasting skill comes from a loose flare definition.** The detector labels
   66 % of observed time as "in a flare". Scored against *significant* flares (≥10× or ≥30× background),
   the model's ability to rank which windows lead to a flare (AUC 0.72–0.84) is **no better than ranking
   by the X-ray flux already reached** (AUC 0.73–0.82). TSS falls to 0.33–0.52.
3. **Peak size of a flare that is already rising is genuinely forecastable**: 36 % lower error than
   assuming the flux already reached, 44 % lower than climatology. The choice of architecture does not
   matter: TCN, GRU, transformer and a linear encoder are tied within fold-to-fold scatter.
4. **HEL1OS hard X-rays do not yet add detectable skill.** Peak error drops 9 %, but the 95 % interval
   includes zero. Where HEL1OS was observing, removing it slightly *raised* nowcast TSS (0.696 → 0.715).
   Hard X-rays alone carry signal (AUC 0.80), but it is redundant with the soft X-rays at this data volume.
5. **The SoLEXS energy scale was wrong and is now corrected** to the published calibration, checked
   against the onboard Fe-55 lines (+20 / +26 eV).
6. **None of these scores can be put next to published GOES ≥C/≥M results yet**, because the events are
   not GOES classes and the forecast horizon is minutes, not a day. Section 3 compares approaches and
   findings, not numbers. Section 4 lists what would make the numbers comparable.

### Skill on significant flares (post-hoc re-scoring)

Same trained model and outputs; only the truth changes. Operating thresholds were re-chosen on
validation, and TSS intervals resample whole days. *AUC current flux* ranks windows by the flux
already reached, a reference available in real time.

| truth: flares ≥ | base rate (in progress) | TSS in progress [95 % CI] | TSS within 60 min [95 % CI] | AUC within 60 min | AUC current flux |
|---|---:|---:|---:|---:|---:|
| all detected (training labels) | 0.562 | 0.706 [0.690, 0.721] | 0.561 [0.534, 0.589] | 0.853 | 0.816 |
| 3× background | 0.443 | 0.624 [0.603, 0.645] | 0.430 [0.405, 0.456] | 0.779 | 0.740 |
| 10× background | 0.261 | 0.517 [0.489, 0.546] | 0.348 [0.318, 0.376] | 0.729 | 0.733 |
| 30× background | 0.153 | 0.461 [0.425, 0.498] | 0.335 [0.298, 0.367] | 0.719 | 0.749 |

A hindsight reference — "a flare of that size is in progress at the forecast moment", using the label
the model is trying to predict — reaches TSS 0.65–0.76 for "within 60 min" at every definition, above
the model. Most of the occurrence skill is therefore *"a flare is already happening"*. That reference
is not available in real time, so it is an upper bound, not a baseline to beat.

## 1. What the run shows

### 1.1 Nowcasting "is a flare in progress" works — but on a very loose event definition

| Test period 2026-03-24 → 2026-09-12 | TSS | HSS | AUC | Brier skill |
|---|---:|---:|---:|---:|
| SoLEXHEL-Net (TCN, deep) | **0.706** | 0.689 | 0.931 | 0.526 |
| Gradient-boosted trees on window summaries | 0.700 | 0.683 | 0.928 | **0.570** |
| Logistic regression | 0.643 | 0.627 | 0.885 | 0.394 |

Thresholds were chosen on validation (2025-09 → 2026-03) and frozen. Model and trees are tied;
the trees are better calibrated. With both instruments blanked the model scores AUC 0.500, so the
skill comes from the X-ray data, not from memorising time.

### 1.2 Short-horizon occurrence and flux forecasts beat persistence, modestly

| Flare within | base rate | TSS deep | TSS trees | Brier skill deep |
|---|---:|---:|---:|---:|
| 15 min | 0.678 | **0.606** | 0.605 | 0.334 |
| 30 min | 0.765 | **0.569** | 0.557 | 0.228 |
| 60 min | 0.870 | **0.561** | 0.459 | 0.049 |

| log-flux forecast | 1 min | 5 min | 15 min | 30 min | 60 min |
|---|---:|---:|---:|---:|---:|
| RMSE deep | 0.161 | 0.323 | **0.521** | **0.639** | **0.738** |
| RMSE trees | **0.138** | **0.319** | 0.544 | 0.712 | 0.761 |
| RMSE persistence | 0.158 | 0.341 | 0.576 | 0.736 | 0.874 |
| q10–q90 coverage (nominal 0.80) | 0.76 | 0.74 | 0.71 | 0.70 | 0.68 |

The deep model's edge is at longer horizons (TSS +0.10 over the trees at 60 min; flux error 3–10 %
lower at 15–60 min). At 1 min it is *worse* than persistence. Its 80 % intervals cover only 68–76 %: they are
too narrow and should not be used as calibrated uncertainty without recalibration.

### 1.3 Rise-phase peak forecasting: real skill over references, no winning architecture

8 910 flares, 272 997 rise samples, rolling-origin CV grouped by flare (3 folds, always training on
earlier flares).

| encoder | peak log-MAE (± sd over folds) | skill vs current level | skill vs climatology | time-to-peak skill | large-flare AUC |
|---|---:|---:|---:|---:|---:|
| transformer | 0.843 ± 0.036 | 0.364 | 0.446 | 0.128 | 0.888 |
| TCN | 0.848 ± 0.043 | 0.360 | 0.442 | 0.050 | 0.881 |
| linear | 0.854 ± 0.030 | 0.356 | 0.439 | 0.059 | 0.885 |
| GRU | 0.861 ± 0.015 | 0.351 | 0.434 | 0.027 | 0.868 |
| *current flux level (reference)* | 1.326 | — | — | — | 0.858 |
| *training climatology (reference)* | 1.526 | — | — | — | — |

* Every encoder beats both references on peak size by a wide margin (skill 0.35–0.45).
* The spread between encoders (0.018) is smaller than any encoder's fold-to-fold scatter
  (0.015–0.043): **the ranking is not meaningful**, and a linear encoder is within 0.011 of the best.
* Timing is hard: time-to-peak skill over climatology is only 0.03–0.13.
* Ranking large flares barely improves on the flux already reached (AUC 0.87–0.89 vs 0.858).
* Peak error is 0.84 in natural-log rate — a typical factor of e^0.84 ≈ 2.3, or ≈ 0.37 dex.

### 1.4 HEL1OS does not yet add detectable skill

Paired ablation on the 871 flares with HEL1OS on the rise: same TCN, same folds and seeds, hard
X-rays present vs blanked, per-sample error difference bootstrapped over flares.

| | soft + hard | soft only |
|---|---:|---:|
| peak log-MAE | 0.808 | 0.890 |
| large-flare AUC | 0.789 | **0.842** |
| time-to-peak MAE (min) | 71.6 | **57.7** |

Error reduction from adding HEL1OS: **+0.081, 95 % CI −0.030 to +0.243** (+9.2 %), and it is lower
on only **43 %** of flares — the mean is carried by a minority of events. Large-flare ranking and
timing are *worse* with HEL1OS. On the nowcast test set, blanking HEL1OS changes TSS from 0.706 to
0.714. Verdict: **no detectable benefit at this sample size**; the point estimate is positive, so
this is "not yet shown", not "shown useless".

The nowcast ablation restricted to the 51 688 test windows where HEL1OS observed the forecast
moment tells the same story:

| input at inference | TSS | AUC |
|---|---:|---:|
| both instruments | 0.696 | 0.931 |
| SoLEXS only | **0.715** | **0.935** |
| HEL1OS only | 0.459 | 0.798 |
| neither | 0.000 | 0.500 |

HEL1OS on its own is informative, but the network gains nothing from it once SoLEXS is present. The
nowcast model was trained with only ~4 simultaneous days (Apr–May 2024), because the chronological
split put the 2026 overlap in the test period. The fusion ablation above, which trains on the overlap,
is the fairer test.

## 2. Three problems found in the data and methods — and what was done

**(a) The flare labels are far looser than GOES classes.** The detector (≥1.4× a 6 h 10th-percentile
background, ≥4σ, merge gaps < 5 min, end at half the peak excess) marks **65.8 % of all observed
SoLEXS time as "in a flare"**: 11.8 events per day, mean duration 81 min, 39 % below 2× background,
and one "rise" lasting 80 h. At solar maximum the Sun rarely returns to its 10th-percentile level, so
activity chains into long merged events. Consequences: "flare within 60 min" is true 87 % of the
time; time-to-peak errors of hours come from merged events; and **none of the skill scores above can
be compared numerically with GOES ≥C/≥M studies** (§3). The stricter re-scoring in the box at the
top is the first step; the fix is to calibrate events against GOES classes (§4).

**(b) The SoLEXS energy scale was wrong, and is now corrected.** The pipeline assumed channel 41 =
1 keV and channel 339 = 22 keV (70.5 eV/ch). The published calibration (Sarwade et al. 2025) is
47.75 eV/ch to channel 168 and 94.5 eV/ch above, with a ~2 keV threshold. This archive confirms it
independently: the two strongest quiet-Sun lines, at channels 122.05 and 134.65, are the onboard
Fe-55 source's Mn Kα/Kβ — the published scale places them at 5.915 / 6.516 keV (+20 / +26 eV), the
old one 707 eV off. So the old "1–2 keV" band was really ~2.0–2.7 keV and every band label was
shifted. The band *integrals* are unchanged, so this run's results stand as ML results, but the keV
labels in them are wrong. The code now defaults to the published scale; `legacy_linear` reproduces
this run bit-for-bit (verified on 2026-09-10) and is pinned in forward-test `v1`.

**(c) HEL1OS reading.** CZT1/CZT2 had been sharing columns (CZT2 reads 0.9–1.5× CZT1); CdTe bands with
different lengths were silently dropped; zero-count samples were treated as missing (inflating faint
CdTe bands up to 10×). All fixed before this run. The CdTe "5–20 keV" L1 band effectively starts near
the documented 8 keV CdTe threshold.

## 3. Comparison with the literature

**Most published flare forecasts answer a different question.** The benchmark studies forecast
whether an *active region* (or the disk) produces a ≥C or ≥M GOES flare in the next **24 h**, from
**magnetograms**. This project forecasts the next **15–60 minutes** from **full-disk X-ray light
curves**, with events defined relative to background. TSS is less sensitive to base rate than
accuracy, but it still depends strongly on the event definition — which is why no row of the table
below is a like-for-like comparison.

| Study | Data → target | Headline | Relation to this work |
|---|---|---|---|
| Bobra & Couvidat 2015 | HMI SHARP, SVM → ≥M in 24/48 h | TSS-centred evaluation of 2 071 ARs | Different question; established TSS as the metric used here |
| Nishizuka et al. 2018 (DeFN) | HMI/AIA/GOES features, DNN → 24 h | TSS 0.80 (≥M), 0.63 (≥C); chronological 2010-14 / 2015 split | Our nowcast TSS 0.71 sits in the same range but on far looser events — not comparable |
| Nishizuka et al. 2021 | DeFN operational, Jan 2019–Jun 2020 | TSS 0.70 (≥C, 50 % threshold) | Real-time result; our `forward-test` is built to produce the equivalent |
| Muranushi et al. 2015 (UFCORIN) | HMI time series → GOES classes | TSS 0.75 (X), 0.48 (≥M), 0.56 (≥C) | Same order as our occurrence TSS 0.56–0.61, different horizon |
| Sinha et al. 2022 | HMI, classical ML | Logistic regression TSS 0.967 ± 0.018 | Simple models competitive — **matches our finding** that trees ≈ deep net and linear ≈ transformer |
| Barnes et al. 2016 | All-Clear workshop, common data | ≥M skill weakly positive, none substantially better than climatology | Our Brier skill falls to 0.05 at 60 min: **the same pattern at long horizon** |
| Leka et al. 2019 | Operational systems | Many above no-skill; "no single winner"; ranking depends on event definition and metric | **Matches**: no encoder wins, and our scores hinge on the event definition |
| Cinto et al. 2020 | XGBoost on time series → ≥C up to 72 h | TSS gains of 0.37 / 0.13 / 0.36 over baselines | Supports boosted trees as the baseline to beat |
| Ahmadzadeh et al. 2021 | SWAN-SF benchmark | Random splits "spuriously enhance" skill | Our chronological split, embargo and flare-grouped folds follow this |
| Telikicherla, Woods & Schwab 2025 (HOPE) | DAXSS + GOES-XRS, 162 flares → alert before peak | Alerts 5–15 min before peak; 17.9 min earlier than NOAA R3 alerts | **Closest in spirit** (minutes, X-ray only); they use a hot-onset temperature signature that SoLEXS spectra can provide directly |
| Yi et al. 2026 | GOES-XRS 1997–2024, attention LSTM → peak of an ongoing flare | RMSE 0.26 / 0.45 / 0.87 dex for ≥C / ≥M / X | **Closest task**. Our peak MAE ≈ 0.37 dex is larger, but they use GOES-class events, quarterly folds pooled across years (not forward in time), and report **no persistence baseline** — ours beats current-level by 36 % |
| Aschwanden 2020 | Scaling laws → GOES class upper limit | CCC ≈ 0.7 for 172 M/X flares | Physics-based alternative for magnitude |
| Veronig et al. 2002 (Neupert effect) | >1 000 GOES/BATSE flares | Timing consistent with HXR ∝ d(SXR)/dt in large flares; many exceptions | The physical reason HEL1OS *should* help; our null result says the network is not yet extracting it on 76 days |
| Adithya et al. 2026 (Aditya-L1 SUIT/SoLEXS/HEL1OS) | 7 M/X flares | X-ray enhancement ~3 min before impulsive phase; 28 % of pre-flare transients with HEL1OS 10–30 keV counterparts | Precursor signal exists in HEL1OS but is sparse — consistent with a weak, unproven fusion gain |

**Where the findings agree with the field:** simple models match deep ones; skill at long horizons
barely beats climatology; architecture rankings are within noise; evaluation design (chronological
splits, references, uncertainty) matters more than the network.

**Where this work adds something:** our literature search (not exhaustive) found no published ML
flare-forecasting evaluation on the combined SoLEXS + HEL1OS archive. This one is leakage-controlled,
tests fusion with a paired design, reports current-level and climatology references for every
magnitude forecast, and checks the published energy calibration against the Fe-55 lines.

**Where it falls short of the field:** events are not GOES-calibrated; the fusion test rests on
76 simultaneous days; there is no real-time forward score yet (DeFN has one).

## 4. What to do next, in order

1. **Define events on GOES classes.** Cross-calibrate SoLEXS against GOES-XRS (Sarwade et al. report a
   linear relation, ~15 % low at low flux) or import the NOAA flare list, then relabel and retrain.
   Without it no number here can be put next to a published TSS. `forward-test score --goes-events`
   already reads SWPC/HEK/CSV lists; it needs the file downloaded.
2. **Retrain on the published energy scale** (now the default). Expect small changes; the band labels
   become physically correct.
3. **Run the forward test** as new days arrive (`forward-test predict --name v1`, then `score`).
4. **Revisit fusion only after 1–3**, and first with physics features (hard X-ray vs d(soft)/dt lag,
   HOPE-style temperature onset from SoLEXS spectra) before cross-attention.
5. **Recalibrate** the flux-forecast quantiles (coverage 0.68–0.76 vs 0.80).
6. LightGBM and PatchTST are reasonable additions, but §1.3 predicts they will land inside the current
   fold scatter; they should come after the event definition is fixed.

## References

* Bobra, M. G. & Couvidat, S. 2015, ApJ 798, 135 — [arXiv:1411.1405](https://arxiv.org/abs/1411.1405)
* Nishizuka, N. et al. 2018, ApJ 858, 113 — [arXiv:1805.03421](https://arxiv.org/abs/1805.03421)
* Nishizuka, N. et al. 2021, Earth Planets Space — [arXiv:2112.00977](https://arxiv.org/abs/2112.00977)
* Muranushi, T. et al. 2015, UFCORIN — [arXiv:1507.08011](https://arxiv.org/abs/1507.08011)
* Sinha, S. et al. 2022, ApJ 935, 45 — [arXiv:2204.05910](https://arxiv.org/abs/2204.05910)
* Barnes, G. et al. 2016, ApJ 829, 89 — [arXiv:1608.06319](https://arxiv.org/abs/1608.06319)
* Leka, K. D. et al. 2019, ApJS 243, 36 — [arXiv:1907.02905](https://arxiv.org/abs/1907.02905)
* Cinto, T. et al. 2020, Sol. Phys. 295, 93 — [arXiv:2004.13299](https://arxiv.org/abs/2004.13299)
* Ahmadzadeh, A. et al. 2021, ApJS 254 — [arXiv:2103.07542](https://arxiv.org/abs/2103.07542)
* Camporeale, E. 2019, Space Weather — [arXiv:1903.05192](https://arxiv.org/abs/1903.05192)
* Telikicherla, A., Woods, T. N. & Schwab, B. D. 2025, ApJ — [arXiv:2509.05234](https://arxiv.org/abs/2509.05234)
* Yi, K. et al. 2026 — [arXiv:2608.20062](https://arxiv.org/abs/2608.20062)
* Aschwanden, M. J. 2020, ApJ — [doi:10.3847/1538-4357/ab9630](https://iopscience.iop.org/article/10.3847/1538-4357/ab9630)
* Veronig, A. et al. 2002 / 2005 (Neupert effect) — [ADS 2005ApJ...621..482V](https://ui.adsabs.harvard.edu/abs/2005ApJ...621..482V/abstract)
* Sarwade, A. R. et al. 2025, JATIS 11(4), 045005 (SoLEXS calibration) — [arXiv:2509.26292](https://arxiv.org/abs/2509.26292)
* SoLEXS iron fluorescence in X-class flares, Sol. Phys. 2026 — [arXiv:2605.22573](https://arxiv.org/abs/2605.22573)
* HEL1OS instrument paper, Sol. Phys. 2025 — [arXiv:2512.12679](https://arxiv.org/abs/2512.12679)
* Adithya H. N. et al. 2026, MNRAS (pre-flare evolution with Aditya-L1) — [arXiv:2607.26171](https://arxiv.org/abs/2607.26171)
* Murray, S. A. et al. 2017, Space Weather (Met Office verification) — [arXiv:1703.06754](https://arxiv.org/abs/1703.06754)
