# Forecasting flares hours ahead from SoLEXS activity

Generated 2026-09-18 19:39 UTC by `scripts/multihour_forecast.py`. Hourly origins; features from the SoLEXS light curve and the master catalogue only (no magnetograms). Fitted on the training period; the model (logistic regression or gradient-boosted trees) and its threshold chosen on validation; scored on the test period (from 2026-03-24). 95% intervals resample whole weeks.

| target | test hours | base rate | model | AUC [95% CI] | persistence AUC | AUC gain [95% CI] | TSS | persistence TSS | BSS vs training climatology |
|---|---|---|---|---|---|---|---|---|---|
| >=C1 within 2 h | 3394 | 0.39 | logistic | 0.724 [0.679, 0.766] | 0.702 | +0.022 [0.011, 0.037] | 0.338 | 0.287 | 0.212 |
| >=C1 within 6 h | 3397 | 0.65 | logistic | 0.719 [0.655, 0.772] | 0.711 | +0.008 [-0.01, 0.027] | 0.333 | 0.297 | 0.239 |
| >=C1 within 12 h | 3492 | 0.82 | logistic | 0.703 [0.623, 0.778] | 0.73 | -0.027 [-0.072, 0.025] | 0.303 | 0.309 | 0.138 |
| >=C1 within 24 h | 3492 | 0.94 | trees | 0.738 [0.594, 0.848] | 0.796 | -0.058 [-0.086, -0.019] | 0.414 | 0.43 | -0.039 |
| >=M1 within 2 h | 3394 | 0.06 | logistic | 0.842 [0.743, 0.889] | 0.785 | +0.057 [0.019, 0.118] | 0.481 | 0.467 | 0.279 |
| >=M1 within 6 h | 3397 | 0.13 | logistic | 0.822 [0.743, 0.871] | 0.765 | +0.057 [0.016, 0.128] | 0.47 | 0.464 | 0.423 |
| >=M1 within 12 h | 3492 | 0.21 | logistic | 0.806 [0.719, 0.864] | 0.74 | +0.065 [0.015, 0.128] | 0.413 | 0.424 | 0.467 |
| >=M1 within 24 h | 3492 | 0.30 | logistic | 0.785 [0.692, 0.856] | 0.686 | +0.099 [0.038, 0.171] | 0.395 | 0.325 | 0.474 |

*Persistence* is a logistic fit on one number: flares of the class SoLEXS saw in the last 24 h. BSS is against the training-period base rate; the test period is more active than training, which penalises every model's calibration.

## What carries the signal

Drop in test AUC when one feature is shuffled (selected model):

- >=C1 within 2 h: flux_now (+0.078), n_C_6h (+0.076), flux_max_1h (+0.037), flux_mean_24h (+0.020)
- >=C1 within 6 h: flux_mean_24h (+0.065), n_C_72h (+0.061), flux_now (+0.055), flux_max_1h (+0.037)
- >=C1 within 12 h: flux_mean_24h (+0.087), n_C_72h (+0.067), flux_now (+0.064), n_B_6h (+0.054)
- >=C1 within 24 h: flux_min_6h (+0.033), flux_now (+0.025), flux_max_6h (+0.016), flux_mean_6h (+0.013)
- >=M1 within 2 h: flux_now (+0.199), flux_mean_6h (+0.038), n_B_24h (+0.014), n_B_72h (+0.010)
- >=M1 within 6 h: flux_now (+0.084), n_C_6h (+0.063), n_B_72h (+0.043), n_C_72h (+0.024)
- >=M1 within 12 h: flux_max_24h (+0.083), n_B_72h (+0.036), n_C_72h (+0.023), n_C_6h (+0.022)
- >=M1 within 24 h: n_B_72h (+0.101), flux_max_24h (+0.088), n_C_72h (+0.043), n_C_6h (+0.038)

## Reading it

- Significant gain over persistence: >=C1 within 2 h, >=M1 within 2 h, >=M1 within 6 h, >=M1 within 12 h, >=M1 within 24 h. The extra skill comes from the soft X-ray level (flux now, the 24 h maximum) and from how many small flares, down to B-class microflares, SoLEXS counted over the last 1-3 days: an active region announces itself before its big flares, and a count of big flares alone misses that.
- For >= C1 at 6-24 h nearly every window contains a flare in this active period (base rate 0.65-0.94), so there is little left to forecast.
- Without magnetic-field data (SDO/HMI SHARP) this is a lower bound on what a multi-hour forecast can do.
