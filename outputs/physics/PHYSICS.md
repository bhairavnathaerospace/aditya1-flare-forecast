# Phase 2 — hot onset and Neupert effect across the archive

Run 2026-09-15 by `scripts/physics_catalog.py`; numbers from `physics_summary_dec120s.json` and
`physics_summary_dec40s.json`. The analysis covers 6 848 GOES-18 flares ≥ C1.0 that started inside a
SoLEXS day, using the published energy scale. 5 925 were analysable:

| not analysable | count |
|---|---:|
| SoLEXS gap on the rise | 542 |
| no significant SoLEXS onset | 377 |
| no pre-flare background | 4 |

**Method.** Background: linear fit over the 15 min before the GOES start, following a decaying
earlier flare but never extrapolating a rise. Onset: causal, the first 3 consecutive 20 s bins above
3σ.

Temperature proxies:
- **SoLEXS hardness:** net 6–8 keV over net 3–4 keV counts. It is only computed with ≥ 200 / 20 counts.
- **GOES XRS ratio:** net short channel over net long channel.

Neither is converted to MK, because there is no response matrix in the L1 zips.

## 1. Hot onsets are the rule

| | C | M | X |
|---|---:|---:|---:|
| flares (with enough onset counts) | 4 833 (2 481) | 1 025 (818) | 67 (54) |
| hardness: pre-flare background | 0.000 | 0.010 | 0.014 |
| hardness: first 2 min after onset | 0.106 | 0.129 | 0.194 |
| hardness: around the peak | 0.086 | 0.140 | 0.240 |
| onset / peak hardness (median) | 1.20 | 1.00 | 0.87 |
| onset / steepest-rise hardness (median) | 0.95 | 0.79 | 0.58 |
| onset ≥ 80 % of peak hardness | 89 % | 65 % | 57 % |
| GOES ratio onset / peak (independent) | 1.05 | 0.87 | 0.70 |
| onset → steepest rise (median) | 2.7 min | 5.3 min | 8.0 min |
| onset → SoLEXS peak (median) | 5.7 min | 9.3 min | 12.7 min |

The new emission is already about as hot in its first two minutes as it is at peak, as
Hudson et al. (2021) describe. Larger flares then heat further during the impulsive phase. Two
instruments agree on this: the SoLEXS hardness and the GOES ratio.

**Caveat:** only flares bright enough at onset have an onset hardness (51 % of C, 80 % of M).

**Example, 2024-05-05 X1.2.** The friend's XSPEC fits showed 8–9 MK, then ~17 MK at onset, then ~24 MK
in the impulsive phase. This analysis finds:
- causal onset 11:43, 2 min after the GOES start;
- hardness 0.003 (background), 0.277 (onset), 0.359 (steepest rise), 0.255 (peak);
- GOES ratio 0.33 at onset and 0.34 at peak;
- steepest rise 7.3 min and peak 11.3 min after onset.

The pattern matches the XSPEC fits.

## 2. Neupert effect in HEL1OS

451 flares had HEL1OS observing. 45 of them (12 C, 31 M, 2 X) had CZT 20–40 keV hard X-rays at ≥ 5σ.
For those 45:
- the median best correlation with d(SoLEXS)/dt is **r = 0.77**, and 87 % have r > 0.5;
- the median lag is **0 s** (interquartile range 0–40 s, soft derivative after hard X-rays);
- the hard X-ray peak comes a median 20 s before the steepest soft X-ray rise.

HEL1OS timing behaves as the Neupert effect predicts, which also validates the HEL1OS reading pipeline.

## 3. Do these signatures help predict how big the flare gets? No — not beyond the flux seen so far

- **Test design:** gradient-boosted trees, rolling-origin folds in time, and a paired bootstrap that
  resamples whole days.
- **Baseline features:** SoLEXS net rate, background and rise so far.
- **Target:** log10 GOES peak flux, only for flares whose peak is still ahead.

| decision time | flares tested | flux so far | + onset hardness | + GOES ratio | current GOES level | training median |
|---|---:|---:|---:|---:|---:|---:|
| onset + 40 s | 4 352 | 0.220 dex | 0.218 | (no full GOES minute yet) | 0.367 | 0.341 |
| onset + 2 min | 3 676 | 0.219 dex | 0.220 | 0.219 | 0.344 | 0.342 |

Gains from adding each feature (95 % CI):

| feature added | onset + 40 s | onset + 2 min |
|---|---|---|
| onset hardness | +0.0014 dex [+0.0002, +0.0024] | −0.0009 [−0.0028, +0.0011] |
| GOES ratio | — | −0.0005 [−0.0021, +0.0011] |
| early hard X-rays (HEL1OS flares, 284 tested) | — | −0.0019 [−0.0109, +0.0071] |

**Will a flare still below M1 at onset + 2 min reach M1?** 625 of the 3 560 tested did.

| features | AUC |
|---|---:|
| flux so far | 0.783 |
| + onset hardness | 0.780 |
| + GOES ratio | 0.779 |
| current GOES level | 0.803 |

**Verdict.** Hot onsets are real and nearly universal. The hardness at onset does grow with the
eventual class, but that information is already carried by how bright and how fast the flare is so
far. Adding temperature or early hard X-rays changes peak error by less than 1 %. At the 40 s
decision the gain is statistically detectable but negligible. The intervals are tight, so this is a
measured null, not an underpowered one.

Where these signatures may still matter, not yet tested:
- earlier *detection* of the onset from the hardness jump;
- timing of the impulsive phase;
- an absolute temperature from calibrated fits rather than a band-ratio proxy.
