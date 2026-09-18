# Hard X-ray spectral index from HEL1OS photon lists

Generated 2026-09-18 16:48 UTC by `scripts/hxr_spectra.py`. 80 flares have a CZT 20-40 keV burst in the master catalogue; 44 give a reliable power-law fit at the hard X-ray peak (>= 300 net counts above 30 keV, 1.5 <= gamma <= 10, error < 0.5).

| fit outcome | flares |
|---|---|
| no pre-flare background | 6 |
| ok | 44 |
| ok, unreliable | 10 |
| too few counts above 30 keV | 20 |

## Spectral index at the hard X-ray peak

Count spectral index gamma (N(E) ~ E^-gamma, 30 keV up to the last 3-sigma bin, 16 s at the 30-60 keV peak): median **4.11** (IQR 3.23-4.84, range 2.62-8.05), typical error 0.1 (statistical, scaled by sqrt(chi2/dof) where the fit is poor). Thick-target electron index delta = gamma + 1.

- **Two independent detectors agree**: CZT1 vs CZT2 on 37 flares, median difference +0.088, 97% within 2 sigma (rms pull 1.02).
- **Energy scale**: the onboard Am-241 line sits at 59.25 +- 0.3 keV across 73 flares (true 59.54 keV).
- **Thermal contamination**: starting the fit at 25 keV instead of 30 keV changes gamma by +0.070 (median); 66% of flares come out steeper, as expected where hot thermal emission still contributes at 25-30 keV.
- Across flares, gamma vs GOES peak flux shows a weak trend (Spearman +0.31, p = 0.0457) (n = 42); gamma vs the 35 keV peak flux shows a weak trend (Spearman -0.32, p = 0.0356). A positive Spearman means bigger flares have softer spectra.

## Soft-hard-soft (photons of both detectors)

50 flares with >= 5 time bins through the impulsive phase: gamma falls as the flux rises in **80%** of them (56% significant at p < 0.05, against 14% significantly the other way). Median Spearman -0.461, median slope d(gamma)/d(log10 F) = -1.183.

## Soft-hard-soft (gamma from CZT1, flux from CZT2 (independent noise))

41 flares with >= 5 time bins through the impulsive phase: gamma falls as the flux rises in **88%** of them (66% significant at p < 0.05, against 2% significantly the other way). Median Spearman -0.565, median slope d(gamma)/d(log10 F) = -1.429.

## Caveats

- No response matrix ships with the L1 products, so gamma is the *count* index. It tracks the photon index where CZT efficiency is flat (30-150 keV); K-escape and hole tailing are not modelled.
- Errors are statistical; for the brightest flares a single power law is a poor fit (chi2/dof up to ~70), and the quoted error is scaled up accordingly.
- HEL1OS reads events out in batches (scripts/hxr_timing.py): below ~2,000 events/s photons are displaced in time by up to the batch spacing (seconds). Spectral shapes do not care; the time-resolved bins (>= 4 s, impulsive-phase rates) are affected only slightly.
- 'ok, unreliable' fits (flat gamma < 1.5 or few counts) are mostly HEL1OS-only events with no GOES flare: likely background or particle fluctuations, not flares.
