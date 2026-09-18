# Flare temperatures from SoLEXS spectra

Generated 2026-09-18 20:09 UTC by `scripts/solexs_temperature.py`: 2001 of 2580 SoLEXS flares >= 5e-06 W/m^2 fitted (65791 spectra of 20 s). Isothermal continuum temperature from the line-free 4.3-6.2 and 8.6-12 keV windows (no response matrix or atomic database available); the background is the 2 min before each flare.

| fit outcome | flares |
|---|---|
| no fittable bins | 567 |
| no pre-flare background | 12 |
| ok | 2001 |

## Temperature by class

| SoLEXS class | flares | T at the flux peak, MK (IQR) | hottest, MK | hottest before the peak | by (median) | Fe XXV EW at peak, keV |
|---|---|---|---|---|---|---|
| C | 905 | 15.9 (14.3-17.7) | 19.2 | 79% | 1.2 min | 1.74 |
| M | 1032 | 17.4 (15.6-19.1) | 22.0 | 96% | 3.2 min | 1.81 |
| X | 64 | 23.0 (20.8-25.5) | 27.5 | 98% | 4.0 min | 1.34 |

Bigger flares are hotter: Spearman 0.388 between peak temperature and peak flux, 4.76 MK per decade of flux. The plasma is hottest before the flux peaks, as the hot-onset study found.

## Checks

- **GOES**: the XRS-A/XRS-B ratio rises with temperature. Across 1965 flares its rank correlation with the SoLEXS peak temperature is 0.591 (p = 5.1e-185).
- Within flares, minute by minute: median Spearman 0.75 over 1204 flares, positive in 98%.
- **HEL1OS CdTe** (independent detector, 9-20 keV, same model) at the peak minute: 135 flares, CdTe/SoLEXS temperature ratio 1.909 (median), Spearman 0.828 (p = 2.9e-35). The two rank flares alike; the absolute scale differs because the 9-20 keV photons weigh the hottest plasma more in a multi-thermal flare and because the CdTe response (window, threshold) is unknown here, so only the ranking is a check.
- **Model systematic**: replacing E^-1.3 by E^-1.0 or E^-1.6 changes the peak temperature by -8% / +7% (median).
- **Fe XXV 6.7 keV line**: its equivalent width grows with temperature up to 20-23 MK (1.75 keV) and falls beyond, the shape the Fe XXV ionisation balance predicts: 0.87 keV at 6-8 MK, 1.49 keV at 8-10 MK, 1.4 keV at 10-12 MK, 1.42 keV at 12-14 MK, 1.53 keV at 14-16 MK, 1.64 keV at 16-18 MK, 1.69 keV at 18-20 MK, 1.75 keV at 20-23 MK, 1.75 keV at 23-26 MK, 1.7 keV at 26-30 MK, 1.46 keV at 30-40 MK.

## Caveats

- A continuum-slope temperature, not a full spectral fit: the E^-1.3 shape, the 450 um silicon efficiency and ignored free-bound edges set a systematic of roughly the size quoted above. Emission measures need the effective area and are not given.
- Very bright peaks (> ~5,000 counts/s, the largest X flares) may carry pile-up, which hardens the spectrum and pushes T up; the 30-43 MK peaks of the X3-X5 flares should be read with that in mind.
- A CHIANTI-based fit (e.g. sunkit-spex with its atomic tables) would replace the approximation; it needs a download.
