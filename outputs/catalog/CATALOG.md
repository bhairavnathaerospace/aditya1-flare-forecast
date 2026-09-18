# Aditya-L1 master flare catalogue

Generated 2026-09-18 19:38 UTC by `scripts/master_catalog.py`. Period 2024-02-01 00:00:00 → 2026-09-13 00:00:00; SoLEXS observed 758.8 days, HEL1OS 111.9 days. GOES-18 is used only to score, never to detect.

**9102 catalogue entries**: 471 seen by both instruments, 7854 by SoLEXS only, 777 by HEL1OS only (547 of them while SoLEXS was not observing).

Are the HEL1OS-only flares real? 230 occurred while SoLEXS was observing; only 16 are in the GOES list, yet 60% show a simultaneous >3 sigma rise in SoLEXS 6-12 keV (34% above 5 sigma), against 6% (3%) at random quiet times: about half are real small hot flares that neither the GOES list nor the SoLEXS rule records.

SoLEXS-derived classes: A 0, B 820, C 6398, M 1042, X 65.


## Whole archive

### Detection rate (recall) by GOES class

| GOES class | SoLEXS | HEL1OS (chance) | HEL1OS saw non-thermal (≥20 keV) | both observing: SoLEXS alone | both observing: combined (chance) |
|---|---|---|---|---|---|
| B | 66% (n=439) | 27% (4%) (n=126) | 0% | 73% (n=108) | 81% (75%) |
| C | 84% (n=5157) | 67% (15%) (n=691) | 0% | 86% (n=407) | 91% (88%) |
| M | 90% (n=1084) | 91% (21%) (n=127) | 32% | 94% (n=77) | 97% (95%) |
| X | 99% (n=67) | 100% (29%) (n=7) | 100% | 100% (n=4) | 100% (100%) |

*Chance*: the same numbers with every HEL1OS event moved 2 h earlier or later (mean of both). HEL1OS's real contribution is the gap between a figure and its chance level.

Ceiling — the same rule on GOES's own flux: B 66%, C 80%, M 91%, X 99%; precision 82%.


### False alarms (SoLEXS flares with no GOES-listed flare within 5 min)

| SoLEXS class | flares | matched to GOES | unmatched per observed day |
|---|---|---|---|
| >=B | 8132 | 70% | 3.259 |
| >=C | 7334 | 73% | 2.57 |
| >=M | 1087 | 94% | 0.08 |

HEL1OS events inside a GOES flare: 55% of 1301; with non-thermal (CZT 20–40 keV) emission: 63% of 83.


### Class from SoLEXS alone vs GOES

5659 matched flares: same letter 97%, median error 0.026 dex (bias +0.001), within a factor 1.5: 99%.
Without the peak-level correction: same letter 95%, median error 0.059 dex (bias +0.059).

| GOES class | n | same letter | median error (dex) | bias (dex) |
|---|---|---|---|---|
| B | 293 | 84% | 0.051 | +0.051 |
| C | 4320 | 99% | 0.027 | +0.001 |
| M | 980 | 94% | 0.02 | -0.003 |
| X | 66 | 94% | 0.031 | -0.028 |

### Alert time before the GOES peak (flares both instruments observed)

| GOES class | SoLEXS alert: median min before peak | alerted before peak | combined alert: median | before peak | HEL1OS fired first | by (median min) |
|---|---|---|---|---|---|---|
| B | 1.0 | 54% | 1.0 | 57% | 62% | 1.0 |
| C | 2.0 | 74% | 3.0 | 81% | 50% | 0.0 |
| M | 4.0 | 83% | 5.0 | 89% | 47% | 0.0 |
| X | 6.5 | 100% | 7.5 | 100% | 25% | -3.0 |

## Test period only (from 2026-03-24)

### Detection rate (recall) by GOES class

| GOES class | SoLEXS | HEL1OS (chance) | HEL1OS saw non-thermal (≥20 keV) | both observing: SoLEXS alone | both observing: combined (chance) |
|---|---|---|---|---|---|
| B | 69% (n=252) | 27% (4%) (n=126) | 0% | 73% (n=108) | 81% (75%) |
| C | 86% (n=891) | 69% (14%) (n=390) | 0% | 87% (n=373) | 92% (89%) |
| M | 93% (n=112) | 94% (22%) (n=71) | 37% | 94% (n=70) | 97% (96%) |
| X | 100% (n=6) | 100% (50%) (n=2) | 100% | 100% (n=2) | 100% (100%) |

*Chance*: the same numbers with every HEL1OS event moved 2 h earlier or later (mean of both). HEL1OS's real contribution is the gap between a figure and its chance level.

### False alarms (SoLEXS flares with no GOES-listed flare within 5 min)

| SoLEXS class | flares | matched to GOES | unmatched per observed day |
|---|---|---|---|
| >=B | 1595 | 66% | 3.563 |
| >=C | 1137 | 78% | 1.62 |
| >=M | 108 | 97% | 0.02 |

HEL1OS events inside a GOES flare: 61% of 710; with non-thermal (CZT 20–40 keV) emission: 81% of 37.


### Class from SoLEXS alone vs GOES

1054 matched flares: same letter 96%, median error 0.033 dex (bias +0.002), within a factor 1.5: 100%.
Without the peak-level correction: same letter 91%, median error 0.068 dex (bias +0.067).

| GOES class | n | same letter | median error (dex) | bias (dex) |
|---|---|---|---|---|
| B | 174 | 87% | 0.054 | +0.054 |
| C | 770 | 98% | 0.031 | -0.005 |
| M | 104 | 94% | 0.022 | -0.017 |
| X | 6 | 67% | 0.049 | -0.049 |

### Alert time before the GOES peak (flares both instruments observed)

| GOES class | SoLEXS alert: median min before peak | alerted before peak | combined alert: median | before peak | HEL1OS fired first | by (median min) |
|---|---|---|---|---|---|---|
| B | 1.0 | 54% | 1.0 | 57% | 62% | 1.0 |
| C | 2.0 | 74% | 3.0 | 81% | 49% | 0.0 |
| M | 4.0 | 83% | 5.0 | 90% | 47% | 0.0 |
| X | 6.0 | 100% | 7.0 | 100% | 50% | -0.5 |
