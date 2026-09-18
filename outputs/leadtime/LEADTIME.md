# Lead time before the flare peak (v3)

Generated 2026-09-18 19:40 UTC by `scripts/lead_time.py`. Test period 2026-03-24 05:03 -> 2026-09-13 00:00 UTC, 150.2 days with SoLEXS and model output. Every threshold was fixed on the validation period; the test period only scores it.

*Warned*: the alert was on at some minute from 30 min before the GOES start (or the previous flare's peak) to the GOES peak. *Chance*: the same windows moved 2 h, where no flare happened. *Event TSS* = warned - chance. *Lead*: GOES peak minus the first alert minute.

## Alert: a GOES >= C1 flare within 15 min

996 GOES flares, 1114 flare-free chance windows.

### Same false-alarm rate: 2 per day on validation

| method | warned | chance | event TSS | median lead (IQR), min | lead >= 5 min | false alarms / day (test) | time on |
|---|---|---|---|---|---|---|---|
| v3 network | 92% | 9% | 0.83 | 6.0 (2.0-18.0) | 50% | 2.15 | 6% |
| SoLEXS trend (15-min extrapolation) | 82% | 9% | 0.72 | 5.0 (1.0-16.0) | 41% | 1.39 | 4% |
| SoLEXS flux now | 64% | 18% | 0.46 | 10.0 (2.0-34.0) | 41% | 0.84 | 12% |
| catalogue rise rule | 75% | 13% | 0.62 | 7.0 (2.0-18.0) | 46% | 4.00 | 4% |
| rise rule, only at >= C1 flux | 72% | 9% | 0.63 | 6.0 (2.0-16.0) | 42% | 1.87 | 3% |

### Each method's best-TSS threshold (validation minutes)

| method | warned | chance | event TSS | median lead (IQR), min | lead >= 5 min | false alarms / day (test) | time on |
|---|---|---|---|---|---|---|---|
| v3 network | 100% | 47% | 0.52 | 20.0 (8.0-34.0) | 82% | 14.62 | 14% |
| SoLEXS trend (15-min extrapolation) | 100% | 60% | 0.40 | 22.0 (9.0-36.0) | 86% | 12.24 | 24% |
| SoLEXS flux now | 89% | 34% | 0.55 | 14.0 (3.0-35.0) | 61% | 1.64 | 22% |

By GOES class, at 2 false alarms/day (warned / median lead, min):

| method | C | M | X |
|---|---|---|---|
| v3 network | 90% / 5.0 (n=880) | 100% / 14.0 (n=111) | 100% / 9.0 (n=5) |
| SoLEXS trend (15-min extrapolation) | 79% / 3.0 (n=880) | 100% / 16.0 (n=111) | 100% / 24.0 (n=5) |
| SoLEXS flux now | 60% / 7.0 (n=880) | 100% / 27.0 (n=111) | 100% / 42.0 (n=5) |
| catalogue rise rule | 74% / 7.0 (n=880) | 86% / 10.0 (n=111) | 100% / 7.0 (n=5) |
| rise rule, only at >= C1 flux | 70% / 6.0 (n=880) | 86% / 10.0 (n=111) | 100% / 7.0 (n=5) |

- Network minus SoLEXS trend (15-min extrapolation) (same flares, day-block 95% CI): warned +0.100 [0.074, 0.131], mean lead +2.18 min [1.062, 3.14] (a missed flare counts as 0).
- Network minus SoLEXS flux now (same flares, day-block 95% CI): warned +0.272 [0.223, 0.323], mean lead -0.05 min [-1.892, 1.643] (a missed flare counts as 0).
- HEL1OS ablation (442 flares with HEL1OS for >= half the window, same thresholds): warned 96% with vs 93% without [0.015, 0.05]; mean lead gain +1.04 min [0.546, 1.529]; false alarms/day 2.90 vs 1.85 (72.0 days with HEL1OS).
- The same ablation with the HEL1OS-hidden threshold moved to the same false-alarm rate (2.96/day): warned 96% with vs 95% without [-0.002, 0.022]; mean lead gain +0.15 min [-0.303, 0.66].

## Alert: GOES flux will reach M1 within 30 min

116 GOES flares, 178 flare-free chance windows.

### Same false-alarm rate: 0.5 per day on validation

| method | warned | chance | event TSS | median lead (IQR), min | lead >= 5 min | false alarms / day (test) | time on |
|---|---|---|---|---|---|---|---|
| v3 network | 91% | 9% | 0.82 | 6.0 (2.0-14.0) | 53% | 0.35 | 1% |
| SoLEXS trend (15-min extrapolation) | 72% | 2% | 0.70 | 3.0 (1.0-7.0) | 25% | 0.55 | 0% |
| SoLEXS flux now | 98% | 8% | 0.90 | 5.0 (1.0-13.0) | 50% | 0.26 | 1% |

### Each method's best-TSS threshold (validation minutes)

| method | warned | chance | event TSS | median lead (IQR), min | lead >= 5 min | false alarms / day (test) | time on |
|---|---|---|---|---|---|---|---|
| v3 network | 98% | 48% | 0.51 | 22.5 (4.2-39.8) | 73% | 1.96 | 6% |
| SoLEXS trend (15-min extrapolation) | 100% | 66% | 0.34 | 31.0 (14.0-41.0) | 88% | 8.56 | 8% |
| SoLEXS flux now | 100% | 51% | 0.49 | 22.5 (4.0-40.0) | 74% | 2.18 | 8% |

By GOES class, at 0.5 false alarms/day (warned / median lead, min):

| method | M | X |
|---|---|---|
| v3 network | 91% / 6.0 (n=111) | 100% / 5.0 (n=5) |
| SoLEXS trend (15-min extrapolation) | 71% / 2.0 (n=111) | 100% / 5.0 (n=5) |
| SoLEXS flux now | 98% / 4.0 (n=111) | 100% / 5.0 (n=5) |

Before GOES first reached M1.0:

| method | alert before M1 | median lead, min | lead >= 5 min |
|---|---|---|---|
| v3 network | 46% | 0.5 | 24% |
| SoLEXS trend (15-min extrapolation) | 25% | 0.0 | 6% |
| SoLEXS flux now | 34% | 0.0 | 24% |

- Network minus SoLEXS trend (15-min extrapolation) (same flares, day-block 95% CI): warned +0.190 [0.066, 0.291], mean lead +6.17 min [3.01, 9.322] (a missed flare counts as 0).
- Network minus SoLEXS flux now (same flares, day-block 95% CI): warned -0.069 [-0.125, -0.026], mean lead -0.21 min [-1.597, 0.99] (a missed flare counts as 0).
- HEL1OS ablation (70 flares with HEL1OS for >= half the window, same thresholds): warned 94% with vs 89% without [0.012, 0.118]; mean lead gain +0.34 min [0.227, 0.473]; false alarms/day 0.60 vs 0.43 (72.0 days with HEL1OS).
- The same ablation with the HEL1OS-hidden threshold moved to the same false-alarm rate (0.64/day): warned 94% with vs 97% without [-0.102, 0.024]; mean lead gain -1.09 min [-1.848, -0.176].

Notes:

- A false alarm is an alert episode with no GOES flare of the class from its start to 15 min (C) or 30 min (M) after it ends. 'Time on' is the share of minutes the alert was raised.
- The M alert's lead before the peak includes the flare's own rise from C level; the lead before GOES first reached M1.0 is the operationally useful number.
- Trade-off curves over all thresholds: leadtime_summary.json (curves_test) and leadtime.png.
