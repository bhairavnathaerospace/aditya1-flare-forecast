# HEL1OS timing audit and sub-second hard X-rays

Generated 2026-09-18 16:58 UTC by `scripts/hxr_timing.py` on 44 flares with a reliable CZT spectrum (+-150 s around the CZT peak; quiet reference 400-700 s before it). Onboard 10 ms ticks throughout: the UTC column is a per-packet stamp that jumps by up to +-1 s between packets.

## Events come in readout batches

Before the flares (64 events/s summed over the four detectors, median), only **4.3%** of 10 ms ticks hold an event, against 47% if the same events were spread at random. They come in batches of about **484 events**, each ~0.4 s long, one every 2.2-8.0 s (median 8.0 s; faster at higher rates, Spearman -0.819, never longer than 8.0 s), in all four detectors at once. The official 1 s light-curve product shows the same batches with exactly the same counts. This is the signature of a buffer read out when it fills (or on a timeout): at ordinary rates the time stamps are readout times, and timing is only as good as the batch spacing. Averaged over 20 s this also makes HEL1OS look over-dispersed, the 2-10x Poisson noise the master catalogue had to measure.

| events/s (all detectors) | seconds | ticks occupied (median) | continuous seconds (>= 90% occupied) |
|---|---|---|---|
| 0-300 | 3626 | 0% | 0% |
| 300-1,000 | 3788 | 32% | 0% |
| 1,000-2,000 | 2139 | 65% | 9% |
| 2,000-3,000 | 957 | 98% | 77% |
| 3,000-5,000 | 1060 | 89% | 44% |
| 5,000-8,000 | 723 | 96% | 96% |
| 8,000-12,000 | 683 | 98% | 100% |
| 12,000-and above | 224 | 98% | 100% |

Only bright flare peaks fill the stream continuously; everything below is measured there only.

## Inside continuous stretches (13 flares)

Median stretch 94 s at 9,313 events/s. Fractional residuals from a 4 s running mean. *Monitor*: CdTe1 x CdTe2 at 6-12 keV (thermal; anything above zero is instrumental). *HXR raw*: CZT1 x CZT2 at 40-100 keV. *HXR corrected*: after removing each CZT's own CdTe monitor; above zero would be real sub-second hard X-ray structure. Share of flares whose 95% interval is above zero in brackets.

| scale | monitor | HXR raw | HXR corrected | noise sd | 95% upper limit on real rms |
|---|---|---|---|---|---|
| 50ms | +0.0427 (100%) | +0.0601 (50%) | +0.0239 (10%) | 0.02639 | 28% |
| 100ms | +0.0286 (100%) | +0.0362 (40%) | +0.0080 (10%) | 0.01866 | 21% |
| 200ms | +0.0162 (100%) | +0.0189 (40%) | +0.0036 (10%) | 0.01319 | 17% |
| 500ms | +0.0050 (100%) | +0.0065 (30%) | +0.0047 (10%) | 0.00834 | 15% |

After the monitor correction only 10% of flares keep a covariance above zero at 100 ms, and the median sits inside its counting noise: **no sub-second hard X-ray structure is detected** beyond what the instrument itself imprints. The thermal monitors, which cannot vary that fast, share a ~17% rms modulation at 100 ms.

**Energy-dependent delay** (60-150 vs 25-40 keV, 2 flares with a clear cross-correlation): weighted mean +0.2 +- 1.4 ms, median +0.2 ms; 0% individually beyond 2 sigma. A readout stamp shared by all energies would also give zero, so this does not show that the energies are simultaneous.

## What this means

- Sub-second hard X-ray science is not available from the L1 event lists in general: below ~2,000 events/s the timing is set by the readout batches (seconds). It needs the instrument team's description of how events are time-tagged.
- HEL1OS light curves are reliable on >= 10-20 s bins, which is what the forecasting pipeline uses.
- Spectral shapes (scripts/hxr_spectra.py) do not depend on timing and stand.
