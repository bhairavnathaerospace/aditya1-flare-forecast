# Solar Flare Nowcasting and Forecasting from Aditya-L1 (SoLEXS + HEL1OS)

A deep learning system that ingests Aditya-L1 soft X-ray (SoLEXS) and hard X-ray
(HEL1OS) Level-1 products, detects flares, and predicts what happens next.

```bash
pip install -r requirements.txt
# On Windows, plain `pip install torch` is CPU-only. If you have an NVIDIA GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.cuda.is_available())"   # must print True

python -m solarflare.cli inspect     # what is in the data
python -m solarflare.cli train       # preprocess, train, evaluate, plot
```

The model is 20 stacked dilated 1-D convolution blocks — memory-bound with poor
cache reuse, roughly the worst case for a CPU. Measured on this machine
(8-core CPU vs RTX 3050 6GB laptop): **72 s/epoch → 7 s/epoch, a 9x speedup**,
so a full 60-epoch run takes under two minutes. The numbers are identical
either way, only the wall clock changes. `--device` overrides the auto-choice.

> **Current results use GOES-18 flare classes as truth:
> [`outputs/archive_goes/reports/CONCLUSIONS.md`](outputs/archive_goes/reports/CONCLUSIONS.md)**
> (2024-02 → 2026-09, 7 187 GOES ≥ C flares). Each point is on the unseen 2026 test period:
> - **Flare in progress:** nowcast TSS 0.77 for ≥ C and 0.79 for ≥ M; trees 0.745.
> - **"Flare within 60 min":** TSS only 0.26. For ≥ M and ≥ X, the flux already reached ranks
>   upcoming flares better than the model.
> - **Flux forecast:** the anchored model (`--anchor-flux`, frozen as v3) beats "no change" in
>   calibrated SoLEXS flux from 15 min out; without the anchor it only wins at 30–60 min.
> - **Peak size of a rising flare:** 16–20 % better than the level reached. Transformer, linear, GRU
>   and TCN tie; **PatchTST is clearly worse** (0.253 dex, no skill over the level reached).
> - **HEL1OS:** cuts peak error by **15.4 % ± 4.4 %** over four seeds, every interval excluding zero,
>   but only 3.9 % with a linear encoder — the gain needs the *shape* of the hard X-ray rise.
> - **Probabilities:** raw occurrence heads are miscalibrated (Brier skill < 0); isotonic calibration
>   on validation fixes that (+0.34 at 15 min) without changing detection (`scripts/recalibrate.py`).
> - **More data?** Skill flattens past ~50 % of the SoLEXS archive. For HEL1OS, recent data beats more
>   data, and the two instruments overlap on only 76 of 950 days — the main limitation of every
>   hard X-ray result here.
> - **Physics:** hot onsets and the Neupert effect are confirmed, but add no forecast skill beyond
>   the flux so far ([`outputs/physics/PHYSICS.md`](outputs/physics/PHYSICS.md)).
>
> The first archive run on SoLEXS-detector labels is kept for the record in
> [`outputs/archive/reports/CONCLUSIONS.md`](outputs/archive/reports/CONCLUSIONS.md). Sections 1–2 below
> describe the original two-day sample, where the project began.

---

## 1. Read this first: what the supplied data can and cannot support

Four properties of the two folders in this repository shape everything below.

### The two instruments do not overlap in time

| Product | Coverage | Cadence |
|---|---|---|
| SoLEXS SDD2 | `2026-09-10 00:00:00 → 23:59:59` (86 400 s) | 1 s |
| HEL1OS CZT1 | `2026-09-11 12:00:03 → 23:59:58` (43 188 s) | 1 s nominal |

There is **zero simultaneous soft+hard coverage**. A model cannot learn a
soft↔hard relationship that is never observed. The fusion pathway is built,
tested and ready, but on this data it can only be exercised one modality at a
time. `inspect` prints the overlap explicitly, and the evaluation reports a
modality ablation that shows the hard X-ray channel contributing nothing to
soft-flare nowcasting *because it was never simultaneously available*.

**To enable real fusion, add days where both instruments observed.** Nothing in
the code changes — drop the folders in and re-run.

### HEL1OS light curves are 83% fill

Rows with `CTR == 0` **and** `STAT_ERR == 0` mean *no telemetry for that
second*, not zero counts. Only 7 430 of 43 188 seconds carry real samples.

| | 18–160 keV mean rate |
|---|---|
| Naive read (fill treated as zeros) | 26.0 cts/s |
| Correctly masked | **151.0 cts/s** |

A model trained on the naive version learns the telemetry duty cycle. Masking
is unconditional in `io/hel1os.py`, and per-bin coverage is passed to the
network as a feature so it can discount thinly-sampled bins.

### One day of soft X-ray data is enough for nowcasting, not for forecasting

The SoLEXS day contains a dozen-plus detected flares — good signal, but it is
one day.

* **Nowcasting** (is a flare happening, what phase, how bright, where is it
  heading) is genuinely learnable from this and is what the reported numbers
  measure.
* **Operational forecasting** in the space-weather sense — "probability of an
  M-class flare in the next 24 h" — needs months to years of data spanning many
  active regions. The architecture supports it; the data does not. Treat the
  ≤60 min occurrence heads as a demonstration of the mechanism, and see
  §7 for the degenerate 60-minute base rate.

### SDD1 is empty

`AL1_SLX_L1_20260910_v1.0/.../SDD1/` holds a zero-row GTI and no `.lc`/`.pi`.
The detector produced no science data. The reader returns `None` for it and the
pipeline continues on SDD2.

---

## 2. What the data actually contains

Run `python -m solarflare.cli inspect` for the current listing. On 2026-09-10
SoLEXS shows two substantial events — peaking at **00:38** and **17:11 UTC**, the
latter reaching ~20 cts/s in the 1.55–12.4 keV band and decaying for over two
hours — plus roughly a dozen smaller ones, against a quiescent background of
~0.5 cts/s in that band (~1.8 cts/s across the full usable ~2–24 keV range).
(Band energies in this section were quoted on the old, wrong energy scale — see below.)

HEL1OS on 2026-09-11 12:00–24:00 is flat: 18–160 keV sits at 151 ± 24 cts/s
with the Sun in the field of view throughout, HV and detector temperature
stable. No flares. The exact event list, with rise times and durations, is
printed by `inspect` and recorded in `outputs/reports/data_meta.json`.

### A verified calibration detail

The SoLEXS `.pi` file carries **86 400 × 340-channel spectra at 1 s** — far more
information than the light curve. Channels 0–40 hold the electronic noise peak.
We verified that the pipeline's own `.lc` equals the sum of channels 41–339
**exactly** (max absolute difference `0.0` over 86 376 bins), so the same
threshold is applied everywhere. That check runs on every ingest; if a future
release changes the convention it fails loudly rather than silently corrupting
every feature.

### Persistent instrumental lines — a ruler for the energy scale

`inspect --lines` sums the quietest 52 366 s and finds narrow lines that are
present at **constant strength during quiet Sun**:

| channel | net counts | significance |
|---:|---:|---:|
| 122.05 | 425 | 49.6σ |
| 134.65 | 71 | 7.5σ |
| 93.96 | 52 | 4.5σ |
| 246.10 | 30 | 8.5σ |

A line that does not vary with solar activity cannot be solar (Fe XXV at
6.7 keV appears only in flares) — these are instrumental: an onboard Fe-55
calibration source plus fluorescence. That makes them a ruler for the
channel-to-energy gain.

### The energy scale: first assumed wrong, now the published calibration

The first version of this pipeline assumed channel 41 = 1.0 keV and channel
339 = 22.0 keV (70.47 eV/channel). It chose the line pairing "ch 93.96 / 99.93
as Ti Kβ − Kα" because that pairing agreed with the assumption. **That was
wrong.** The published SoLEXS calibration ([Sarwade et al. 2025, JATIS 11(4)
045005](https://arxiv.org/abs/2509.26292)) is E = gain × channel + offset:
47.75 eV/channel up to channel 168, 94.5 eV/channel above, offset 86.66 eV for
SDD2, threshold ~2 keV. The strongest lines above then identify themselves:

| line | channel | published scale | old assumed scale |
|---|---:|---:|---:|
| Fe-55 Mn Kα 5.895 keV | 122.05 | 5.915 keV (+20 eV) | 6.747 keV — misidentified |
| Fe-55 Mn Kβ 6.490 keV | 134.65 | 6.516 keV (+26 eV) | — |
| Kβ − Kα spacing | 12.60 ch | 47.2 eV/ch (published 47.75) | — |
| pipeline threshold | 41 | 2.04 keV (published ~2 keV) | 1.0 keV |

`inspect --lines` now runs this Fe-55 check on the first archive day and reports
the residual, then tracks the strongest line's channel across six days for gain drift. The old "1–2 keV" band really covered ~2.0–2.7 keV, and
every band label was shifted.

The scale is `PreprocessConfig.solexs_energy_scale`:
- **`sarwade2025` (default).** Bands 2–3, 3–4, 4–6, 6–8, 8–12 and 12–24 keV.
- **`legacy_linear`.** Kept only to reproduce models trained on it: the 2024–2026 archive run in
  `outputs/archive` and forward-test `v1`. It reproduces their cached rates bit for bit (verified on
  2026-09-10). Pass `--energy-scale legacy_linear` to any command working on those outputs. Frozen
  forward-test models record their scale automatically.

---

## 3. Architecture: SoLEXHEL-Net

```
SoLEXS 24 feat ─┐
   + mask       ├─→ causal TCN encoder ─┐
                │                        │
                │                        ├─→ gated mask-aware fusion ─→ TCN trunk ─┐
                │                        │                                          │
HEL1OS 48 feat ─┤                        │                          causal attention pool
   + mask       └─→ causal TCN encoder ─┘                                          │
                                                                                    ▼
                            ┌───────────────┬──────────────┬──────────────┬────────────────┐
                        flare phase     P(flare now)    log-flux        forecast        P(flare
                        (4-class)                       nowcast      q10/q50/q90 at     within
                                                                   1/5/15/30/60 min   15/30/60 min)
                                                                              + time-to-peak
                                                                              + peak magnitude
```

**Every convolution is strictly causal** (left-padded only). This is not
stylistic: a symmetric kernel leaks a few future steps per layer, and stacked
across a dozen layers the model reads the answer it is supposed to forecast.

Key design choices and why:

| Choice | Reason |
|---|---|
| **Dilated TCN** (dilations 1→128, RF 541 steps > 360-step window) | Flare context spans tens of minutes; dilation reaches it without RNN sequentiality. |
| **Modality dropout (p=0.25)** | Randomly blanks a whole instrument during training, so one network handles soft-only, hard-only, and fused inputs. This is what makes non-overlapping data usable at all. |
| **Mask as an input channel** | Missing steps are zero-filled; the mask tells the network which zeros are measurements and which are absences. Without it, a dead link looks like a quiet Sun. |
| **Gated fusion conditioned on availability** | When HEL1OS is absent the gate routes the trunk to the soft stream instead of averaging in a zero tensor. |
| **Causal attention pooling** | Flares are localised; a mean over two hours dilutes the onset. The last step queries the window, which keeps it causal. |
| **Monotone quantile head** | q10 ≤ q50 ≤ q90 by construction (cumulative softplus), so intervals never cross. |
| **Learned per-task log-variance** | Five heads on different scales; hand-tuned weights are guesswork (Kendall & Gal). |
| **Focal loss** | Long quiet stretches are easy negatives that swamp plain BCE. |
| **Running *percentile* background** | A running mean over a window containing a flare is dragged up by the flare and subtraction erases the event. |
| **`ChannelNorm1d`, not `GroupNorm`** | GroupNorm pools statistics across the time axis and would break causality — see §6. |

461 503 parameters — deliberately small, because the dataset is one day.

---

## 4. Features

**SoLEXS (24 channels)** derived from the 340-channel spectra: six band rates
(2–3, 3–4, 4–6, 6–8, 8–12, 12–24 keV on the published energy scale; the archive
run used the legacy 1–2 … 12–22 keV labels, which were mislabelled — §2); GOES 1–8 Å and 0.5–4 Å analogues; total
rate; background-subtracted excess; ratio to background; five adjacent-band
hardness ratios; the GOES temperature ratio; counts-weighted mean energy;
causal Δlog-flux over 1/5/15 min; trailing log means over 5/30 min; coverage.

**HEL1OS (48 channels = 12 per detector × CZT1, CZT2, CdTe1, CdTe2)**: five
band rates (CZT 20–40, 40–60, 60–80, 80–150, 18–160 keV; CdTe 5–20, 20–30,
30–40, 40–60, 1.8–90 keV); three hardness ratios; wide-band ratio to a trailing
background; causal Δlog over 1/5 min; that detector's coverage.

Every feature is computed from **one detector's columns only**. CZT2 reads
0.9–1.5× CZT1 depending on the band (same seconds, 2026-09-13), so any feature
that let the two share a column would record which detector happened to be
sampled as solar variability. CdTe is the natural bridge to SoLEXS (2–22 keV):
its L1 files carry a 5–20 keV band, although the instrument paper gives CdTe a
nominal range of 8–70 keV, so that band effectively measures ~8–20 keV.

Rates go through `log1p` — flare flux spans decades, and in linear space the
loss is dominated by a handful of peak samples. Spectral *shape* (hardness,
mean energy) is kept separate from *amplitude*: it carries the plasma
temperature information that distinguishes a real onset from a rate bump, and
it stays informative even where the absolute calibration is uncertain.

---

## 5. Labels

No flare catalogue ships with these files, so labels are derived. The detector
follows operational GOES logic rather than a bare threshold (which either
drowns in noise at B-level or misses everything when the background is
elevated):

1. Running 10th-percentile background over ±1 h.
2. Candidate onset where rate ≥ 1.4× background **and** excess ≥ 4σ Poisson,
   sustained ≥ 120 s.
3. Walk back to 10% of peak excess for true onset; forward to 50% decay for end.
4. Merge events < 5 min apart; drop events < 5 min long.

Produces per-bin `phase` (quiet/rise/peak/decay), `in_flare`, `log_flux`,
time-to-peak, peak magnitude, and forward-looking occurrence labels.

Magnitude buckets are expressed as **excess-over-background ratio**, not
W/m², because these are counts with no absolute calibration. `MAGNITUDE_EDGES`
in `labels.py` is the single place to recalibrate against a real GOES catalogue.

---

## 6. Avoiding the ways this goes wrong

Five traps were found and closed while building this. Four are covered by
tests in `tests/test_correctness.py` (38 checks, all passing); the fifth was
caught by the ablation at evaluation time.

**Split leakage.** Windows overlap, so a random split puts near-duplicates in
train and test and yields a model that looks excellent and forecasts nothing.
Splits are chronological with a 1-hour embargo (≥ the longest horizon).

Critically, the split is applied **per segment, not across the pooled
timeline**. A global sort handed the entire hard-only HEL1OS day to the test
set, producing a test split with `in_flare_rate = 0.0` and every label masked —
metrics that would have looked fine and meant nothing.

**Normalisation over time.** `GroupNorm` on a `(B, C, T)` tensor pools
statistics across the **time** axis, so step *t* is rescaled using steps after
*t*. It is the natural choice and it silently destroys causality. Replaced with
`ChannelNorm1d`, which normalises per time step. Invisible in training curves;
caught only by an explicit perturbation test.

**Centred background as a feature.** The quiescent background is a running
10th percentile. A *centred* window is the right estimator for defining ground
truth, but using it as a model **input** leaks up to half a window of the
future into every sample. There are now two functions — `trailing_percentile`
for features, `running_percentile` for labels — and a test that perturbs the
tail of an observation and asserts no earlier feature moves.

**Background inflated by the flare itself.** With a 1-hour window, the
128-minute 17:00 event drove its own estimated background from 1.2 to 6 cts/s,
erasing most of its amplitude. The window is now 6 hours; background at that
peak settles at 0.80 cts/s, matching the observed pre-flare level.

**Time-of-day features memorised the day.** This one survived into a full
training run and was only caught by the ablation. `tod_sin`/`tod_cos` were
included to absorb spacecraft-periodic systematics. On a single day they
uniquely identify every sample, so the network used them as a lookup table:
with **both instruments blanked** it still reached AUC 0.82 on flare-in-progress,
while the full model fell from validation TSS 0.864 to test TSS 0.078 and the
occurrence heads dropped below AUC 0.5 (anti-correlated), because the memorised
clock mapping is wrong for held-out times. Disabling them (`use_clock = False`)
took `clock_only` to AUC 0.500 exactly, test TSS to 0.244, and nowcast R2 from
0.03 to 0.54. The `clock_only` ablation row now exists to catch any recurrence.

**Normalisation statistics.** Median/IQR (flare peaks are genuine outliers that
would otherwise set the scale), fitted on training windows only, frozen for
val/test/inference.

---

## 7. Evaluation

Accuracy is meaningless here — always predicting "no flare" scores 90%+ on a
quiet week. The reported metrics are the operational standard:

* **TSS** (True Skill Statistic) — insensitive to class balance; the primary
  model-selection criterion.
* **HSS**, POD, FAR, CSI, F1, full contingency table.
* **Brier score** and **BSS vs climatology** for probability quality.
* **Reliability curve** — a 30% forecast should verify 30% of the time.
* **Skill vs persistence *and* climatology** for every regression horizon,
  plus the forecast-observation correlation. Both references are required.
  Persistence is strong at short range but *weak* at long range, because it
  extrapolates a decaying flare — so beating it at +60 min is easy and means
  nothing. A model that beats persistence, loses to climatology, and correlates
  ~0 with truth has simply learned to predict the mean. That is exactly what
  happens here beyond ~15 min, and the report says so.
* **`clock_only` ablation** — both instruments blanked. Any skill remaining
  comes from non-instrument inputs, i.e. memorisation. Warns automatically
  above AUC 0.6.
* **Interval coverage** vs nominal, for the quantile forecasts.
* **Modality ablation** — rescore with each instrument masked off.

Operating thresholds are fitted on **validation** and held fixed for test; 0.5
is correct only for balanced classes, which these never are.

**A known degeneracy.** With flares roughly every two hours in this single day,
the 60-minute occurrence base rate is ~0.97 — almost every window has a flare
within an hour. BSS against climatology is therefore strongly negative at that
horizon and the metric is not informative. This is a property of a one-day
dataset, not of the model; it resolves with more data. It is reported rather
than hidden.

Full measured numbers: `outputs/reports/RESULTS.md`. Headline, test split
(362 windows, 19:48–23:00 UTC on 2026-09-10):

| task | result | verdict |
|---|---|---|
| Flare in progress | TSS 0.244, AUC 0.739, FAR 0.000 | works, conservative |
| Nowcast log-flux | R² 0.536, MAE 0.156 | works |
| Forecast +1 / +5 min | corr 0.83 / 0.54, beats climatology | works |
| Forecast +15 min | corr 0.21 | marginal |
| Forecast +30 / +60 min | corr ≈ 0, loses to climatology | **predicts the mean** |
| Flare occurrence within H | AUC 0.21–0.44 (below chance) | **fails** |
| Time-to-peak | MAE 106 min | **fails** |

Two heads genuinely do not work on this data, and it is worth being precise
about why rather than tuning until the numbers look better:

* **Occurrence** (`will a flare start within H?`). Test base rates are 0.44 /
  0.60 / 0.78, and 0.97 at 60 min on the earlier split — there is barely a
  negative class left to discriminate. Worse, the chronological split puts a
  quiet training stretch against a flare-clustered test stretch, so the
  train base rate (0.22–0.34) and test base rate differ by 2–3x. AUC below 0.5
  means the ranking is actively inverted under that shift. This needs more
  days, not a better model.
* **Time-to-peak** needs many labelled rise phases; this day supplies 9 `rise`
  windows in test. The `rise` class F1 is 0.000 for the same reason.

**The deep model does not beat gradient boosting here.** On flare-in-progress,
GBDT reaches TSS 0.345 against the network's 0.244 (the network has the better
AUC, 0.739 vs 0.636, so it ranks better but transfers its threshold worse).
That is the expected outcome at this data volume and is precisely why the
baselines are in the repo. A 461k-parameter sequence model needs more than one
day to justify itself over a tree on summary statistics.

---

## 8. Layout

```
solarflare/
  config.py              all instrument constants and hyperparameters
  io/
    solexs.py            .lc/.pi/.gti reader + channel-threshold verification
    hel1os.py            per-detector light curves (CZT + CdTe), band alignment,
                         per-slot fill masking, product grouping, housekeeping
    discover.py          walk a data root, find every observation (small samples)
  preprocess/
    cache.py             per-file preprocessing cache: index, dedupe, build, load
    timeline.py          stitch per-file series into continuous timelines
    grid.py              common time grid, mask-aware rebinning, chunked backgrounds
    features.py          spectra -> physically meaningful causal features
    labels.py            flare detection, phases, forecast targets
    dataset.py           segments, windowing, splits, normalisation
    events.py            event-centric rise-phase dataset, grouped by flare
  models/
    blocks.py            causal conv, SE, TCN block, attention pooling
    net.py               SoLEXHEL-Net (nowcasting)
    riseflare.py         RiseNet (rise-phase forecasting) + its loss
    ssm.py               selective state-space (Mamba-style) encoder, pure PyTorch
    zoo.py               swappable causal encoders: tcn/ssm/gru/transformer/
                         patchtst/linear
    losses.py            focal, pinball, masked Huber, multi-task weighting
  physics/
    onset.py             hot-onset hardness, causal onset, Neupert lag, GOES XRS ratio
  torch_data.py          Dataset wrapper
  pipeline.py            raw FITS -> loaders
  train.py               training loop
  evaluate.py            verification + ablation
  forecast.py            rise-phase grouped CV, reference forecasts, event bootstrap,
                         paired soft-only vs soft+hard fusion ablation
  forward.py             prospective test: freeze / predict new data / score (+ GOES)
  predict.py             streaming inference (same code path as live telemetry)
  baselines.py           climatology / persistence / logistic / GBDT
  calibration.py         instrumental-line energy-scale diagnostics
  metrics.py             TSS/HSS/POD/FAR/Brier/reliability/...
  plots.py               light curves, spectrogram, timeline, reliability
  report.py              renders RESULTS.md from the JSON reports
  cli.py                 cache / inspect / train / baselines / forecast / fusion /
                         forward-test / evaluate / predict / report
scripts/
  run_all.py             the whole pipeline in one command, one log per stage
  unzip_archive.py       safe, resumable extraction of PRADAN zips (CRC-checked)
  verify_extract.py      read-only check of an extraction against its zips
  posthoc_event_definitions.py  re-score a trained model on stricter flare definitions
  goes_crosscheck.py     SoLEXS vs GOES-18: flux calibration and label agreement
  fair_references.py     flux forecast vs "no change" in calibrated SoLEXS flux
  physics_catalog.py     per-flare hot-onset / Neupert catalogue + leakage-free test
  learning_curve.py      skill vs how much training history is used
  recalibrate.py         isotonic calibration of the probability heads
  fusion_seeds_summary.py  the HEL1OS ablation across seeds and encoders
  dashboard_export.py    replay one day through a frozen model
  dashboard_build.py     bake that replay into the Flare Watch page
  share_card.py          one summary image of the results
  stay_awake.py          hold a power request while a long run is in progress
  chains/                the multi-hour run queues, one per experiment (see its README)
tests/
  test_correctness.py    the science: causality, leakage, masking, splits
  test_robustness.py     the real world: bad files, failure paths, every CLI path
  test_scale.py          archive runs: cache, stitching, dedupe, calendar split, thinning
  test_extract.py        extraction: corrupt zips, traversal, disk floor, shared date trees
  test_hel1os.py         HEL1OS archive: band alignment, zero counts, detectors, versions
  test_forward.py        forward test: frozen normaliser, cutoff, append-only, GOES truth
  test_goes.py           GOES truth: flare list, targets, thresholds, scoring masks
  test_physics.py        hot onset, Neupert lag, and no leakage at the decision time
dashboard/
  flare_watch.template.html   the replay console, data baked in at build time
outputs/                 one folder per experiment -- see outputs/README.md
  archive/               first run (SoLEXS labels) + the shared 200 MB cache
  archive_goes/          main GOES-labelled run: CONCLUSIONS.md, v2
  archive_goes_anchor/   anchored-flux retrain: best forecasts, v3
  archive_goes_patchtst/ PatchTST cross-validation
  physics/               hot-onset / Neupert catalogue, PHYSICS.md
  dashboard/  share/     built pages and images
```

## 9. Tests

```bash
python -m tests.test_correctness
```

```bash
python -m tests.test_robustness
```

**Eight suites, all run in CI** (`.github/workflows/ci.yml`) alongside `ruff`
and `pyflakes`. Counts below are checks per suite.

* `test_correctness.py` (58) - the science. Causality of every layer, the
  end-to-end feature matrix, HEL1OS fill-row detection, mask-aware rebinning
  and row-level coverage, background robustness, target alignment, split
  hygiene with embargo, and the metric implementations.
* `test_robustness.py` (40) - the real world. Missing `.lc`/`.gti`, absent
  detectors, zero-row GTIs, all-fill HEL1OS files, truncated FITS, empty data
  trees, failed figure writes, a full synthetic-data run of **every** CLI
  path including `predict`, the rise-phase forecasting module end to end on
  synthetic flares across all five encoders, and a GPU exit-code guard for the
  cuDNN GRU crash below (skipped on machines without CUDA).
* `test_scale.py` (31) - the archive. Incremental cache builds and correct
  invalidation, version de-duplication, `.zip.part` exclusion, zip == extracted
  reads, chunked percentiles == exact, a flare crossing midnight detected once
  (and shown to be split without stitching), the calendar split's embargo, and
  quiet-window thinning that never drops a flare window, the sliding-window
  percentile == `np.nanpercentile`, and the flare bootstrap unchanged but fast
  at archive scale.
* `test_extract.py` (21) - extraction into a live download folder. CRC
  verification, corrupt and truncated zips, path traversal, in-progress
  `.part` files, the free-disk floor, and the HEL1OS layouts that nearly
  deleted data: a shared `YYYY/MM/DD` tree, a zip holding two products, and
  one product shipped in two zips.
* `test_hel1os.py` (27) - the HEL1OS archive. CdTe bands of different lengths and
  sub-second phases all kept, zero-count samples kept as zeros, CZT1/CZT2
  never mixed, an unknown detector is an error, the newer processing version
  wins, one source per product, cache → features end to end.
* `test_forward.py` (26) - the forward test. The frozen normaliser is the
  training one bit for bit (and a refit on grown data provably differs), an old
  checkpoint refuses to freeze after the data changed, frozen models are never
  overwritten, forecasts exist only after the cutoff and re-running never
  rewrites them, unsettled truth is held back, and GOES/HEK/SWPC flare lists parse.
* `test_goes.py` (30) - GOES truth. The NOAA flare summary is read as written (epoch, missing ends),
  flagged fluxes are never truth, events are exactly the GOES list, targets are log10 W/m², the
  large-flare threshold is M1.0, and persistence is scored only where its origin value is real.
* `test_physics.py` (19) - hot onset and Neupert diagnostics. A decaying pre-flare background is
  followed but a rising one never extrapolated, the causal onset is known only at its last bin,
  the Neupert lag has the right sign, and decision-time features ignore every later sample.

That second suite exists because defects kept reaching "done" without it: a
failed PNG write destroyed a completed training run, `predict` shipped having
never been executed, and every run containing a GRU exited with
`0xC0000409` after finishing its work.

**The GRU crash, for anyone who meets it elsewhere.** `nn.GRU(num_layers=2,
dropout=0.1)` on CUDA kills the process at interpreter exit with
`STATUS_STACK_BUFFER_OVERRUN` on Windows + torch 2.14 cu130. It reproduces in
a ten-line script with nothing else loaded, and not with dropout 0 or a single
layer, so the trigger is cuDNN's RNN dropout state. `GRUEncoder` stacks
single-layer GRUs with `nn.Dropout` between them instead, which is
mathematically the same regularisation.

## 10. Commands

Everything, in order, with a log per stage (a failing stage is recorded and the
rest still run):

```bash
python scripts/run_all.py
```

Or stage by stage:

```bash
python -m solarflare.cli inspect                                   # survey the data
python -m solarflare.cli inspect --lines                           # energy-scale diagnostics
python -m solarflare.cli train --epochs 60                         # nowcast: train + evaluate + plot
python -m solarflare.cli baselines                                 # climatology/persistence/GBDT
python -m solarflare.cli forecast --encoders tcn,ssm,gru,transformer,linear --folds 3
python -m solarflare.cli fusion --encoder tcn --folds 3            # does HEL1OS add skill?
python scripts/posthoc_event_definitions.py --data-root D:/Data --out-dir outputs/archive  # skill on >=3x/10x/30x flares
python -m solarflare.cli report                                    # render RESULTS.md
python -m solarflare.cli evaluate --checkpoint outputs/checkpoints/best.pt
python -m solarflare.cli predict --output outputs/reports/predictions.csv
python -m solarflare.cli train --dt 10 --window 3600 --epochs 40   # finer grid
```

`predict` runs the model along the observation using only the trailing window at
each step — the same code path a live telemetry buffer would use.

### Forward test: forecasting data that did not exist yet

A held-out test period is still picked after the data exist, and every choice
made while looking at test scores leaks into it. The one test nothing can have
been tuned on is data downloaded **after** the model was fixed.

Freeze once, straight after training and before downloading anything new:

```bash
python -m solarflare.cli forward-test freeze --name v1 --data-root "D:/Data" --out-dir outputs/archive
```

Forecast whenever new days have been downloaded. It is append-only and only
writes forecasts after the data cutoff:

```bash
python -m solarflare.cli forward-test predict --name v1 --data-root "D:/Data" --out-dir outputs/archive
```

Score whatever has become verifiable, optionally against an independent GOES
flare list:

```bash
python -m solarflare.cli forward-test score --name v1 --data-root "D:/Data" --out-dir outputs/archive --goes-events goes_flares.json --min-class C1.0
```

What `freeze` pins, and why:

| Pinned | Why |
|---|---|
| Weights + SHA-256 | Every forecast row carries the hash; a later model cannot pass as the frozen one. |
| **Input normaliser** | `prepare` refits it from the training split of whatever is on disk. Once new days arrive the split moves and every input is scaled differently. Checkpoints now store it; an older checkpoint may only be frozen if the split is verified identical to training. |
| Decision thresholds | Chosen on validation at freeze time, never re-tuned on forward data. |
| Training climatology | Base rates for the reference forecasts, known at freeze time. |
| Data cutoff | The last observed bin. Forecasts exist only for later moments. |

`score` holds back forecasts whose truth is not final yet. That takes 3 h for
the centred background plus the longest horizon. It reports TSS with day-block
bootstrap intervals, Brier skill against training climatology, and flux skill
against persistence.

With `--goes-events` it also checks the forecasts against an independent flare
list (NOAA SWPC JSON, a HEK export, or CSV) and reports how many GOES flares
our SoLEXS detector found. Without that list, the truth comes from this
project's own detector.

## 11. Running on the mission archive

Point `--data-root` at the download folder. The PRADAN zips are read
**directly** — do not unzip them; the members are already gzip-compressed
FITS, so extracting would only duplicate GBs on disk.

```bash
python -m solarflare.cli cache --data-root "D:/Data" --out-dir outputs/archive
```

```bash
python scripts/run_all.py --data-root "D:/Data" --out-dir outputs/archive
```

What changes at scale, and why:

| Concern | What happens |
|---|---|
| **Memory.** A SoLEXS day is ~470 MB of spectra and peaks ~1.3 GB to read; the original design held every day in RAM and would fail near 100 days on a 24 GB machine. | Stage 1 runs **once per file** and writes ~100 KB of gridded band rates to `out-dir/cache/` (`preprocess/cache.py`). The whole mission is ~100 MB. |
| **Re-runs while downloading.** | The cache is keyed on file path, size, mtime *and* every raw-rate setting. Finished files are skipped, `*.zip.part` files are ignored, a changed file or setting rebuilds only what it affects, and a corrupt zip is recorded in `manifest.json` and retried next run — never fatal. |
| **Duplicates.** PRADAN re-processes days (v1.0 → v1.1). | Only the highest version of each day is used, so no flare is counted twice. |
| **Midnight.** Backgrounds, trailing means and flare detection are window operations; run per day they restart at 00:00 and a flare decaying past midnight is cut in two. | Days are **stitched** into continuous timelines first (`preprocess/timeline.py`); gaps up to 6 h stay inside a segment as unobserved bins. `tests/test_scale.py` demonstrates the per-day answer is wrong. |
| **Rolling percentiles.** An unchunked 6 h window over a nine-month segment needs ~11 GB, and re-sorting 1080 samples per step is slow. | Short series: fixed-size chunks of `np.nanpercentile`. From 50 000 rows: a sliding sorted window, ~30× faster. Checked against numpy on a 62-day SoLEXS segment: identical NaN pattern, worst relative difference 3.5e-16. |
| **Split leakage.** Splitting each of hundreds of segments 60/20/20 interleaves train and test in calendar time. | Above `large_data_days` (30) the split becomes **one calendar cut** with an embargo. Train and test then sit at different phases of the solar cycle — test skill includes that shift. |
| **Training cost.** ~2 M windows at a 40 s stride. | Archive mode: 120 s stride, quiet training windows thinned to a 600 s stride (every flare-adjacent window kept; **val/test never thinned**), 400 random batches per epoch. |
| **HEL1OS-only time.** Its phase labels are placeholders (no soft X-ray truth). | Phase loss and class-balance statistics now mask it; before, it was trained as "quiet Sun". |

Every value applied is printed at the start of the run and recorded in
`reports/data_meta.json` under `archive_profile`.

**Instruments.** The PRADAN SoLEXS download contains SDD2 spectra only; SDD1
is powered down in every day checked.

### HEL1OS on the archive

The pipeline reads HEL1OS from **extracted** products (unlike SoLEXS zips).
Light curves are all it needs:

```bash
python scripts/unzip_archive.py --data-root "D:/Data" --instrument hel1os --members lightcurves
```

```bash
python scripts/verify_extract.py --data-root "D:/Data"
```

Coverage as downloaded (2026-09-14): 343 products, 111.9 observed days, in
Apr–Jun 2024 and Jun–Sep 2026. **76.3 days are simultaneous with SoLEXS**
(2026-07: 28.4 d, 2026-08: 24.5 d), and HEL1OS watched at least half the rise
of **872 of the 8 944** flares detected in SoLEXS.

What the real products required, each covered by `tests/test_hel1os.py`:

| Found in the archive | Handling |
|---|---|
| Each file holds one detector, and CZT1/CZT2 share band names; the old per-file layout stitched them into the same columns although CZT2 reads 0.9–1.5× CZT1. | One source per **product**; every detector (CZT1, CZT2, CdTe1, CdTe2) has its own columns and coverage in a fixed 24-column raw layout. |
| CdTe band extensions have different lengths and sub-second phases; the old reader silently dropped every band of a different length. | Bands are mapped onto the longest band's 1 s grid by nearest slot. |
| `STAT_ERR` is √counts, so a zero-count sample is written like "no telemetry". Deciding per band inflated faint CdTe bands up to 10×. | A slot is sampled if *any* band of the detector reports; zeros in the other bands are then real zeros. The residual bias (a slot where every band read zero) keeps the quiet CdTe wide band 3–15% high on typical days, up to ~2× on the faintest. Negligible during flares. |
| The same interval ships as V111 and V211, or V211 and V212; overlapping products carry identical telemetry. | Stitching gives the higher processing version priority in every bin it covers, then better coverage. |

**Does HEL1OS help?** `python -m solarflare.cli fusion` answers it directly.
It is restricted to the flares HEL1OS observed. The same encoder is
cross-validated twice over identical folds and seeds, once with the hard
X-rays and once with them blanked, and the per-flare difference in peak
error is bootstrapped. Two MAEs quoted side by side can overlap even when
one arm wins on the same flares; the paired difference cannot hide that.

The nowcast model's calendar split puts nearly all simultaneous data (Jun–Sep
2026) in the **test** period. Its modality ablation therefore measures fusion
out of time, trained on only ~4 simultaneous days from 2024, which is why the
fusion ablation above is the primary evidence.
#   a d i t y a 1 - f l a r e - f o r e c a s t  
 