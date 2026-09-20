# Solar flare nowcasting and forecasting from Aditya-L1 (SoLEXS + HEL1OS)

A deep-learning system for ISRO Problem Statement 15. It reads Aditya-L1 soft X-ray (SoLEXS) and hard
X-ray (HEL1OS) Level-1 products and does four things:

- detects flares in each instrument and merges the two lists into one catalogue;
- nowcasts whether a flare is in progress and how bright it is;
- forecasts the next 1–60 minutes (flux, flare occurrence, peak) with a measured lead time;
- raises alerts at a chosen false-alarm rate, shown in a desktop console.

GOES-18 XRS is the reference truth for training and scoring only. It is never a model input.

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu130   # GPU build (see requirements.txt)
# set data paths in config/project.toml, then:
python -m solarflare pipeline            # the whole study into outputs/, resumable
```

Results land in **`outputs/RESULTS.md`**. The page is generated from the measured JSON reports, so it
cannot drift from what the code produced.

---

## What the problem statement asks, and where it is answered

| Asked for | Answered by | Output |
|---|---|---|
| Flare detection in SoLEXS and in HEL1OS | Two independent detectors: a SoLEXS rise rule and HEL1OS coincidence bursts. Each is scored against the GOES flare list, with a time-shifted chance control. | `outputs/catalog/` |
| A merged flare catalogue | Soft and hard detections are merged into one list, with GOES class estimated from SoLEXS alone. | `outputs/catalog/master_catalog.csv` |
| A forecasting model with quantified lead time | SoLEXHEL-Net (below) forecasts 1–60 min ahead. The alert study reports the lead time before each flare's peak, and before GOES first reaches M1. | `outputs/alerts/LEADTIME.md` |
| High TPR, low FAR | Every alert threshold is fixed on the validation period at a set false-alarm rate (C: 2/day, M: 0.5/day) and scored once on the test period. Hit rates are shown next to a chance baseline (the same windows moved 2 h). | `outputs/alerts/alert_rules.json` |
| An interface with light curves and visual alerts | The desktop console's *Flare Watch* tab replays the frozen model day by day: light curves, calibrated probabilities, alert shading. | `Training Console.exe` |
| Hours-ahead forecasting | Day-ahead models (2–24 h) are built from SoLEXS activity, from SDO/HMI SHARP magnetic parameters, and from both, all compared on the same hours. | `outputs/dayahead/DAYAHEAD.md` |

---

## Layout

```
config/project.toml        every path and model setting; nothing else in the code names a folder
solarflare/                the package (python -m solarflare <command>)
  io/                      readers: SoLEXS, HEL1OS (light curves + event lists), GOES-R XRS, SHARP
  preprocess/              cache, gridding, features, labels, windows, chronological splits
  models/                  SoLEXHEL-Net and the encoder zoo used in the ablations
  train.py evaluate.py     training (smoothed model selection, per-part gradient clipping), test metrics
  forward.py probcal.py    freezing a model with its normaliser, calibration and thresholds
  catalog/                 the two detectors, the merge, the GOES scoring, the overview figure
  products/                alerts (lead time), dayahead, references, calibration report, HEL1OS
                           value, hard X-ray spectra and timing, temperatures, onset study
  runall.py summary.py     the pipeline runner and the one-page results
  settings.py cli.py       project.toml -> settings; the command line
dashboard/                 the desktop console (Tkinter) and its detached job runner
scripts/                   data chores: unzip PRADAN archives, verify them, data status, SHARP download
tests/                     11 suites (below)
outputs/                   everything the pipeline writes (reports, catalogues, figures and the frozen
                           model are kept in git; predictions, checkpoints and logs are rebuilt)
cache/                     preprocessed products, rebuilt only for new files (not in git)
```

`outputs/` after a full run:

```
outputs/
  quality/      copied SoLEXS days, masked in training
  model/        the network: reports, figures, checkpoint
    forward/final/   the frozen model: weights, normaliser, calibration, thresholds, SHA-256
  ablations/    sharp/ (with vs without SHARP inputs), hel1os/seed_*/ (paired soft vs soft+hard)
  catalog/      master catalogue and its scores
  alerts/       minute-by-minute predictions, lead times, alert rules, the Flare Watch replay
  dayahead/     2–24 h forecasts
  physics/      temperatures, hard X-ray spectra and timing, hot onsets, Neupert effect
  pipeline/     state.json and one log per stage
  RESULTS.md    one page over all of it
```

---

## Data

| Source | What | Where (project.toml) |
|---|---|---|
| PRADAN (ISSDC) | SoLEXS L1 day files (340-channel spectra, 1 s) and HEL1OS L1 products (light curves, event lists), as downloaded zips | `data_root` |
| NOAA NCEI | GOES-18 XRS L2 flare summary (`xrsf-l2-flsum`) and 1-minute flux (`xrsf-l2-avg1m`), used as truth | `goes_dir` |
| JSOC | SDO/HMI SHARP keywords (`hmi.sharp_cea_720s`), hourly, numbers only: `python scripts/download_sharp.py` | `sharp_dir` |

```bash
python scripts/unzip_archive.py --data-root D:/Data --instrument solexs      # never deletes a zip
python scripts/unzip_archive.py --data-root D:/Data --instrument hel1os --members all
python scripts/verify_extract.py --data-root D:/Data
python scripts/data_status.py --data-root D:/Data                             # what is there, what is missing
```

Extraction stops before the disk falls below `min_free_gb`. A product split across two zips is
extracted once. A half-downloaded file is skipped, never read.

---

## The pipeline

`python -m solarflare pipeline` runs each stage as `python -m solarflare <command>` in its own
process. State is kept in `outputs/pipeline/state.json`, so an interrupted run resumes where it
stopped. `--list` shows the state of every stage. `--redo <stage>` reruns that stage and everything
built on it. A frozen model is never overwritten; the old one is moved aside.

| Stage | Command | Does |
|---|---|---|
| cache | `cache` | Grids every SoLEXS/HEL1OS product once (only new files are read). |
| quality | `quality` | Finds PRADAN SoLEXS day files that repeat the previous day. |
| train | `train` | Trains the network on X-ray inputs. |
| train-sharp | `train --sharp` | The same network with SHARP inputs (only when `use_sharp = "auto"`). |
| choose-inputs | — | Keeps SHARP only if it raises the validation score by `sharp_min_gain`. The test period plays no part. |
| baselines, calibration, references | `baselines`, `calibration-report`, `references` | Classical baselines, probability calibration, and "no change" flux references. |
| freeze | `forward-test freeze --name final` | Fixes the weights, normaliser, calibration and thresholds, all fitted on validation. |
| catalog, catalog-figure | `catalog`, `catalog-figure` | The master catalogue, scored against GOES. |
| alerts-predict(-nohard), alerts | `alerts --predict [--blank-hard]`, `alerts` | Frozen-model predictions at 1-min stride, lead time vs false alarms, the HEL1OS-hidden ablation, alert rules. |
| dayahead | `dayahead` | 2–24 h forecasts: SoLEXS activity, SHARP, and both. |
| hel1os-seed-N, hel1os-value | `fusion --seed N`, `hel1os-value` | Paired soft-only vs soft+hard ablation, repeated over seeds. |
| temperature, hxr-spectra, hxr-timing, onset-study | same names | Flare physics from both instruments. |
| report, summary | `report` | `outputs/model/reports/RESULTS.md`, then `outputs/RESULTS.md`. |

Before a long run, `config/rehearsal.toml` runs every stage on the real data with one training epoch
(about 2.5 h), so a wiring mistake shows up early:

```bash
SOLARFLARE_CONFIG=config/rehearsal.toml python -m solarflare pipeline --keep-going
```

Any command also runs on its own. `python -m solarflare <command> --help` lists its options. The
model commands also take `--set section.key=value`, which overrides any setting for one run (for
example `--set train.balance_head_gradients=false`). Commands that act on a trained run (evaluate,
freeze, predict, report, baselines) reload that run's own `reports/config.json`. A later edit to
`project.toml` therefore cannot change the inputs a saved model is fed.

---

## The network: SoLEXHEL-Net

```
SoLEXS 24 feat (+12 SHARP) + mask ─→ causal TCN encoder ─┐
                                                          ├─→ gated, mask-aware fusion ─→ TCN trunk ─→ causal attention pool
HEL1OS 48 feat + mask ─────────────→ causal TCN encoder ─┘                                                   │
          ┌──────────────┬───────────────┬─────────────────────┬─────────────────────────┬─────────────────────┘
      flare phase     P(flare now)    GOES flux now      flux q10/q50/q90 at         P(>= C1 flare within
      (4 classes)                     (anchored)         +1/5/15/30/60 min           15/30/60 min), peak size/time
```

The input is a 2-hour window on a 20 s grid (360 steps). The network is small on purpose (~0.46 M
parameters).

| Choice | Why |
|---|---|
| Every convolution is causal (left-padded only) | A symmetric kernel leaks a few future steps per layer; stacked over a dozen layers, the model would read the answer. |
| Modality dropout, mask as an input channel | The two instruments overlap on a minority of days. One network handles soft-only, hard-only and fused input, and the mask tells a dead link from a quiet Sun. |
| Gated fusion conditioned on availability | When HEL1OS is absent, the trunk is routed to the soft stream instead of averaging in zeros. |
| Anchored flux heads | Flux = calibrated SoLEXS flux now + a learned change. The anchor is refitted on the training period of every run. |
| Monotone quantile head | q10 ≤ q50 ≤ q90 by construction. |
| Learned per-task loss weights, focal loss | Five heads on different scales; long quiet stretches would otherwise swamp the loss. |
| `ChannelNorm1d`, not `GroupNorm` | GroupNorm pools statistics over time and breaks causality. |

**Features.**
- **SoLEXS (24):** band rates on the published energy scale (Sarwade et al. 2025); GOES 1–8 Å and
  0.5–4 Å analogues; excess and ratio over a *trailing* background; hardness ratios; the GOES
  temperature ratio; mean energy; causal Δlog-flux; coverage.
- **HEL1OS (48 = 12 × CZT1, CZT2, CdTe1, CdTe2):** band rates, hardness ratios, ratio to background,
  Δlog. Each feature comes from one detector only, because the detectors disagree by 0.9–1.5×.
- **SHARP (12, optional):** disk totals and maxima of unsigned flux, current helicity, free energy,
  R, shear and area. Only regions within 70° of the central meridian are used. Values are as an
  operator would have had them: the latest hour ≤ t − 2 h, with a freshness flag.

**Labels.** GOES-18 flare list ≥ C1.0 and 1-minute XRS-B flux. Targets: phase, in-flare, log flux
now, flux at each horizon, occurrence within 15/30/60 min, and peak size and time.

---

## Guarding against the usual ways this goes wrong

- **Leakage through the split.** Splits are chronological and global, with a 1 h embargo (at least
  the longest horizon). Every product reads the split dates from the trained run, never from code.
- **Leakage through features.** Backgrounds used as inputs are trailing, never centred. HEL1OS
  readout smoothing is a trailing mean. Tests perturb the future and assert that no earlier feature
  moves.
- **Thresholds tuned on the test period.** All thresholds, calibrations and the SHARP decision use
  validation only. The test period is scored once.
- **Probabilities that are not probabilities.** Occurrence heads are calibrated by isotonic
  regression on validation. The frozen model stores the knots, and its thresholds refer to
  calibrated values.
- **A lucky epoch.** Model selection and early stopping follow a trailing mean of the validation
  score over 3 epochs. One earlier run picked a noisy epoch-5 spike on the raw score and stopped.
- **One head drowning the others.** Gradients are clipped per network part. The peak head's gradient
  had been about 10× the rest.
- **HEL1OS readout batching.** Events arrive in batches of about 500–650, one every 2–8 s. Rates
  are averaged over 60 s (trailing) before use.
- **Copied data.** Four PRADAN SoLEXS day files (2024-07-26, 2024-09-01, 2024-10-02, 2025-02-20)
  repeat 6–12 h of the previous day under their own date. `quality` finds them, with GOES as
  referee, and training masks them.
- **Unfair baselines.** Every method is compared at the same false-alarm rate. Hit rates are shown
  with their chance level, and gains with day- or week-block bootstrap intervals.
- **One seed.** The HEL1OS gain is quoted across seeds, with a verdict that says "mixed" when the
  seeds disagree.

---

## The console

`Training Console.exe` in the project folder (or `pythonw dashboard/mission_control.pyw`) is a
desktop window that refreshes every 5 s.

- **Pipeline:** every stage with its state (done, running, failed, needs redoing) and its duration.
- **Training:** epoch progress, losses, validation skill, GPU, and a diagram of the network lit by
  gradient size per part.
- **Flare Watch:** the frozen model replayed one UTC day at a time. It shows three panels:
  - SoLEXS flux in GOES units, with GOES for comparison and the GOES flare list;
  - the calibrated flare probability, with the C alert;
  - the M signal, with the M alert.

  Each value is drawn at the minute it became known. Days are marked as validation or test.
- **Actions:** run the full pipeline, redo a stage, check or extract data, update the cache, run the
  tests. Jobs run detached (closing the window does not stop them) and keep the laptop awake.
- **Terminal and watchdog:** job, stage and training logs. The watchdog flags failed stages, a
  stalled or diverging run, an idle or hot GPU, and low disk.

Rebuild the exe after changing `dashboard/`:

```bash
python -m PyInstaller --onefile --windowed --name "Training Console" --icon dashboard/console.ico --paths . --paths dashboard --collect-submodules console --exclude-module torch --exclude-module scipy --exclude-module pandas --exclude-module sklearn --exclude-module astropy dashboard/mission_control.pyw
```

---

## Tests

```bash
python -m tests.test_correctness    # causality, leakage, masking, splits, metrics
python -m tests.test_robustness     # bad files, failure paths, the full CLI on synthetic data
python -m tests.test_scale          # cache, stitching, archive splits, zip reading
python -m tests.test_extract        # corrupt zips, path traversal, disk floor, HEL1OS layout
python -m tests.test_hel1os         # band alignment, zero counts, detector separation, versions
python -m tests.test_forward        # frozen normaliser, cutoff, append-only forecasts, GOES truth
python -m tests.test_goes           # flare summary reader, labels, targets
python -m tests.test_physics        # background, causal onset, Neupert lag
python -m tests.test_catalog        # soft rule, HEL1OS bursts, calibration, merge
python -m tests.test_products       # lead time, HXR spectra, timing, temperature, day-ahead
python -m tests.test_pipeline       # settings, calibration, SHARP, smoothing, clipping, runner, summary
```

CI (`.github/workflows/ci.yml`) runs lint and all eleven suites on Python 3.12 and 3.13, with CPU
torch.

---

## Limits worth knowing

- SoLEXS and HEL1OS overlap on a minority of days. Every hard X-ray result rests on that overlap,
  and the reports quote their sample sizes.
- HEL1OS readout batching means the L1 event lists resolve seconds, not milliseconds, below about
  2 000 events/s.
- Definitive SHARP data are published weeks after observation. The latest month may be missing, and
  the network then holds the last values with a "stale" flag.
- GOES is the truth, so a flare GOES misses counts as a false alarm. The catalogue checks HEL1OS-only
  events against SoLEXS 6–12 keV to show that some are real.
