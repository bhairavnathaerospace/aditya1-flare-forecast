# What is in `outputs/`

One folder per experiment. Each holds `reports/` (JSON + Markdown + logs),
`checkpoints/`, `figures/`, and where a model was frozen, `forward/`.

| Folder | What it is | Labels / model | Keep because |
|---|---|---|---|
| `archive/` | First full-archive run, 2026-09-14 | SoLEXS detector labels, legacy energy scale | The baseline the GOES run is compared against; also holds the **shared preprocessing cache** (200 MB) every run reads |
| `archive_goes/` | Main run, 2026-09-15 | GOES ≥C1.0 classes, published energy scale | The headline results and `CONCLUSIONS.md` |
| `archive_goes_anchor/` | Anchored-flux retrain, 2026-09-16 | Same, flux heads predict "calibrated SoLEXS flux now + a change" | Best flux forecasts; frozen as **v3** |
| `archive_goes_patchtst/` | PatchTST rise-phase CV, 2026-09-16 | Same folds and seeds as the other encoders | The negative result on PatchTST |
| `physics/` | Hot-onset and Neupert catalogue over 5 925 flares | — | `PHYSICS.md` and the per-flare CSVs |
| `dashboard/` | Flare Watch replay data and the built page | Model v2 | Rebuilt by `scripts/dashboard_*.py` |
| `share/` | Summary image for posting | — | Rebuilt by `scripts/share_card.py` |
| `_early_two_day_sample/` | The original two-day-sample outputs from before the archive existed | — | Historical only; safe to delete |

**Frozen models** (weights + normaliser + thresholds + climatology + data cutoff):

| Model | Where | Labels | Note |
|---|---|---|---|
| v1 | `archive/forward/v1` | SoLEXS detector, legacy scale | Superseded; `_superseded/` holds the first freeze, whose climatology came from a thinned training set |
| v2 | `archive_goes/forward/v2` | GOES classes | Drives the dashboard |
| v3 | `archive_goes_anchor/forward/v3` | GOES classes, anchored flux | **Current best**; use this for forward tests |

**Regenerable, delete freely:** `dashboard/cache/`, `share/`, any `figures/`.
**Do not delete:** `archive/cache/` (200 MB, ~4 h to rebuild) and any `reports/`.
