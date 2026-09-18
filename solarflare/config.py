"""Central configuration for the Aditya-L1 soft+hard X-ray flare model.

Every number that encodes an instrument convention or a scientific choice lives
here so it can be audited and overridden, rather than being buried in code.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections.abc import Sequence
import json

# --------------------------------------------------------------------------
# Instrument conventions
# --------------------------------------------------------------------------

# SoLEXS L1 PHA-II files carry DETCHANS = 340 PI channels.  The pipeline's own
# light curve (.lc) is exactly the sum of channels 41..339 -- verified against
# this dataset -- because channels 0..40 hold the electronic noise peak.  We
# adopt the same lower threshold.
SOLEXS_N_CHANNELS = 340
SOLEXS_CH_LO = 41
SOLEXS_CH_HI = 340  # exclusive

# Channel -> energy.  Two scales, selected by PreprocessConfig.solexs_energy_scale.
#
# "sarwade2025" (default): the published SoLEXS calibration, Sarwade et al. 2025,
#   J. Astron. Telesc. Instrum. Syst. 11(4) 045005 -- E = gain * channel + offset,
#   47.75 eV/channel for channels 1-168 and 94.5 eV/channel for 169-340 (adaptive
#   binning), offset 86.66 eV for SDD2 and -33.88 eV for SDD1, lower threshold
#   ~2 keV.  The upper segment is taken as continuous at channel 168, which is
#   what the 340 channels spanning 2-24 keV require.
#   Independently confirmed on this archive: the two strongest quiet-Sun lines
#   sit at channels 122.05 and 134.65; as the onboard Fe-55 source's Mn Ka
#   (5.895 keV) and Kb (6.490 keV) they imply 0.0472 keV/channel, and this scale
#   places them at 5.914 and 6.516 keV -- within 30 eV, a sixth of the 170 eV
#   resolution. It also puts the pipeline's channel-41 threshold at 2.04 keV,
#   matching the published ~2 keV.
#
# "legacy_linear": what this project assumed before finding that paper --
#   channel 41 at 1.0 keV, channel 339 at 22.0 keV, 70.47 eV/channel. It is
#   wrong (every band edge was mislabelled; "1-2 keV" is really ~2.0-2.7 keV),
#   and is kept only so models trained with it stay reproducible: the
#   2024-02 -> 2026-09 archive run and forward-test "v1" were trained on it.
SOLEXS_CAL_SARWADE2025 = {
    "gain_lo_keV": 0.04775,
    "gain_hi_keV": 0.0945,
    "break_channel": 168,
    "offset_keV": {"SDD1": -0.03388, "SDD2": 0.08666},
}
SOLEXS_LEGACY_GAIN_KEV = (22.0 - 1.0) / (339 - 41)  # ~0.07047 keV / channel
SOLEXS_LEGACY_OFFSET_KEV = 1.0 - SOLEXS_LEGACY_GAIN_KEV * SOLEXS_CH_LO
SOLEXS_ENERGY_SCALES = ("sarwade2025", "legacy_linear")

# Soft X-ray sub-bands (keV) integrated from the 340-channel spectra, per scale.
# The published scale starts at ~2 keV and reaches ~24 keV.
SOLEXS_BANDS_BY_SCALE: dict[str, Sequence[tuple[float, float]]] = {
    "sarwade2025": ((2.0, 3.0), (3.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 12.0), (12.0, 24.0)),
    "legacy_linear": ((1.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 12.0), (12.0, 22.0)),
}

# GOES 1-8 Angstrom == 1.55-12.4 keV.  Used to build a GOES-analogue channel;
# SoLEXS sees only its part above the ~2 keV threshold.
GOES_LONG_KEV = (1.55, 12.4)
# GOES 0.5-4 A == 3.1-24.8 keV, clipped to the SoLEXS upper bound.
GOES_SHORT_BY_SCALE: dict[str, tuple[float, float]] = {
    "sarwade2025": (3.1, 24.8),
    "legacy_linear": (3.1, 22.0),
}

# Back-compatible names for the legacy scale (used by old notebooks and tests).
SOLEXS_BANDS_KEV = SOLEXS_BANDS_BY_SCALE["legacy_linear"]
GOES_SHORT_KEV = GOES_SHORT_BY_SCALE["legacy_linear"]

# HEL1OS CZT1 hard X-ray bands as written by the L1 pipeline (keV).
HEL1OS_BANDS_KEV: Sequence[tuple[float, float]] = (
    (20.0, 40.0),
    (40.0, 60.0),
    (60.0, 80.0),
    (80.0, 150.0),
    (18.0, 160.0),  # wide band, kept last
)

HEL1OS_CDTE_BANDS_KEV: Sequence[tuple[float, float]] = (
    (5.0, 20.0),
    (20.0, 30.0),
    (30.0, 40.0),
    (40.0, 60.0),
    (1.8, 90.0),  # wide band, kept last
)

#: Every HEL1OS detector the pipeline reads, in model-input order, with the
#: product subfolder its light curve lives in and its band layout.
#:
#: Each detector keeps its own columns. The two CZTs are nominally identical,
#: but on the mission archive CZT2 reads 0.9x-1.5x CZT1 depending on the band
#: (2026-09-13, same seconds). Letting them share columns -- which is what
#: happened when both files carried the same band names -- made a bin's value
#: depend on which detector happened to be sampled, a step the model would
#: learn as solar variability. CdTe is the closest to SoLEXS in energy: its L1
#: files have a 5-20 keV band, though the instrument paper gives CdTe a nominal
#: 8-70 keV range, so that band effectively starts near 8 keV.
HEL1OS_DETECTORS: dict[str, tuple[str, Sequence[tuple[float, float]]]] = {
    "czt1": ("czt", HEL1OS_BANDS_KEV),
    "czt2": ("czt", HEL1OS_BANDS_KEV),
    "cdte1": ("cdte", HEL1OS_CDTE_BANDS_KEV),
    "cdte2": ("cdte", HEL1OS_CDTE_BANDS_KEV),
}

# In HEL1OS L1 light curves a row with CTR == 0 AND STAT_ERR == 0 means "no
# telemetry for this second", not "zero counts".  Roughly 83% of rows in a
# typical file are such fill rows.  Treating them as real zeros biases the
# background low by ~6x, so they are always masked out.
HEL1OS_FILL_IS_MISSING = True

MJD_UNIX_EPOCH = 40587.0  # MJD of 1970-01-01, for MJD <-> Unix conversion


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    #: SoLEXS channel-to-energy scale: "sarwade2025" (published calibration) or
    #: "legacy_linear" (the earlier, wrong assumption -- only for reproducing
    #: models trained with it). See SOLEXS_CAL_SARWADE2025 above.
    solexs_energy_scale: str = "sarwade2025"

    #: Where flare truth comes from.
    #: "solexs": this project's detector on the SoLEXS light curve (below). On the
    #:   2024-2026 archive it marks 66% of observed time as "in a flare" and finds
    #:   ~50% of GOES C/M flares -- kept for reproducing earlier runs.
    #: "goes": the NOAA GOES-R XRS flare summary and 1-minute XRS-B flux in
    #:   ``goes_dir``. Events, phases and occurrence labels come from the GOES
    #:   list (>= ``goes_min_class``); flux and peak targets are log10(W/m^2);
    #:   "large flare" means reaching ``goes_exceed_class``. Model inputs are
    #:   unchanged -- SoLEXS and HEL1OS only.
    label_source: str = "solexs"
    #: SoLEXS GOES-long count rate -> log10 GOES XRS-B flux, fitted on the
    #: training period only (scripts/fair_references.py, 2026-09-15: 0.062 dex).
    #: Used by ModelConfig.anchor_flux and never as a target.
    flux_anchor_intercept: float = -6.714
    flux_anchor_slope: float = 0.637
    #: Fallback when SoLEXS is not observing at the forecast origin: the
    #: training-period mean log10 flux.
    flux_anchor_default: float = -5.64
    goes_dir: str = ""
    goes_min_class: str = "C1.0"
    goes_exceed_class: str = "M1.0"

    #: Common resampling grid, seconds.  The shortest flare rise in this
    #: dataset is ~2 min, so 20 s still resolves every onset with ~6 samples,
    #: and it gives HEL1OS ~3.4 real samples per bin instead of ~1.7.
    dt_seconds: float = 20.0

    #: Full width of the running-percentile background window, seconds.
    #: Must comfortably exceed the longest flare, or the event inflates its own
    #: background: the 2026-09-10 17:00 event lasts 128 min, and a 1 h window
    #: pushed the estimated background from 1.2 to 6 cts/s right under the peak.
    background_window_s: float = 21600.0  # 6 h
    #: Percentile used as the quiescent background level.
    background_percentile: float = 10.0

    #: Minimum fraction of valid 1 s samples for a rebinned bin to count as
    #: observed.  Below this the bin is marked missing.
    min_valid_fraction: float = 0.20

    #: Flare detection (operates on the background-subtracted soft rate).
    #: A candidate starts when the rate exceeds background by this factor ...
    flare_rise_factor: float = 1.4
    #: ... and by at least this many Poisson sigma.
    flare_rise_sigma: float = 4.0
    #: ... for at least this many consecutive seconds.
    flare_min_rise_s: float = 120.0
    #: An event ends when it falls back to this fraction of its peak excess.
    flare_end_fraction: float = 0.5
    #: Events shorter than this are discarded as spikes.
    flare_min_duration_s: float = 300.0
    #: Events closer together than this are merged.
    flare_merge_gap_s: float = 300.0

    #: Smoothing applied before detection, seconds.
    smooth_s: float = 60.0

    #: Files closer together than this are stitched into one continuous
    #: timeline (the gap becomes unobserved bins); a longer gap starts a new
    #: segment. Six hours matches the background window: a shorter outage
    #: leaves the background estimate well supported on both sides, a longer
    #: one would let a single window straddle genuinely separate epochs.
    max_stitch_gap_s: float = 21600.0

    #: Parallel processes building the per-file cache. Reading one SoLEXS day
    #: peaks near 1.5 GB, so this is a memory knob as much as a speed knob:
    #: 3 workers stays well inside a 24 GB machine.
    cache_workers: int = 3


# --------------------------------------------------------------------------
# Windowing / targets
# --------------------------------------------------------------------------

@dataclass
class WindowConfig:
    #: Length of the causal input window, seconds (2 h at dt=20 s -> 360 steps).
    #: Two hours comfortably contains the longest rise phase plus context.
    input_seconds: float = 7200.0
    #: Stride between consecutive training windows, seconds.
    stride_seconds: float = 40.0

    #: Regression horizons for forecasting the soft flux, seconds ahead.
    forecast_horizons_s: Sequence[float] = (60.0, 300.0, 900.0, 1800.0, 3600.0)
    #: Quantiles predicted at each horizon (pinball loss).
    quantiles: Sequence[float] = (0.1, 0.5, 0.9)

    #: "Will a flare be in progress within the next H seconds?" horizons.
    occurrence_horizons_s: Sequence[float] = (900.0, 1800.0, 3600.0)

    #: Fraction of the window that must be observed for it to be usable.
    min_observed_fraction: float = 0.5

    #: Above this many observed days the dataset is treated as an archive:
    #: calendar-time split, coarser window stride, quiet-window thinning in
    #: training, capped epochs. Below it the small-sample behaviour is kept
    #: unchanged, so results on short samples stay comparable.
    large_data_days: float = 30.0
    #: Window stride for archives (all splits). 40 s over ~1000 days is ~2M
    #: windows; 120 s still samples every rise phase several times.
    large_stride_seconds: float = 120.0
    #: Quiet (flare-free) training windows are kept only on this stride in
    #: archive mode. Validation and test are never thinned.
    quiet_train_stride_seconds: float = 600.0


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

@dataclass
class ModelConfig:
    hidden: int = 64
    #: Dilation schedule of the causal TCN.  1..128 over kernel-3 convs gives a
    #: receptive field of 511 steps, longer than the 360-step window, so the
    #: last step can attend back over the whole input.
    dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128)
    kernel_size: int = 3
    dropout: float = 0.1
    #: Probability of dropping an entire modality during training.  This is what
    #: makes the network usable when only one instrument is available.
    modality_dropout: float = 0.25
    attn_heads: int = 4
    n_phase_classes: int = 4  # quiet / rise / peak / decay

    #: Encoder depth for the non-TCN architectures in models/zoo.py.
    ssm_layers: int = 4
    #: State dimension of the selective SSM.  Mamba defaults to 16; 8 is used
    #: here because the scan keeps O(T x d_inner x d_state) activations per
    #: layer and eight stacked layers overran a 6 GB card at 16.  With only
    #: 36 input channels, 8 states is not the binding constraint on capacity.
    ssm_state: int = 8
    rnn_layers: int = 2
    tf_layers: int = 3
    #: Predict flux as "calibrated SoLEXS flux now + a learned change" instead
    #: of an absolute level. Measured on the 2026-09 GOES run: the network read
    #: the current GOES flux 0.092 dex from SoLEXS while a one-line calibration
    #: managed 0.069, so the absolute level was costing accuracy the change
    #: prediction did not need. Only meaningful with label_source="goes".
    anchor_flux: bool = False

    #: PatchTST patch length in steps (20 steps = 400 s at dt=20 s). Attention
    #: then runs over 18 patches instead of 360 steps.
    patch_len: int = 20

    #: Feed time-of-day (sin/cos) to the network.  **Off by default.**
    #:
    #: The intent was to let the model absorb spacecraft-periodic systematics.
    #: On a single day of data it does the opposite: the clock uniquely
    #: identifies every moment, so it becomes a lookup table into that day.
    #: Measured on this dataset with clock features ON, a model given *no*
    #: instrument data at all still reached AUC 0.82 on the flare-now head --
    #: pure memorisation -- while the full model collapsed from validation TSS
    #: 0.864 to test TSS 0.078, and the occurrence heads went below AUC 0.5
    #: (anti-correlated) because the memorised clock mapping is wrong for
    #: held-out times.
    #:
    #: Only turn this on with many days spanning many active regions, where
    #: time of day no longer identifies the sample.  The `clock_only` row of
    #: the modality ablation is there to catch this if you do.
    use_clock: bool = False


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    seed: int = 1337
    #: "auto" prefers CUDA when a GPU is visible.  This model is ~20x faster
    #: on even a modest laptop GPU, so check torch.cuda.is_available() before
    #: assuming a long run is normal -- a CPU-only torch build is the usual
    #: cause of a slow one.
    device: str = "auto"
    #: CPU thread count; 0 means "use every core".  Left configurable because
    #: oversubscribing a shared machine is slower than using half of it.
    torch_threads: int = 0
    #: Chronological split fractions (train, val, test).
    split: Sequence[float] = (0.6, 0.2, 0.2)
    #: Embargo between splits, seconds.  Must be >= the longest horizon so no
    #: training window can see into a validation/test target.
    embargo_s: float = 3600.0
    early_stop_patience: int = 12
    #: "per_segment", "global", or "auto" (global once the data exceeds
    #: WindowConfig.large_data_days). See dataset.chronological_split.
    split_mode: str = "auto"
    #: Upper bound on training batches per epoch; 0 means no cap. In archive
    #: mode the cap below applies: each epoch draws a fresh random subset, so
    #: every window is still seen across epochs, but early stopping gets a
    #: validation signal every few minutes instead of every hour.
    max_batches_per_epoch: int = 0
    large_max_batches_per_epoch: int = 400
    #: Focal-loss focusing parameter for the rare-positive heads.
    focal_gamma: float = 2.0
    #: Loss weights; the model additionally learns per-task uncertainty.
    w_phase: float = 1.0
    w_occurrence: float = 1.0
    w_nowcast: float = 1.0
    w_forecast: float = 1.0


@dataclass
class Config:
    data_root: Path = Path(".")
    out_dir: Path = Path("outputs")
    #: Where per-file preprocessed products live; defaults to out_dir/cache.
    #: Point several out_dirs at one cache to avoid re-processing the archive.
    cache_dir: Path | None = None
    pre: PreprocessConfig = field(default_factory=PreprocessConfig)
    win: WindowConfig = field(default_factory=WindowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_json(self, path: Path) -> None:
        d = asdict(self)
        d["data_root"] = str(self.data_root)
        d["out_dir"] = str(self.out_dir)
        d["cache_dir"] = str(self.cache_dir) if self.cache_dir else None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d, indent=2, default=list), encoding="utf-8")

    @property
    def steps_per_window(self) -> int:
        return int(round(self.win.input_seconds / self.pre.dt_seconds))
