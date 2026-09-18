"""Event-centric dataset: predict a flare's outcome from its early rise.

This is the reframed target, and it is the one these instruments can actually
support.  "Will a flare start in the next hour" needs magnetograms -- flare
onset is driven by magnetic free energy that X-ray flux does not observe, and
the measured occurrence heads confirmed it (AUC below 0.5 on held-out data).

What full-disk X-ray *does* determine is what happens once a flare has begun.
So for every detected event we ask, at each moment during the rise:

    given everything up to now, how bright will this get, when will it peak,
    will it exceed a warning threshold, and will it be long-duration?

That is operationally useful -- satellite operators and HF-radio users need
minutes of warning about magnitude, not days of warning about occurrence -- and
it is answerable from rise morphology plus the hard X-ray impulsive signature.

**Evaluation must be grouped by event.** One flare yields many rise samples; a
random split would put the same flare on both sides and report a model that has
memorised thirteen events. `group` carries the event id for exactly this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config
from .dataset import Segment
from .labels import flux_target

#: Time-to-peak is regressed in hours so it shares a scale with log-flux.
PEAK_TIME_SCALE_S = 3600.0

#: An event counts as long-duration (a CME-association proxy) when it lasts
#: longer than this.  Long-duration events are statistically associated with
#: eruption; impulsive ones are more often confined.  Predicting this early is
#: higher space-weather value than predicting occurrence, because eruption is
#: what drives geomagnetic consequences.
LDE_DURATION_S = 3600.0


@dataclass
class RiseSample:
    seg: int
    end: int            # exclusive index of the last observed step
    event_id: int       # grouping key -- never split a flare across folds
    lead_s: float       # seconds already elapsed since onset
    y_log_peak: float   # log1p peak rate of this event
    y_time_to_peak: float   # hours until the peak
    y_exceeds: float    # will this event exceed the warning threshold
    y_lde: float        # will this event be long-duration
    t_unix: float
    #: log1p of the latest raw flux at the prediction moment. Not a target --
    #: it is the "current level" (persistence) reference forecast: a model that
    #: cannot beat reporting the value already reached is not forecasting the
    #: peak at all.
    #:
    #: Deliberately the raw last value, not a smoothed one. The peak target is
    #: defined on the 60 s smoothed curve, so Poisson noise lets the raw value
    #: sit above that peak in a few percent of samples (3.8% on 2026-09-10).
    #: A trailing-smoothed current level avoids that, but it lags during a
    #: rise and was measured to be a 5% *easier* reference. The harder
    #: reference is the honest one, so raw it is.
    y_current: float = 0.0


@dataclass
class RiseDataset:
    samples: list[RiseSample]
    n_events: int
    threshold_rate: float

    def __len__(self) -> int:
        return len(self.samples)

    def groups(self) -> np.ndarray:
        return np.array([s.event_id for s in self.samples], dtype=np.int64)


def format_threshold(thr: float) -> str:
    """Counts (SoLEXS labels) vs W/m^2 (GOES labels): GOES thresholds are tiny."""
    if thr == thr and 0 < thr < 1e-2:
        from ..io.goes import _CLASS_SCALE
        letter = max((k for k, v in _CLASS_SCALE.items() if thr >= v * 0.999), key=_CLASS_SCALE.get,
                     default="A")
        return f"{thr:.1e} W/m^2 (GOES {letter}{thr / _CLASS_SCALE[letter]:.1f})"
    return f"{thr:.2f} cts/s"


def build_rise_dataset(
    segments: list[Segment],
    cfg: Config,
    min_lead_s: float = 40.0,
    max_lead_frac: float = 1.0,
    stride_s: float | None = None,
    exceed_percentile: float = 75.0,
) -> RiseDataset:
    """One sample per (event, moment-during-rise).

    Parameters
    ----------
    min_lead_s
        Skip the first moments after onset: with only one or two bins there is
        no morphology to read, and those samples would dominate by count.
    max_lead_frac
        Fraction of the rise to sample up to. 1.0 includes the peak itself;
        lower values make the task strictly harder and more honest.
    exceed_percentile
        The "large flare" threshold is set at this percentile of observed peak
        rates, so the positive class stays populated regardless of how active
        the period was.  With an absolute GOES calibration, replace this with a
        real class boundary (C1.0, M1.0, ...).
    """
    dt = cfg.pre.dt_seconds
    stride = max(int((stride_s or dt) / dt), 1)
    min_lead = max(int(min_lead_s / dt), 1)

    peaks = [e.peak_rate for s in segments for e in s.events]
    if cfg.pre.label_source == "goes":
        # A real class boundary, not a percentile of whatever happened.
        from ..io.goes import class_flux
        threshold = class_flux(cfg.pre.goes_exceed_class)
    else:
        threshold = float(np.percentile(peaks, exceed_percentile)) if peaks else np.inf

    samples: list[RiseSample] = []
    event_id = 0
    for si, seg in enumerate(segments):
        for ev in seg.events:
            # A rise that began before this segment's data, or peaks after it,
            # would get a wrong lead time or peak: skip it.
            if (ev.start_unix < float(seg.time_unix[0])
                    or ev.peak_unix > float(seg.time_unix[-1])):
                event_id += 1
                continue
            rise_steps = ev.peak_idx - ev.start_idx
            if rise_steps < min_lead:
                event_id += 1
                continue
            last = ev.start_idx + max(int(rise_steps * max_lead_frac), min_lead)
            last = min(last, ev.peak_idx, len(seg) - 1)

            for end in range(ev.start_idx + min_lead, last + 1, stride):
                if seg.target_valid[end - 1] <= 0:
                    continue
                samples.append(RiseSample(
                    seg=si,
                    end=end,
                    event_id=event_id,
                    lead_s=float((end - ev.start_idx) * dt),
                    y_log_peak=float(flux_target(ev.peak_rate, cfg.pre.label_source)),
                    y_time_to_peak=float((ev.peak_idx - end) * dt / PEAK_TIME_SCALE_S),
                    y_exceeds=float(ev.peak_rate >= threshold),
                    y_lde=float(ev.duration_s >= LDE_DURATION_S),
                    t_unix=float(seg.time_unix[end - 1]),
                    y_current=float(seg.log_flux[end - 1]),
                ))
            event_id += 1

    return RiseDataset(samples=samples, n_events=event_id, threshold_rate=threshold)


def summarise(ds: RiseDataset, dt: float) -> str:
    if not ds.samples:
        return "  (no rise-phase samples -- no events long enough to sample)"
    leads = np.array([s.lead_s for s in ds.samples])
    exceeds = np.array([s.y_exceeds for s in ds.samples])
    lde = np.array([s.y_lde for s in ds.samples])
    per_event = {}
    for s in ds.samples:
        per_event.setdefault(s.event_id, 0)
        per_event[s.event_id] += 1
    return "\n".join([
        f"  rise samples : {len(ds.samples)} drawn from "
        f"{len(per_event)} events (of {ds.n_events} detected)",
        f"  lead time    : median {np.median(leads) / 60:.1f} min, "
        f"max {leads.max() / 60:.1f} min",
        f"  large-flare threshold : {format_threshold(ds.threshold_rate)} "
        f"({100 * exceeds.mean():.0f}% of samples positive)",
        f"  long-duration events  : {100 * lde.mean():.0f}% of samples",
        f"  samples per event     : min {min(per_event.values())}, "
        f"max {max(per_event.values())}",
    ])


def event_level_split(ds: RiseDataset, frac_train: float = 0.6,
                      frac_val: float = 0.2, seed: int = 0
                      ) -> tuple[list[int], list[int], list[int]]:
    """Split by **event**, chronologically.

    Grouping is not optional here. Rise samples from one flare are nearly
    identical to each other; splitting them independently would leak the answer
    and inflate every score.
    """
    order: dict[int, float] = {}
    for s in ds.samples:
        order.setdefault(s.event_id, s.t_unix)
    ids = sorted(order, key=lambda e: order[e])
    n = len(ids)
    i_tr = int(round(n * frac_train))
    i_va = int(round(n * (frac_train + frac_val)))
    tr_ids, va_ids, te_ids = set(ids[:i_tr]), set(ids[i_tr:i_va]), set(ids[i_va:])

    tr = [i for i, s in enumerate(ds.samples) if s.event_id in tr_ids]
    va = [i for i, s in enumerate(ds.samples) if s.event_id in va_ids]
    te = [i for i, s in enumerate(ds.samples) if s.event_id in te_ids]
    return tr, va, te
