"""Algorithmic nowcasting: flares found in SoLEXS and HEL1OS independently, then merged.

``detect``   the detectors (SoLEXS rise rule, HEL1OS coincidence bursts) and the
             SoLEXS -> GOES flux calibration
``build``    runs them over the whole archive and scores the result against GOES
             (``python -m solarflare catalog``)
``figures``  the overview figure
"""

from .detect import (
    HARD_BEFORE_SOFT_S, HardBurst, HardEvent, MasterFlare, PiecewiseCalibration, SoftEvent,
    as_row, attach_bands, flux_class, hxr_bursts, merge, noaa_events, rise_events_as_bursts,
    solexs_to_goes_flux, to_minutes, trailing_background, trailing_noise,
)

__all__ = [
    "HARD_BEFORE_SOFT_S", "HardBurst", "HardEvent", "MasterFlare", "PiecewiseCalibration", "SoftEvent",
    "as_row", "attach_bands", "flux_class", "hxr_bursts", "merge", "noaa_events", "rise_events_as_bursts",
    "solexs_to_goes_flux", "to_minutes", "trailing_background", "trailing_noise",
]
