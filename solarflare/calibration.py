"""Energy-scale diagnostics from persistent instrumental lines.

The SoLEXS quiet-Sun spectrum in this dataset contains narrow lines that are
present at constant strength even when the Sun is doing nothing.  Solar lines
(Fe XXV at 6.7 keV, say) come and go with flares; a line that never varies is
instrumental -- almost always an onboard radioactive calibration source.

That makes it a ruler.  Identify the line and the channel-to-energy gain
follows, replacing the inferred value in ``config.py`` with a measured one.

This module does **not** guess the identification.  It measures the line
channels and, given an assignment you supply, solves for gain and offset and
reports the residuals so a wrong assignment is visible rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SOLEXS_CH_LO
from .io.solexs import SolexsObservation, channel_energies

#: Common calibration-source and fluorescence lines, keV.  For reference when
#: assigning detected channels -- not applied automatically.  The flag marks
#: lines that are solar in origin: those vary with flares, so they cannot
#: explain a line that is present at constant strength during quiet Sun, and
#: they are excluded from the suggestions below.
KNOWN_LINES: dict[str, tuple[float, bool]] = {
    # name: (energy keV, is_solar)
    "Si Ka": (1.740, False),
    "Ti Ka": (4.511, False),
    "Ti Kb": (4.932, False),
    "Cr Ka": (5.415, False),
    "Fe55 Mn Ka": (5.895, False),
    "Fe Ka (fluorescence)": (6.404, False),
    "Fe55 Mn Kb": (6.490, False),
    "Fe XXV": (6.700, True),
    "Cu Ka": (8.048, False),
    "Au La": (9.713, False),
}

#: Characteristic line separations, keV.  A detected pair whose spacing matches
#: one of these is a candidate identification, and the spacing alone fixes the
#: gain without needing the absolute energies.
LINE_PAIRS: dict[str, float] = {
    "Mn Kb - Mn Ka": 6.490 - 5.895,
    "Ti Kb - Ti Ka": 4.932 - 4.511,
    "Si escape (Mn Ka)": 1.740,
}


@dataclass
class DetectedLine:
    channel: float          # centroid
    net_counts: float       # continuum-subtracted
    significance: float     # net / sqrt(continuum)
    energy_current_cal: float


def quiet_spectrum(obs: SolexsObservation, quiet_mask: np.ndarray | None = None,
                   quantile: float = 0.5) -> tuple[np.ndarray, int]:
    """Summed spectrum over the quietest part of an observation.

    Defaults to the lowest-``quantile`` fraction of seconds by total rate, so
    flare emission does not contaminate the line search.
    """
    total = obs.spectra[:, SOLEXS_CH_LO:].sum(axis=1)
    good = obs.valid & np.isfinite(total)
    if quiet_mask is None:
        if not good.any():
            return np.zeros(obs.spectra.shape[1]), 0
        thr = np.quantile(total[good], quantile)
        quiet_mask = good & (total <= thr)
    else:
        quiet_mask = quiet_mask & good
    if not quiet_mask.any():
        return np.zeros(obs.spectra.shape[1]), 0
    return obs.spectra[quiet_mask].astype(np.float64).sum(axis=0), int(quiet_mask.sum())


def find_lines(spectrum: np.ndarray, ch_lo: int = 60, ch_hi: int = 260,
               min_sigma: float = 3.0, continuum_width: int = 31,
               energy_scale: str = "sarwade2025", detector: str = "SDD2",
               ) -> list[DetectedLine]:
    """Locate narrow excesses over a median-filtered continuum. Energies are
    reported on ``energy_scale`` (see io.solexs.channel_energies)."""
    from scipy.ndimage import median_filter, gaussian_filter1d

    cont = median_filter(spectrum, size=continuum_width)
    resid = spectrum - cont
    smooth = gaussian_filter1d(resid, 1.0)

    energies = channel_energies(energy_scale, detector)
    out: list[DetectedLine] = []
    for c in range(max(ch_lo, 1), min(ch_hi, spectrum.size - 1)):
        if not (smooth[c] > smooth[c - 1] and smooth[c] >= smooth[c + 1]):
            continue
        sig = resid[c] / np.sqrt(max(cont[c], 1.0))
        if sig < min_sigma:
            continue
        w = slice(max(c - 4, 0), min(c + 5, spectrum.size))
        chs = np.arange(spectrum.size)[w]
        wts = np.clip(resid[w], 0.0, None)
        if wts.sum() <= 0:
            continue
        centroid = float((chs * wts).sum() / wts.sum())
        out.append(DetectedLine(
            channel=centroid,
            net_counts=float(wts.sum()),
            significance=float(sig),
            energy_current_cal=float(np.interp(centroid, np.arange(energies.size),
                                               energies)),
        ))
    out.sort(key=lambda d: -d.net_counts)
    return out


def solve_gain(assignments: list[tuple[float, float]]) -> dict:
    """Least-squares fit of E = gain * channel + offset.

    ``assignments`` is [(channel, energy_keV), ...].  Two lines determine the
    scale exactly; three or more also give residuals, which is the only way to
    notice that an assignment is wrong.
    """
    if len(assignments) < 2:
        return {"error": "need at least two (channel, energy) pairs"}
    ch = np.array([a[0] for a in assignments], dtype=float)
    en = np.array([a[1] for a in assignments], dtype=float)
    A = np.column_stack([ch, np.ones_like(ch)])
    (gain, offset), *_ = np.linalg.lstsq(A, en, rcond=None)
    pred = gain * ch + offset
    resid = en - pred
    return {
        "gain_keV_per_channel": float(gain),
        "offset_keV": float(offset),
        "residuals_keV": resid.tolist(),
        "max_abs_residual_keV": float(np.abs(resid).max()),
        "band_lo_keV": float(gain * SOLEXS_CH_LO + offset),
        "band_hi_keV": float(gain * 339 + offset),
    }


def report(obs: SolexsObservation, energy_scale: str = "sarwade2025") -> str:
    """Human-readable line report for the `inspect` command.

    Checks the configured energy scale against the onboard Fe-55 source: its
    Mn Ka / Kb lines (5.895 / 6.490 keV) are the strongest persistent lines in
    quiet-Sun SoLEXS spectra. Both the absolute positions and the Kb-Ka
    spacing are reported, so a wrong scale shows up as a residual.
    """
    spec, n = quiet_spectrum(obs)
    if n == 0:
        return "  (no quiet data for a line search)"
    lines = find_lines(spec, energy_scale=energy_scale, detector=obs.detector)
    if not lines:
        return "  (no significant persistent lines found)"

    out = [f"  Persistent lines in the quietest {n} s (instrumental, not solar):",
           f"    {'channel':>8}  {'net':>7}  {'sigma':>6}  {'E (' + energy_scale + ')':>22}"]
    for d in lines[:6]:
        out.append(f"    {d.channel:8.2f}  {d.net_counts:7.0f}  "
                   f"{d.significance:6.1f}  {d.energy_current_cal:19.3f} keV")

    ka, kb = KNOWN_LINES["Fe55 Mn Ka"][0], KNOWN_LINES["Fe55 Mn Kb"][0]
    near_ka = min(lines[:6], key=lambda d: abs(d.energy_current_cal - ka))
    near_kb = min(lines[:6], key=lambda d: abs(d.energy_current_cal - kb))
    out.append("")
    out.append("  Fe-55 check (onboard calibration source):")
    out.append(f"    Mn Ka {ka:.3f} keV -> line at channel {near_ka.channel:.2f}, "
               f"placed at {near_ka.energy_current_cal:.3f} keV "
               f"({1000 * (near_ka.energy_current_cal - ka):+.0f} eV)")
    out.append(f"    Mn Kb {kb:.3f} keV -> line at channel {near_kb.channel:.2f}, "
               f"placed at {near_kb.energy_current_cal:.3f} keV "
               f"({1000 * (near_kb.energy_current_cal - kb):+.0f} eV)")
    if near_kb.channel - near_ka.channel >= 2:
        g = (kb - ka) / (near_kb.channel - near_ka.channel)
        out.append(f"    Kb-Ka spacing implies {1000 * g:.2f} eV/channel "
                   f"(published: 47.75 eV/channel below channel 168)")
    worst = max(abs(near_ka.energy_current_cal - ka), abs(near_kb.energy_current_cal - kb))
    out.append("    -> consistent with this scale (within SoLEXS' 170 eV resolution)"
               if worst < 0.085 else
               "    -> NOT consistent with this scale: check the energy calibration")
    return "\n".join(out)
