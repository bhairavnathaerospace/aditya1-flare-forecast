"""Reader for Aditya-L1 SoLEXS L1 products (.lc / .pi / .gti, plain or gzipped).

The scientifically useful product is the Type-II PHA file: one 340-channel
spectrum per second.  The light curve is derivable from it (and we verify that),
so the spectra are the primary input and the .lc is used as a cross-check.
"""

from __future__ import annotations

import gzip
import io as _io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..config import (
    SOLEXS_CAL_SARWADE2025,
    SOLEXS_CH_HI,
    SOLEXS_CH_LO,
    SOLEXS_ENERGY_SCALES,
    SOLEXS_LEGACY_GAIN_KEV,
    SOLEXS_LEGACY_OFFSET_KEV,
)


def _open_fits(path: Path) -> fits.HDUList:
    """Open a FITS file that may or may not be gzip-compressed."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:
        raw = fh.read()
    return fits.open(_io.BytesIO(raw), memmap=False)


def channel_energies(scale: str = "sarwade2025", detector: str = "SDD2") -> np.ndarray:
    """Energy (keV) assigned to every PI channel under ``scale``.

    "sarwade2025": E = gain * channel + offset with the published piecewise gain
    (see config.SOLEXS_CAL_SARWADE2025), continuous at the break channel.
    "legacy_linear": the project's earlier assumption, bin mid-points.
    """
    ch = np.arange(SOLEXS_CH_HI, dtype=np.float64)
    if scale == "legacy_linear":
        return SOLEXS_LEGACY_GAIN_KEV * (ch + 0.5) + SOLEXS_LEGACY_OFFSET_KEV
    if scale != "sarwade2025":
        raise ValueError(f"unknown SoLEXS energy scale {scale!r}; "
                         f"choose from {SOLEXS_ENERGY_SCALES}")
    c = SOLEXS_CAL_SARWADE2025
    off = c["offset_keV"].get(detector.upper(), c["offset_keV"]["SDD2"])
    brk = c["break_channel"]
    e_break = c["gain_lo_keV"] * brk + off
    return np.where(ch <= brk, c["gain_lo_keV"] * ch + off,
                    e_break + c["gain_hi_keV"] * (ch - brk))


@dataclass
class SolexsObservation:
    """One SoLEXS SDD observation, on its native 1 s grid."""

    detector: str                 # "SDD1" / "SDD2"
    time_unix: np.ndarray         # (N,) seconds since 1970-01-01 UTC
    spectra: np.ndarray           # (N, 340) counts per second per channel
    exposure: np.ndarray          # (N,) seconds
    valid: np.ndarray             # (N,) bool -- inside a GTI and not NaN
    gti: np.ndarray               # (G, 2) unix start/stop
    lc_counts: np.ndarray | None  # (N,) pipeline light curve, or None
    date_obs: str

    @property
    def n(self) -> int:
        return self.time_unix.size

    def band_rate(self, e_lo: float, e_hi: float,
                  energies: np.ndarray | None = None) -> np.ndarray:
        """Counts/s integrated over [e_lo, e_hi) keV, honouring the channel
        threshold that the SoLEXS pipeline itself applies. ``energies`` is the
        channel energy scale (default: the published calibration)."""
        e = channel_energies(detector=self.detector) if energies is None else energies
        sel = (e >= e_lo) & (e < e_hi)
        sel[:SOLEXS_CH_LO] = False  # never trust the electronic noise peak
        if not sel.any():
            return np.zeros(self.n, dtype=np.float64)
        counts = self.spectra[:, sel].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(self.exposure > 0, counts / self.exposure, np.nan)
        return rate

    def total_rate(self) -> np.ndarray:
        """Counts/s over the full usable band (channels 41..339)."""
        counts = self.spectra[:, SOLEXS_CH_LO:SOLEXS_CH_HI].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.exposure > 0, counts / self.exposure, np.nan)


def _gti_mask(time_unix: np.ndarray, gti: np.ndarray) -> np.ndarray:
    if gti.size == 0:
        return np.zeros(time_unix.size, dtype=bool)
    mask = np.zeros(time_unix.size, dtype=bool)
    for start, stop in gti:
        mask |= (time_unix >= start) & (time_unix <= stop)
    return mask


def _gti_from_hdul(hdul: fits.HDUList) -> np.ndarray:
    data = hdul["GTI"].data
    if data is None or len(data) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.column_stack(
        [np.asarray(data["START"], dtype=np.float64),
         np.asarray(data["STOP"], dtype=np.float64)]
    )


def read_gti(path: Path) -> np.ndarray:
    """Return an (G, 2) array of unix start/stop pairs; empty if the file has
    no good-time intervals (which is how a powered-down SDD shows up)."""
    with _open_fits(path) as hdul:
        return _gti_from_hdul(hdul)


def read_solexs(sdd_dir: Path) -> SolexsObservation | None:
    """Read one extracted SDD directory (e.g. ``.../SDD2``).

    Returns ``None`` when the detector produced no science data -- SDD1 is
    frequently in that state, carrying only a zero-row GTI.
    """
    sdd_dir = Path(sdd_dir)

    def open_member(ext: str) -> fits.HDUList | None:
        for pattern in (f"*.{ext}", f"*.{ext}.gz"):
            hits = sorted(sdd_dir.glob(pattern))
            if hits:
                return _open_fits(hits[0])
        return None

    return _read_solexs(open_member, sdd_dir.name)


def list_zip_detectors(zip_path: Path) -> list[str]:
    """SDD names inside a PRADAN SoLEXS zip that carry a spectrum file."""
    import zipfile

    with zipfile.ZipFile(zip_path) as z:
        dets = {n.split("/")[-2] for n in z.namelist()
                if n.endswith((".pi", ".pi.gz")) and "/" in n}
    return sorted(dets)


def read_solexs_zip(zip_path: Path, detector: str = "SDD2") -> SolexsObservation | None:
    """Read one SDD straight out of a PRADAN daily zip, without extracting it.

    A mission's worth of SoLEXS zips is several GB; unpacking them would
    duplicate all of it on disk for no benefit, since the members are already
    gzip-compressed FITS that astropy reads from memory.
    """
    import zipfile

    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        members = z.namelist()

        def open_member(ext: str) -> fits.HDUList | None:
            for suffix in (f".{ext}", f".{ext}.gz"):
                hits = sorted(n for n in members
                              if f"/{detector}/" in n and n.endswith(suffix))
                if hits:
                    raw = z.read(hits[0])
                    if suffix.endswith(".gz"):
                        raw = gzip.decompress(raw)
                    return fits.open(_io.BytesIO(raw), memmap=False)
            return None

        return _read_solexs(open_member, detector)


def _read_solexs(open_member, detector: str) -> SolexsObservation | None:
    """Shared reader; ``open_member(ext)`` returns an HDUList or None."""
    hdul = open_member("pi")
    if hdul is None:
        return None
    with hdul:
        data = hdul["SPECTRUM"].data
        time_unix = np.asarray(data["TSTART"], dtype=np.float64)
        spectra = np.asarray(data["COUNTS"], dtype=np.float32)
        exposure = np.asarray(data["EXPOSURE"], dtype=np.float64)
        date_obs = str(hdul[0].header.get("OBS_DATE", ""))

    if time_unix.size == 0:
        return None

    gti_hdul = open_member("gti")
    if gti_hdul is not None:
        with gti_hdul:
            gti = _gti_from_hdul(gti_hdul)
    else:
        gti = np.zeros((0, 2))

    lc_counts = None
    lc_hdul = open_member("lc")
    if lc_hdul is not None:
        with lc_hdul:
            lc = lc_hdul["RATE"].data
            lc_time = np.asarray(lc["TIME"], dtype=np.float64)
            lc_vals = np.asarray(lc["COUNTS"], dtype=np.float64)
        if lc_time.size == time_unix.size and np.allclose(lc_time, time_unix):
            lc_counts = lc_vals

    finite = np.isfinite(spectra).all(axis=1) & (exposure > 0)
    valid = finite & _gti_mask(time_unix, gti)

    return SolexsObservation(
        detector=detector,
        time_unix=time_unix,
        spectra=spectra,
        exposure=exposure,
        valid=valid,
        gti=gti,
        lc_counts=lc_counts,
        date_obs=date_obs,
    )


def verify_lc_consistency(obs: SolexsObservation) -> dict:
    """Cross-check the pipeline light curve against our channel integration.

    A mismatch means the assumed channel threshold is wrong for this release,
    which would silently corrupt every derived feature -- so it is worth
    asserting on every ingest rather than trusting the constant.
    """
    if obs.lc_counts is None:
        return {"checked": False}
    ours = obs.spectra[:, SOLEXS_CH_LO:SOLEXS_CH_HI].sum(axis=1)
    good = np.isfinite(obs.lc_counts)
    if not good.any():
        return {"checked": False}
    diff = np.abs(ours[good] - obs.lc_counts[good])
    return {
        "checked": True,
        "max_abs_diff": float(diff.max()),
        "exact": bool(diff.max() == 0),
        "n_compared": int(good.sum()),
    }
