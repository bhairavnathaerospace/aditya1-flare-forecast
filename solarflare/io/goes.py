"""Independent truth: NOAA GOES-R XRS science-quality L2 products.

Two products, both netCDF-4 (read with h5py):

* ``sci_xrsf-l2-flsum_*`` -- the flare summary. One record per flare state
  change: EVENT_START, EVENT_PEAK (carrying the GOES class), EVENT_END,
  POST_EVENT, tied together by ``flare_id``.
* ``sci_xrsf-l2-avg1m_*`` -- 1-minute XRS-B (1-8 Angstrom) flux in W/m^2.

Why this exists: the project's own SoLEXS flare detector labels 66% of
observed time as "in a flare" (GOES: 9.5% over the same 2024-02 -> 2026-09
period) and finds only ~50% of GOES C/M flares, so skill measured against it
is not comparable with anything published. GOES classes are the community
definition.

Times in these files are "seconds since 2000-01-01 12:00:00 UTC", neglecting
leap seconds -- i.e. a fixed offset from Unix time.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

#: 2000-01-01 12:00:00 UTC as a Unix timestamp.
GOES_EPOCH_UNIX = 946728000.0

_CLASS_SCALE = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}


def class_flux(cls: str) -> float:
    """"M2.5" -> 2.5e-5 W/m^2; unparseable -> nan."""
    import re
    m = re.match(r"\s*([ABCMX])\s*([\d.]+)?", str(cls or "").upper())
    if not m:
        return float("nan")
    return _CLASS_SCALE[m.group(1)] * float(m.group(2) or 1.0)


@dataclass
class GoesFlare:
    flare_id: int
    start_unix: float
    peak_unix: float
    end_unix: float
    end_estimated: bool      # the summary had no EVENT_END for this flare
    goes_class: str          # e.g. "M1.4"
    peak_flux: float         # W/m^2 (from the class, i.e. background included)
    background_flux: float   # W/m^2 at the peak record


def _decode(a) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def read_flare_summary(path: Path) -> list[GoesFlare]:
    """Flares from one flsum file, sorted by peak time.

    About 13% of flares on GOES-18 2022-2026 have no EVENT_END record --
    usually because the next flare started before this one decayed. Their end
    is estimated as the earlier of the next flare's start and
    peak + 2 x (peak - start), at least 10 min after the peak, and flagged.
    """
    import h5py

    with h5py.File(path, "r") as h:
        t = h["time"][:].astype(np.float64) + GOES_EPOCH_UNIX
        status = _decode(h["status"][:])
        cls = _decode(h["flare_class"][:])
        fid = h["flare_id"][:].astype(np.int64)
        bg = h["background_flux"][:].astype(np.float64)

    starts: dict[int, float] = {}
    peaks: dict[int, tuple[float, str, float]] = {}
    ends: dict[int, float] = {}
    for ti, si, ci, fi, bi in zip(t, status, cls, fid, bg):
        if si == "EVENT_START":
            starts.setdefault(int(fi), float(ti))
        elif si == "EVENT_PEAK":
            peaks[int(fi)] = (float(ti), ci, float(bi))
        elif si == "EVENT_END":
            ends.setdefault(int(fi), float(ti))

    rows = []
    for fi, (tp, ci, bi) in peaks.items():
        ts = starts.get(fi, tp)
        rows.append([fi, ts, tp, ends.get(fi), ci, bi])
    rows.sort(key=lambda r: r[2])

    out: list[GoesFlare] = []
    for k, (fi, ts, tp, te, ci, bi) in enumerate(rows):
        estimated = te is None
        if estimated:
            te = tp + max(2.0 * (tp - ts), 600.0)
            if k + 1 < len(rows):
                te = min(te, max(rows[k + 1][1], tp + 60.0))
        out.append(GoesFlare(int(fi), float(ts), float(tp), float(max(te, tp)), estimated,
                             ci, class_flux(ci), float(bi)))
    return out


def read_xrs_1min(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated (time_unix, XRS-B flux W/m^2) with bad samples as NaN.

    A sample is kept only if its quality flag is 0, the flux is finite and
    positive. Records are time-sorted and de-duplicated.
    """
    import h5py

    ts, fs = [], []
    for p in sorted(paths):
        with h5py.File(p, "r") as h:
            t = h["time"][:].astype(np.float64) + GOES_EPOCH_UNIX
            f = h["xrsb_flux"][:].astype(np.float64)
            flag = h["xrsb_flag"][:] if "xrsb_flag" in h else np.zeros(t.size, np.uint8)
        good = (flag == 0) & np.isfinite(f) & (f > 0)
        ts.append(t)
        fs.append(np.where(good, f, np.nan))
    if not ts:
        return np.zeros(0), np.zeros(0)
    t = np.concatenate(ts)
    f = np.concatenate(fs)
    order = np.argsort(t, kind="stable")
    t, f = t[order], f[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    return t[keep], f[keep]


@dataclass
class GoesTruth:
    flares: list[GoesFlare]
    time_unix: np.ndarray   # 1-min record starts
    xrsb: np.ndarray        # W/m^2, NaN where bad
    source_files: list[str]

    def flux_on_grid(self, grid: np.ndarray) -> np.ndarray:
        """XRS-B flux for each bin of ``grid``: the 1-min record containing
        the bin's left edge, NaN outside coverage or where flagged."""
        out = np.full(grid.size, np.nan)
        if self.time_unix.size == 0:
            return out
        i = np.searchsorted(self.time_unix, grid, side="right") - 1
        ok = (i >= 0) & (grid - self.time_unix[np.clip(i, 0, None)] < 60.0)
        out[ok] = self.xrsb[i[ok]]
        return out


def find_goes_files(goes_dir: Path) -> tuple[list[Path], list[Path]]:
    d = Path(goes_dir)
    flsum = sorted(d.glob("*xrsf-l2-flsum*.nc"))
    avg1m = sorted(d.glob("*xrsf-l2-avg1m*.nc"))
    return flsum, avg1m


@lru_cache(maxsize=4)
def _load_cached(goes_dir: str, stamp: tuple) -> GoesTruth:
    flsum, avg1m = find_goes_files(Path(goes_dir))
    flares: list[GoesFlare] = []
    seen: set[int] = set()
    for p in flsum:
        for fl in read_flare_summary(p):
            if fl.flare_id not in seen:
                seen.add(fl.flare_id)
                flares.append(fl)
    flares.sort(key=lambda f: f.peak_unix)
    t, f = read_xrs_1min(avg1m)
    return GoesTruth(flares, t, f, [str(p) for p in flsum + avg1m])


def load_goes(goes_dir: Path) -> GoesTruth:
    """All GOES products in a folder, cached per folder contents."""
    flsum, avg1m = find_goes_files(Path(goes_dir))
    if not flsum or not avg1m:
        raise FileNotFoundError(
            f"GOES truth needs both a flare summary (*xrsf-l2-flsum*.nc) and 1-minute "
            f"XRS data (*xrsf-l2-avg1m*.nc) in {goes_dir}; found {len(flsum)} and {len(avg1m)}")
    stamp = tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in flsum + avg1m)
    return _load_cached(str(Path(goes_dir).resolve()), stamp)
