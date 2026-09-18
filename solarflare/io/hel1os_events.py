"""HEL1OS L1 photon event lists (``events/evt.fits``): finding and reading them.

Measured on the real products (2026-09-18):

* One HDU per detector (``CZT1-EVENTS``, ``CZT2-EVENTS``, ``CDTE1-EVENTS``,
  ``CDTE2-EVENTS``) with ``mjd`` (UTC), ``hlsobt`` (onboard seconds), ``ener``
  (keV, already calibrated) and, for CZT, ``pix``.
* Onboard times are 10 ms ticks: ``hlsobt`` takes 100 fractional values. Bins
  that are not whole ticks alias the quantisation into a fake ~20% modulation,
  so every series here is built on the tick grid.
* The ``mjd`` column is a per-packet stamp: against ``hlsobt`` it jumps by up to
  +-1 s from one packet to the next, which turns a smooth 0.1 s light curve
  into noise 50x Poisson. Relative timing uses ``hlsobt``; UTC comes from the
  median offset over the requested interval (good to ~1 s).
* Rows are packet-ordered, not time-ordered (about 4% out of order).
* At high rates the stream has gaps: at an M6 peak, 10-80 ms with no event in
  either CZT, about every 0.65 s. ``live`` marks ticks where any detector
  recorded an event.
* CZT carries an Am-241 source: a 59.5 keV line in every file, useful as an
  energy-scale check. ``aux/cztdis/czt?dispix.txt`` lists pixels disabled onboard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
from astropy.io import fits

TICK_S = 0.01
DETECTORS = ("CZT1", "CZT2", "CDTE1", "CDTE2")
_NAME = re.compile(r"HLS_(\d{8})_(\d{6})_(\d+)sec_lev1_V(\d+)")
MJD_UNIX0 = 40587.0


@dataclass(frozen=True)
class Product:
    path: Path
    t_start: float
    t_stop: float
    version: int


@lru_cache(maxsize=2)
def product_index(root: str) -> tuple[Product, ...]:
    """Every extracted HEL1OS product under ``root`` that has an event list."""
    out = []
    for d in Path(root).glob("*/*/*/HLS_*"):
        m = _NAME.fullmatch(d.name)
        if not m or not (d / "events" / "evt.fits").exists():
            continue
        t0 = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=UTC).timestamp()
        out.append(Product(d, t0, t0 + float(m.group(3)), int(m.group(4))))
    return tuple(sorted(out, key=lambda p: (p.t_start, -p.version)))


def product_for(root: str | Path, t0: float, t1: float) -> Product | None:
    """The product covering the most of [t0, t1]; ties go to the higher version
    (overlapping versions carry identical telemetry, see hel1os.py)."""
    best, best_key = None, None
    for p in product_index(str(root)):
        ov = min(p.t_stop, t1) - max(p.t_start, t0)
        if ov <= 0:
            continue
        key = (round(ov, 0), p.version)
        if best_key is None or key > best_key:
            best, best_key = p, key
    return best


def disabled_pixels(product: Product, det: str) -> set[int]:
    f = product.path / "aux" / "cztdis" / f"{det.lower()}dispix.txt"
    if not f.exists():
        return set()
    return {int(x) for x in f.read_text().split() if x.strip().lstrip("-").isdigit()}


@dataclass
class Events:
    det: str
    tick: np.ndarray       # int64 onboard 10 ms ticks, sorted
    energy: np.ndarray     # keV
    pix: np.ndarray | None
    utc_offset: float      # UTC = tick * TICK_S + utc_offset

    def utc(self) -> np.ndarray:
        return self.tick * TICK_S + self.utc_offset


def read_events(product: Product, det: str, t0: float, t1: float,
                e_lo: float = 0.0, e_hi: float = np.inf) -> Events:
    """Events of one detector with UTC in [t0, t1) and energy in [e_lo, e_hi)."""
    det = det.upper()
    with fits.open(product.path / "events" / "evt.fits", memmap=True) as h:
        d = h[f"{det}-EVENTS"].data
        obt = np.asarray(d["hlsobt"], dtype=np.float64)
        mjd = np.asarray(d["mjd"], dtype=np.float64)
        utc = (mjd - MJD_UNIX0) * 86400.0
        near = (utc >= t0 - 120.0) & (utc < t1 + 120.0)
        if not near.any():
            return Events(det, np.zeros(0, np.int64), np.zeros(0), None, np.nan)
        offset = float(np.median(utc[near] - obt[near]))
        tick = np.rint(obt / TICK_S).astype(np.int64)
        k0, k1 = int(np.floor((t0 - offset) / TICK_S)), int(np.ceil((t1 - offset) / TICK_S))
        e = np.asarray(d["ener"], dtype=np.float64)
        sel = (tick >= k0) & (tick < k1) & (e >= e_lo) & (e < e_hi)
        pix = np.asarray(d["pix"], dtype=np.int16)[sel] if "pix" in d.names else None
        tick, e = tick[sel], e[sel]
    if pix is not None:
        bad = disabled_pixels(product, det)
        if bad:
            ok = ~np.isin(pix, list(bad))
            tick, e, pix = tick[ok], e[ok], pix[ok]
    order = np.argsort(tick, kind="stable")
    return Events(det, tick[order], e[order], None if pix is None else pix[order], offset)


def tick_counts(ev: Events, k0: int, n: int, e_lo: float = 0.0, e_hi: float = np.inf) -> np.ndarray:
    """Counts per 10 ms tick on ticks k0 .. k0+n-1."""
    m = (ev.energy >= e_lo) & (ev.energy < e_hi) & (ev.tick >= k0) & (ev.tick < k0 + n)
    return np.bincount(ev.tick[m] - k0, minlength=n).astype(np.float64)
