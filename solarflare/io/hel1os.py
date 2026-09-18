"""Reader for Aditya-L1 HEL1OS L1 hard X-ray products.

The one thing that matters here: HEL1OS L1 light curves are written on a dense
1 s grid but are only *sampled* every few seconds.  Seconds with no telemetry
are written as ``CTR = 0`` with ``STAT_ERR = 0``.  In the reference dataset that
is 83% of all rows.  A sampled slot that saw zero photons is written the same
way (STAT_ERR is sqrt(counts)), so the decision is made per slot across all
bands of a detector -- see ``read_hel1os_lightcurve``.

Reading the file naively gives a mean rate of ~26 cts/s; masking correctly gives
~151 cts/s.  A model trained on the former learns the telemetry duty cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..config import HEL1OS_FILL_IS_MISSING, MJD_UNIX_EPOCH


def mjd_to_unix(mjd: np.ndarray) -> np.ndarray:
    return (np.asarray(mjd, dtype=np.float64) - MJD_UNIX_EPOCH) * 86400.0


@dataclass
class Hel1osObservation:
    """One HEL1OS detector light curve on its native 1 s grid."""

    detector: str              # "CZT1", "CZT2", "CdTe1", ... as in DETNAM
    time_unix: np.ndarray      # (N,)
    rates: np.ndarray          # (N, B) counts/s, NaN where not sampled
    errors: np.ndarray         # (N, B) statistical error, NaN where not sampled
    valid: np.ndarray          # (N, B) bool -- True where a real sample exists
    bands_kev: np.ndarray      # (B, 2) energy edges
    gti: np.ndarray            # (G, 2) unix
    iso_start: str
    iso_stop: str
    #: Product folder the file came from (``HLS_<start>_<dur>sec_lev1_V<ver>``),
    #: so that the detectors of one observation can be grouped back together.
    product: str = ""

    @property
    def detector_key(self) -> str:
        """Lower-case detector id used by the feature layout: "czt1", "cdte2"."""
        return self.detector.lower()

    @property
    def n(self) -> int:
        return self.time_unix.size

    @property
    def any_valid(self) -> np.ndarray:
        """(N,) True where at least one band was sampled."""
        return self.valid.any(axis=1)

    def band_index(self, e_lo: float, e_hi: float) -> int | None:
        for i, (lo, hi) in enumerate(self.bands_kev):
            if abs(lo - e_lo) < 1e-6 and abs(hi - e_hi) < 1e-6:
                return i
        return None

    @property
    def wide_band_index(self) -> int:
        """Index of the widest band, used as the hard X-ray summary channel."""
        widths = self.bands_kev[:, 1] - self.bands_kev[:, 0]
        return int(np.argmax(widths))


def read_gti_fits(path: Path) -> np.ndarray:
    if not Path(path).exists():
        return np.zeros((0, 2), dtype=np.float64)
    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data
        if data is None or len(data) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        names = {n.lower(): n for n in data.columns.names}
        start = np.asarray(data[names["tstart"]], dtype=np.float64)
        stop = np.asarray(data[names["tstop"]], dtype=np.float64)
    return np.column_stack([mjd_to_unix(start), mjd_to_unix(stop)])


def read_hel1os_lightcurve(path: Path, gti_path: Path | None = None
                           ) -> Hel1osObservation | None:
    """Read a ``lightcurve_<detector>.fits`` file (all energy-band extensions).

    **Band alignment.** Each band extension has its own 1 s grid. In CZT files
    the grids coincide; in CdTe files each band's grid starts at that band's
    first photon, so bands differ in length (CdTe1 on 2024-04-13: 39558,
    39518, 39534, 39571 rows) and in sub-second phase (0.09 s, 0.53 s,
    0.73 s). Every band is mapped onto the grid of the longest band by
    nearest 1 s slot. (The first version assumed one shared grid and silently
    dropped every band of a different length.)

    **What counts as sampled.** ``STAT_ERR`` is sqrt(counts), so a telemetry
    sample with zero counts is written exactly like a slot with no telemetry:
    CTR = 0, STAT_ERR = 0. Deciding per band therefore throws away every
    genuine zero and biases faint bands upward: on 2026-09-13 the per-band
    rule gave CdTe1 5-20 keV a mean of 1.15 cts/s against 0.19 with zeros kept.
    So the decision is made per *slot*: a slot is sampled if any band of the
    detector is non-zero there, and a zero in another band is then a real
    zero count. CZT moves by <5% (its bands are bright); CdTe by up to 10x.

    Still undetectable: a sampled slot where *every* band read zero. That
    only matters for the faint CdTe wide band. A zero-truncated Poisson fit
    puts its quiet-Sun level 3-15% high on typical days and ~2x high on the
    faintest (CdTe1, 2026-09-13, 1.3 counts per sample). The bias falls
    quickly as the rate rises, so it is negligible during flares, but quiet
    CdTe levels should not be compared across days at better than that.
    """
    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        prim = hdul[0].header
        band_hdus = [h for h in hdul[1:] if h.data is not None and len(h.data)]
        if not band_hdus:
            return None

        band_times = [mjd_to_unix(np.asarray(h.data["MJD"], dtype=np.float64))
                      for h in band_hdus]
        ref = max(range(len(band_times)), key=lambda i: (band_times[i].size, i))
        ref_t = band_times[ref]
        step = float(np.median(np.diff(ref_t))) if ref_t.size > 1 else 1.0
        step = step if step > 0 else 1.0
        slots = [np.rint((t - ref_t[0]) / step).astype(np.int64) for t in band_times]
        k_min = min(int(s.min()) for s in slots)
        k_max = max(int(s.max()) for s in slots)
        n = k_max - k_min + 1
        time_unix = ref_t[0] + step * np.arange(k_min, k_max + 1, dtype=np.float64)

        n_b = len(band_hdus)
        ctr_all = np.zeros((n, n_b), dtype=np.float64)
        err_all = np.zeros((n, n_b), dtype=np.float64)
        finite = np.ones((n, n_b), dtype=bool)
        edges = np.zeros((n_b, 2), dtype=np.float64)
        detector = str(band_hdus[0].header.get("DETNAM", "")) or \
            path.stem.replace("lightcurve_", "").upper()

        for i, hdu in enumerate(band_hdus):
            rows = slots[i] - k_min
            ctr = np.asarray(hdu.data["CTR"], dtype=np.float64)
            err = np.asarray(hdu.data["STAT_ERR"], dtype=np.float64)
            ok = np.isfinite(ctr) & np.isfinite(err)
            ctr_all[rows, i] = np.where(ok, ctr, 0.0)
            err_all[rows, i] = np.where(ok, err, 0.0)
            finite[rows[~ok], i] = False
            edges[i] = (float(hdu.header.get("ELOW", np.nan)),
                        float(hdu.header.get("EHIGH", np.nan)))

        if HEL1OS_FILL_IS_MISSING:
            slot_sampled = ((ctr_all != 0.0) | (err_all != 0.0)).any(axis=1)
            valid = slot_sampled[:, None] & finite
        else:
            valid = finite
        rates = np.where(valid, ctr_all, np.nan).astype(np.float32)
        errors = np.where(valid, err_all, np.nan).astype(np.float32)

        iso_start = str(prim.get("ISOSTART", ""))
        iso_stop = str(prim.get("ISOSTOP", ""))

    gti = read_gti_fits(gti_path) if gti_path else np.zeros((0, 2))
    if gti.size:
        inside = np.zeros(n, dtype=bool)
        for a, b in gti:
            inside |= (time_unix >= a) & (time_unix <= b)
        valid &= inside[:, None]

    return Hel1osObservation(
        detector=detector,
        time_unix=time_unix,
        rates=rates,
        errors=errors,
        valid=valid,
        bands_kev=edges,
        gti=gti,
        iso_start=iso_start,
        iso_stop=iso_stop,
        product=str(product_dir_of(path)),
    )


def product_dir_of(lightcurve: Path) -> Path:
    """``.../HLS_<...>/czt/lightcurve_czt1.fits`` -> ``.../HLS_<...>``.

    A file not inside a ``czt``/``cdte`` subfolder is its own product (its
    parent folder), which keeps loose files usable."""
    p = Path(lightcurve).parent
    return p.parent if p.name.lower() in ("czt", "cdte") else p


def gti_path_for(lightcurve: Path) -> Path:
    """The GTI written next to a light curve: ``<product>/aux/gti<det>.fits``."""
    det = Path(lightcurve).stem.replace("lightcurve_", "")
    return product_dir_of(lightcurve) / "aux" / f"gti{det}.fits"


def product_lightcurves(product_dir: Path) -> list[Path]:
    """Every detector light curve of one product, in a stable order."""
    d = Path(product_dir)
    files = sorted(d.glob("czt/lightcurve_*.fits")) + sorted(d.glob("cdte/lightcurve_*.fits"))
    return files or sorted(d.glob("lightcurve_*.fits"))


def read_hel1os_product(product_dir: Path) -> list[Hel1osObservation]:
    """Read all detectors of one product, each with its own GTI."""
    out = []
    for lc in product_lightcurves(product_dir):
        gti = gti_path_for(lc)
        obs = read_hel1os_lightcurve(lc, gti if gti.exists() else None)
        if obs is not None and obs.any_valid.any():
            out.append(obs)
    return out


def read_housekeeping(path: Path) -> dict[str, np.ndarray] | None:
    """Read the HEL1OS HK table.

    Useful as model context and as a data-quality gate: ``suninfov`` tells us
    whether the Sun was actually in the field of view, and temperature / HV
    excursions flag periods whose count rates should not be trusted.
    """
    path = Path(path)
    if not path.exists():
        return None
    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data
        cols = {n.lower(): n for n in data.columns.names}
        out: dict[str, np.ndarray] = {}
        if "mjd" in cols:
            out["time_unix"] = mjd_to_unix(np.asarray(data[cols["mjd"]], dtype=np.float64))
        for key in ("czt1ctr", "czt2ctr", "czt1temp", "czt2temp", "czthvmon",
                    "suninfov", "czt1hotpixcnt", "czt1satctr1"):
            if key in cols:
                out[key] = np.asarray(data[cols[key]], dtype=np.float64)
    return out or None
