"""SDO/HMI SHARP active-region magnetic parameters, as whole-Sun hourly indicators.

The files come from scripts/download_sharp.py: one CSV per month of the JSOC
series ``hmi.sharp_cea_720s`` sampled hourly, one row per active region (HARP).

Flares happen in regions whose magnetic field is large, sheared and twisted;
those parameters (Bobra & Couvidat 2015) exist hours before any X-rays. Aditya-L1
sees the whole Sun, so the regions are combined into disk totals and maxima.

Kept: rows with QUALITY 0 and a flux-weighted centre within 70 degrees of the
central meridian -- nearer the limb the line-of-sight projection corrupts the
field and the parameters stop meaning what they say.

Causal by construction: the value at time t is the latest hour with T_REC <= t
(and not older than ``max_age_s``); nothing after t is ever used. Callers shift t
back by ``LATENCY_S`` as well: an operator gets the near-real-time SHARP series
some time after the observation, not at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

#: TAI - UTC since 2017 (JSOC times are TAI).
TAI_MINUS_UTC_S = 37.0
MAX_LON_DEG = 70.0
#: Assumed delay between an HMI observation and its SHARP parameters reaching a
#: forecaster (the near-real-time series). Conservative; no result depends on it
#: at the level of an hour.
LATENCY_S = 2 * 3600.0

#: (feature name, source keyword, how regions combine, log10?)
FEATURES = (
    ("sharp_n_regions", None, "count", False),
    ("sharp_log_usflux_sum", "USFLUX", "sum", True),
    ("sharp_log_usflux_max", "USFLUX", "max", True),
    ("sharp_log_totusjh_sum", "TOTUSJH", "sum", True),
    ("sharp_log_totusjh_max", "TOTUSJH", "max", True),
    ("sharp_log_totpot_sum", "TOTPOT", "sum", True),
    ("sharp_log_totpot_max", "TOTPOT", "max", True),
    ("sharp_log_savncpp_sum", "SAVNCPP", "sum", True),
    ("sharp_r_value_max", "R_VALUE", "max", False),
    ("sharp_shrgt45_max", "SHRGT45", "max", False),
    ("sharp_log_area_sum", "AREA_ACR", "sum", True),
)
FEATURE_NAMES = [f[0] for f in FEATURES]


def parse_trec(s: str) -> float:
    """'2024.02.01_13:00:00_TAI' -> unix seconds (UTC)."""
    s = str(s).replace("_TAI", "")
    for fmt in ("%Y.%m.%d_%H:%M:%S", "%Y.%m.%d_%H:%M"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=UTC).timestamp() - TAI_MINUS_UTC_S
        except ValueError:
            continue
    return float("nan")


def _quality_ok(q) -> np.ndarray:
    """QUALITY arrives as an int or a hex string such as '0x00000000'."""
    out = []
    for v in q:
        try:
            out.append(int(str(v), 0) == 0)
        except ValueError:
            out.append(False)
    return np.array(out, bool)


@dataclass
class SharpDisk:
    """Hourly whole-Sun indicators: ``values[i]`` belongs to ``time_unix[i]``."""
    time_unix: np.ndarray
    values: np.ndarray            # (hours, len(FEATURE_NAMES))
    names: list[str]
    n_rows_read: int
    n_rows_kept: int

    def at(self, t: np.ndarray, max_age_s: float = 3 * 3600.0) -> np.ndarray:
        """Indicators known at each time in ``t``: latest hour <= t, NaN if older than max_age_s."""
        t = np.asarray(t, dtype=np.float64)
        out = np.full((t.size, len(self.names)), np.nan)
        if self.time_unix.size == 0:
            return out
        i = np.searchsorted(self.time_unix, t, side="right") - 1
        ok = (i >= 0)
        ok[ok] &= (t[ok] - self.time_unix[i[ok]]) <= max_age_s
        out[ok] = self.values[i[ok]]
        return out


def load_sharp(sharp_dir: str | Path) -> SharpDisk | None:
    """Read every ``sharp_*.csv`` under ``sharp_dir``; None if there are none."""
    import pandas as pd

    frames = []
    for f in sorted(Path(sharp_dir).glob("sharp_*.csv")):
        try:
            d = pd.read_csv(f)
        except pd.errors.EmptyDataError:        # a month JSOC has not published yet
            continue
        if len(d) and "T_REC" in d:
            frames.append(d)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    n_read = len(df)
    df["t"] = [parse_trec(s) for s in df["T_REC"]]
    keep = np.isfinite(df["t"].to_numpy())
    if "QUALITY" in df:
        keep &= _quality_ok(df["QUALITY"].to_numpy())
    if "LON_FWT" in df:
        keep &= np.abs(pd.to_numeric(df["LON_FWT"], errors="coerce").to_numpy()) <= MAX_LON_DEG
    df = df[keep]
    for _, key, _, _ in FEATURES:
        if key and key in df:
            df[key] = pd.to_numeric(df[key], errors="coerce").abs()
    g = df.groupby("t")
    cols = []
    for name, key, how, log in FEATURES:
        if key is None:
            v = g.size().astype(float)
        elif key not in df:
            v = g.size() * np.nan
        else:
            v = getattr(g[key], how)()
        v = v.astype(float)
        if log:
            v = np.log10(v.where(v > 0))
        cols.append(v.rename(name))
    table = pd.concat(cols, axis=1).sort_index()
    table = table[~table.index.duplicated()]
    return SharpDisk(table.index.to_numpy(dtype=np.float64), table.to_numpy(dtype=np.float64),
                     list(FEATURE_NAMES), n_read, int(keep.sum()))
