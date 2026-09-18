"""Walk a data root and find every SoLEXS and HEL1OS observation in it.

Both missions bury their products a few directories deep, and the depth differs
between releases, so we search by filename signature rather than by fixed path.
Adding more days means dropping more folders in -- nothing here changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .solexs import read_solexs, SolexsObservation
from .hel1os import (Hel1osObservation, product_dir_of, read_hel1os_product,
                     read_housekeeping)


@dataclass
class DataInventory:
    solexs: list[SolexsObservation]
    hel1os: list[Hel1osObservation]
    hk: list[dict]

    def summary(self) -> str:
        lines = ["SoLEXS observations:"]
        if not self.solexs:
            lines.append("  (none)")
        for o in self.solexs:
            lines.append(
                f"  {o.detector} {o.date_obs}  n={o.n}  valid={int(o.valid.sum())}"
                f"  ({100 * o.valid.mean():.1f}%)"
            )
        lines.append("HEL1OS observations:")
        if not self.hel1os:
            lines.append("  (none)")
        for o in self.hel1os:
            frac = 100 * o.any_valid.mean()
            lines.append(
                f"  {o.detector} {o.iso_start[:19]} -> {o.iso_stop[:19]}"
                f"  n={o.n}  sampled={frac:.1f}%  bands={len(o.bands_kev)}"
            )
        return "\n".join(lines)


def discover(data_root: Path) -> DataInventory:
    data_root = Path(data_root)

    # --- SoLEXS: any directory holding a *.pi / *.pi.gz -----------------
    solexs_dirs: set[Path] = set()
    for pattern in ("**/*.pi", "**/*.pi.gz"):
        for p in data_root.glob(pattern):
            solexs_dirs.add(p.parent)

    solexs: list[SolexsObservation] = []
    for d in sorted(solexs_dirs):
        obs = read_solexs(d)
        if obs is not None and obs.valid.any():
            solexs.append(obs)

    # --- HEL1OS: every detector light curve of every product ------------
    hel1os: list[Hel1osObservation] = []
    hk: list[dict] = []
    products = sorted({product_dir_of(p) for p in data_root.glob("**/lightcurve_*.fits")
                       if not any(part.startswith(".tmp_") for part in p.parts)})
    for prod in products:
        hel1os.extend(read_hel1os_product(prod))
        h = read_housekeeping(prod / "aux" / "hk.fits")
        if h is not None:
            hk.append(h)

    solexs.sort(key=lambda o: o.time_unix[0])
    hel1os.sort(key=lambda o: o.time_unix[0])
    return DataInventory(solexs=solexs, hel1os=hel1os, hk=hk)
