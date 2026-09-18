"""Verify an extraction made by unzip_archive.py against the original zips.

    python scripts/verify_extract.py --data-root D:/Data

For every zip: each product folder exists, carries a completion marker whose
content fingerprint matches the zip, and every extracted file has the size and
CRC-32 recorded in the zip's own directory. Read-only; changes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unzip_archive import MARKER, PATTERNS, _fingerprint, _product_units  # noqa: E402


def _crc(path: Path) -> int:
    c = 0
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            c = zlib.crc32(chunk, c)
    return c & 0xFFFFFFFF


def check_zip(z: Path, dest: Path) -> tuple[str, int, int, list[str]]:
    with zipfile.ZipFile(z) as f:
        infos = [i for i in f.infolist() if not i.is_dir()]
    units, root_mode = _product_units([i.filename for i in infos], z.stem)
    problems: list[str] = []
    for u in units:
        m = dest / PurePosixPath(u) / MARKER
        if not m.exists():
            problems.append(f"no marker: {u}")
        elif json.loads(m.read_text(encoding="utf-8")).get("fingerprint") != \
                _fingerprint(infos, u, root_mode):
            problems.append(f"marker fingerprint differs from zip: {u}")
    nbytes = 0
    for i in infos:
        rel = PurePosixPath(z.stem) / i.filename if root_mode else PurePosixPath(i.filename)
        p = dest / rel
        if not p.exists():
            problems.append(f"missing: {i.filename}")
        elif p.stat().st_size != i.file_size:
            problems.append(f"size differs: {i.filename}")
        elif _crc(p) != i.CRC:
            problems.append(f"CRC differs: {i.filename}")
        else:
            nbytes += i.file_size
    return z.name, len(units), nbytes, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--instrument", choices=sorted(PATTERNS) + ["all"], default="all")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    root = Path(args.data_root)
    kinds = sorted(PATTERNS) if args.instrument == "all" else [args.instrument]

    total_problems = 0
    for kind in kinds:
        dest = root / "extracted" / kind
        zips = [z for z in sorted(root.rglob(PATTERNS[kind]))
                if z.suffix == ".zip" and dest not in z.parents]
        t0 = time.time()
        with ThreadPoolExecutor(max(args.threads, 1)) as ex:
            res = list(ex.map(check_zip, zips, [dest] * len(zips)))
        bad = [(n, p) for n, _, _, p in res if p]
        total_problems += len(bad)
        print(f"{kind}: {len(zips)} zips, {sum(r[1] for r in res)} product folders, "
              f"{sum(r[2] for r in res) / 1e9:.1f} GB CRC-verified in "
              f"{time.time() - t0:.0f}s -> {len(bad)} zip(s) with problems")
        for n, p in bad[:20]:
            print(f"  {n}: {p[:3]}")
    return 1 if total_problems else 0


if __name__ == "__main__":
    sys.exit(main())
