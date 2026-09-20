"""Safely unzip downloaded PRADAN archives.

    python scripts/unzip_archive.py --data-root D:/Data --instrument solexs
    python scripts/unzip_archive.py --data-root D:/Data --instrument hel1os --members lightcurves

Designed to be run repeatedly while a download is still in progress:

* **Never deletes or modifies a zip.**
* **Skips work already done.** Each extracted product gets a marker recording
  the zip's size and modification time; a re-downloaded zip is re-extracted.
* **Never exposes a half-extracted product.** Files go to a hidden temporary
  folder first and are renamed into place only once every member has been
  written, so the pipeline can index the destination at any moment.
* **Verifies integrity.** Members are streamed through zipfile, which checks
  each CRC-32 as it reads; a corrupt or truncated zip is reported and skipped.
* **Ignores in-progress downloads** (``*.zip.part``).
* **Never narrows a full extraction.** A product extracted with ``--members all``
  counts as done for any narrower filter, so a light-curves-only pass cannot
  delete event lists already on disk.
* **Refuses path traversal.** A member whose path would land outside the
  destination is rejected rather than written.
* **Guards disk space.** Before each zip it checks that extracting it would
  still leave ``--min-free-gb`` free, and stops cleanly otherwise. HEL1OS event
  lists expand to several GB per file; the full mission does not fit on a
  typical drive if they are included.

Nothing here is needed by the pipeline, which reads SoLEXS zips directly and
gets identical data either way; this exists for people who want the FITS files
on disk for other tools.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

PATTERNS = {
    "solexs": "AL1_SLX_L1_*.zip",
    "hel1os": "HLS_*.zip",
}

#: Member filters. "lightcurves" keeps exactly what the pipeline reads for
#: HEL1OS; "no-events" drops the multi-GB photon event lists.
MEMBER_FILTERS = {
    "all": None,
    "no-events": re.compile(r"^(?!.*/events/)"),
    "lightcurves": re.compile(
        r"/(czt|cdte)/lightcurve_[a-z0-9]+\.fits$|/aux/gti[a-z0-9]+\.fits$|/aux/hk\.fits$"),
}

MARKER = ".extracted.json"
_TMP_PREFIX = ".tmp_"


def _safe_target(root: Path, member: str) -> Path:
    """Resolve a member path under root, rejecting absolute or '..' paths."""
    p = PurePosixPath(member)
    if p.is_absolute() or ".." in p.parts or (p.parts and ":" in p.parts[0]):
        raise ValueError(f"unsafe path in zip: {member!r}")
    target = (root / Path(*p.parts)).resolve()
    if not str(target).startswith(str(root.resolve())):
        raise ValueError(f"path escapes destination: {member!r}")
    return target


#: A folder named like a mission product. Only folders matching this are ever
#: replaced wholesale; shared parents (years, months, days) never are.
_PRODUCT_RE = re.compile(r"^(HLS_|AL1_SLX_L1_)")


def _product_units(names: list[str], zip_stem: str) -> tuple[list[str], bool]:
    """The product folders this zip holds, as paths relative to the zip root.

    Two layouts had to be learned the hard way:

    * HEL1OS zips unpack into a shared date tree (``2024/06/08/HLS_<...>/``).
      Taking the first path component as "the product" meant every new zip
      replaced ``2024/`` -- deleting all earlier extractions of that year.
    * Some HEL1OS zips hold *two* products (``HLS_20260705_030536...zip`` also
      contains ``HLS_20260705_000010...``). Their deepest common folder is the
      shared day folder ``2026/07/05``, so taking the common path as the
      product would delete any other zip's data for that day on re-extraction.

    So a product is the first folder named like one, and a zip may hold
    several. Returns (units, root_mode): in root mode no member sits under a
    product-named folder, and the whole zip gets a folder named after itself.
    """
    units: set[str] = set()
    for n in names:
        parts = PurePosixPath(n).parts
        idx = next((i for i, p in enumerate(parts[:-1]) if _PRODUCT_RE.match(p)), None)
        if idx is None:
            return [zip_stem], True
        units.add("/".join(parts[:idx + 1]))
    return sorted(units), False


def _fingerprint(infos, unit: str, root_mode: bool) -> str:
    """Content fingerprint of one product: member paths, sizes and CRC-32s.

    Read from the zip's central directory, so it costs no decompression. It
    makes "already extracted?" a question about content rather than about which
    zip wrote the folder: some HEL1OS products appear in two different zips, and
    marker-by-zip made each extraction invalidate the other's forever.
    """
    import hashlib

    prefix = "" if root_mode else unit.rstrip("/") + "/"
    rows = sorted((i.filename[len(prefix):], i.file_size, i.CRC)
                  for i in infos if root_mode or i.filename.startswith(prefix))
    return hashlib.sha1(repr(rows).encode()).hexdigest()


_reserve_lock = threading.Lock()
_reserved_bytes = 0


def extract_one(zip_path: Path, dest: Path, member_filter, min_free: int) -> dict:
    global _reserved_bytes
    t0 = time.time()
    st = zip_path.stat()
    rec = {"zip": str(zip_path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    need = 0
    reserved = False
    try:
        with zipfile.ZipFile(zip_path) as z:
            infos = [i for i in z.infolist() if not i.is_dir()]
            if member_filter is not None:
                infos = [i for i in infos if member_filter.search(i.filename)]
            names = [i.filename for i in infos]
            need = sum(i.file_size for i in infos)
            units, root_mode = _product_units(names, zip_path.stem)
            finals = []
            for u in units:
                _safe_target(dest, u + "/x")  # reject traversal in product paths too
                finals.append(dest / Path(*PurePosixPath(u).parts))

            prints = {u: _fingerprint(infos, u, root_mode) for u in units}

            def _current(u: str, final: Path) -> bool:
                m = final / MARKER
                if not m.exists():
                    return False
                old = json.loads(m.read_text(encoding="utf-8"))
                # A full extraction already holds every member a narrower filter
                # would write. Re-extracting would replace it with the narrower
                # set and delete the rest (e.g. 60 GB of HEL1OS event lists).
                if member_filter is not None and old.get("filter") == "all":
                    return True
                if "fingerprint" in old:
                    return old["fingerprint"] == prints[u]
                # Marker from before fingerprints: trust it only for the zip
                # that wrote it, and upgrade it so the next run is content-based.
                if old.get("size") == st.st_size and old.get("mtime_ns") == st.st_mtime_ns:
                    m.write_text(json.dumps({**old, "fingerprint": prints[u]}),
                                 encoding="utf-8")
                    return True
                return False

            if all(_current(u, f) for u, f in zip(units, finals)):
                return {**rec, "status": "skipped"}

            # Reserve space under a lock: with several threads, each could see
            # enough free space on its own while together overrunning the floor.
            with _reserve_lock:
                free = shutil.disk_usage(dest).free - _reserved_bytes
                if free - need < min_free:
                    return {**rec, "status": "no-space",
                            "error": f"needs {need / 1e9:.2f} GB, only "
                                     f"{(free - min_free) / 1e9:.2f} GB above the floor"}
                _reserved_bytes += need
                reserved = True

            tmp = dest / f"{_TMP_PREFIX}{zip_path.stem}"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            for info in infos:
                target = _safe_target(tmp, info.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                # Streaming read verifies the CRC-32 and raises BadZipFile on
                # mismatch at the end of each member.
                with z.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out, length=4 << 20)

        for u, final in zip(units, finals):
            produced = tmp if root_mode else tmp / Path(*PurePosixPath(u).parts)
            if final.exists():
                # A product-named folder (or this zip's own folder) -- never a
                # shared year/month/day parent.
                shutil.rmtree(final)
            final.parent.mkdir(parents=True, exist_ok=True)
            produced.rename(final)
            (final / MARKER).write_text(json.dumps({
                **rec, "members": len(names), "bytes": need, "products": units,
                "fingerprint": prints[u],
                "filter": getattr(member_filter, "pattern", "all"),
            }), encoding="utf-8")
        shutil.rmtree(tmp, ignore_errors=True)
        return {**rec, "status": "ok", "bytes": need, "products": len(units),
                "seconds": round(time.time() - t0, 2)}
    except (zipfile.BadZipFile, OSError, ValueError, EOFError) as exc:
        shutil.rmtree(dest / f"{_TMP_PREFIX}{zip_path.stem}", ignore_errors=True)
        return {**rec, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if reserved:
            with _reserve_lock:
                _reserved_bytes -= need


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True, help="folder containing the downloaded zips")
    ap.add_argument("--dest", default=None, help="default: <data-root>/extracted/<instrument>")
    ap.add_argument("--instrument", choices=sorted(PATTERNS), default="solexs")
    ap.add_argument("--members", choices=sorted(MEMBER_FILTERS), default=None,
                    help="default: 'all' for solexs, 'lightcurves' for hel1os")
    ap.add_argument("--min-free-gb", type=float, default=25.0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    root = Path(args.data_root)
    dest = Path(args.dest) if args.dest else root / "extracted" / args.instrument
    dest.mkdir(parents=True, exist_ok=True)
    members = args.members or ("all" if args.instrument == "solexs" else "lightcurves")
    member_filter = MEMBER_FILTERS[members]
    min_free = int(args.min_free_gb * 1e9)

    zips = sorted(p for p in root.rglob(PATTERNS[args.instrument])
                  if p.suffix == ".zip" and dest not in p.parents)
    print(f"{len(zips)} {args.instrument} zips under {root} -> {dest}  "
          f"(members: {members}, keep >= {args.min_free_gb:g} GB free)", flush=True)

    counts: dict[str, int] = {}
    written = 0
    failures = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(args.threads, 1)) as ex:
        futs = [ex.submit(extract_one, z, dest, member_filter, min_free) for z in zips]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            written += r.get("bytes", 0) if r["status"] == "ok" else 0
            if r["status"] in ("failed", "no-space"):
                failures.append(r)
            if i % 50 == 0 or i == len(zips):
                print(f"  [{i}/{len(zips)}] {counts}  {written / 1e9:.2f} GB written  "
                      f"({time.time() - t0:.0f}s)", flush=True)

    print(f"\ndone: {counts}, {written / 1e9:.2f} GB written, "
          f"{shutil.disk_usage(dest).free / 1e9:.0f} GB free")
    for r in failures[:20]:
        print(f"  {r['status']}: {Path(r['zip']).name}: {r.get('error')}")
    if counts.get("no-space"):
        print("Stopped extracting some files to protect free disk space; "
              "free space or lower --min-free-gb and re-run to continue.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
