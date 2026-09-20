"""Per-file preprocessing cache: raw mission products -> compact gridded rates.

A SoLEXS day is ~470 MB of spectra once decompressed and ~1.3 GB of peak
memory to read; the gridded band rates the model actually needs are ~0.4 MB.
Holding every day's spectra in memory -- which is what the original pipeline
did -- stops working at roughly a hundred days on a 24 GB machine, while the
mission archive is ~1000 days.

So stage 1 of preprocessing runs once per downloaded file and writes a small
``.npz``. Properties that matter for a download still in progress:

* **Incremental.** A file already cached is skipped, so re-running after more
  data arrives processes only the new files.
* **Invalidated correctly.** The cache key covers the file's path, size and
  modification time *and* every setting that changes the raw rates (grid,
  bands, gain, channel threshold, ...). Change one and exactly the affected
  entries rebuild; change a model setting and nothing does.
* **Failure-tolerant.** A truncated or corrupt zip is recorded in the manifest
  and skipped; it never aborts the run, and it is retried next time.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .. import config as C
from ..config import PreprocessConfig
from .grid import GriddedSeries

#: Bump when the *meaning* of cached arrays changes, to invalidate everything.
CACHE_VERSION = 1


@dataclass
class Source:
    kind: str        # "solexs" | "hel1os"
    fmt: str         # "zip" | "dir" | "fits"
    path: str
    detector: str
    date: str        # YYYYMMDD where known, else ""
    version: str     # product version where known, e.g. "v1.1"


_SLX_ZIP = re.compile(r"AL1_SLX_L1_(\d{8})_(v[\d.]+)\.zip$")
_HLS_PRODUCT = re.compile(r"HLS_(\d{8})_\d{6}_\d+sec_lev\d+_V(\d+)")

#: Bump when the HEL1OS reader or raw layout changes meaning. Kept separate
#: from CACHE_VERSION so a HEL1OS change does not rebuild ~800 SoLEXS days.
HEL1OS_READER_VERSION = 2


def version_priority(version: str) -> float:
    """"V212" -> 212.0, "v1.1" -> 1.1; unknown -> 0. Higher wins in stitching."""
    m = re.search(r"\d+(?:\.\d+)?", version or "")
    return float(m.group(0)) if m else 0.0


def index_sources(roots: list[Path]) -> list[Source]:
    """Find every usable product under the given roots, without reading data.

    Partially downloaded files (``*.zip.part``) never match. When the same
    SoLEXS day exists more than once -- a re-processed version, or both a zip
    and an extracted copy -- only the highest version is kept, because feeding
    two copies of one day would double-count its flares.
    """
    from ..io.hel1os import product_dir_of
    from ..io.solexs import list_zip_detectors

    found: list[Source] = []
    for root in roots:
        root = Path(root)
        for z in root.rglob("AL1_SLX_L1_*.zip"):
            m = _SLX_ZIP.search(z.name)
            if not m:
                continue
            try:
                dets = list_zip_detectors(z)
            except Exception:  # noqa: BLE001 - corrupt/incomplete zip
                dets = ["SDD2"]  # let the build step record the failure
            for det in dets:
                found.append(Source("solexs", "zip", str(z), det, m.group(1), m.group(2)))

        for pi in list(root.rglob("*.pi")) + list(root.rglob("*.pi.gz")):
            # scripts/unzip_archive.py writes into hidden ".tmp_*" folders and
            # renames them only when complete; never index a half-written day.
            if any(part.startswith(".tmp_") for part in pi.parts):
                continue
            sdd = pi.parent
            m = re.search(r"AL1_SLX_L1_(\d{8})_(v[\d.]+)", str(sdd))
            found.append(Source("solexs", "dir", str(sdd), sdd.name,
                                m.group(1) if m else "", m.group(2) if m else ""))

        # HEL1OS: one source per *product* (all detectors of one observation),
        # so the detectors are merged column-by-column before any stitching.
        products: set[Path] = set()
        for lc in root.rglob("lightcurve_*.fits"):
            if any(part.startswith(".tmp_") for part in lc.parts):
                continue  # extraction still in progress
            products.add(product_dir_of(lc))
        for prod in products:
            m = _HLS_PRODUCT.search(prod.name)
            found.append(Source("hel1os", "product", str(prod), "ALL",
                                m.group(1) if m else "", f"V{m.group(2)}" if m else ""))

    # De-duplicate SoLEXS days: keep the highest version, prefer zip on ties
    # (it is the unmodified download).
    def vkey(s: Source):
        nums = tuple(int(x) for x in re.findall(r"\d+", s.version)) or (0,)
        return (nums, s.fmt == "zip")

    best: dict[tuple, Source] = {}
    others: list[Source] = []
    for s in found:
        if s.kind == "solexs" and s.date:
            k = (s.date, s.detector)
            if k not in best or vkey(s) > vkey(best[k]):
                best[k] = s
        else:
            others.append(s)
    seen_paths: set[str] = set()
    out = []
    for s in sorted(list(best.values()) + others, key=lambda s: (s.kind, s.date, s.path)):
        key = (s.kind, s.fmt, s.path, s.detector)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        out.append(s)
    return out


def _settings_fingerprint(cfg: PreprocessConfig) -> dict:
    """Everything that changes the *raw* cached rates -- and nothing else.

    The legacy energy scale reproduces the fingerprint exactly as it was
    before scales were selectable, so caches built for models trained on it
    stay valid; the published scale adds its calibration, so switching scale
    rebuilds only SoLEXS entries.
    """
    scale = cfg.solexs_energy_scale
    fp = {
        "cache_version": CACHE_VERSION,
        "dt_seconds": cfg.dt_seconds,
        "min_valid_fraction": cfg.min_valid_fraction,
        "solexs_bands": [list(b) for b in C.SOLEXS_BANDS_BY_SCALE[scale]],
        "goes_long": list(C.GOES_LONG_KEV),
        "goes_short": list(C.GOES_SHORT_BY_SCALE[scale]),
        "ch_lo": C.SOLEXS_CH_LO,
        "ch_hi": C.SOLEXS_CH_HI,
        "hel1os_fill_is_missing": C.HEL1OS_FILL_IS_MISSING,
    }
    if scale == "legacy_linear":
        fp["gain"] = C.SOLEXS_LEGACY_GAIN_KEV
        fp["offset"] = C.SOLEXS_LEGACY_OFFSET_KEV
    else:
        fp["energy_scale"] = scale
        fp["calibration"] = C.SOLEXS_CAL_SARWADE2025
    return fp


def cache_key(src: Source, cfg: PreprocessConfig) -> str:
    p = Path(src.path)
    settings = _settings_fingerprint(cfg)
    if src.kind == "hel1os" and src.fmt == "product":
        from ..io.hel1os import gti_path_for, product_lightcurves
        from .features import HEL1OS_RAW_NAMES
        files = product_lightcurves(p)
        files += [g for g in map(gti_path_for, files) if g.exists()]
        if not files:
            raise FileNotFoundError(f"no light curves left in {p}")
        stats = [f.stat() for f in files]
        size = [s.st_size for s in stats]
        mtime = [s.st_mtime_ns for s in stats]
        settings = {**settings, "hel1os_reader": HEL1OS_READER_VERSION,
                    "hel1os_layout": HEL1OS_RAW_NAMES,
                    "files": [f.relative_to(p).as_posix() for f in files]}
    else:
        st = p.stat()
        # For a directory, size/mtime of the spectrum file is what matters.
        if p.is_dir():
            pis = sorted(list(p.glob("*.pi")) + list(p.glob("*.pi.gz")))
            st = pis[0].stat() if pis else st
        size, mtime = st.st_size, st.st_mtime_ns
    blob = json.dumps({
        "src": [src.kind, src.fmt, str(p.resolve()).lower(), src.detector],
        "size": size,
        "mtime_ns": mtime,
        "settings": settings,
    }, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:20]


def _process(src: Source, cfg: PreprocessConfig, out_path: str) -> dict:
    """Worker: read one product, write its gridded raw rates. Top-level so it
    pickles into a subprocess on Windows."""
    t0 = time.time()
    entry = {"source": asdict(src), "key": Path(out_path).stem}
    try:
        if src.kind == "solexs":
            from ..io.solexs import read_solexs, read_solexs_zip, verify_lc_consistency
            from .features import solexs_raw
            obs = (read_solexs_zip(Path(src.path), src.detector) if src.fmt == "zip"
                   else read_solexs(Path(src.path)))
            if obs is None or not obs.valid.any():
                entry.update(status="empty", seconds=round(time.time() - t0, 2))
                return entry
            lc = verify_lc_consistency(obs)
            raw = solexs_raw(obs, cfg)
            entry["lc_check"] = lc
            del obs
        else:
            from ..io.hel1os import read_hel1os_product
            from .features import hel1os_product_raw
            observations = read_hel1os_product(Path(src.path))
            if not observations:
                entry.update(status="empty", seconds=round(time.time() - t0, 2))
                return entry
            raw = hel1os_product_raw(observations, cfg)
            entry["detectors"] = sorted(o.detector_key for o in observations)
            del observations

        np.savez_compressed(
            out_path,
            time_unix=raw.time_unix, values=raw.values, coverage=raw.coverage,
            names=np.array(raw.names),
        )
        observed = raw.coverage > 0
        entry.update(
            status="ok",
            t_start=float(raw.time_unix[0]),
            t_stop=float(raw.time_unix[-1] + cfg.dt_seconds),
            n_bins=int(raw.time_unix.size),
            observed_fraction=float(observed.mean()),
            seconds=round(time.time() - t0, 2),
        )
    except Exception as exc:  # noqa: BLE001 - record, never abort the sweep
        entry.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                     seconds=round(time.time() - t0, 2))
    return entry


def build_cache(sources: list[Source], cfg: PreprocessConfig, cache_dir: Path,
                workers: int = 3, verbose: bool = True) -> list[dict]:
    """Bring the cache up to date for ``sources``; return manifest entries.

    Previously cached entries are reused without touching the source file.
    ``workers`` bounds memory: each worker peaks around 1.5 GB on a SoLEXS day.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    old = {}
    if manifest_path.exists():
        try:
            old = {e["key"]: e for e in json.loads(manifest_path.read_text("utf-8"))}
        except (json.JSONDecodeError, KeyError, TypeError):
            old = {}

    entries: list[dict] = []
    todo: list[tuple[Source, str]] = []
    for src in sources:
        try:
            key = cache_key(src, cfg)
        except OSError as exc:  # vanished mid-download, permissions, ...
            entries.append({"source": asdict(src), "key": "", "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}"})
            continue
        out = cache_dir / f"{key}.npz"
        prev = old.get(key)
        if out.exists() and prev and prev.get("status") == "ok" or prev and prev.get("status") == "empty":
            entries.append(prev)
        else:
            todo.append((src, str(out)))

    if verbose:
        print(f"cache: {len(sources)} sources, {len(entries)} up to date, "
              f"{len(todo)} to process with {workers} worker(s)")

    def _save_manifest():
        # Keep entries built under other settings (another energy scale, a
        # different grid) whose files still exist. Writing only this run's
        # entries made two configurations sharing a cache folder erase each
        # other's index, so each run rebuilt everything the other had cached.
        current = {e.get("key") for e in entries}
        kept = [e for k, e in old.items()
                if k not in current and (cache_dir / f"{k}.npz").exists()]
        manifest_path.write_text(json.dumps(entries + kept, indent=1), encoding="utf-8")

    t0 = time.time()
    if todo:
        if workers <= 1:
            results = (_process(s, cfg, o) for s, o in todo)
            for i, e in enumerate(results, 1):
                entries.append(e)
                _report(e, i, len(todo), t0, verbose)
                if i % 25 == 0:
                    _save_manifest()
        else:
            from concurrent.futures.process import BrokenProcessPool

            done_paths: set[str] = set()
            try:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futs = {ex.submit(_process, s, cfg, o): o for s, o in todo}
                    for i, f in enumerate(as_completed(futs), 1):
                        e = f.result()
                        entries.append(e)
                        done_paths.add(futs[f])
                        _report(e, i, len(todo), t0, verbose)
                        if i % 25 == 0:
                            _save_manifest()
            except BrokenProcessPool as exc:
                # A worker died hard (out of memory, or an interpreter that
                # cannot spawn children). Keep everything already finished and
                # carry on in-process rather than discarding the run.
                remaining = [(s, o) for s, o in todo if o not in done_paths]
                if verbose:
                    print(f"cache: worker pool broke ({exc}); finishing "
                          f"{len(remaining)} file(s) in-process")
                for j, (s, o) in enumerate(remaining, 1):
                    e = _process(s, cfg, o)
                    entries.append(e)
                    _report(e, len(done_paths) + j, len(todo), t0, verbose)
                    if j % 25 == 0:
                        _save_manifest()

    _save_manifest()
    if verbose:
        n_ok = sum(e.get("status") == "ok" for e in entries)
        n_fail = sum(e.get("status") == "failed" for e in entries)
        n_empty = sum(e.get("status") == "empty" for e in entries)
        print(f"cache: {n_ok} ok, {n_empty} empty, {n_fail} failed "
              f"({time.time() - t0:.0f}s this run)")
        bad_lc = [e for e in entries if e.get("lc_check", {}).get("checked")
                  and not e["lc_check"].get("exact")]
        if bad_lc:
            print(f"cache: WARNING {len(bad_lc)} day(s) where the pipeline light "
                  f"curve != sum of channels {C.SOLEXS_CH_LO}+ -- the channel "
                  f"threshold may differ for that release; inspect before trusting.")
    return entries


def _report(e: dict, i: int, n: int, t0: float, verbose: bool) -> None:
    if not verbose:
        return
    if e.get("status") == "failed" or i == n or i % 10 == 0:
        rate = i / max(time.time() - t0, 1e-9)
        eta = (n - i) / max(rate, 1e-9)
        src = Path(e["source"]["path"]).name
        extra = f" FAILED {e.get('error')}" if e.get("status") == "failed" else ""
        print(f"  [{i}/{n}] {src}{extra}   ({rate * 60:.0f} files/min, ~{eta / 60:.0f} min left)",
              flush=True)


def mask_intervals(series: list[GriddedSeries], path: str | Path) -> int:
    """Blank (NaN value, zero coverage) every sample inside the intervals listed in
    ``path`` -- a solexs_duplicates.json-style file. Returns the samples blanked."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    intervals = [(float(a), float(b)) for d in data.get("duplicates", []) for a, b in d["intervals_unix"]]
    n = 0
    for s in series:
        bad = np.zeros(s.time_unix.size, bool)
        for a, b in intervals:
            bad |= (s.time_unix >= a) & (s.time_unix < b)
        if bad.any():
            s.values[bad] = np.nan
            s.coverage[bad] = 0
            n += int(bad.sum())
    return n


def load_cached(entries: list[dict], cache_dir: Path, kind: str) -> list[GriddedSeries]:
    """Load every successfully cached product of one instrument, time-sorted."""
    out = []
    for e in entries:
        if e.get("status") != "ok" or e["source"]["kind"] != kind:
            continue
        z = np.load(Path(cache_dir) / f"{e['key']}.npz")
        out.append(GriddedSeries(z["time_unix"], z["values"], z["coverage"],
                                 [str(n) for n in z["names"]],
                                 priority=version_priority(e["source"].get("version", ""))))
    out.sort(key=lambda s: float(s.time_unix[0]))
    return out
