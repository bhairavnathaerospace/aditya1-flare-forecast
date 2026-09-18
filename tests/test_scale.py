"""Tests for running on a mission archive rather than a two-day sample.

Each check targets a failure that only appears at scale, and that the
correctness and robustness suites could not see:

* a flare crossing midnight split in two by per-day processing
* the cache re-processing everything, or failing to notice a changed file
* a re-processed day (v1.0 and v1.1) counted twice
* zip reading diverging from reading extracted files
* the chunked rolling percentile disagreeing with the exact one
* per-segment splitting leaking the future across hundreds of segments
* quiet-window thinning dropping flare windows, or touching val/test
* HEL1OS-only windows being trained as "quiet Sun"

    python -m tests.test_scale
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarflare.config import Config, PreprocessConfig  # noqa: E402

FAILURES: list[str] = []
DAY = 86400.0


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Synthetic PRADAN-style SoLEXS day, with flares, written as a zip
# ---------------------------------------------------------------------------

def _solexs_day(t0: float, flares=(), seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """One day of 1 s, 340-channel spectra. Flares: (peak_s, rise_s, decay_s, amp)."""
    rng = np.random.default_rng(seed)
    n = 86400
    spec = rng.poisson(0.02, size=(n, 340)).astype(np.float64)
    sec = t0 + np.arange(n, dtype=float)
    chans = np.arange(45, 161)
    w = np.exp(-(chans - 45) / 30.0)
    w /= w.sum()
    for peak, rise, decay, amp in flares:
        prof = np.where(sec <= peak, np.clip(1.0 - (peak - sec) / rise, 0, None),
                        np.exp(-(sec - peak) / decay))
        spec[:, chans] += rng.poisson(amp * prof[:, None] * w[None, :])
    return sec, spec


def _write_day_zip(dirpath: Path, date: str, t0: float, flares=(), version="v1.0",
                   seed: int = 0) -> Path:
    from astropy.io import fits
    import gzip
    import io

    sec, spec = _solexs_day(t0, flares, seed)
    n = sec.size
    base = f"AL1_SLX_L1_{date}_{version}"

    def gz_fits(hdul) -> bytes:
        buf = io.BytesIO()
        hdul.writeto(buf)
        return gzip.compress(buf.getvalue(), compresslevel=1)

    pi = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(fits.ColDefs([
        fits.Column(name="TSTART", format="D", array=sec),
        fits.Column(name="TELAPSE", format="D", array=np.ones(n)),
        fits.Column(name="SPEC_NUM", format="J", array=np.arange(n)),
        fits.Column(name="CHANNEL", format="340K", array=np.tile(np.arange(340), (n, 1))),
        fits.Column(name="COUNTS", format="340D", array=spec),
        fits.Column(name="EXPOSURE", format="D", array=np.ones(n)),
    ]), name="SPECTRUM")])
    pi[0].header["OBS_DATE"] = date
    lc = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(fits.ColDefs([
        fits.Column(name="TIME", format="D", array=sec),
        fits.Column(name="COUNTS", format="D", array=spec[:, 41:].sum(axis=1)),
    ]), name="RATE")])
    gti = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(fits.ColDefs([
        fits.Column(name="START", format="D", array=np.array([sec[0]])),
        fits.Column(name="STOP", format="D", array=np.array([sec[-1]])),
    ]), name="GTI")])

    dirpath.mkdir(parents=True, exist_ok=True)
    out = dirpath / f"{base}.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr(f"{base}/SDD1/AL1_SOLEXS_{date}_SDD1_L1.gti.gz", gz_fits(gti))
        z.writestr(f"{base}/SDD2/AL1_SOLEXS_{date}_SDD2_L1.pi.gz", gz_fits(pi))
        z.writestr(f"{base}/SDD2/AL1_SOLEXS_{date}_SDD2_L1.lc.gz", gz_fits(lc))
        z.writestr(f"{base}/SDD2/AL1_SOLEXS_{date}_SDD2_L1.gti.gz", gz_fits(gti))
    return out


# 2026-01-01 00:00:00 UTC, on a 20 s boundary.
T0 = 1767225600.0


def _archive(root: Path, n_days: int, flares_per_day: dict[int, tuple] | None = None):
    flares_per_day = flares_per_day or {}
    for d in range(n_days):
        t0 = T0 + d * DAY
        date = time.strftime("%Y%m%d", time.gmtime(t0))
        _write_day_zip(root, date, t0, flares_per_day.get(d, ()), seed=d)


# ---------------------------------------------------------------------------

def test_chunked_percentile_is_exact():
    import solarflare.preprocess.grid as g
    rng = np.random.default_rng(1)
    x = rng.lognormal(size=5000)
    x[rng.random(5000) < 0.2] = np.nan
    saved = g._PERCENTILE_CHUNK
    try:
        g._PERCENTILE_CHUNK = 1 << 30
        ref_t = g.trailing_percentile(x, 777, 10)
        ref_c = g.running_percentile(x, 777, 10)
        g._PERCENTILE_CHUNK = 333          # many chunk boundaries
        got_t = g.trailing_percentile(x, 777, 10)
        got_c = g.running_percentile(x, 777, 10)
    finally:
        g._PERCENTILE_CHUNK = saved
    check("chunked trailing percentile == single-chunk", np.allclose(ref_t, got_t, equal_nan=True))
    check("chunked centred percentile == single-chunk", np.allclose(ref_c, got_c, equal_nan=True))


def test_zip_reader_matches_directory_reader():
    from solarflare.io.solexs import read_solexs, read_solexs_zip, list_zip_detectors
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        z = _write_day_zip(root, "20260101", T0, ((T0 + 40000, 600, 900, 60),))
        with zipfile.ZipFile(z) as zz:
            zz.extractall(root / "x")
        ex = next((root / "x").rglob("SDD2"))
        a, b = read_solexs_zip(z, "SDD2"), read_solexs(ex)
        same = all(np.array_equal(getattr(a, f), getattr(b, f), equal_nan=True)
                   for f in ("time_unix", "spectra", "exposure", "valid", "gti", "lc_counts"))
        check("zip and extracted-directory reads are identical", same)
        check("zip detector listing ignores SDDs without spectra",
              list_zip_detectors(z) == ["SDD2"], str(list_zip_detectors(z)))


def test_index_deduplicates_versions_and_ignores_partial_downloads():
    from solarflare.preprocess.cache import index_sources
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_day_zip(root / "a", "20260101", T0, version="v1.0")
        _write_day_zip(root / "b", "20260101", T0, version="v1.1")
        part = _write_day_zip(root / "c", "20260102", T0 + DAY)
        part.rename(part.with_name(part.name + ".part"))
        src = [s for s in index_sources([root]) if s.kind == "solexs"]
        check("two versions of one day are indexed once", len(src) == 1, str(len(src)))
        check("...and the higher version wins",
              bool(src) and src[0].version == "v1.1", str([s.version for s in src]))
        check("a .zip.part (download in progress) is never indexed",
              not any(s.date == "20260102" for s in src))


def test_cache_is_incremental_and_invalidates_correctly():
    from solarflare.preprocess.cache import index_sources, build_cache
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _archive(root / "data", 3)
        cfg = PreprocessConfig()
        cdir = root / "cache"
        src = index_sources([root / "data"])

        e1 = build_cache(src, cfg, cdir, workers=1, verbose=False)
        keys1 = {e["key"] for e in e1}
        check("first build caches every day",
              sum(e["status"] == "ok" for e in e1) == 3, str([e["status"] for e in e1]))

        t = time.time()
        e2 = build_cache(src, cfg, cdir, workers=1, verbose=False)
        check("second build reuses every entry", {e["key"] for e in e2} == keys1)
        check("...without re-reading the data", time.time() - t < 1.0,
              f"{time.time() - t:.2f}s")

        z = Path(src[0].path)
        st = z.stat()
        os.utime(z, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # "re-downloaded"
        e3 = build_cache(src, cfg, cdir, workers=1, verbose=False)
        check("a changed file (new mtime) rebuilds exactly that one entry",
              len({e["key"] for e in e3} - keys1) == 1)

        cfg2 = PreprocessConfig(dt_seconds=60.0)
        e4 = build_cache(src, cfg2, cdir, workers=1, verbose=False)
        check("changing a raw-rate setting (grid) rebuilds everything",
              not ({e["key"] for e in e4} & {e["key"] for e in e3}))

        cfg3 = PreprocessConfig(background_window_s=3600.0)  # derive-stage only
        e5 = build_cache(src, cfg3, cdir, workers=1, verbose=False)
        check("changing a derive-stage setting (background) rebuilds nothing",
              {e["key"] for e in e5} == {e["key"] for e in e3})


def test_flare_across_midnight_is_one_event():
    """The reason stitching exists."""
    from solarflare.preprocess.cache import index_sources, build_cache, load_cached
    from solarflare.preprocess.dataset import build_segments_from_raw

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        # Peaks 5 minutes before midnight, decays for an hour into day 2.
        peak = T0 + DAY - 300
        _archive(root / "data", 2, {0: ((peak, 900, 1800, 150),),
                                    1: ((peak, 900, 1800, 150),)})
        cfg = Config(data_root=root / "data", out_dir=root / "out")
        src = index_sources([root / "data"])
        entries = build_cache(src, cfg.pre, root / "cache", workers=1, verbose=False)
        raws = load_cached(entries, root / "cache", "solexs")

        stitched, _ = build_segments_from_raw(raws, [], cfg)
        per_day = [build_segments_from_raw([r], [], cfg)[0] for r in raws]

        ev_stitched = [e for s in stitched for e in s.events]
        near = [e for e in ev_stitched if abs(e.peak_unix - peak) < 600]
        check("stitched days form one continuous segment", len(stitched) == 1,
              str(len(stitched)))
        check("the midnight flare is detected exactly once", len(near) == 1,
              f"{len(near)} events near the peak")
        if near:
            check("...and its decay continues past midnight",
                  near[0].end_unix > T0 + DAY + 600,
                  f"ends {near[0].end_unix - (T0 + DAY):.0f} s after midnight")
        pieces = sum(len(s[0].events) for s in per_day)
        check("processing each day separately does NOT give the same answer "
              "(this is what stitching fixes)", pieces != len(near),
              f"per-day events {pieces}")


def test_global_split_and_thinning():
    from solarflare.preprocess.cache import index_sources, build_cache, load_cached
    from solarflare.preprocess.dataset import (
        build_segments_from_raw, enumerate_windows, chronological_split,
        thin_quiet_training_windows, resolve_split_mode,
    )

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        n_days = 6
        flares = {k: ((T0 + k * DAY + 30000 + 5000 * (k % 3), 600, 1200, 80 + 20 * k),)
                  for k in range(n_days)}
        _archive(root / "data", n_days, flares)
        cfg = Config(data_root=root / "data", out_dir=root / "out")
        cfg.win.large_data_days = 3.0            # force archive behaviour
        cfg.win.stride_seconds = 120.0
        src = index_sources([root / "data"])
        entries = build_cache(src, cfg.pre, root / "cache", workers=1, verbose=False)
        segs, _ = build_segments_from_raw(load_cached(entries, root / "cache", "solexs"),
                                          [], cfg)
        win = enumerate_windows(segs, cfg)
        mode = resolve_split_mode(segs, cfg)
        check("auto split mode switches to global above the archive threshold",
              mode == "global", mode)

        tr, va, te = chronological_split(win, cfg, mode="global")
        t_tr = max(win[i].t_unix for i in tr)
        check("every validation window is after all training + embargo",
              all(win[i].t_unix > t_tr + cfg.train.embargo_s for i in va))
        check("every test window is after all validation + embargo",
              all(win[i].t_unix > max(win[j].t_unix for j in va) + cfg.train.embargo_s
                  for i in te))
        check("splits are disjoint", not (set(tr) & set(va) | set(va) & set(te)))

        thin = thin_quiet_training_windows(segs, win, tr, cfg)
        L = cfg.steps_per_window
        from solarflare.preprocess.dataset import _max_horizon_steps
        H = _max_horizon_steps(cfg)

        def active(i):
            w = win[i]
            s = segs[w.seg]
            return s.in_flare[w.end - L: min(w.end + H, len(s))].any()

        act = [i for i in tr if active(i)]
        check("thinning keeps every flare-adjacent training window",
              set(act) <= set(thin), f"dropped {len(set(act) - set(thin))}")
        quiet_all = len(tr) - len(act)
        quiet_kept = len(thin) - len(act)
        check("thinning removes most quiet training windows",
              quiet_all == 0 or quiet_kept < 0.5 * quiet_all,
              f"kept {quiet_kept}/{quiet_all}")


def test_phase_loss_ignores_windows_without_soft_truth():
    import torch
    from solarflare.models.losses import focal_ce
    logits = torch.tensor([[5.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([2, 0])            # second row: placeholder "quiet"
    mask = torch.tensor([1.0, 0.0])          # ...with no soft X-ray truth
    only_first = focal_ce(logits[:1], target[:1])
    masked = focal_ce(logits, target, mask=mask)
    check("masked phase loss equals the loss on truthful windows only",
          torch.allclose(masked, only_first), f"{masked.item()} vs {only_first.item()}")


def test_prepare_end_to_end_archive_mode():
    """prepare() on a small synthetic archive, with archive mode forced."""
    from solarflare.pipeline import prepare
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        flares = {k: ((T0 + k * DAY + 43000, 600, 1200, 100),) for k in range(5)}
        _archive(root / "data", 5, flares)
        cfg = Config(data_root=root / "data", out_dir=root / "out")
        cfg.win.large_data_days = 2.0
        prep = prepare(cfg, verbose=False)
        check("archive mode engages", prep.archive)
        check("stride raised to the archive stride",
              cfg.win.stride_seconds == cfg.win.large_stride_seconds)
        check("epoch cap applied", cfg.train.max_batches_per_epoch ==
              cfg.train.large_max_batches_per_epoch)
        check("all five days detected their flare",
              prep.meta["n_events"] >= 5, str(prep.meta["n_events"]))
        check("meta records the split mode", prep.meta["split_mode"] == "global")
        shutil.rmtree(root / "out", ignore_errors=True)


def test_bootstrap_is_fast_and_unchanged_at_archive_scale():
    """The per-event bootstrap must give the same interval as the direct
    implementation, and finish in seconds for thousands of events."""
    from solarflare.forecast import bootstrap_ci
    rng = np.random.default_rng(5)
    groups = rng.integers(0, 60, 3000)
    vals = rng.normal(size=3000) + groups * 0.01

    def stat(idx):
        return float(np.mean(vals[idx]))

    def direct(n_boot=200, seed=0):
        r = np.random.default_rng(seed)
        uniq = np.unique(groups)
        draws = []
        for _ in range(n_boot):
            pick = r.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.where(groups == g)[0] for g in pick])
            draws.append(stat(idx))
        return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    _, lo, hi = bootstrap_ci(None, groups, stat, n_boot=200)
    rlo, rhi = direct()
    check("bootstrap interval identical to the direct implementation",
          abs(lo - rlo) < 1e-12 and abs(hi - rhi) < 1e-12, f"{lo},{hi} vs {rlo},{rhi}")
    big = rng.integers(0, 3000, 60000)
    t = time.time()
    bootstrap_ci(None, big, lambda idx: float(np.mean(big[idx])), n_boot=100)
    check("3000 events x 60k samples bootstraps in under 20 s", time.time() - t < 20,
          f"{time.time() - t:.1f}s")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} scale test groups\n")
    for t in tests:
        print(f"{t.__name__}:")
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  ERROR {t.__name__}: {type(exc).__name__}: {exc}")
            FAILURES.append(f"{t.__name__} (exception)")
        print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("All scale checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
