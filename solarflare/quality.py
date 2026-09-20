"""Find SoLEXS day files that carry a copy of the previous day's data, and which copy is real.

    python -m solarflare quality

Found 2026-09-19: four PRADAN daily files hold 5-12 h of the previous day's light
curve re-stamped with their own date (the X7 flare of 2024-10-01 22:20 appears
again on 2024-10-02). For every pair of consecutive days in the cache, samples
at the same time of day that agree within 1% are flagged when they form >= 1 h of
matches; GOES-18 XRS-B decides which date the data belong to (the real one
correlates at r ~ 0.99, the copy at r ~ 0). The copies' intervals are written
to outputs/quality/solexs_duplicates.json, which the catalogue
masks when it loads SoLEXS.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from solarflare.settings import load_settings

import numpy as np


from solarflare.io.goes import load_goes

DT = 20
MIN_MATCH_S = 3600


def day_series(cache: Path, e: dict) -> tuple[float, np.ndarray]:
    with np.load(cache / f"{e['key']}.npz") as z:
        names = [str(n) for n in z["names"]]
        v = z["values"][:, names.index("slx_goes_long")].astype(float)
        t = z["time_unix"].astype(float)
        cov = z["coverage"]
    day0 = np.floor(t[0] / 86400.0) * 86400.0
    x = np.full(86400 // DT, np.nan)
    k = ((t - day0) // DT).astype(int)
    ok = (k >= 0) & (k < x.size) & (cov > 0)
    x[k[ok]] = v[ok]
    return day0, x


def goes_r(goes, day0: float, x: np.ndarray, idx: np.ndarray) -> float:
    g = goes.flux_on_grid(day0 + idx * DT)
    ok = np.isfinite(g) & (g > 0) & np.isfinite(x[idx]) & (x[idx] > 0)
    return float(np.corrcoef(np.log10(x[idx][ok]), np.log10(g[ok]))[0, 1]) if ok.sum() > 20 else float("nan")


def main(argv=None) -> int:
    S = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(S.cache))
    ap.add_argument("--goes-dir", default=str(S.goes_dir))
    ap.add_argument("--out", default=str(S.copied_days))
    args = ap.parse_args(argv)
    cache = Path(args.cache_dir)
    entries = [e for e in json.loads((cache / "manifest.json").read_text("utf-8"))
               if e["source"]["kind"] == "solexs" and e.get("status") == "ok"]
    by_date = {e["source"]["date"]: e for e in entries}
    goes = load_goes(Path(args.goes_dir))
    found = []
    dates = sorted(by_date)
    for a, b in zip(dates[:-1], dates[1:]):
        da, xa = day_series(cache, by_date[a])
        db, xb = day_series(cache, by_date[b])
        if db - da != 86400.0:
            continue
        same = np.isfinite(xa) & np.isfinite(xb) & (np.abs(xa - xb) <= 0.01 * np.maximum(np.abs(xa), 1.0))
        if same.sum() * DT < MIN_MATCH_S:
            continue
        idx = np.flatnonzero(same)
        ra, rb = goes_r(goes, da, xa, idx), goes_r(goes, db, xb, idx)
        copy_day, copy_d0, real = (b, db, a) if ra >= rb else (a, da, b)
        # contiguous runs of copied samples (gaps of a few bins bridged)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 15) + 1)
        intervals = [[copy_d0 + r[0] * DT, copy_d0 + (r[-1] + 1) * DT] for r in runs if r.size * DT >= 600]
        found.append({"copy_date": copy_day, "real_date": real, "file": Path(by_date[copy_day]["source"]["path"]).name,
                      "matched_hours": round(same.sum() * DT / 3600.0, 1),
                      "goes_r_real": round(max(ra, rb), 3), "goes_r_copy": round(min(ra, rb), 3),
                      "intervals_unix": intervals,
                      "intervals_utc": [[datetime.fromtimestamp(s, UTC).strftime("%Y-%m-%d %H:%M"),
                                         datetime.fromtimestamp(e, UTC).strftime("%Y-%m-%d %H:%M")] for s, e in intervals]})
        print(f"{copy_day} carries {same.sum() * DT / 3600:.1f} h of {real} (GOES r: real {max(ra, rb):.3f}, "
              f"copy {min(ra, rb):.3f}) in {found[-1]['file']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"generated_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
                                          "days_checked": len(dates), "duplicates": found}, indent=2), encoding="utf-8")
    print(f"{len(found)} copied day(s) among {len(dates)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
