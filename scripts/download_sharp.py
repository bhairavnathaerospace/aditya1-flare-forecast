"""Download SDO/HMI SHARP active-region magnetic parameters (numbers only, no images).

    pip install drms
    python scripts/download_sharp.py                     # 2024-02 -> 2026-09, hourly, into D:/Data/sharp
    python scripts/download_sharp.py --start 2024-02 --end 2026-09 --out D:/Data/sharp

Source: the Joint Science Operations Center (JSOC, Stanford), series
``hmi.sharp_cea_720s``: one row per active region (HARP) every 12 min, with the
magnetic summary parameters used in flare forecasting (Bobra & Couvidat 2015).
Sampled hourly (``@1h``) -- enough for hours-to-a-day forecasts and ~50x smaller.
Keyword queries need no JSOC account.

One CSV per month (``sharp_YYYY-MM.csv``, roughly 1-2 MB each). Months already
downloaded are skipped, so the script can be stopped and re-run safely. A month
that fails is reported and can be retried.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

SERIES = "hmi.sharp_cea_720s"
KEYS = ["T_REC", "HARPNUM", "NOAA_AR", "NOAA_NUM", "NOAA_ARS", "QUALITY", "LAT_FWT", "LON_FWT",
        "USFLUX", "MEANGAM", "MEANGBT", "MEANGBZ", "MEANGBH", "MEANJZD", "TOTUSJZ", "MEANALP",
        "MEANJZH", "TOTUSJH", "ABSNJZH", "SAVNCPP", "MEANPOT", "TOTPOT", "MEANSHR", "SHRGT45",
        "R_VALUE", "AREA_ACR"]


def months(start: str, end: str):
    y, m = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    while (y, m) <= (ye, me):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-02", help="first month, YYYY-MM")
    ap.add_argument("--end", default=f"{date.today():%Y-%m}", help="last month, YYYY-MM")
    ap.add_argument("--out", default="D:/Data/sharp")
    ap.add_argument("--cadence", default="1h", help="JSOC sampling, e.g. 1h or 12m (native)")
    args = ap.parse_args()
    try:
        import drms
    except ImportError:
        print("This needs the 'drms' package:  pip install drms")
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = drms.Client()
    failed = []
    for y, m in months(args.start, args.end):
        dest = out / f"sharp_{y:04d}-{m:02d}.csv"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"{dest.name}: already here, skipped")
            continue
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        query = f"{SERIES}[][{y:04d}.{m:02d}.01_00:00_TAI-{ny:04d}.{nm:02d}.01_00:00_TAI@{args.cadence}]"
        for attempt in range(3):
            try:
                t0 = time.time()
                df = client.query(query, key=",".join(KEYS))
                tmp = dest.with_suffix(".csv.part")
                df.to_csv(tmp, index=False)
                tmp.replace(dest)
                print(f"{dest.name}: {len(df)} rows, {df['HARPNUM'].nunique() if len(df) else 0} regions "
                      f"({time.time() - t0:.0f} s)", flush=True)
                break
            except Exception as exc:                 # network hiccups: retry, then move on
                print(f"{dest.name}: attempt {attempt + 1} failed: {exc}", flush=True)
                time.sleep(20 * (attempt + 1))
        else:
            failed.append(dest.name)
    if failed:
        print(f"\n{len(failed)} month(s) failed; re-run the same command to retry: {', '.join(failed)}")
        return 1
    print(f"\nDone: {len(list(out.glob('sharp_*.csv')))} monthly files in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
