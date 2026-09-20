"""What data is on disk, and what still needs doing -- read only, a few seconds.

    python scripts/data_status.py [--data-root D:/Data]

SoLEXS days, HEL1OS zips against extracted products (so you can see what an
"extract" would add), half-finished downloads, GOES and SHARP files, the
copied-day list, free disk space, and how much SoLEXS/HEL1OS overlap there is.
Paths default to config/project.toml.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from solarflare.settings import load_settings

    S = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(S.data_root))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    data = Path(args.data_root)
    dl = data / "pradan1.issdc.gov.in" / "al1" / "protected" / "downloadData"
    ex = data / "extracted"

    print(f"Data status  {datetime.now(UTC):%Y-%m-%d %H:%M} UTC   root {data}\n")
    free = shutil.disk_usage(data).free / 1e9
    print(f"Disk: {free:.0f} GB free on {data.drive or data}" + ("   << below the 30 GB extraction floor" if free < 30 else ""))

    solexs = sorted(dl.glob("solexs/**/AL1_SLX_L1_*.zip"))
    days = sorted({re.search(r"_(\d{8})_", p.name).group(1) for p in solexs if re.search(r"_(\d{8})_", p.name)})
    print(f"\nSoLEXS: {len(solexs)} zips, {len(days)} days" + (f", {days[0]} -> {days[-1]}" if days else ""))
    months = sorted({d[:6] for d in days})
    print("  months: " + " ".join(f"{m[:4]}-{m[4:]}" for m in months))

    hz = sorted(dl.glob("hel1os/**/HLS_*.zip"))
    parts = sorted(dl.glob("**/*.zip.part"))
    extracted = {p.name for p in ex.glob("hel1os/*/*/*/HLS_*") if (p / ".extracted.json").exists()}
    pending = [p for p in hz if p.stem not in extracted]
    by_month: dict[str, list[int]] = {}
    for p in hz:
        m = re.search(r"HLS_(\d{6})", p.name).group(1)
        by_month.setdefault(m, [0, 0])
        by_month[m][0] += 1
        by_month[m][1] += p.stem in extracted
    print(f"\nHEL1OS: {len(hz)} zips, {len(extracted)} extracted products, {len(pending)} zips not yet extracted")
    for m, (n, e) in sorted(by_month.items()):
        print(f"  {m[:4]}-{m[4:]}: {n:4d} zips, {e:4d} extracted" + ("" if e >= n else f"   <- {n - e} to extract"))
    if parts:
        print(f"\nHalf-finished downloads (.zip.part): {len(parts)}")
        for p in parts[:10]:
            print(f"  {p.relative_to(dl)}")

    goes = sorted(S.goes_dir.glob("*.nc"))
    print(f"\nGOES: {len(goes)} files " + ", ".join(p.name[:40] for p in goes[:4]))
    sharp = sorted(S.sharp_dir.glob("sharp_*.csv"))
    empty = [p.stem[6:] for p in sharp if p.stat().st_size < 10]
    print(f"SHARP: {len(sharp)} monthly files" + (f" ({sharp[0].stem[6:]} to {sharp[-1].stem[6:]})" if sharp else
                                                    " (python scripts/download_sharp.py)")
          + (f"; empty (not yet published): {', '.join(empty)}" if empty else ""))

    dup = S.copied_days
    if dup.exists():
        d = json.loads(dup.read_text("utf-8"))["duplicates"]
        print(f"\nCopied SoLEXS days (skipped in training): {', '.join(x['copy_date'] for x in d) or 'none'}")
    else:
        print("\nCopied SoLEXS days: not checked yet (run python -m solarflare quality)")

    try:
        from solarflare.io.hel1os_events import product_index
        prods = product_index(str(ex / "hel1os"))
        minutes = {m for p in prods for m in range(int(p.t_start // 60), int(p.t_stop // 60))}
        sdays = set(days)
        per_day: dict[str, int] = {}
        for m in minutes:
            k = datetime.fromtimestamp(m * 60, UTC).strftime("%Y%m%d")
            per_day[k] = per_day.get(k, 0) + 1
        both = sum(v for k, v in per_day.items() if k in sdays) / 1440
        print(f"\nHEL1OS products span {len(minutes) / 1440:.0f} days; {both:.0f} of them on SoLEXS days "
              "(scheduled time, before gaps)")
    except Exception as exc:                          # informational only
        print(f"\n(overlap not computed: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
