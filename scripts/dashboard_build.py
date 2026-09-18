"""Bake a replay JSON (scripts/dashboard_export.py) into the self-contained dashboard page.

    python scripts/dashboard_build.py --replay outputs/dashboard/replay_20260704_v1.json

The page has no server and fetches nothing: the day's data is embedded, so the one
HTML file can be opened locally or published as-is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--template", default=str(ROOT / "dashboard" / "flare_watch.template.html"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "dashboard" / "flare_watch.html"))
    args = ap.parse_args()

    data = json.loads(Path(args.replay).read_text("utf-8"))
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = Path(args.template).read_text("utf-8")
    if "__REPLAY__" not in html:
        raise SystemExit("template has no __REPLAY__ placeholder")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("__REPLAY__", payload), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB) for {data['day']}, model {data['model']['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
