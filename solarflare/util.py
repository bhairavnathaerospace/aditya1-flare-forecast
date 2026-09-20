"""Small helpers shared by the analysis modules."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

TIME_FMT = "%Y-%m-%d %H:%M:%S"


def read_rows(path: str | Path) -> list[dict]:
    """All rows of a CSV file as dicts (the file is closed on return)."""
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def ts(s: str) -> float:
    """'YYYY-MM-DD HH:MM:SS' (UTC) -> unix seconds."""
    return datetime.strptime(s, TIME_FMT).replace(tzinfo=UTC).timestamp()


def utc(x: float, fmt: str = TIME_FMT) -> str:
    """unix seconds -> UTC string ('' for None/NaN)."""
    if x is None or x != x:
        return ""
    return datetime.fromtimestamp(float(x), UTC).strftime(fmt)
