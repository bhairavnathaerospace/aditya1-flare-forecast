"""Run the whole pipeline end to end, one command.

    python scripts/run_all.py
    python scripts/run_all.py --skip tests --encoders tcn,gru

Each stage runs as its own process with its own log in outputs/reports/logs/.
A failing stage is recorded and the run continues: a crash in one architecture
or a locked figure file must not throw away hours of finished work -- both of
those have happened on this project.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs" / "reports" / "logs"


def stages(encoders: str, folds: int, epochs: int, common: list[str]
           ) -> list[tuple[str, list[str]]]:
    py = [sys.executable, "-u"]
    cli = py + ["-m", "solarflare.cli"]
    return [
        ("tests-correctness", py + ["-m", "tests.test_correctness"]),
        ("tests-robustness", py + ["-m", "tests.test_robustness"]),
        ("tests-scale", py + ["-m", "tests.test_scale"]),
        ("tests-extract", py + ["-m", "tests.test_extract"]),
        ("tests-hel1os", py + ["-m", "tests.test_hel1os"]),
        ("tests-forward", py + ["-m", "tests.test_forward"]),
        ("tests-goes", py + ["-m", "tests.test_goes"]),
        # Cache first and on its own: it is the long I/O step, and every later
        # stage then reuses it instead of re-reading the archive.
        ("cache", cli + ["cache"] + common),
        ("inspect", cli + ["inspect", "--lines"] + common),
        ("nowcast-train", cli + ["train", "--epochs", str(epochs)] + common),
        ("baselines", cli + ["baselines"] + common),
        ("forecast-cv", cli + ["forecast", "--encoders", encoders,
                               "--folds", str(folds)] + common),
        ("fusion", cli + ["fusion", "--folds", str(folds)] + common),
        ("report", cli + ["report"] + common),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="tcn,ssm,gru,transformer,linear")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--skip", default="", help="comma-separated stage names")
    ap.add_argument("--data-root", default=".", help="folder with the mission products")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--workers", type=int, default=None,
                    help="cache worker processes (~1.5 GB RAM each)")
    ap.add_argument("--cli-args", default="",
                    help='extra flags for every solarflare.cli stage, e.g. '
                         '"--labels goes --goes-dir D:/Data/goes --cache-dir outputs/archive/cache"')
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    global LOGS
    LOGS = (ROOT / args.out_dir / "reports" / "logs").resolve()
    LOGS.mkdir(parents=True, exist_ok=True)

    summary = []
    t_all = time.time()
    common = ["--data-root", args.data_root, "--out-dir", args.out_dir]
    if args.workers:
        common += ["--workers", str(args.workers)]
    if args.cli_args:
        import shlex
        common += shlex.split(args.cli_args, posix=True)
    for name, cmd in stages(args.encoders, args.folds, args.epochs, common):
        if name in skip:
            summary.append((name, "skipped", 0.0))
            continue
        log = LOGS / f"{name}.log"
        print(f"[{time.strftime('%H:%M:%S')}] {name:18s} ...", flush=True)
        t0 = time.time()
        with log.open("w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        status = "ok" if rc == 0 else f"FAILED (exit {rc})"
        summary.append((name, status, dt))
        print(f"[{time.strftime('%H:%M:%S')}] {name:18s} {status}  "
              f"{dt:6.1f}s  -> {log}", flush=True)

    print("\n" + "=" * 60)
    for name, status, dt in summary:
        print(f"  {name:18s} {status:18s} {dt:7.1f}s")
    print(f"  {'total':18s} {'':18s} {time.time() - t_all:7.1f}s")
    failed = [n for n, s, _ in summary if s.startswith("FAILED")]
    print("=" * 60)
    print("ALL STAGES OK" if not failed else f"FAILED STAGES: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
