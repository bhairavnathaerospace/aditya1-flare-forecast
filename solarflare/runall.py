"""The whole study, stage by stage, into outputs/.

    python -m solarflare pipeline                 # run what is not done yet
    python -m solarflare pipeline --list          # stages and their state
    python -m solarflare pipeline --only alerts   # one stage (its inputs must exist)
    python -m solarflare pipeline --redo train    # a stage again, and everything built on it
    python -m solarflare pipeline --dry-run       # what would run

Each stage is one ``python -m solarflare <command>`` in its own process, so a
crash in one cannot corrupt another and memory is returned between stages.
State lives in outputs/pipeline/state.json and each stage's output in
outputs/pipeline/logs/<stage>.log; an interrupted run resumes where it stopped.
A stage whose inputs were rebuilt is run again; a frozen model is never
overwritten (the old one is moved aside, not deleted).

Stages, in order:

  cache            per-file preprocessing cache (skips files already done)
  quality          SoLEXS days that repeat the previous day (masked in training)
  train            the network -> outputs/model (use_sharp "auto": outputs/ablations/sharp/xray_only)
  train-sharp      the same with SHARP inputs -> outputs/ablations/sharp/with_sharp  ["auto" only]
  choose-inputs    keep SHARP only if it wins on VALIDATION by pipeline.sharp_min_gain;
                   the winner is copied to outputs/model                              ["auto" only]
  baselines        classical baselines on the model's splits
  calibration      probabilities before/after calibration
  references       flux forecast vs "no change" references
  freeze           the final model: weights, normaliser, calibration, thresholds
  catalog          master flare catalogue (SoLEXS + HEL1OS), scored against GOES
  catalog-figure   its overview figure
  alerts-predict   the frozen model minute by minute (and again with HEL1OS hidden)
  alerts           lead time vs false alarms; alert rules for the console
  dayahead         2-24 h forecasts: SoLEXS activity, SHARP, both
  hel1os-seed-N    paired soft-only vs soft+hard ablation, one per seed
  hel1os-value     the HEL1OS gain across seeds
  temperature, hxr-spectra, hxr-timing, onset-study   flare physics
  report           the model's RESULTS.md
  summary          outputs/RESULTS.md, one page over everything
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .settings import Settings, load_settings


@dataclass
class Stage:
    name: str
    title: str
    run: list[str] | Callable[[Settings], None]
    needs: tuple[str, ...] = ()
    produces: tuple[Path, ...] = ()
    #: a stage that refuses to overwrite its product (freeze): an existing product counts as done
    keep_existing: bool = False


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def sharp_available(s: Settings) -> bool:
    return any(s.sharp_dir.glob("sharp_*.csv")) if s.sharp_dir.exists() else False


def use_sharp_mode(s: Settings) -> str:
    """'auto' (train both and choose on validation), 'on' or 'off'."""
    v = s.model.get("use_sharp", "auto")
    if v is True or str(v).lower() in ("true", "on", "yes"):
        return "on"
    if v is False or str(v).lower() in ("false", "off", "no"):
        return "off"
    return "auto" if sharp_available(s) else "off"


def sharp_runs(s: Settings) -> tuple[Path, Path]:
    return s.ablations / "sharp" / "xray_only", s.ablations / "sharp" / "with_sharp"


def stages(s: Settings) -> list[Stage]:
    mode = use_sharp_mode(s)
    model_ready = "choose-inputs" if mode == "auto" else "train"
    seeds = [int(x) for x in s.pipeline.get("seeds", [1337, 7, 42])]
    phys = s.physics
    L = [
        Stage("cache", "preprocessing cache", ["cache"], produces=(s.cache / "manifest.json",)),
        Stage("quality", "copied SoLEXS days", ["quality"], ("cache",), (s.copied_days,)),
    ]
    if mode == "auto":
        xray_run, sharp_run = sharp_runs(s)
        L += [
            Stage("train", "network, X-ray inputs", ["train", "--out-dir", str(xray_run)], ("quality",),
                  (xray_run / "reports" / "evaluation.json",)),
            Stage("train-sharp", "network, X-ray + SHARP inputs", ["train", "--sharp", "--out-dir", str(sharp_run)],
                  ("quality",), (sharp_run / "reports" / "evaluation.json",)),
            Stage("choose-inputs", "SHARP in or out, on validation", choose_inputs, ("train", "train-sharp"),
                  (s.ablations / "sharp" / "decision.json", s.model_dir / "checkpoints" / "best.pt")),
        ]
    else:
        L.append(Stage("train", "network" + (" with SHARP" if mode == "on" else ""),
                       ["train"] + (["--sharp"] if mode == "on" else []), ("quality",),
                       (s.model_dir / "checkpoints" / "best.pt", s.model_dir / "reports" / "evaluation.json")))
    L += [
        Stage("baselines", "classical baselines", ["baselines"], (model_ready,),
              (s.model_dir / "reports" / "baselines.json",)),
        Stage("calibration", "probability calibration report", ["calibration-report"], (model_ready,),
              (s.model_dir / "reports" / "calibration.json",)),
        Stage("references", "flux forecast vs references", ["references"], (model_ready,),
              (s.model_dir / "reports" / "fair_references.json",)),
        Stage("freeze", "freeze the final model", ["forward-test", "freeze", "--name", "final"], (model_ready,),
              (s.frozen_dir / "frozen.pt",), keep_existing=True),
        Stage("catalog", "master flare catalogue", ["catalog"], (model_ready,),
              (s.catalog / "master_catalog.csv",)),
        Stage("catalog-figure", "catalogue figure", ["catalog-figure"], ("catalog",)),
        Stage("alerts-predict", "alerts: minute-by-minute predictions", ["alerts", "--predict"], ("freeze",),
              (s.alerts / "pred_final.npz",)),
        Stage("alerts-predict-nohard", "alerts: predictions with HEL1OS hidden",
              ["alerts", "--predict", "--blank-hard"], ("freeze",), (s.alerts / "pred_final_nohard.npz",)),
        Stage("alerts", "alerts: lead time and false alarms", ["alerts"],
              ("alerts-predict", "alerts-predict-nohard", "catalog"),
              (s.alerts / "leadtime_summary.json", s.alerts / "alert_rules.json")),
        Stage("dayahead", "2-24 h forecasts", ["dayahead"], ("catalog",), (s.dayahead / "dayahead_summary.json",)),
    ]
    for seed in seeds:
        d = s.ablations / "hel1os" / f"seed_{seed}"
        L.append(Stage(f"hel1os-seed-{seed}", f"HEL1OS ablation, seed {seed}",
                       ["fusion", "--seed", str(seed), "--out-dir", str(d)], ("quality",),
                       (d / "reports" / "fusion_ablation.json",)))
    L += [
        Stage("hel1os-value", "HEL1OS gain across seeds", ["hel1os-value"],
              tuple(f"hel1os-seed-{x}" for x in seeds), (s.ablations / "hel1os" / "hel1os_value.json",)),
        Stage("temperature", "SoLEXS flare temperatures", ["temperature"], ("catalog",)),
        Stage("hxr-spectra", "HEL1OS spectra", ["hxr-spectra"], ("catalog",), (phys / "hxr_spectra.csv",)),
        Stage("hxr-timing", "HEL1OS timing", ["hxr-timing"], ("hxr-spectra",)),
        Stage("onset-study", "hot onsets and the Neupert effect", ["onset-study"], ("cache",)),
        Stage("report", "model RESULTS.md", ["report"], ("baselines", "calibration")),
        Stage("summary", "outputs/RESULTS.md", write_summary, ()),
    ]
    return L


# ---------------------------------------------------------------------------
# internal stages
# ---------------------------------------------------------------------------

def _best_val(run: Path) -> tuple[float, int]:
    ev = json.loads((run / "reports" / "evaluation.json").read_text("utf-8"))
    t = ev.get("training", {})
    return float(t["best_score"]), int(t.get("best_epoch", -1))


def choose_inputs(s: Settings) -> None:
    """Keep SHARP as a network input only if its run scores higher on validation
    by at least pipeline.sharp_min_gain. The test period plays no part, and the
    simpler model wins a tie. The winning run is copied to outputs/model; an
    earlier outputs/model is moved to archive/superseded, never deleted."""
    xray, sharp = sharp_runs(s)
    v0, e0 = _best_val(xray)
    v1, e1 = _best_val(sharp)
    margin = float(s.pipeline.get("sharp_min_gain", 0.01))
    use = v1 >= v0 + margin
    win = sharp if use else xray
    if s.model_dir.exists() and any(s.model_dir.iterdir()):
        aside = s.root / "archive" / "superseded" / f"model_{datetime.now(UTC):%Y%m%d_%H%M%S}"
        aside.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s.model_dir), str(aside))
        print(f"moved the previous outputs/model to {aside}")
    shutil.copytree(win, s.model_dir, ignore=shutil.ignore_patterns("live.json", "forward"))
    cf = s.model_dir / "reports" / "config.json"
    d = json.loads(cf.read_text("utf-8"))
    d["out_dir"] = str(s.model_dir)
    cf.write_text(json.dumps(d, indent=2), encoding="utf-8")
    dec = {"decided_utc": _now(), "criterion": f"validation score (smoothed) gain >= {margin}",
           "xray_only": {"run": str(xray), "val_score": round(v0, 4), "best_epoch": e0},
           "with_sharp": {"run": str(sharp), "val_score": round(v1, 4), "best_epoch": e1},
           "gain": round(v1 - v0, 4), "use_sharp": bool(use), "copied_to": str(s.model_dir)}
    path = s.ablations / "sharp" / "decision.json"
    path.write_text(json.dumps(dec, indent=2), encoding="utf-8")
    print(f"validation score: X-ray only {v0:.4f}, with SHARP {v1:.4f} (gain {v1 - v0:+.4f}, needed "
          f"{margin:+.4f}) -> {'SHARP kept' if use else 'SHARP left out'}; {win.name} copied to {s.model_dir}")


def write_summary(s: Settings) -> None:
    from .summary import build

    print(f"wrote {build(s)}")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

class State:
    def __init__(self, path: Path):
        self.path = path
        self.d = json.loads(path.read_text("utf-8")) if path.exists() else {"stages": {}}

    def get(self, name: str) -> dict:
        return self.d["stages"].get(name, {})

    def set(self, name: str, **kw) -> None:
        self.d["stages"].setdefault(name, {}).update(kw)
        self.save()

    def clear(self, name: str) -> None:
        self.d["stages"].pop(name, None)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


def is_done(st: Stage, state: State) -> bool:
    status = state.get(st.name).get("status")
    if st.keep_existing and status != "stale" and st.produces and all(p.exists() for p in st.produces):
        return True
    return state.get(st.name).get("status") == "done" and all(p.exists() for p in st.produces)


def dependents(name: str, all_stages: list[Stage]) -> set[str]:
    out, frontier = set(), {name}
    while frontier:
        nxt = {st.name for st in all_stages if set(st.needs) & frontier} - out
        out |= nxt
        frontier = nxt
    return out


def free_gb(path: Path) -> float:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def keep_awake(on: bool) -> None:
    """Stop Windows sleeping while stages run (the console does the same for its jobs)."""
    if sys.platform != "win32":
        return
    import ctypes

    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def run_stage(st: Stage, s: Settings, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {st.name}: {st.title}  ({_now()} UTC) =====\n")
        if callable(st.run):
            import contextlib
            import io

            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    st.run(s)
                rc = 0
            except Exception as e:                      # reported, not raised: the runner decides
                buf.write(f"FAILED: {type(e).__name__}: {e}\n")
                rc = 1
            fh.write(buf.getvalue())
            sys.stdout.write(buf.getvalue())
            return rc
        cmd = [sys.executable, "-u", "-m", "solarflare", *st.run]
        fh.write("$ " + " ".join(cmd[2:]) + "\n")
        fh.flush()
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        p = subprocess.Popen(cmd, cwd=s.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                             text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            fh.write(line)
            fh.flush()
            sys.stdout.write(f"  {line}")
            sys.stdout.flush()
        return p.wait()


def main(argv=None) -> int:
    s = load_settings()
    ap = argparse.ArgumentParser(prog="solarflare pipeline", description="the whole study, resumable")
    ap.add_argument("--list", action="store_true", help="show stages and their state, run nothing")
    ap.add_argument("--only", default="", help="comma-separated stages to run (their inputs must be done)")
    ap.add_argument("--from", dest="start", default="", help="skip the stages before this one")
    ap.add_argument("--redo", default="", help="comma-separated stages to run again, with everything after them")
    ap.add_argument("--skip", default="", help="comma-separated stages to leave out this time")
    ap.add_argument("--keep-going", action="store_true", help="after a failure, continue with independent stages")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    all_stages = stages(s)
    names = [st.name for st in all_stages]
    by_name = {st.name: st for st in all_stages}
    pdir = s.outputs / "pipeline"
    state = State(pdir / "state.json")

    def pick(csv_: str) -> list[str]:
        out = [x.strip() for x in csv_.split(",") if x.strip()]
        bad = [x for x in out if x not in by_name]
        if bad:
            raise SystemExit(f"unknown stage(s) {bad}; stages: {', '.join(names)}")
        return out

    only, redo, skip = pick(args.only), pick(args.redo), pick(args.skip)
    if args.start:
        pick(args.start)

    if args.list:
        mode = use_sharp_mode(s)
        print(f"SHARP as network input: {mode}; seeds {s.pipeline.get('seeds')}; outputs {s.outputs}")
        for st in all_stages:
            info = state.get(st.name)
            status = "done" if is_done(st, state) else info.get("status", "-")
            took = f"{info['seconds'] / 60:.1f} min" if info.get("seconds") else ""
            print(f"  {st.name:<24} {status:<12} {took:>10}  {st.title}")
        return 0

    stale: set[str] = set()
    for r in redo:
        stale |= {r} | dependents(r, all_stages)
    started = not args.start
    plan = []
    for st in all_stages:
        started = started or st.name == args.start
        if not started or st.name in skip or (only and st.name not in only):
            continue
        if st.name in stale or not is_done(st, state):
            plan.append(st)
    if not plan:
        print("nothing to do: every stage is done (--list shows them; --redo <stage> runs one again)")
        return 0
    print(f"pipeline: {len(plan)} stage(s): {', '.join(st.name for st in plan)}")
    if args.dry_run:
        for st in plan:
            print(f"  {st.name:<24} " + (" ".join(st.run) if isinstance(st.run, list) else f"<{st.run.__name__}>"))
        return 0

    keep_awake(True)
    failed: set[str] = set()
    t_all = time.time()
    try:
        for st in plan:
            blocked = [n for n in st.needs if n in failed
                       or (n in by_name and not is_done(by_name[n], state) and n not in [p.name for p in plan])]
            if blocked:
                print(f"\n[skip] {st.name}: needs {', '.join(blocked)}")
                failed.add(st.name)
                continue
            gb = free_gb(s.outputs)
            if gb < s.min_free_gb:
                print(f"\n[stop] only {gb:.0f} GB free on the outputs drive (floor {s.min_free_gb:g} GB)")
                return 2
            if st.keep_existing:
                for p in st.produces:
                    if p.exists():                      # never overwrite a frozen model: move it aside
                        aside = p.parent.with_name(f"{p.parent.name}_superseded_{int(time.time())}")
                        shutil.move(str(p.parent), str(aside))
                        print(f"moved the previous {st.name} product to {aside}")
            print(f"\n[{names.index(st.name) + 1}/{len(names)}] {st.name}: {st.title}  ({_now()} UTC)", flush=True)
            state.set(st.name, status="running", started=_now(), log=str(pdir / "logs" / f"{st.name}.log"))
            t0 = time.time()
            try:
                rc = run_stage(st, s, pdir / "logs" / f"{st.name}.log")
            except KeyboardInterrupt:
                state.set(st.name, status="interrupted", seconds=round(time.time() - t0, 1))
                print(f"\n[interrupted] {st.name}; run the pipeline again to resume")
                return 130
            missing = [str(p) for p in st.produces if not p.exists()]
            secs = round(time.time() - t0, 1)
            if rc == 0 and not missing:
                state.set(st.name, status="done", finished=_now(), seconds=secs)
                # anything built on this stage is now out of date
                for dep in dependents(st.name, all_stages):
                    if state.get(dep).get("status") == "done" and dep not in [p.name for p in plan]:
                        state.set(dep, status="stale")
                print(f"[done] {st.name} in {secs / 60:.1f} min")
            else:
                why = f"exit code {rc}" if rc else f"did not write {', '.join(missing)}"
                state.set(st.name, status="failed", finished=_now(), seconds=secs, error=why)
                print(f"[FAILED] {st.name}: {why}; log {pdir / 'logs' / (st.name + '.log')}")
                failed.add(st.name)
                if not args.keep_going:
                    return 1
    finally:
        keep_awake(False)
    print(f"\npipeline finished in {(time.time() - t_all) / 3600:.1f} h"
          + (f"; failed or skipped: {', '.join(sorted(failed))}" if failed else "; every stage done"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
