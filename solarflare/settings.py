"""Project settings: where the data lives and where every result goes.

Read from ``config/project.toml`` (or the file named by ``SOLARFLARE_CONFIG``).
Everything that needs a path asks here, so moving the data means editing one
file. ``outputs`` is laid out as::

    outputs/
      quality/      data status, copied SoLEXS days
      model/        the trained network: reports, checkpoints, live status
        forward/final/   the frozen model used for alerts and forward tests
      ablations/    paired comparisons (HEL1OS, SHARP, settings)
      catalog/      master flare catalogue (soft + hard detections, GOES scores)
      alerts/       minute-by-minute predictions over validation + test, lead times
      dayahead/     2-24 h forecaster (X-ray activity + SHARP)
      physics/      temperatures, hard X-ray spectra, HEL1OS timing, hot onsets
      RESULTS.md    one page tying it together
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILE = ROOT / "config" / "project.toml"


@dataclass(frozen=True)
class Settings:
    root: Path
    data_root: Path
    goes_dir: Path
    sharp_dir: Path
    hel1os_extracted: Path
    cache: Path
    outputs: Path
    min_free_gb: float = 30.0
    model: dict = field(default_factory=dict)
    pipeline: dict = field(default_factory=dict)

    # ---- output locations ------------------------------------------------------
    @property
    def quality(self) -> Path:
        return self.outputs / "quality"

    @property
    def copied_days(self) -> Path:
        """Intervals PRADAN SoLEXS files repeat from the previous day."""
        return self.quality / "solexs_duplicates.json"

    @property
    def model_dir(self) -> Path:
        return self.outputs / "model"

    @property
    def frozen_dir(self) -> Path:
        return self.model_dir / "forward" / "final"

    @property
    def ablations(self) -> Path:
        return self.outputs / "ablations"

    @property
    def catalog(self) -> Path:
        return self.outputs / "catalog"

    @property
    def alerts(self) -> Path:
        return self.outputs / "alerts"

    @property
    def dayahead(self) -> Path:
        return self.outputs / "dayahead"

    @property
    def physics(self) -> Path:
        return self.outputs / "physics"

    def split_dates(self, run_dir: Path | None = None) -> dict[str, float]:
        """Training end and test start (unix s) of a trained run: its data_meta.json.

        Every product scores on the same test period the model was held out on,
        so the dates are read from the run rather than written into code."""
        import json

        meta = (run_dir or self.model_dir) / "reports" / "data_meta.json"
        if not meta.exists():
            raise FileNotFoundError(f"{meta} not found: train the model first "
                                    "(python -m solarflare train)")
        d = json.loads(meta.read_text("utf-8"))["split_dates"]
        return {"train_end": float(d["train_end"]), "test_start": float(d["test_start"])}


def _resolve(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p)


@lru_cache(maxsize=4)
def load_settings(path: str | None = None, root: str | None = None) -> Settings:
    """Settings from ``path`` (default: SOLARFLARE_CONFIG, else config/project.toml).
    Relative paths resolve against ``root`` (default: this checkout); the packaged
    console passes the project folder it found, since it runs from elsewhere."""
    root = Path(root) if root else ROOT
    f = Path(path or os.environ.get("SOLARFLARE_CONFIG") or root / "config" / "project.toml")
    raw = tomllib.loads(f.read_text("utf-8")) if f.exists() else {}
    p = raw.get("paths", {})
    return Settings(
        root=root,
        data_root=_resolve(p.get("data_root", "data"), root),
        goes_dir=_resolve(p.get("goes_dir", "data/goes"), root),
        sharp_dir=_resolve(p.get("sharp_dir", "data/sharp"), root),
        hel1os_extracted=_resolve(p.get("hel1os_extracted", "data/extracted/hel1os"), root),
        cache=_resolve(p.get("cache", "cache"), root),
        outputs=_resolve(p.get("outputs", "outputs"), root),
        min_free_gb=float(raw.get("data", {}).get("min_free_gb", 30)),
        model=dict(raw.get("model", {})),
        pipeline=dict(raw.get("pipeline", {})),
    )


def model_config(s: Settings, out_dir: Path | None = None, sharp: bool | None = None):
    """The final model's configuration: the [model] section of project.toml on top
    of the defaults in config.py (which reproduce the runs up to v4)."""
    from .config import Config

    m = s.model
    cfg = Config(data_root=s.data_root, out_dir=Path(out_dir) if out_dir else s.model_dir)
    cfg.cache_dir = s.cache
    cfg.pre.label_source = m.get("labels", "goes")
    cfg.pre.goes_dir = str(s.goes_dir)
    cfg.pre.goes_min_class = m.get("goes_min_class", "C1.0")
    cfg.pre.fit_flux_anchor = True
    cfg.model.anchor_flux = bool(m.get("anchor_flux", True))
    if m.get("skip_copied_days", True) and s.copied_days.exists():
        cfg.pre.exclude_intervals = str(s.copied_days)
    cfg.pre.hel1os_smooth_s = float(m.get("hel1os_smooth_s", 0.0))
    cfg.train.select_smooth_epochs = int(m.get("select_smooth_epochs", 1))
    cfg.train.early_stop_patience = int(m.get("early_stop_patience", 12))
    cfg.train.balance_head_gradients = bool(m.get("balance_head_gradients", False))
    use = m.get("use_sharp", "auto") if sharp is None else sharp
    if use is True:
        cfg.pre.sharp_dir = str(s.sharp_dir)
    # any other setting, as "section.key=value" strings (e.g. ["train.epochs=40"])
    return apply_overrides(cfg, list(m.get("set", [])))


def apply_overrides(cfg, pairs: list[str]):
    """``section.key=value`` strings (e.g. ``train.balance_head_gradients=false``)
    applied to a Config, with the value parsed to the field's type."""
    import json

    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        section, _, name = key.strip().partition(".")
        obj = getattr(cfg, section, None)
        if obj is None or not hasattr(obj, name):
            raise SystemExit(f"--set {pair}: no setting {key!r} (sections: pre, win, model, train)")
        cur = getattr(obj, name)
        raw = raw.strip()
        if isinstance(cur, bool):
            val = raw.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            val = int(raw)
        elif isinstance(cur, float):
            val = float(raw)
        elif isinstance(cur, (tuple, list)):
            val = tuple(json.loads(raw))
        else:
            val = raw
        setattr(obj, name, val)
    return cfg
