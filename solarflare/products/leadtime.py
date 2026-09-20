"""Lead time per flare: how long before the GOES peak an alert fires, and at what false-alarm cost.

    python -m solarflare alerts --predict                 # once, GPU ~13 min: the frozen model at a 1-min stride
    python -m solarflare alerts --predict --blank-hard    # the same with HEL1OS hidden (paired ablation)
    python -m solarflare alerts                           # scoring, figure and alert rules, ~1 min

Every alert is causal: it is ON at minute A using only data available by A (the
model's origin bin ends 20 s before A; a SoLEXS minute [A-60, A) ends at A).

  network (C)  calibrated P(a GOES >= C1 flare is in progress within 15 min)
  network (M)  median forecast of GOES flux, highest of +5/+15/+30 min
  combined (M) the higher of the network's M signal and the SoLEXS flux now
  trend        SoLEXS calibrated flux extrapolated 15 min along its last-5-min rise
  flux now     SoLEXS calibrated flux (training-period calibration)
  rise rule    the master catalogue's SoLEXS rule (fixed; C alerts)

Operating points are chosen on the model's validation period and only scored on
its test period: (a) the best-TSS threshold on minute labels (the criterion the
frozen thresholds used; the network's own C threshold is the frozen one), and
(b) the threshold giving a set number of false alarms per day, so every method
is compared at the same false-alarm cost. The rules the console raises alerts
with are written to alert_rules.json.

Per GOES flare in the test period with SoLEXS and model output for >= 80% of
its window:
  window    30 min before the GOES start (or the previous flare's peak, if later)
            to the GOES peak
  warned    the alert was ON at some minute of the window
  lead      GOES peak minus the first ON minute; for M alerts also the minute
            GOES first reached M1.0 minus that minute
  chance    the same windows moved 2 h earlier and later, kept only where no
            flare of the class occurs: how often the alert is ON anyway.
            Event TSS = warned - chance.
False alarms: ON episodes (gaps <= 5 min merged) with no GOES flare of the class
from the episode start to 15 min (C) / 30 min (M) after its end, per day of data.

Writes outputs/alerts/{lead_times.csv, leadtime_summary.json, LEADTIME.md, leadtime.png,
alert_rules.json, watch.npz}; the last two drive the console's Flare Watch replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from solarflare.settings import load_settings
from solarflare.util import read_rows, ts

#: The model's split (<run>/reports/data_meta.json), set in main().
TRAIN_END = float("nan")
TEST_START = float("nan")
EMBARGO_S = 3600.0
PRE_START_MIN = 30
MERGE_GAP_MIN = 5
SHIFT_MIN = 120
M1 = -5.0                             # log10 W/m^2
C1 = -6.0
COVER = 0.8
HORIZON_MIN = {"C": 15, "M": 30}
FA_TARGETS = {"C": (1.0, 2.0, 5.0), "M": (0.25, 0.5, 1.0)}
PRIMARY = {"C": 2.0, "M": 0.5}
#: threshold candidates as signal quantiles, dense in the upper tail where rare alerts live
QGRID = 1.0 - np.geomspace(0.5, 2e-5, 160)
NAMES = {"model": "network", "model_no_hel1os": "network, HEL1OS hidden",
         "combined": "network + SoLEXS flux now (higher of the two)",
         "trend": "SoLEXS trend (15-min extrapolation)", "current": "SoLEXS flux now",
         "rule": "catalogue rise rule", "rule_C_level": "rise rule, only at >= C1 flux"}


def utc(x) -> str:
    return datetime.fromtimestamp(float(x), UTC).strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# 1. predictions
# ---------------------------------------------------------------------------

def predict(args) -> None:
    import torch
    from numpy.lib.stride_tricks import sliding_window_view

    from solarflare import probcal
    from solarflare.config import Config
    from solarflare.forward import MANIFEST, _segments, load_frozen

    fdir = Path(args.forward_dir)
    cfg = Config(data_root=Path(args.data_root), out_dir=Path(args.run_dir))
    cfg.cache_dir = Path(args.cache_dir)
    frozen, model, norm, device = load_frozen(fdir, cfg)
    cfg.pre.goes_dir = args.goes_dir
    manifest = json.loads((fdir / MANIFEST).read_text("utf-8"))
    dt, L = cfg.pre.dt_seconds, cfg.steps_per_window
    stride = max(int(round(60.0 / dt)), 1)
    t_from, t_to = TRAIN_END + EMBARGO_S, float(frozen["data_cutoff_unix"])
    t0 = time.time()
    segments, _ = _segments(cfg, verbose=True)
    print(f"{len(segments)} segments in {time.time() - t0:.0f} s; predicting on {device}"
          + (" with HEL1OS hidden" if args.blank_hard else ""), flush=True)

    rec = {k: [] for k in ("origin", "p_now", "p_occ", "now", "fc", "hard_frac")}
    with torch.no_grad():
        for seg in segments:
            if (seg.goes_long is None or len(seg) < L or seg.time_unix[-1] < t_from
                    or seg.time_unix[0] > t_to):
                continue
            hard_mask = np.zeros_like(seg.hard_mask) if args.blank_hard else seg.hard_mask
            avail = np.maximum(seg.soft_mask, hard_mask)
            csum = np.concatenate([[0.0], np.cumsum(avail)])
            hsum = np.concatenate([[0.0], np.cumsum(hard_mask)])
            ends = np.arange(L, len(seg) + 1)
            origin = seg.time_unix[ends - 1]
            keep = ((origin >= t_from) & (origin <= t_to)
                    & (np.rint(origin / dt).astype(np.int64) % stride == 0)
                    & (seg.soft_mask[ends - 1] > 0)
                    & ((csum[ends] - csum[ends - L]) / L >= cfg.win.min_observed_fraction))
            ends = ends[keep]
            if ends.size == 0:
                continue
            soft_n = np.nan_to_num(norm.apply_soft(seg.soft), nan=0.0, posinf=0.0,
                                   neginf=0.0).astype(np.float32)
            hard_n = np.nan_to_num(norm.apply_hard(seg.hard), nan=0.0, posinf=0.0,
                                   neginf=0.0).astype(np.float32)
            if args.blank_hard:
                hard_n[:] = 0.0
            views = {"soft": sliding_window_view(soft_n, L, axis=0),
                     "hard": sliding_window_view(hard_n, L, axis=0),
                     "clock": sliding_window_view(seg.clock.astype(np.float32), L, axis=0),
                     "ms": sliding_window_view(seg.soft_mask.astype(np.float32), L),
                     "mh": sliding_window_view(hard_mask.astype(np.float32), L)}
            for b0 in range(0, ends.size, args.batch):
                s = ends[b0:b0 + args.batch] - L

                def tens(k, _s=s, _views=views):
                    a = _views[k][_s]
                    if a.ndim == 3:
                        a = a.transpose(0, 2, 1)
                    return torch.from_numpy(np.ascontiguousarray(a)).to(device)

                o = model(tens("soft"), tens("ms"), tens("hard"), tens("mh"), tens("clock"))
                rec["p_now"].append(torch.sigmoid(o["in_flare"]).cpu().numpy())
                p_occ = torch.sigmoid(o["occurrence"]).cpu().numpy()
                cal = frozen.get("calibration")
                if cal:           # the frozen thresholds refer to calibrated probabilities
                    p_occ = np.column_stack([probcal.apply(cal["occurrence"][h], p_occ[:, h])
                                             for h in range(p_occ.shape[1])])
                rec["p_occ"].append(p_occ)
                rec["now"].append(o["nowcast"].cpu().numpy())
                rec["fc"].append(o["forecast"].cpu().numpy())
            rec["origin"].append(seg.time_unix[ends - 1])
            rec["hard_frac"].append(((hsum[ends] - hsum[ends - L]) / L).astype(np.float32))
    out = {k: np.concatenate(v) for k, v in rec.items()}
    order = np.argsort(out["origin"], kind="stable")
    out = {k: v[order] for k, v in out.items()}
    dest = Path(args.out) / f"pred_{manifest['name']}{'_nohard' if args.blank_hard else ''}.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out, horizons_s=np.asarray(cfg.win.forecast_horizons_s),
                        occ_horizons_s=np.asarray(cfg.win.occurrence_horizons_s),
                        quantiles=np.asarray(cfg.win.quantiles),
                        threshold_occ=np.asarray(manifest["thresholds"]["occurrence"]),
                        sha=np.asarray(manifest["checkpoint_sha256"][:16]))
    print(f"{out['origin'].size} forecasts {utc(out['origin'][0])} -> {utc(out['origin'][-1])} "
          f"in {time.time() - t0:.0f} s -> {dest}")


# ---------------------------------------------------------------------------
# 2. scoring helpers
# ---------------------------------------------------------------------------

def tss_threshold(y: np.ndarray, s: np.ndarray, n: int = 400) -> tuple[float, float]:
    """Threshold on score ``s`` (ON when s >= thr) with the best TSS."""
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order].astype(np.float64), s[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    P, N = max(tp[-1], 1), max(fp[-1], 1)
    best, best_t = -np.inf, float(ss[-1])
    for c in np.unique(np.quantile(ss, np.linspace(0.0, 1.0, n))):
        k = np.searchsorted(-ss, -c, side="right")   # entries with s >= c
        if k == 0:
            continue
        t = tp[k - 1] / P - fp[k - 1] / N
        if t > best:
            best, best_t = t, float(c)
    return best_t, float(best)


def episodes(on: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Runs of ON minutes, gaps of <= MERGE_GAP_MIN minutes merged: (starts, stops), inclusive."""
    idx = np.flatnonzero(on)
    if idx.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    cut = np.flatnonzero(np.diff(idx) > MERGE_GAP_MIN + 1)
    return np.concatenate([[idx[0]], idx[cut + 1]]), np.concatenate([idx[cut], [idx[-1]]])


def next_on(on: np.ndarray) -> np.ndarray:
    """Index of the first ON minute at or after each minute (len(on) if none)."""
    n = on.size
    return np.minimum.accumulate(np.where(on, np.arange(n), n)[::-1])[::-1]


class Grid:
    """One-minute availability grid A_k = g0 + 60 k."""

    def __init__(self, t0: float, t1: float):
        self.g0 = np.floor(t0 / 60.0) * 60.0
        self.n = int((t1 - self.g0) // 60.0) + 1
        self.t = self.g0 + 60.0 * np.arange(self.n)

    def idx(self, t) -> np.ndarray:
        return np.rint((np.asarray(t, np.float64) - self.g0) / 60.0).astype(np.int64)

    def put(self, t, v) -> np.ndarray:
        out = np.full(self.n, np.nan)
        k = self.idx(t)
        ok = (k >= 0) & (k < self.n)
        out[k[ok]] = np.asarray(v, np.float64)[ok]
        return out

    def mark(self, a, b) -> np.ndarray:
        """True on minutes in [a, b] (times), for many intervals."""
        d = np.zeros(self.n + 1)
        ka = np.clip(np.ceil((np.asarray(a, float) - self.g0) / 60.0).astype(np.int64), 0, self.n)
        kb = np.clip(np.floor((np.asarray(b, float) - self.g0) / 60.0).astype(np.int64) + 1, 0, self.n)
        good = kb > ka
        np.add.at(d, ka[good], 1)
        np.add.at(d, kb[good], -1)
        return np.cumsum(d)[:-1] > 0


class Flares:
    """Flare intervals of one class, for overlap queries."""

    def __init__(self, st: np.ndarray, en: np.ndarray):
        o = np.argsort(st)
        self.st, self.en = st[o], en[o]
        self.en_max = np.maximum.accumulate(self.en) if self.en.size else self.en

    def overlaps(self, a, b) -> np.ndarray:
        """Does any flare interval overlap [a_i, b_i]?"""
        a, b = np.atleast_1d(np.asarray(a, float)), np.atleast_1d(np.asarray(b, float))
        k = np.searchsorted(self.st, b, side="right")
        out = np.zeros(a.size, bool)
        ok = k > 0
        out[ok] = self.en_max[k[ok] - 1] >= a[ok]
        return out


def lead_gain(a: np.ndarray, b: np.ndarray):
    """Statistic for day_bootstrap: mean lead of a minus b (missed flares count 0)."""
    return lambda i: float(np.mean(np.nan_to_num(a[i])) - np.mean(np.nan_to_num(b[i])))


def rate_gain(a: np.ndarray, b: np.ndarray):
    """Statistic for day_bootstrap: warned share of a minus b."""
    return lambda i: float(np.mean(a[i]) - np.mean(b[i]))


def day_bootstrap(days: np.ndarray, stat, n_boot: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    u = np.unique(days)
    groups = {d: np.flatnonzero(days == d) for d in u}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[d] for d in rng.choice(u, u.size, replace=True)])
        v = stat(idx)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return None
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def _r(v, nd=3):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


# ---------------------------------------------------------------------------
# 3. scoring
# ---------------------------------------------------------------------------

def score(args) -> int:
    from numpy.lib.stride_tricks import sliding_window_view

    from solarflare.catalog.build import load_solexs
    from solarflare.catalog.detect import PiecewiseCalibration, to_minutes
    from solarflare.io.goes import load_goes

    out = Path(args.out)
    P = dict(np.load(out / f"pred_{args.name}.npz"))
    nh = out / f"pred_{args.name}_nohard.npz"
    PH = dict(np.load(nh)) if nh.exists() else None
    hs = list(P["horizons_s"])
    q50 = int(np.argmin(np.abs(P["quantiles"] - 0.5)))
    thr_c_frozen = float(P["threshold_occ"][0])
    g = Grid(TRAIN_END + EMBARGO_S, float(P["origin"][-1]) + 60.0)

    def m_signal(p):
        return np.max(p["fc"][:, [hs.index(h) for h in (300.0, 900.0, 1800.0)], q50], axis=1)

    sig = {"model": {"C": g.put(P["origin"] + 60.0, P["p_occ"][:, 0]),
                     "M": g.put(P["origin"] + 60.0, m_signal(P))}}
    hard_frac = g.put(P["origin"] + 60.0, P["hard_frac"])
    if PH is not None:
        sig["model_no_hel1os"] = {"C": g.put(PH["origin"] + 60.0, PH["p_occ"][:, 0]),
                                  "M": g.put(PH["origin"] + 60.0, m_signal(PH))}

    # SoLEXS minute flux; piecewise calibration fitted on training minutes only
    truth = load_goes(Path(args.goes_dir))
    _, t, r, _, ok = load_solexs(Path(args.cache_dir))
    tm, rm, _, vm = to_minutes(t, r, ok, 20.0)
    gmin = truth.flux_on_grid(tm)
    fit_on = vm & (tm <= TRAIN_END) & np.isfinite(gmin) & (gmin > 0) & (rm > 0)
    cal = PiecewiseCalibration.fit(np.log10(rm[fit_on]), np.log10(gmin[fit_on]))
    with np.errstate(divide="ignore", invalid="ignore"):
        f = g.put(tm + 60.0, np.log10(cal(np.where(vm, rm, np.nan))))
    f5 = np.concatenate([np.full(5, np.nan), f[:-5]])
    trend = f + 15.0 * np.maximum((f - f5) / 5.0, 0.0)
    sig["trend"] = {"C": trend, "M": trend}
    sig["current"] = {"C": f, "M": f}
    sig["combined"] = {"M": np.fmax(sig["model"]["M"], f)}

    # GOES truth
    fl = [x for x in truth.flares if x.goes_class[:1] in "CMX" and g.t[0] - 7200 <= x.peak_unix <= g.t[-1]]
    st = np.array([x.start_unix for x in fl])
    pk = np.array([x.peak_unix for x in fl])
    en = np.array([x.end_unix for x in fl])
    ism = np.array([x.goes_class[:1] in "MX" for x in fl])
    flares = {"C": Flares(st, en), "M": Flares(st[ism], en[ism])}
    with np.errstate(divide="ignore", invalid="ignore"):
        lx = np.log10(truth.flux_on_grid(g.t))            # record [A, A+60): the future at A
    win = sliding_window_view(np.concatenate([lx, np.full(30, np.nan)]), 30)[:g.n]
    with np.errstate(all="ignore"):
        fmax30 = np.where(np.isfinite(win).sum(1) >= 20,
                          np.nanmax(np.where(np.isfinite(win), win, -np.inf), 1), np.nan)
    labels = {"C": (g.mark(st - 900.0, en).astype(float), np.isfinite(lx)),
              "M": ((fmax30 >= M1).astype(float), np.isfinite(fmax30))}

    val = g.t < TEST_START
    test = g.t >= TEST_START + EMBARGO_S
    base = np.isfinite(sig["model"]["C"]) & np.isfinite(f) & np.isfinite(trend)
    if PH is not None:
        base &= np.isfinite(sig["model_no_hel1os"]["C"])

    # the catalogue rule: ON from the minute after its alert minute to the SoLEXS peak
    cat = read_rows(Path(args.catalog) / "master_catalog.csv")
    pairs = [(ts(x["soft_alert_utc"]) + 60.0, ts(x["peak_utc"]) + 60.0) for x in cat if x["soft_alert_utc"]]
    pairs = [(a, max(a, p)) for a, p in pairs if 0 <= g.idx(a) < g.n]
    pairs_c = [(a, p) for a, p in pairs if np.nan_to_num(f[g.idx(a)], nan=-99.0) >= C1]
    sig["rule"] = {"C": g.mark([a for a, _ in pairs], [p for _, p in pairs]).astype(float)}
    sig["rule_C_level"] = {"C": g.mark([a for a, _ in pairs_c], [p for _, p in pairs_c]).astype(float)}

    methods = {"C": ["model", "trend", "current", "rule", "rule_C_level"],
               "M": ["model", "combined", "trend", "current"]}

    # ---- windows ----------------------------------------------------------------
    avail = base.astype(float)
    csum = np.concatenate([[0.0], np.cumsum(avail)])

    def cover(k0, k1):
        return (csum[k1 + 1] - csum[k0]) / (k1 - k0 + 1)

    def windows(c):
        same = ism if c == "M" else np.ones(len(fl), bool)
        rows = []
        for i, x in enumerate(fl):
            if not same[i] or x.peak_unix < TEST_START + EMBARGO_S or x.peak_unix > g.t[-1] - 3600:
                continue
            prev = pk[(pk < x.peak_unix) & same]
            w0t = max(x.start_unix - 60.0 * PRE_START_MIN, (prev.max() + 60.0) if prev.size else -np.inf)
            k0, k1 = int(g.idx(np.ceil(w0t / 60.0) * 60.0)), int(g.idx(x.peak_unix))
            if k1 <= k0 or k0 < 0 or k1 >= g.n or cover(k0, k1) < COVER:
                continue
            cross = x.peak_unix
            if c == "M":
                j = np.flatnonzero(np.nan_to_num(lx[g.idx(x.start_unix):k1 + 1], nan=-99) >= M1)
                cross = x.start_unix + 60.0 * j[0] if j.size else x.peak_unix
            rows.append({"i": i, "k0": k0, "k1": k1, "cross_k": int(g.idx(cross)),
                         "cls": x.goes_class, "day": int(x.peak_unix // 86400),
                         "hel1os": float(np.nanmean(hard_frac[k0:k1 + 1])) if np.isfinite(hard_frac[k0:k1 + 1]).any() else 0.0})
        W = {k: np.array([r[k] for r in rows]) for k in ("i", "k0", "k1", "cross_k", "day", "hel1os")}
        W["cls"] = np.array([r["cls"] for r in rows])
        return W

    Wc = {c: windows(c) for c in ("C", "M")}

    def shifted(c):
        """Chance windows: moved +-SHIFT_MIN, flare-free for the class, with data."""
        W = Wc[c]
        k0s, k1s = [], []
        for sh in (-SHIFT_MIN, SHIFT_MIN):
            a, b = W["k0"] + sh, W["k1"] + sh
            ok = (a >= 0) & (b < g.n)
            a, b = a[ok], b[ok]
            quiet = ~flares[c].overlaps(g.t[a], g.t[b] + 60.0 * HORIZON_MIN[c])
            covd = np.array([cover(x, y) >= COVER for x, y in zip(a, b)], bool)
            k0s.append(a[quiet & covd])
            k1s.append(b[quiet & covd])
        return np.concatenate(k0s), np.concatenate(k1s)

    Sc = {c: shifted(c) for c in ("C", "M")}

    # ---- scoring an alert series ---------------------------------------------------
    def false_alarms(c, on, mask):
        s, e = episodes(on & mask)
        days = mask.sum() / 1440.0
        false = ~flares[c].overlaps(g.t[s], g.t[e] + 60.0 * HORIZON_MIN[c]) if s.size else np.zeros(0, bool)
        return {"false_per_day": _r(false.sum() / days) if days else None, "episodes": int(s.size),
                "false": int(false.sum()), "duty_cycle": _r((on & mask).sum() / max(mask.sum(), 1), 4)}

    def leads(on, W, key="k1"):
        nx = next_on(on)
        first = nx[W["k0"]]
        warned = first <= W["k1"]
        return np.where(warned, (W[key] - first).astype(float), np.nan), warned

    def chance(c, on):
        a, b = Sc[c]
        nx = next_on(on)
        return float(np.mean(nx[a] <= b)) if a.size else np.nan

    def summarise(c, on, mask_fa):
        W = Wc[c]
        ld, warned = leads(on, W)
        d = {"n": int(W["k0"].size), "warned": int(warned.sum()), "TPR": _r(warned.mean()),
             "chance": _r(chance(c, on)), "median_lead_min": _r(np.nanmedian(ld), 1) if warned.any() else None,
             "q25_lead_min": _r(np.nanpercentile(ld, 25), 1) if warned.any() else None,
             "q75_lead_min": _r(np.nanpercentile(ld, 75), 1) if warned.any() else None,
             "lead_ge_5min": _r(np.mean(np.nan_to_num(ld, nan=-1) >= 5)),
             "lead_ge_10min": _r(np.mean(np.nan_to_num(ld, nan=-1) >= 10))}
        d["event_TSS"] = _r(d["TPR"] - d["chance"]) if d["chance"] is not None else None
        d.update(false_alarms(c, on, mask_fa))
        for letter in ("C", "M", "X"):
            m = np.array([s[:1] == letter for s in W["cls"]], bool)
            if m.any():
                d[f"class_{letter}"] = {"n": int(m.sum()), "TPR": _r(warned[m].mean()),
                                        "median_lead_min": _r(np.nanmedian(ld[m]), 1) if warned[m].any() else None}
        if c == "M":
            lm, wm = leads(on, W, "cross_k")
            d["before_M1"] = {"median_lead_min": _r(np.nanmedian(lm), 1) if wm.any() else None,
                              "lead_ge_5min": _r(np.mean(np.nan_to_num(lm, nan=-99) >= 5)),
                              "alert_before_M1": _r(np.mean(np.nan_to_num(lm, nan=-99) > 0))}
        return d, ld

    # ---- operating points (validation only) -------------------------------------------
    ops: dict = {"C": {}, "M": {}}
    for c in ("C", "M"):
        y, yok = labels[c]
        for mth in methods[c] + (["model_no_hel1os"] if PH is not None else []):
            s = sig[mth][c]
            if mth.startswith("rule"):
                ops[c][mth] = {"fixed": {"threshold": 0.5}}
                continue
            if mth == "model_no_hel1os":
                continue                      # scored at the network's own thresholds
            o: dict = {}
            m = val & base & yok
            if mth == "model" and c == "C":
                o["best_TSS"] = {"threshold": thr_c_frozen, "source": "frozen at freeze time"}
            else:
                tt, best = tss_threshold(y[m], s[m])
                o["best_TSS"] = {"threshold": round(tt, 4), "val_TSS": round(best, 3)}
            cands = np.unique(np.quantile(s[val & base], QGRID))
            fa_val = np.array([false_alarms(c, np.nan_to_num(s, nan=-np.inf) >= q, val & base)["false_per_day"]
                               for q in cands])
            for target in FA_TARGETS[c]:
                okq = np.flatnonzero(fa_val <= target)
                q = float(cands[okq[0]]) if okq.size else float(cands[-1])
                o[f"fa_{target:g}"] = {"threshold": round(q, 4),
                                       "val_false_per_day": _r(fa_val[okq[0]] if okq.size else fa_val[-1])}
            ops[c][mth] = o
        if PH is not None:
            ops[c]["model_no_hel1os"] = ops[c]["model"]

    # ---- test scores ----------------------------------------------------------------
    common = test & base
    summary: dict = {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        "model": args.name, "model_sha256": str(P["sha"]),
        "test_period": [utc(TEST_START + EMBARGO_S), utc(g.t[-1])],
        "test_days_with_data": round(float(common.sum()) / 1440.0, 1),
        "primary_false_alarms_per_day": PRIMARY, "results": {"C": {}, "M": {}},
    }
    lead_cols: dict = {}
    for c in ("C", "M"):
        for mth in methods[c]:
            res = {}
            for op, spec in ops[c][mth].items():
                on = np.nan_to_num(sig[mth][c], nan=-np.inf) >= spec["threshold"]
                d, ld = summarise(c, on, common)
                d["threshold"] = spec["threshold"]
                res[op] = d
                if op in ("fixed", f"fa_{PRIMARY[c]:g}"):
                    lead_cols[(c, mth)] = ld
            summary["results"][c][mth] = res
        summary["results"][c]["flares"] = int(Wc[c]["k0"].size)
        summary["results"][c]["chance_windows"] = int(Sc[c][0].size)

        # paired: network vs each baseline at the primary false-alarm rate
        W = Wc[c]
        op = f"fa_{PRIMARY[c]:g}"
        comp = {}
        for other in [m for m in methods[c] if m not in ("model",) and not m.startswith("rule")]:
            a, b = lead_cols[(c, "model")], lead_cols[(c, other)]
            comp[other] = {
                "warned_diff": _r(np.mean(np.isfinite(a)) - np.mean(np.isfinite(b))),
                "warned_diff_ci": day_bootstrap(W["day"], lambda i, a=a, b=b: np.mean(np.isfinite(a[i])) - np.mean(np.isfinite(b[i]))),
                "mean_lead_diff_min": _r(np.mean(np.nan_to_num(a)) - np.mean(np.nan_to_num(b)), 2),
                "mean_lead_diff_ci": day_bootstrap(W["day"], lambda i, a=a, b=b: np.mean(np.nan_to_num(a[i])) - np.mean(np.nan_to_num(b[i]))),
            }
        summary["results"][c]["network_minus"] = {"operating_point": op, **comp}

        if PH is not None:
            hsel = W["hel1os"] >= 0.5
            spec = ops[c]["model"][op]
            on_w = np.nan_to_num(sig["model"][c], nan=-np.inf) >= spec["threshold"]
            on_wo = np.nan_to_num(sig["model_no_hel1os"][c], nan=-np.inf) >= spec["threshold"]
            Wh = {k: v[hsel] for k, v in W.items()}
            lw, ww = leads(on_w, Wh)
            lo_, wo = leads(on_wo, Wh)
            hmask = common & (np.nan_to_num(hard_frac) >= 0.5)
            summary["results"][c]["hel1os_ablation"] = {
                "operating_point": op, "flares": int(hsel.sum()),
                "warned_with": _r(ww.mean()), "warned_without": _r(wo.mean()),
                "median_lead_with": _r(np.nanmedian(lw), 1) if ww.any() else None,
                "median_lead_without": _r(np.nanmedian(lo_), 1) if wo.any() else None,
                "mean_lead_gain_min": _r(np.mean(np.nan_to_num(lw)) - np.mean(np.nan_to_num(lo_)), 2),
                "mean_lead_gain_ci": day_bootstrap(Wh["day"], lead_gain(lw, lo_)),
                "warned_gain_ci": day_bootstrap(Wh["day"], rate_gain(ww, wo)),
                "false_alarms_with": false_alarms(c, on_w, hmask),
                "false_alarms_without": false_alarms(c, on_wo, hmask),
                "days_with_hel1os": round(float(hmask.sum()) / 1440.0, 1),
            }
            # Same threshold is not a fair fight if HEL1OS also changes the false-alarm
            # rate: move the HEL1OS-hidden threshold until its false alarms on these
            # days match the full model's, then compare again.
            s_wo = sig["model_no_hel1os"][c]
            target = false_alarms(c, on_w, hmask)["false_per_day"] or 0.0
            cands = np.unique(np.quantile(s_wo[hmask], QGRID))
            fas = np.array([false_alarms(c, np.nan_to_num(s_wo, nan=-np.inf) >= q, hmask)["false_per_day"]
                            for q in cands])
            q = float(cands[int(np.argmin(np.abs(fas - target)))])
            on_wm = np.nan_to_num(s_wo, nan=-np.inf) >= q
            lm_, wm_ = leads(on_wm, Wh)
            summary["results"][c]["hel1os_ablation"]["matched_false_alarms"] = {
                "threshold_without": round(q, 4),
                "false_per_day_without": false_alarms(c, on_wm, hmask)["false_per_day"],
                "warned_without": _r(wm_.mean()),
                "median_lead_without": _r(np.nanmedian(lm_), 1) if wm_.any() else None,
                "warned_gain_ci": day_bootstrap(Wh["day"], rate_gain(ww, wm_)),
                "mean_lead_gain_min": _r(np.mean(np.nan_to_num(lw)) - np.mean(np.nan_to_num(lm_)), 2),
                "mean_lead_gain_ci": day_bootstrap(Wh["day"], lead_gain(lw, lm_)),
            }

    # ---- trade-off curves on the test period (figure) ------------------------------------
    curves: dict = {"C": {}, "M": {}}
    for c in ("C", "M"):
        for mth in [m for m in methods[c] if not m.startswith("rule")]:
            s = sig[mth][c]
            pts = []
            for q in np.unique(np.quantile(s[common], QGRID[::2])):
                on = np.nan_to_num(s, nan=-np.inf) >= q
                ld, warned = leads(on, Wc[c])
                pts.append({"threshold": round(float(q), 4),
                            "false_per_day": false_alarms(c, on, common)["false_per_day"],
                            "TPR": _r(warned.mean()), "chance": _r(chance(c, on)),
                            "median_lead_min": _r(np.nanmedian(ld), 1) if warned.any() else None})
            curves[c][mth] = pts
    summary["curves_test"] = curves

    # ---- per-flare table -------------------------------------------------------------------
    rows = []
    for c in ("C", "M"):
        W = Wc[c]
        for j in range(W["k0"].size):
            x = fl[W["i"][j]]
            row = {"alert_type": c, "goes_class": x.goes_class, "start_utc": utc(x.start_unix),
                   "peak_utc": utc(x.peak_unix), "window_min": int(W["k1"][j] - W["k0"][j]),
                   "hel1os_frac": round(float(W["hel1os"][j]), 2)}
            if c == "M":
                row["m1_reached_utc"] = utc(g.t[W["cross_k"][j]])
            for mth in methods[c]:
                v = lead_cols[(c, mth)][j]
                row[f"lead_min_{mth}"] = "" if not np.isfinite(v) else int(v)
            rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "lead_times.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    write_alert_rules(out / "alert_rules.json", summary, ops, str(P["sha"]))
    write_watch(out / "watch.npz", g, f, lx, sig, hard_frac, fl, TEST_START)
    (out / "leadtime_summary.json").write_text(json.dumps({**summary, "operating_points": ops}, indent=2),
                                               encoding="utf-8")
    md = render(summary)
    (out / "LEADTIME.md").write_text(md, encoding="utf-8")
    figure(summary, out / "leadtime.png")
    print(md)
    return 0


# ---------------------------------------------------------------------------
# 4. report and figure
# ---------------------------------------------------------------------------

def write_alert_rules(path: Path, summary: dict, ops: dict, sha: str) -> None:
    """The rules the console raises alerts with, fixed on validation.

    C (a >= C1 flare within 15 min): the network's calibrated probability at the
    threshold that gave PRIMARY["C"] false alarms a day on validation.
    M (flux reaching M1 within 30 min): the combined signal (the higher of the
    network's flux forecast and the flux now) at the threshold that gave
    PRIMARY["M"] false alarms a day on validation. The choice is made before any
    scoring: an operator wants the alarm on when either says M, and the network
    alone can sit below a flare that is already at M level. The test scores of
    the network alone are in LEADTIME.md for comparison."""
    rc, rm = ops["C"]["model"][f"fa_{PRIMARY['C']:g}"], ops["M"]["combined"][f"fa_{PRIMARY['M']:g}"]
    test_c = summary["results"]["C"]["model"][f"fa_{PRIMARY['C']:g}"]
    test_m = summary["results"]["M"]["combined"][f"fa_{PRIMARY['M']:g}"]
    rules = {
        "model_sha256": sha,
        "C": {"signal": "calibrated P(>= C1 flare in progress within 15 min)", "threshold": rc["threshold"],
              "false_alarms_per_day_validation": rc.get("val_false_per_day"),
              "test": {k: test_c.get(k) for k in ("TPR", "chance", "event_TSS", "median_lead_min", "false_per_day")}},
        "M": {"signal": "max(network median flux forecast +5/+15/+30 min, SoLEXS flux now), log10 W/m^2",
              "threshold": rm["threshold"], "false_alarms_per_day_validation": rm.get("val_false_per_day"),
              "test": {k: test_m.get(k) for k in ("TPR", "chance", "event_TSS", "median_lead_min", "false_per_day")}},
    }
    path.write_text(json.dumps(rules, indent=2), encoding="utf-8")


def write_watch(path: Path, g, flux, goes, sig: dict, hard_frac, flares, test_start: float) -> None:
    """Minute series for the console's Flare Watch: what the alert rules saw,
    and GOES for comparison (never an input). Every value sits at the minute it
    became known: SoLEXS and the network at the end of their minute, GOES
    likewise (its record [A, A+60) is plotted at A+60)."""
    f32 = np.float32
    goes_known = np.concatenate([[np.nan], goes[:-1]])
    np.savez_compressed(
        path, t=g.t, test_start=np.float64(test_start),
        flux=flux.astype(f32), goes=goes_known.astype(f32), hard_frac=hard_frac.astype(f32),
        p_c=sig["model"]["C"].astype(f32), m_network=sig["model"]["M"].astype(f32),
        m_combined=sig["combined"]["M"].astype(f32),
        flare_start=np.array([x.start_unix for x in flares]), flare_peak=np.array([x.peak_unix for x in flares]),
        flare_end=np.array([x.end_unix for x in flares]), flare_class=np.array([x.goes_class for x in flares]))


def render(s: dict) -> str:
    def f(v, nd=1):
        return "--" if v is None else f"{v:.{nd}f}"

    def pct(v):
        return "--" if v is None else f"{100 * v:.0f}%"

    L = [f"# Lead time before the flare peak ({s['model']})", "",
         f"Generated {s['generated_utc']} UTC by `python -m solarflare alerts`. Test period {s['test_period'][0]} -> "
         f"{s['test_period'][1]} UTC, {s['test_days_with_data']} days with SoLEXS and model output. "
         "Every threshold was fixed on the validation period; the test period only scores it.", "",
         "*Warned*: the alert was on at some minute from 30 min before the GOES start (or the previous flare's peak) "
         "to the GOES peak. *Chance*: the same windows moved 2 h, where no flare happened. "
         "*Event TSS* = warned - chance. *Lead*: GOES peak minus the first alert minute.", ""]
    for c, title in (("C", "Alert: a GOES >= C1 flare within 15 min"), ("M", "Alert: GOES flux will reach M1 within 30 min")):
        R = s["results"][c]
        L += [f"## {title}", "", f"{R['flares']} GOES flares, {R['chance_windows']} flare-free chance windows.", ""]
        for op_title, op in ((f"Same false-alarm rate: {s['primary_false_alarms_per_day'][c]:g} per day on validation",
                              f"fa_{s['primary_false_alarms_per_day'][c]:g}"),
                             ("Each method's best-TSS threshold (validation minutes)", "best_TSS")):
            L += [f"### {op_title}", "",
                  "| method | warned | chance | event TSS | median lead (IQR), min | lead >= 5 min | false alarms / day (test) | time on |",
                  "|---|---|---|---|---|---|---|---|"]
            for mth, res in R.items():
                if not isinstance(res, dict) or mth in ("network_minus", "hel1os_ablation"):
                    continue
                d = res.get(op) or res.get("fixed")
                if d is None or (op == "best_TSS" and "fixed" in res):
                    continue
                L.append(f"| {NAMES[mth]} | {pct(d['TPR'])} | {pct(d['chance'])} | {f(d['event_TSS'], 2)} | "
                         f"{f(d['median_lead_min'])} ({f(d['q25_lead_min'])}-{f(d['q75_lead_min'])}) | "
                         f"{pct(d['lead_ge_5min'])} | {f(d['false_per_day'], 2)} | {pct(d['duty_cycle'])} |")
            L += [""]
        op = f"fa_{s['primary_false_alarms_per_day'][c]:g}"
        letters = [k for k in ("C", "M", "X") if f"class_{k}" in R["model"][op]]
        L += [f"By GOES class, at {s['primary_false_alarms_per_day'][c]:g} false alarms/day (warned / median lead, min):", "",
              "| method | " + " | ".join(letters) + " |", "|---|" + "---|" * len(letters)]
        for mth, res in R.items():
            if not isinstance(res, dict) or mth in ("network_minus", "hel1os_ablation"):
                continue
            d = res.get(op) or res.get("fixed")
            L.append(f"| {NAMES[mth]} | " + " | ".join(
                f"{pct(d[f'class_{k}']['TPR'])} / {f(d[f'class_{k}']['median_lead_min'])} (n={d[f'class_{k}']['n']})"
                for k in letters) + " |")
        L += [""]
        if c == "M":
            L += ["Before GOES first reached M1.0:", "", "| method | alert before M1 | median lead, min | lead >= 5 min |",
                  "|---|---|---|---|"]
            for mth in ("model", "combined", "trend", "current"):
                d = R[mth][op]["before_M1"]
                L.append(f"| {NAMES[mth]} | {pct(d['alert_before_M1'])} | {f(d['median_lead_min'])} | {pct(d['lead_ge_5min'])} |")
            L += [""]
        nm = R.get("network_minus", {})
        for other, d in nm.items():
            if other == "operating_point":
                continue
            L.append(f"- Network minus {NAMES[other]} (same flares, day-block 95% CI): warned {d['warned_diff']:+.3f} "
                     f"{d['warned_diff_ci']}, mean lead {d['mean_lead_diff_min']:+.2f} min {d['mean_lead_diff_ci']} "
                     "(a missed flare counts as 0).")
        h = R.get("hel1os_ablation")
        if h:
            L.append(f"- HEL1OS ablation ({h['flares']} flares with HEL1OS for >= half the window, same thresholds): warned "
                     f"{pct(h['warned_with'])} with vs {pct(h['warned_without'])} without {h['warned_gain_ci']}; mean lead "
                     f"gain {h['mean_lead_gain_min']:+.2f} min {h['mean_lead_gain_ci']}; false alarms/day "
                     f"{f(h['false_alarms_with']['false_per_day'], 2)} vs {f(h['false_alarms_without']['false_per_day'], 2)} "
                     f"({h['days_with_hel1os']} days with HEL1OS).")
            mf = h.get("matched_false_alarms")
            if mf:
                L.append(f"- The same ablation with the HEL1OS-hidden threshold moved to the same false-alarm rate "
                         f"({f(mf['false_per_day_without'], 2)}/day): warned {pct(h['warned_with'])} with vs "
                         f"{pct(mf['warned_without'])} without {mf['warned_gain_ci']}; mean lead gain "
                         f"{mf['mean_lead_gain_min']:+.2f} min {mf['mean_lead_gain_ci']}.")
        L += [""]
    L += ["Notes:", "",
          "- A false alarm is an alert episode with no GOES flare of the class from its start to 15 min (C) or "
          "30 min (M) after it ends. 'Time on' is the share of minutes the alert was raised.",
          "- The M alert's lead before the peak includes the flare's own rise from C level; the lead before GOES "
          "first reached M1.0 is the operationally useful number.",
          "- Trade-off curves over all thresholds: leadtime_summary.json (curves_test) and leadtime.png.", ""]
    return "\n".join(L)


def figure(s: dict, dest: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED = "#1d2733", "#98a2ad"
    col = {"model": "#2a8c7c", "combined": "#7a4fd1", "trend": "#d9822b", "current": "#4a6fa5"}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.6), dpi=150)
    for row, c in enumerate(("C", "M")):
        a1, a2 = ax[row]
        for mth, pts in s["curves_test"][c].items():
            pts = [p for p in pts if p["false_per_day"] is not None and p["false_per_day"] > 0]
            fa = [p["false_per_day"] for p in pts]
            a1.plot(fa, [p["TPR"] - (p["chance"] or 0) for p in pts], "-", color=col[mth], lw=1.6, label=NAMES[mth])
            a2.plot(fa, [p["median_lead_min"] if p["median_lead_min"] is not None else np.nan for p in pts], "-",
                    color=col[mth], lw=1.6, label=NAMES[mth])
            d = s["results"][c][mth][f"fa_{s['primary_false_alarms_per_day'][c]:g}"]
            a1.plot([d["false_per_day"]], [d["event_TSS"]], "o", color=col[mth], ms=6, mec="white")
            a2.plot([d["false_per_day"]], [d["median_lead_min"]], "o", color=col[mth], ms=6, mec="white")
        if c == "C":
            for mth, mk in (("rule", "s"), ("rule_C_level", "D")):
                d = s["results"][c][mth]["fixed"]
                a1.plot([d["false_per_day"]], [d["event_TSS"]], mk, color=INK, ms=5, label=NAMES[mth])
                a2.plot([d["false_per_day"]], [d["median_lead_min"]], mk, color=INK, ms=5, label=NAMES[mth])
        for a in (a1, a2):
            a.set_xscale("log")
            a.set_xlabel("false alarms per day (test period)")
            a.axvline(s["primary_false_alarms_per_day"][c], color=MUTED, lw=0.8, ls=":")
            a.grid(alpha=0.25, lw=0.5)
        a1.set_ylabel("event TSS (warned - chance)")
        a2.set_ylabel("median lead before GOES peak, min")
        name = "C1" if c == "C" else "M1"
        a1.set_title(f"{'>= C1 flare within 15 min' if c == 'C' else 'flux reaches M1 within 30 min'}: skill", loc="left",
                     fontsize=10, color=INK)
        a2.set_title(f"{name} alerts: lead time", loc="left", fontsize=10, color=INK)
        a1.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle(f"Alert lead time and false alarms, test period {s['test_period'][0][:10]} to {s['test_period'][1][:10]}"
                 " (dots: thresholds fixed on validation)", x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(dest)
    plt.close(fig)


def main(argv=None) -> int:
    S = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--blank-hard", action="store_true")
    ap.add_argument("--forward-dir", default=str(S.frozen_dir))
    ap.add_argument("--run-dir", default=str(S.model_dir))
    ap.add_argument("--name", default="final")
    ap.add_argument("--data-root", default=str(S.data_root))
    ap.add_argument("--cache-dir", default=str(S.cache))
    ap.add_argument("--goes-dir", default=str(S.goes_dir))
    ap.add_argument("--catalog", default=str(S.catalog))
    ap.add_argument("--out", default=str(S.alerts))
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args(argv)
    global TRAIN_END, TEST_START
    split = S.split_dates(Path(args.run_dir) if getattr(args, "run_dir", None) else None)
    TRAIN_END, TEST_START = split["train_end"], split["test_start"]
    sys.stdout.reconfigure(encoding="utf-8")
    if args.predict:
        predict(args)
        return 0
    return score(args)


if __name__ == "__main__":
    raise SystemExit(main())
