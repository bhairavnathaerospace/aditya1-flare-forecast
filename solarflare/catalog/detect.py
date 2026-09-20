"""Algorithmic nowcasting: independent soft and hard X-ray flare detection, merged.

Two instruments, detectors that know nothing of each other, one catalogue:

* **Soft X-rays (SoLEXS)** -- an SWPC-style event rule on SoLEXS count rates
  converted to GOES 1-8 Angstrom flux (calibration fitted on the training
  period). A flare *starts* at the first of five one-minute samples that rise
  strictly; it is *confirmed* -- and the alert raised -- once the flux reaches
  1.4x its starting level; it *peaks* at the maximum and *ends* half-way back
  down, or when the next flare starts. Its class comes from SoLEXS alone.
  Rule parameters were fixed on GOES's own flux against the GOES flare list,
  never tuned on SoLEXS. A Poisson guard stops count noise passing as a rise.
* **Hard X-rays (HEL1OS)** -- two detectors, two bands:
  CdTe 5-20 keV (background ~1 cts/s, flares reach thousands) runs the same
  rise rule and gives the primary HEL1OS events; CZT 20-40 keV, where the
  non-thermal electrons show, runs a Poisson burst detector over a trailing
  background that demands *coincidence* of CZT1 and CZT2 (a glitch in one
  detector is not a flare). Higher CZT bands attach to those events and record
  the highest energy a flare reached.
* **Merge** -- a HEL1OS event belongs to a SoLEXS flare when it peaks between
  five minutes before the soft start and the soft end, where impulsive
  emission sits (Neupert effect). Unmatched events on either side stay in the
  catalogue with their origin and whether the other instrument was observing,
  so "hard only" and "soft only" are visible rather than lost.

Every event records when its rule could first have fired (``detect_unix``):
the nowcast latency. Detection is causal up to that time; start, peak and end
are filled in with hindsight, as in any catalogue.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..io.goes import _CLASS_SCALE

#: A merged soft flare owns hard bursts that peak from this long before its start.
HARD_BEFORE_SOFT_S = 300.0


# --------------------------------------------------------------------- classes

def flux_class(flux: float) -> str:
    """1.23e-5 -> "M1.2"; below A1.0 or unusable -> ""."""
    if not np.isfinite(flux) or flux < _CLASS_SCALE["A"]:
        return ""
    letters = sorted(_CLASS_SCALE, key=_CLASS_SCALE.get)
    letter = max((k for k, v in _CLASS_SCALE.items() if flux >= v), key=_CLASS_SCALE.get)
    value = round(flux / _CLASS_SCALE[letter], 1)
    if value >= 10.0 and letter != "X":          # 9.97e-7 is C1.0, not B10.0
        letter = letters[letters.index(letter) + 1]
        value = round(flux / _CLASS_SCALE[letter], 1)
    return f"{letter}{value:.1f}"


def solexs_to_goes_flux(rate: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    """SoLEXS GOES-long analogue rate (cts/s) -> GOES 1-8 A flux (W/m^2).

    ``log10 F = intercept + slope * log10 rate``, fitted on the training period
    only (PreprocessConfig.flux_anchor_*). Non-positive rates give NaN.
    """
    r = np.asarray(rate, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r > 0, 10.0 ** (intercept + slope * np.log10(r)), np.nan)


@dataclass
class PiecewiseCalibration:
    """Monotone piecewise-linear log10 rate -> log10 GOES flux.

    SoLEXS under-reads bright flares: against a single power law it is
    +0.2 dex low above M3 and +0.6 dex low at X level, in the training period
    and after it alike (count-rate saturation). Medians of log flux in 0.2-dex
    bins of log rate, forced monotone, follow that bend; beyond the outer knots
    the end segments are extended.
    """
    x: list[float]
    y: list[float]

    @classmethod
    def fit(cls, log_rate: np.ndarray, log_flux: np.ndarray, width: float = 0.2,
            min_n: int = 30) -> PiecewiseCalibration:
        lr = np.asarray(log_rate, dtype=np.float64)
        lf = np.asarray(log_flux, dtype=np.float64)
        edges = np.arange(np.floor(lr.min() / width) * width, lr.max() + width, width)
        xs, ys = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (lr >= a) & (lr < b)
            if m.sum() >= min_n:
                xs.append(float(np.median(lr[m])))
                ys.append(float(np.median(lf[m])))
        ys = list(np.maximum.accumulate(ys))
        return cls(xs, ys)

    def __call__(self, rate: np.ndarray) -> np.ndarray:
        r = np.asarray(rate, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.where(r > 0, np.log10(np.where(r > 0, r, 1.0)), np.nan)
        x, y = np.asarray(self.x), np.asarray(self.y)
        out = np.interp(lr, x, y)
        lo_s = (y[1] - y[0]) / (x[1] - x[0])
        hi_s = (y[-1] - y[-2]) / (x[-1] - x[-2])
        out = np.where(lr < x[0], y[0] + lo_s * (lr - x[0]), out)
        out = np.where(lr > x[-1], y[-1] + hi_s * (lr - x[-1]), out)
        return np.where(np.isfinite(lr), 10.0 ** out, np.nan)


def to_minutes(t: np.ndarray, rate: np.ndarray, ok: np.ndarray, dt: float,
               min_valid: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average a regular ``dt`` grid into 1-minute samples aligned to the minute.

    Returns (minute start, mean rate, counts in the minute, valid). A minute is
    valid when at least ``min_valid`` of its bins were observed.
    """
    per = max(int(round(60.0 / dt)), 1)
    t = np.asarray(t, dtype=np.float64)
    if t.size == 0:
        z = np.zeros(0)
        return z, z, z, np.zeros(0, bool)
    m0 = np.floor(t[0] / 60.0) * 60.0
    idx = np.floor((t - m0) / 60.0).astype(np.int64)
    n = int(idx[-1]) + 1
    good = np.asarray(ok, bool) & np.isfinite(rate)
    s = np.bincount(idx[good], weights=np.asarray(rate)[good], minlength=n)
    c = np.bincount(idx[good], minlength=n).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    valid = c >= min(min_valid, per)
    counts = np.where(valid, mean * 60.0, np.nan)
    return m0 + 60.0 * np.arange(n), np.where(valid, mean, np.nan), counts, valid


# ------------------------------------------------------------ soft X-ray rule

@dataclass
class SoftEvent:
    start_unix: float
    peak_unix: float
    end_unix: float
    detect_unix: float       # end of the 4th rising minute: when the rule fires
    start_flux: float        # W/m^2
    peak_flux: float         # W/m^2, background included (as GOES classes are)
    goes_class: str          # derived from SoLEXS alone
    end_reason: str          # "decay" | "next_flare" | "gap" | "data_end"
    #: peak_flux before a peak-level correction was applied (NaN if none was)
    peak_flux_minute: float = float("nan")


def noaa_events(t_min: np.ndarray, flux: np.ndarray, valid: np.ndarray,
                counts: np.ndarray | None = None, n_rise: int = 5, amp: float = 1.4,
                sigma: float = 3.0, max_gap_min: int = 10,
                noise: np.ndarray | None = None) -> list[SoftEvent]:
    """Flares from a 1-minute flux series: SWPC-style start, confirmation, half-way end.

    * start -- the first of ``n_rise`` observed minutes that rise strictly
      (four consecutive increases by default);
    * confirmed, and the alert raised, at the end of the first minute whose
      flux is ``amp`` x the starting flux (never before the rise is complete);
    * a rise that never confirms is not a flare: it is folded into the flare it
      interrupted, or dropped.

    ``counts`` (counts per minute behind each flux sample) adds a Poisson
    guard: the confirming increase must be ``sigma`` standard deviations.
    ``noise`` (a measured per-minute standard deviation in flux units, e.g.
    from ``trailing_noise``) does the same for over-dispersed data; with both,
    both must hold.

    Defaults were chosen by running the rule on GOES XRS-B itself (Feb-Aug
    2024) against the GOES flare list: peak recall 0.79, precision 0.81 --
    the ceiling any instrument can reach against that list with this rule.
    """
    f = np.asarray(flux, dtype=np.float64)
    cands = _rise_segments(t_min, f, valid, n_rise=n_rise, max_gap_min=max_gap_min)
    c = None if counts is None else np.asarray(counts, dtype=np.float64)
    t0 = float(t_min[0]) if len(t_min) else 0.0
    out: list[SoftEvent] = []
    for ev in cands:
        s = int(round((ev.start_unix - t0) / 60.0))
        e = int(round((ev.end_unix - t0) / 60.0))
        seg = f[s:e + 1]
        with np.errstate(invalid="ignore"):
            hit = (seg >= amp * ev.start_flux)
            if c is not None:
                dc = c[s:e + 1] - c[s]
                hit &= dc >= sigma * np.sqrt(np.maximum(c[s:e + 1] + c[s], 1.0))
            if noise is not None:
                sd0 = noise[s] if np.isfinite(noise[s]) else np.nanmax(noise[s:e + 1], initial=np.nan)
                hit &= (seg - ev.start_flux) >= sigma * np.sqrt(2.0) * sd0
        hit[:n_rise - 1] = False
        k = np.flatnonzero(hit)
        if k.size:
            ev.detect_unix = float(t_min[s + int(k[0])] + 60.0)
            out.append(ev)
        elif out and ev.start_unix <= out[-1].end_unix + 60.0:
            prev = out[-1]            # an unconfirmed bump inside a decay: same flare
            if ev.peak_flux > prev.peak_flux:
                prev.peak_flux, prev.peak_unix = ev.peak_flux, ev.peak_unix
                prev.goes_class = flux_class(ev.peak_flux)
            prev.end_unix, prev.end_reason = ev.end_unix, ev.end_reason
    return out


def _rise_segments(t_min: np.ndarray, f: np.ndarray, valid: np.ndarray, n_rise: int,
                   max_gap_min: int) -> list[SoftEvent]:
    """Partition the series into candidate flares at every monotonic rise.

    A candidate runs from its rise to the half-way decay point, the next rise
    during its decay, or a data gap longer than ``max_gap_min``.
    """
    ok = np.asarray(valid, bool) & np.isfinite(f) & (f > 0)
    n = f.size
    if n < n_rise:
        return []

    # rise_at[i]: minutes i .. i+n_rise-1 are observed and strictly increasing.
    rise_at = np.zeros(n, bool)
    w = n - n_rise + 1
    mono = np.ones(w, bool)
    for k in range(n_rise):
        mono &= ok[k:k + w]
        if k:
            mono &= f[k:k + w] > f[k - 1:k - 1 + w]
    rise_at[:w] = mono

    out: list[SoftEvent] = []
    i = 0
    while i < n:
        if not rise_at[i]:
            i += 1
            continue
        s = i
        f0 = f[s]
        p = s
        j = s + 1
        reason = "data_end"
        gap = 0
        while j < n:
            if not ok[j]:
                gap += 1
                if gap > max_gap_min:
                    reason = "gap"
                    break
                j += 1
                continue
            gap = 0
            if f[j] > f[p]:
                p = j
            elif j > p and rise_at[j]:
                reason = "next_flare"
                break
            elif j > p and f[j] <= 0.5 * (f[p] + f0):
                reason = "decay"
                break
            j += 1
        if reason == "decay":
            e = j
        else:
            e = j - 1
            while e > p and not ok[e]:
                e -= 1
            if reason == "next_flare":
                e = j
        out.append(SoftEvent(
            start_unix=float(t_min[s]), peak_unix=float(t_min[p]), end_unix=float(t_min[min(e, n - 1)]),
            detect_unix=float(t_min[s + n_rise - 1] + 60.0),
            start_flux=float(f0), peak_flux=float(f[p]), goes_class=flux_class(float(f[p])),
            end_reason=reason))
        i = j if reason == "next_flare" else max(j, s + 1)
    return out


# ------------------------------------------------------------ hard X-ray bursts

@dataclass
class HardBurst:
    band: str
    start_unix: float
    peak_unix: float
    end_unix: float
    detect_unix: float       # end of the n_consec-th significant bin
    peak_excess: float       # cts/s over background, detectors summed
    background: float        # cts/s, detectors summed, frozen at the start
    peak_sigma: float
    excess_counts: float     # fluence above background, counts
    detectors: str           # "czt1+czt2", "cdte1", ...


def _trailing(t: np.ndarray, x: np.ndarray, window_s: float, guard_s: float, min_bins: int) -> np.ndarray:
    """Rolling median of x over [t - window_s, t - guard_s), NaN until min_bins are seen."""
    import pandas as pd

    t = np.asarray(t, dtype=np.float64)
    med = pd.Series(x, index=pd.to_datetime(t, unit="s")).rolling(
        f"{int(window_s - guard_s)}s", min_periods=min_bins).median().to_numpy()
    # the value at time u describes (u - width, u]; it is needed at u = t - guard_s
    i = np.searchsorted(t, t - guard_s, side="right") - 1
    out = np.full(t.size, np.nan)
    ok = (i >= 0) & (np.abs(t[np.clip(i, 0, None)] - (t - guard_s)) < 1e-6)
    out[ok] = med[i[ok]]
    return out


def trailing_background(t: np.ndarray, rate: np.ndarray, ok: np.ndarray, window_s: float = 1800.0,
                        guard_s: float = 120.0, min_bins: int = 10) -> np.ndarray:
    """Median of observed bins in [t - window_s, t - guard_s): causal, and blind
    to the last ``guard_s`` so a rising burst does not lift its own background."""
    return _trailing(t, np.where(np.asarray(ok, bool), rate, np.nan), window_s, guard_s, min_bins)


def trailing_noise(t: np.ndarray, rate: np.ndarray, ok: np.ndarray, window_s: float = 1800.0,
                   guard_s: float = 120.0, min_bins: int = 10) -> np.ndarray:
    """Measured bin-to-bin scatter over the same trailing window.

    1.4826 x median |first difference| / sqrt 2, a robust standard deviation.
    HEL1OS light curves are over-dispersed -- CdTe 5-20 keV scatters ~2x and
    CZT 20-40 keV ~10x what counting statistics predict -- so a Poisson
    threshold there flags noise as bursts. The measured scatter does not.
    """
    r = np.where(np.asarray(ok, bool), rate, np.nan)
    d = np.abs(np.diff(r, prepend=np.nan))
    return 1.4826 / np.sqrt(2.0) * _trailing(t, d, window_s, guard_s, min_bins)


def hxr_bursts(t: np.ndarray, rates: list[np.ndarray], covs: list[np.ndarray], dt: float,
               band: str, names: list[str], k_sum: float = 5.0, k_each: float = 2.0,
               n_consec: int = 3, merge_s: float = 120.0, max_len_s: float = 3 * 3600.0) -> list[HardBurst]:
    """Bursts in one energy band seen by one or two detectors.

    Each detector's excess over its trailing background is divided by the
    larger of its measured scatter and its counting noise. A bin is
    *significant* when the summed excess is ``k_sum`` sigma and every detector
    observing that bin shows ``k_each`` sigma on its own (coincidence).
    ``n_consec`` significant bins in a row trigger a burst. Background and
    noise are then frozen at the burst start, so a long flare cannot lift its
    own baseline; the burst ends when the summed excess stays below 1 sigma
    for two bins (or after ``max_len_s``).
    """
    t = np.asarray(t, dtype=np.float64)
    n = t.size
    if n == 0:
        return []
    ons, bgs, sds = [], [], []
    for r, c in zip(rates, covs):
        c = np.asarray(c)
        live = c > 0
        bg = trailing_background(t, r, live)
        emp = trailing_noise(t, r, live)
        with np.errstate(invalid="ignore", divide="ignore"):
            poisson = np.sqrt(np.maximum(bg, 0.0) / (dt * np.maximum(c, 1e-3)))
        sd = np.fmax(emp, poisson)
        on = live & np.isfinite(r) & np.isfinite(bg) & np.isfinite(sd) & (sd > 0)
        ons.append(on)
        bgs.append(bg)
        sds.append(sd)
    R = [np.where(on, r, 0.0) for r, on in zip(rates, ons)]

    def significance(j: int, ref: int | None = None) -> tuple[float, float]:
        """Summed excess (cts/s) and its sigma at bin j, background taken at ref."""
        k = j if ref is None else ref
        ex, var = 0.0, 0.0
        for r, on, bg, sd in zip(R, ons, bgs, sds):
            if not on[j]:
                continue
            b = bg[k] if np.isfinite(bg[k]) else bg[j]
            v = sd[k] if np.isfinite(sd[k]) else sd[j]
            ex += r[j] - b
            var += v * v
        return ex, (ex / np.sqrt(var) if var > 0 else 0.0)

    any_on = np.zeros(n, bool)
    num = np.zeros(n)
    var = np.zeros(n)
    each_ok = np.ones(n, bool)
    for r, on, bg, sd in zip(R, ons, bgs, sds):
        ex = np.where(on, r - np.nan_to_num(bg), 0.0)
        num += ex
        var += np.where(on, np.nan_to_num(sd) ** 2, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            each_ok &= ~on | (ex / np.where(on, sd, 1.0) >= k_each)
        any_on |= on
    with np.errstate(invalid="ignore", divide="ignore"):
        sig = np.where(var > 0, num / np.sqrt(var), 0.0)
    hot = any_on & (sig >= k_sum) & each_ok

    run = np.zeros(n, np.int64)
    for i in range(n):
        run[i] = run[i - 1] + 1 if (hot[i] and i) else int(hot[i])

    bursts: list[HardBurst] = []
    max_len = int(max_len_s / dt)
    i = 0
    while i < n:
        if run[i] < n_consec:
            i += 1
            continue
        trig = i
        s = trig - n_consec + 1
        while s > 0 and any_on[s - 1] and sig[s - 1] >= 1.0:     # already rising
            s -= 1
        j, below = trig, 0
        peak, peak_ex, peak_sig, fluence = trig, -np.inf, 0.0, 0.0
        while j < n and j - s < max_len:
            if not any_on[j]:
                if j > trig and not any_on[j - 1]:
                    break
                j += 1
                continue
            ex, z = significance(j, ref=s)
            fluence += ex * dt
            if ex > peak_ex:
                peak, peak_ex, peak_sig = j, ex, z
            below = below + 1 if z < 1.0 else 0
            if below >= 2:
                break
            j += 1
        e = min(j, n - 1)
        dets = [nm for on, nm in zip(ons, names) if on[s:e + 1].any()]
        bsum = sum(float(bg[s]) for bg, on in zip(bgs, ons) if on[s] and np.isfinite(bg[s]))
        bursts.append(HardBurst(
            band=band, start_unix=float(t[s]), peak_unix=float(t[peak]), end_unix=float(t[e]),
            detect_unix=float(t[trig] + dt), peak_excess=float(peak_ex), background=bsum,
            peak_sigma=float(peak_sig), excess_counts=float(fluence), detectors="+".join(dets)))
        i = e + 1

    merged: list[HardBurst] = []
    for b in bursts:
        if merged and b.start_unix - merged[-1].end_unix <= merge_s:
            a = merged[-1]
            keep = a if a.peak_excess >= b.peak_excess else b
            merged[-1] = HardBurst(
                band=band, start_unix=a.start_unix, peak_unix=keep.peak_unix, end_unix=b.end_unix,
                detect_unix=a.detect_unix, peak_excess=keep.peak_excess, background=keep.background,
                peak_sigma=keep.peak_sigma, excess_counts=a.excess_counts + b.excess_counts,
                detectors="+".join(sorted(set(a.detectors.split("+")) | set(b.detectors.split("+")))))
        else:
            merged.append(b)
    return merged


# ---------------------------------------------------------------- merging

@dataclass
class HardEvent:
    """One HEL1OS flare: a primary detection plus the bands that joined it."""
    start_unix: float
    peak_unix: float          # peak of the primary band
    end_unix: float
    detect_unix: float        # earliest trigger in any band
    bands: list[str] = field(default_factory=list)
    peak_sigma: float = 0.0   # of the primary band
    peak_excess: dict = field(default_factory=dict)      # band -> cts/s over background
    peak_time: dict = field(default_factory=dict)        # band -> unix


def rise_events_as_bursts(events: list[SoftEvent], band: str, dt_min: float = 60.0
                          ) -> list[HardBurst]:
    """Rule-detected HEL1OS events (rates in cts/s) in the burst format."""
    out = []
    for e in events:
        ex = e.peak_flux - e.start_flux
        bg = max(e.start_flux, 1e-9)
        out.append(HardBurst(
            band=band, start_unix=e.start_unix, peak_unix=e.peak_unix, end_unix=e.end_unix,
            detect_unix=e.detect_unix, peak_excess=ex, background=bg,
            peak_sigma=ex * dt_min / np.sqrt(max(bg * dt_min, 1.0)), excess_counts=float("nan"),
            detectors=""))
    return out


def attach_bands(primary: list[HardBurst], others: list[tuple[list[HardBurst], bool]]
                 ) -> list[HardEvent]:
    """Build HEL1OS events.

    ``primary`` bursts each start an event. For every ``(bursts, standalone)``
    in ``others``, in order, a burst joins the latest-starting event whose
    ``[start - HARD_BEFORE_SOFT_S, end]`` holds its peak; otherwise it starts
    an event of its own if ``standalone``, or is dropped. Contiguous primary
    events (one flare ending as the next starts) stay separate, which a plain
    time-overlap union would not do.
    """
    evs = [HardEvent(b.start_unix, b.peak_unix, b.end_unix, b.detect_unix, [b.band], b.peak_sigma,
                     {b.band: b.peak_excess}, {b.band: b.peak_unix})
           for b in sorted(primary, key=lambda x: x.start_unix)]
    for bursts, standalone in others:
        starts = np.array([e.start_unix for e in evs])
        new = []
        for b in bursts:
            k = int(np.searchsorted(starts, b.peak_unix + HARD_BEFORE_SOFT_S, side="right")) - 1
            host = None
            while k >= 0:
                e = evs[k]
                if e.start_unix - HARD_BEFORE_SOFT_S <= b.peak_unix <= e.end_unix:
                    host = e
                    break
                if e.end_unix < b.peak_unix - 6 * 3600:
                    break
                k -= 1
            if host is None:
                if standalone:
                    new.append(HardEvent(b.start_unix, b.peak_unix, b.end_unix, b.detect_unix, [b.band],
                                         b.peak_sigma, {b.band: b.peak_excess}, {b.band: b.peak_unix}))
                continue
            host.detect_unix = min(host.detect_unix, b.detect_unix)
            if b.band not in host.bands:
                host.bands.append(b.band)
            if b.peak_excess > host.peak_excess.get(b.band, -np.inf):
                host.peak_excess[b.band] = b.peak_excess
                host.peak_time[b.band] = b.peak_unix
        evs = sorted(evs + new, key=lambda e: e.start_unix)
    return evs


@dataclass
class MasterFlare:
    origin: str                       # "soft+hard" | "soft" | "hard"
    soft: SoftEvent | None
    hard: HardEvent | None
    n_hard: int = 0                   # hard events attached to this soft flare
    hard_observed: bool | None = None # HEL1OS was observing (soft-origin rows)
    soft_observed: bool | None = None # SoLEXS was observing (hard-origin rows)

    @property
    def detect_unix(self) -> float:
        ts = [x.detect_unix for x in (self.soft, self.hard) if x is not None]
        return float(min(ts))

    @property
    def start_unix(self) -> float:
        return self.soft.start_unix if self.soft else self.hard.start_unix

    @property
    def peak_unix(self) -> float:
        return self.soft.peak_unix if self.soft else self.hard.peak_unix


def merge(soft: list[SoftEvent], hard: list[HardEvent], hard_observing, soft_observing
          ) -> list[MasterFlare]:
    """One catalogue from both detectors.

    ``hard_observing(t0, t1)`` / ``soft_observing(t0, t1)`` say whether the
    other instrument was taking data over an interval, so a missing
    counterpart can be told apart from an unobserved one.
    """
    hard = sorted(hard, key=lambda h: h.peak_unix)
    peaks = np.array([h.peak_unix for h in hard])
    used = np.zeros(len(hard), bool)
    rows: list[MasterFlare] = []
    for s in soft:
        lo = np.searchsorted(peaks, s.start_unix - HARD_BEFORE_SOFT_S, side="left")
        hi = np.searchsorted(peaks, s.end_unix, side="right")
        cand = [k for k in range(lo, hi) if not used[k]]
        if cand:
            best = max(cand, key=lambda k: hard[k].peak_sigma)
            used[cand] = True
            rows.append(MasterFlare("soft+hard", s, hard[best], n_hard=len(cand), hard_observed=True))
        else:
            rows.append(MasterFlare("soft", s, None,
                                    hard_observed=bool(hard_observing(s.start_unix - HARD_BEFORE_SOFT_S,
                                                                      s.peak_unix))))
    for k, h in enumerate(hard):
        if not used[k]:
            rows.append(MasterFlare("hard", None, h, n_hard=1,
                                    soft_observed=bool(soft_observing(h.start_unix, h.end_unix))))
    rows.sort(key=lambda r: r.start_unix)
    return rows


def as_row(m: MasterFlare) -> dict:
    """Flat dict for CSV/JSON."""
    r: dict = {"origin": m.origin, "detect_unix": m.detect_unix, "start_unix": m.start_unix,
               "peak_unix": m.peak_unix, "n_hard": m.n_hard,
               "hard_observed": m.hard_observed, "soft_observed": m.soft_observed}
    if m.soft:
        for k, v in asdict(m.soft).items():
            r[f"soft_{k}"] = v
    if m.hard:
        h = m.hard
        r.update({"hard_start_unix": h.start_unix, "hard_peak_unix": h.peak_unix,
                  "hard_end_unix": h.end_unix, "hard_detect_unix": h.detect_unix,
                  "hard_bands": "+".join(h.bands), "hard_peak_sigma": h.peak_sigma})
        for b, v in h.peak_excess.items():
            r[f"hard_excess_{b}"] = v
            r[f"hard_peak_unix_{b}"] = h.peak_time[b]
    return r
