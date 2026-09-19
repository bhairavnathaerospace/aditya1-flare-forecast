"""Correctness tests for the flare pipeline.

These check the properties that are easy to break silently and fatal when
broken: causality, mask handling, target alignment, and split hygiene.

    python -m tests.test_correctness      (or: pytest tests/)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarflare.config import Config, ModelConfig, WindowConfig
from solarflare.models.net import SolexHelNet
from solarflare.models.blocks import CausalConv1d, TCNEncoder
from solarflare.preprocess.grid import running_percentile, rebin, interpolate_gaps
from solarflare.preprocess.labels import occurrence_labels, forecast_targets
from solarflare.preprocess.dataset import chronological_split, WindowIndex
from solarflare.metrics import skill_scores, roc_auc, brier_skill_score


FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


# ---------------------------------------------------------------------------

def test_causal_conv_is_causal():
    """Output at t must not change when inputs after t change."""
    torch.manual_seed(0)
    conv = CausalConv1d(3, 5, kernel_size=3, dilation=4)
    x = torch.randn(2, 3, 40)
    y1 = conv(x)
    x2 = x.clone()
    x2[:, :, 25:] += 100.0            # perturb the future only
    y2 = conv(x2)
    same = torch.allclose(y1[:, :, :25], y2[:, :, :25], atol=1e-6)
    check("CausalConv1d does not see the future", same)

    # Left-padding only, so the time axis length is preserved.
    check("CausalConv1d preserves sequence length", y1.shape == (2, 5, 40),
          f"got {tuple(y1.shape)}")


def test_tcn_encoder_is_causal():
    torch.manual_seed(0)
    enc = TCNEncoder(4, 16, (1, 2, 4, 8), 3, 0.0).eval()
    x = torch.randn(1, 4, 64)
    with torch.no_grad():
        y1 = enc(x)
        x2 = x.clone()
        x2[:, :, 40:] += 50.0
        y2 = enc(x2)
    check("TCN encoder stack is causal",
          torch.allclose(y1[:, :, :40], y2[:, :, :40], atol=1e-5))


def test_network_last_step_causality():
    """The whole network, including SE gating and attention pooling.

    The prediction origin is the final step, so this only asserts that the
    heads are a function of the full window -- but the SE block uses a
    cumulative mean, which would break causality at intermediate steps if it
    were a global average.  We test the intermediate property directly.
    """
    torch.manual_seed(0)
    from solarflare.models.blocks import SqueezeExcite
    se = SqueezeExcite(8).eval()
    x = torch.randn(1, 8, 30)
    with torch.no_grad():
        y1 = se(x)
        x2 = x.clone()
        x2[:, :, 20:] += 30.0
        y2 = se(x2)
    check("SqueezeExcite gate is causal (cumulative, not global, mean)",
          torch.allclose(y1[:, :, :20], y2[:, :, :20], atol=1e-5))


def test_selective_scan_matches_reference():
    """The fast chunked scan must equal the obvious sequential recurrence.

    This is an optimisation of a numerical kernel, so it needs an independent
    reference rather than a plausibility check: a subtly wrong scan would
    still train, just to the wrong model.
    """
    from solarflare.models.ssm import (
        selective_scan, selective_scan_sequential, SCAN_CHUNK,
    )
    torch.manual_seed(0)
    # Deliberately span T below, equal to, and well above the chunk size, so
    # both the single-block and the carry-across-blocks paths are exercised.
    for t in (7, SCAN_CHUNK, SCAN_CHUNK + 1, 3 * SCAN_CHUNK + 5):
        dA = torch.rand(2, t, 5, 4) * 0.9 + 0.05
        dBx = torch.randn(2, t, 5, 4)
        fast = selective_scan(dA, dBx)
        ref = selective_scan_sequential(dA, dBx)
        check(f"selective scan matches reference at T={t}",
              torch.allclose(fast, ref, atol=1e-4),
              f"max err {(fast - ref).abs().max():.2e}")


def test_ssm_encoder_is_causal():
    from solarflare.models.ssm import SSMEncoder
    torch.manual_seed(0)
    enc = SSMEncoder(4, 16, n_layers=2, d_state=4, dropout=0.0).eval()
    x = torch.randn(1, 4, 80)
    with torch.no_grad():
        y1 = enc(x)
        x2 = x.clone()
        x2[:, :, 50:] += 30.0
        y2 = enc(x2)
    check("SSM encoder is causal",
          torch.allclose(y1[:, :, :50], y2[:, :, :50], atol=1e-4),
          f"max err {(y1[:, :, :50] - y2[:, :, :50]).abs().max():.2e}")


def test_all_encoders_causal_and_shaped():
    """Every interchangeable encoder must honour the same contract."""
    from solarflare.models.zoo import build_encoder, ENCODERS
    torch.manual_seed(0)
    cfg = ModelConfig(hidden=16, dilations=(1, 2, 4), dropout=0.0)
    for kind in ENCODERS:
        enc = build_encoder(kind, 6, 16, cfg).eval()
        x = torch.randn(2, 6, 48)
        with torch.no_grad():
            y1 = enc(x)
            x2 = x.clone()
            x2[:, :, 30:] += 40.0
            y2 = enc(x2)
        ok_shape = y1.shape == (2, 16, 48)
        ok_causal = torch.allclose(y1[:, :, :30], y2[:, :, :30], atol=1e-4)
        check(f"encoder '{kind}' shape and causality", ok_shape and ok_causal,
              f"shape={tuple(y1.shape)} causal={ok_causal}")


def test_quantiles_monotone():
    torch.manual_seed(0)
    win = WindowConfig()
    net = SolexHelNet(6, 4, 2, ModelConfig(hidden=16, dilations=(1, 2)), win).eval()
    b, t = 4, 64
    with torch.no_grad():
        out = net(torch.randn(b, t, 6), torch.ones(b, t),
                  torch.randn(b, t, 4), torch.ones(b, t), torch.randn(b, t, 2))
    f = out["forecast"]
    check("forecast quantiles are non-crossing",
          bool((f[..., 1:] >= f[..., :-1] - 1e-6).all()))


def test_modality_dropout_never_drops_both():
    torch.manual_seed(0)
    cfg = ModelConfig(hidden=16, dilations=(1, 2), modality_dropout=0.9)
    net = SolexHelNet(6, 4, 2, cfg, WindowConfig()).train()
    b, t = 256, 32
    # Reach into the forward path by checking that the output is finite:
    # if both modalities were ever dropped the attention pool would see an
    # all-masked window.
    out = net(torch.randn(b, t, 6), torch.ones(b, t),
              torch.randn(b, t, 4), torch.ones(b, t), torch.randn(b, t, 2))
    ok = all(torch.isfinite(v).all() for k, v in out.items() if v.dtype.is_floating_point)
    check("heavy modality dropout still yields finite outputs", bool(ok))


def test_fully_masked_window_is_finite():
    """A window with no observations anywhere must not produce NaN."""
    torch.manual_seed(0)
    net = SolexHelNet(6, 4, 2, ModelConfig(hidden=16, dilations=(1, 2)),
                      WindowConfig()).eval()
    b, t = 2, 32
    with torch.no_grad():
        out = net(torch.zeros(b, t, 6), torch.zeros(b, t),
                  torch.zeros(b, t, 4), torch.zeros(b, t), torch.zeros(b, t, 2))
    ok = all(torch.isfinite(v).all() for k, v in out.items() if v.dtype.is_floating_point)
    check("fully-masked window produces finite output (no NaN attention)", bool(ok))


def test_running_percentile_ignores_flare():
    """A percentile background must not be dragged up by a flare in-window."""
    x = np.ones(600) * 2.0
    x[300:320] = 100.0                       # a flare
    bg = running_percentile(x, 201, 10.0)
    check("running percentile background is flare-robust",
          np.all(bg < 3.0), f"max bg {bg.max():.2f}")

    mean_bg = np.convolve(x, np.ones(201) / 201, mode="same")
    check("...and a running mean would not have been",
          mean_bg.max() > 5.0, f"max mean {mean_bg.max():.2f}")


def test_trailing_percentile_is_causal():
    """The background used as a *feature* must not see the future."""
    from solarflare.preprocess.grid import trailing_percentile
    x = np.ones(400) * 2.0
    # A downward excursion, because a 10th percentile is deliberately robust to
    # values added at the top -- an upward spike would not move it and would
    # make this test vacuous.
    y = x.copy()
    y[300:] = 0.0
    a = trailing_percentile(x, 101, 10.0)
    b = trailing_percentile(y, 101, 10.0)
    check("trailing background ignores the future",
          np.allclose(a[:300], b[:300]),
          f"max diff {np.abs(a[:300] - b[:300]).max():.3f}")

    c = running_percentile(x, 101, 10.0)
    d = running_percentile(y, 101, 10.0)
    check("...whereas the centred version does not (hence two functions)",
          not np.allclose(c[:300], d[:300]),
          "centred percentile showed no future dependence -- check the window")


def test_feature_matrix_has_no_future_leakage():
    """End-to-end: perturb the tail of an observation, check earlier features.

    This is the test that would have caught the centred background being used
    as a model input.
    """
    from solarflare.config import PreprocessConfig
    from solarflare.preprocess.features import solexs_features
    from solarflare.io.solexs import SolexsObservation

    rng = np.random.default_rng(0)
    n = 4000
    t = 1.789e9 + np.arange(n, dtype=float)
    spec = rng.poisson(0.02, size=(n, 340)).astype(np.float32)
    obs_a = SolexsObservation("SDD2", t, spec.copy(), np.ones(n), np.ones(n, bool),
                              np.array([[t[0], t[-1]]]), None, "test")
    spec_b = spec.copy()
    spec_b[3000:] += 50                  # a large flare, late in the series
    obs_b = SolexsObservation("SDD2", t, spec_b, np.ones(n), np.ones(n, bool),
                              np.array([[t[0], t[-1]]]), None, "test")

    cfg = PreprocessConfig(dt_seconds=20.0)
    fa = solexs_features(obs_a, cfg).series
    fb = solexs_features(obs_b, cfg).series
    cut = 3000 // int(cfg.dt_seconds)    # grid index of the perturbation

    bad = []
    for i, name in enumerate(fa.names):
        if not np.allclose(fa.values[:cut, i], fb.values[:cut, i],
                           atol=1e-5, equal_nan=True):
            bad.append(name)
    check("no SoLEXS feature depends on future data", not bad,
          f"leaking: {bad}")


def test_rebin_masks_missing():
    """Unsampled points must not be averaged in as zeros."""
    t = np.arange(100, dtype=float)
    v = np.full(100, 150.0)
    valid = np.zeros(100, dtype=bool)
    valid[::6] = True                        # HEL1OS-like sparse sampling
    v[~valid] = 0.0
    grid = np.arange(0, 100, 10, dtype=float)
    out, cov = rebin(t, v, valid, grid, 10.0, expected_per_bin=10 / 6)
    check("sparse rebin recovers the true level, not the fill-diluted one",
          np.allclose(np.nanmean(out), 150.0, atol=1e-6),
          f"got {np.nanmean(out):.2f}")
    check("rebin reports coverage", float(np.nanmean(cov)) > 0.5)


def test_coverage_is_row_level_not_per_channel():
    """Coverage must reflect whether the *sample* was telemetered.

    HEL1OS bands do not share a valid mask exactly, so defining coverage from
    one channel understates it. Here two channels are sampled on alternating
    phases: every bin is fully covered, even though no single channel covers
    more than half of it.
    """
    t = np.arange(60, dtype=float)
    v = np.zeros((60, 2))
    valid = np.zeros((60, 2), dtype=bool)
    valid[::6, 0] = True
    v[::6, 0] = 100.0
    valid[3::6, 1] = True
    v[3::6, 1] = 200.0
    grid = np.arange(0, 60, 12, dtype=float)
    out, cov = rebin(t, v, valid, grid, 12.0, expected_per_bin=12 / 3)

    check("coverage counts any-channel-valid rows",
          np.allclose(cov, 1.0), f"got {np.round(cov, 3)}")
    check("each channel still averages only its own valid samples",
          np.allclose(out[:, 0], 100.0) and np.allclose(out[:, 1], 200.0),
          f"ch0={out[:, 0]} ch1={out[:, 1]}")


def test_interpolate_gaps_respects_max():
    x = np.arange(20, dtype=float)
    x[5:7] = np.nan       # short gap -> filled
    x[12:18] = np.nan     # long gap  -> kept
    y = interpolate_gaps(x, max_gap=3)
    check("short gaps interpolated", np.isfinite(y[5:7]).all())
    check("long gaps left missing", np.isnan(y[12:18]).all())


def test_occurrence_labels_look_forward_only():
    x = np.zeros(20, dtype=np.float32)
    x[10] = 1.0
    occ = occurrence_labels(x, [3])[:, 0]
    # Bins 7..10 are within 3 steps before the flare -> positive.
    check("occurrence label fires in the lead-up window",
          bool(occ[7] == 1 and occ[10] == 1))
    check("occurrence label is zero long before", bool(occ[5] == 0))
    check("occurrence label is zero after the event has passed",
          bool(occ[14] == 0), f"occ[14]={occ[14]}")


def test_forecast_targets_alignment():
    x = np.arange(10, dtype=np.float32)
    y, m = forecast_targets(x, [2, 5])
    check("forecast target at h=2 is x[t+2]", bool(y[0, 0] == 2.0 and y[3, 0] == 5.0))
    check("forecast target at h=5 is x[t+5]", bool(y[0, 1] == 5.0))
    check("targets running off the end are masked",
          bool((~m[-2:, 0]).all() and (~m[-5:, 1]).all()))


def test_split_is_chronological_with_embargo():
    cfg = Config()
    cfg.train.embargo_s = 100.0
    cfg.train.split = (0.6, 0.2, 0.2)
    # Two segments, interleaved in absolute time.
    w = [WindowIndex(seg=0, end=i, t_unix=float(i * 10)) for i in range(100)]
    w += [WindowIndex(seg=1, end=i, t_unix=float(5000 + i * 10)) for i in range(100)]
    tr, va, te = chronological_split(w, cfg)

    check("both segments appear in train",
          {w[i].seg for i in tr} == {0, 1})
    check("both segments appear in test",
          {w[i].seg for i in te} == {0, 1},
          f"got {{{', '.join(str(s) for s in sorted({w[i].seg for i in te}))}}}")

    for seg in (0, 1):
        tr_t = [w[i].t_unix for i in tr if w[i].seg == seg]
        va_t = [w[i].t_unix for i in va if w[i].seg == seg]
        te_t = [w[i].t_unix for i in te if w[i].seg == seg]
        if tr_t and va_t:
            check(f"seg{seg}: val starts after train + embargo",
                  min(va_t) > max(tr_t) + cfg.train.embargo_s - 1e-6)
        if va_t and te_t:
            check(f"seg{seg}: test starts after val + embargo",
                  min(te_t) > max(va_t) + cfg.train.embargo_s - 1e-6)
    check("splits are disjoint", not (set(tr) & set(va)) and not (set(va) & set(te)))


def test_metrics():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    p = np.array([1, 0, 0, 0, 1, 1, 0, 0])
    s = skill_scores(y, p)
    check("contingency table", s["TP"] == 2 and s["FN"] == 1 and s["FP"] == 1
          and s["TN"] == 4)
    # TSS = POD - POFD = 2/3 - 1/5
    check("TSS value", abs(s["TSS"] - (2 / 3 - 1 / 5)) < 1e-9, f"got {s['TSS']}")

    check("perfect AUC", abs(roc_auc(np.array([0, 0, 1, 1]),
                                     np.array([0.1, 0.2, 0.8, 0.9])) - 1.0) < 1e-9)
    check("random AUC ~ 0.5", abs(roc_auc(np.array([0, 1, 0, 1]),
                                          np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-9)
    yb = np.array([0, 0, 0, 1])
    check("BSS of climatology forecast is 0",
          abs(brier_skill_score(yb, np.full(4, yb.mean()))) < 1e-9)


def test_hel1os_fill_detection():
    """The single most important data-quality rule in this project."""
    from astropy.io import fits
    from solarflare.io.hel1os import read_hel1os_lightcurve
    import tempfile

    n = 60
    ctr = np.zeros(n)
    err = np.zeros(n)
    sampled = np.arange(0, n, 6)
    ctr[sampled] = 150.0
    err[sampled] = np.sqrt(150.0)
    mjd = 61294.5 + np.arange(n) / 86400.0

    cols = fits.ColDefs([
        fits.Column(name="MJD", format="D", array=mjd),
        fits.Column(name="ISOT", format="30A", array=np.array(["x"] * n)),
        fits.Column(name="CTR", format="D", array=ctr),
        fits.Column(name="STAT_ERR", format="D", array=err),
    ])
    hdu = fits.BinTableHDU.from_columns(cols, name="CZT1_LC_BAND_18.00KEV_TO_160.00KEV")
    hdu.header["ELOW"] = 18.0
    hdu.header["EHIGH"] = 160.0
    hdu.header["DETNAM"] = "CZT1"
    hl = fits.HDUList([fits.PrimaryHDU(), hdu])

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "lightcurve_czt1.fits"
        hl.writeto(p)
        obs = read_hel1os_lightcurve(p)

    check("fill rows identified as missing",
          int(obs.valid.sum()) == len(sampled),
          f"valid={int(obs.valid.sum())} expected={len(sampled)}")
    real = obs.rates[obs.valid[:, 0], 0]
    check("masked mean is the true rate, not the diluted one",
          abs(float(real.mean()) - 150.0) < 1e-6, f"got {float(real.mean())}")
    naive = float(np.nan_to_num(ctr).mean())
    check("...and the naive mean really is wrong", abs(naive - 25.0) < 1.0,
          f"naive {naive:.1f}")


def test_solexs_energy_scale():
    """The published SoLEXS calibration, checked against physics we can verify.

    The first pipeline assumed channel 41 = 1 keV and channel 339 = 22 keV.
    Sarwade et al. (2025) give 47.75 eV/channel (x2 above channel 168) and a
    ~2 keV threshold; the quiet-Sun Fe-55 lines measured on this archive
    (channels 122.05, 134.65) confirm it. These checks pin both scales.
    """
    from solarflare.io.solexs import channel_energies
    from solarflare.config import (SOLEXS_BANDS_BY_SCALE, SOLEXS_CH_LO,
                                   SOLEXS_LEGACY_GAIN_KEV, SOLEXS_LEGACY_OFFSET_KEV)

    e = channel_energies("sarwade2025", "SDD2")
    ch = np.arange(e.size)

    def at(c):
        return float(np.interp(c, ch, e))

    check("published scale: pipeline threshold (channel 41) sits at ~2 keV",
          abs(e[SOLEXS_CH_LO] - 2.0) < 0.1, f"{e[SOLEXS_CH_LO]:.3f} keV")
    check("published scale: measured Fe-55 lines land on Mn Ka / Kb within 30 eV",
          abs(at(122.05) - 5.895) < 0.03 and abs(at(134.65) - 6.490) < 0.03,
          f"{at(122.05):.3f}, {at(134.65):.3f} keV")
    check("published scale: monotonic, continuous at the gain break, top ~24 keV",
          bool(np.all(np.diff(e) > 0)) and abs(e[169] - e[168] - 0.0945) < 1e-9
          and 23.5 < e[339] < 25.0, f"top {e[339]:.2f} keV")
    check("published scale: every soft band selects at least one usable channel",
          all(((e >= lo) & (e < hi) & (ch >= SOLEXS_CH_LO)).any()
              for lo, hi in SOLEXS_BANDS_BY_SCALE["sarwade2025"]))
    leg = channel_energies("legacy_linear")
    check("legacy scale is exactly the old formula (reproducibility of old models)",
          np.array_equal(leg, SOLEXS_LEGACY_GAIN_KEV * (ch + 0.5) + SOLEXS_LEGACY_OFFSET_KEV))
    old_band = (leg >= 1.0) & (leg < 2.0) & (ch >= SOLEXS_CH_LO)
    check("legacy '1-2 keV' band really covers ~2.0-2.7 keV on the true scale",
          abs(e[old_band].min() - 2.04) < 0.05 and abs(e[old_band].max() - 2.7) < 0.1,
          f"{e[old_band].min():.2f}-{e[old_band].max():.2f} keV")
    try:
        channel_energies("made_up")
        check("unknown energy scale is an error", False)
    except ValueError:
        check("unknown energy scale is an error", True)


def test_energy_scale_cache_fingerprint():
    """Old caches must stay valid for old models; a scale change must rebuild."""
    from solarflare.config import PreprocessConfig
    from solarflare.preprocess.cache import _settings_fingerprint
    fl = _settings_fingerprint(PreprocessConfig(solexs_energy_scale="legacy_linear"))
    fn = _settings_fingerprint(PreprocessConfig())
    check("legacy fingerprint has exactly the pre-change keys and values",
          set(fl) == {"cache_version", "dt_seconds", "min_valid_fraction", "solexs_bands",
                      "goes_long", "goes_short", "gain", "offset", "ch_lo", "ch_hi",
                      "hel1os_fill_is_missing"}
          and fl["solexs_bands"][0] == [1.0, 2.0] and fl["goes_short"] == [3.1, 22.0])
    check("published-scale fingerprint differs, so SoLEXS entries rebuild", fl != fn)



def test_flux_anchor_is_the_calibrated_current_flux():
    """anchor_flux must add exactly the calibrated SoLEXS flux at the origin,
    use the fallback where SoLEXS is absent, and never look at other steps."""
    import numpy as np
    from solarflare.config import ModelConfig, PreprocessConfig, WindowConfig
    from solarflare.models.net import SolexHelNet, flux_anchor

    pre = PreprocessConfig()
    pre_names_idx = 6                      # log_goes_long
    mcfg = ModelConfig(hidden=16, dilations=(1, 2), anchor_flux=True)
    mean = np.zeros(24)
    std = np.ones(24)
    a = flux_anchor(mcfg, pre, mean, std)
    check("anchor points at log_goes_long", a.index == pre_names_idx, str(a.index))
    check("anchor is off unless asked", flux_anchor(ModelConfig(), pre, mean, std) is None)

    win = WindowConfig()
    torch.manual_seed(0)
    net = SolexHelNet(24, 4, 2, mcfg, win, anchor=a).eval()
    plain = SolexHelNet(24, 4, 2, ModelConfig(hidden=16, dilations=(1, 2)), win, anchor=None).eval()
    plain.load_state_dict(net.state_dict())

    b, t = 2, 32
    soft = torch.zeros(b, t, 24)
    rate = 5000.0
    soft[0, -1, pre_names_idx] = float(np.log1p(rate))
    soft_mask = torch.ones(b, t)
    soft_mask[1, -1] = 0.0                 # SoLEXS not observing at the origin
    hard, hard_mask, clock = torch.zeros(b, t, 4), torch.zeros(b, t), torch.zeros(b, t, 2)
    with torch.no_grad():
        out = net(soft, soft_mask, hard, hard_mask, clock)
        ref = plain(soft, soft_mask, hard, hard_mask, clock)
    want = a.intercept + a.slope * np.log10(rate)
    got = float(out["nowcast"][0] - ref["nowcast"][0])
    check("observed origin: nowcast shifts by the calibrated flux", abs(got - want) < 2e-3,
          f"{got:.4f} vs {want:.4f}")
    got_fb = float(out["nowcast"][1] - ref["nowcast"][1])
    check("unobserved origin: the training-mean fallback is used",
          abs(got_fb - a.default) < 1e-4, f"{got_fb:.4f} vs {a.default}")
    shift = (out["forecast"][0] - ref["forecast"][0]).numpy()
    check("every forecast quantile shifts by the same anchor",
          np.allclose(shift, want, atol=2e-3), str(shift[:1]))
    check("classification heads are untouched by the anchor",
          torch.allclose(out["in_flare"], ref["in_flare"], atol=1e-6))

    soft2 = soft.clone()
    soft2[:, :-1, :] += 3.0                # change every step except the origin
    with torch.no_grad():
        out2 = net(soft2, soft_mask, hard, hard_mask, clock)
        ref2 = plain(soft2, soft_mask, hard, hard_mask, clock)
    check("the anchor reads only the origin step",
          abs(float((out2["nowcast"][0] - ref2["nowcast"][0]) - got) - 0.0) < 2e-3)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} test groups\n")
    for t in tests:
        print(f"{t.__name__}:")
        try:
            t()
        except Exception as exc:  # keep going, report at the end
            print(f"  ERROR {t.__name__}: {type(exc).__name__}: {exc}")
            FAILURES.append(f"{t.__name__} (exception)")
        print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
