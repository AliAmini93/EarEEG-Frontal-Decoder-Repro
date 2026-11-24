"""Filtering, epoch indexing, baseline, stored common average, 150-uV rejection and channel QC (synthetic)."""
import importlib.util

import numpy as np
import pytest

from eareeg_decoder import montage as mt
from eareeg_decoder import preprocessing as pp
from eareeg_decoder.config import PROTOCOL

NAMES = list(mt.EAR_CHANNELS)


def test_onset_rounding_is_half_up():
    idx, fin = pp.onset_to_sample([0.001, 0.002, 0.003, np.nan], 500.0)
    # 0.5 -> 1, 1.0 -> 1, 1.5 -> 2 (floor(x + 0.5)); NaN -> -1
    assert idx.tolist() == [1, 1, 2, -1]
    assert fin.tolist() == [True, True, True, False]


def test_map_onsets_reasons():
    idx, why = pp.map_onsets([0.1, 5.0, 9.5, 5.0, np.nan], 100.0, 1000)
    assert why.tolist() == ["epoch_before_start", "duplicate_sample", "epoch_after_end", "duplicate_sample", "nonfinite"]


@pytest.mark.parametrize("sfreq", [250.0, 500.0, 1000.0])
def test_epoch_grid_and_time_zero(sfreq):
    n = int(20 * sfreq)
    data = np.zeros((2, n))
    onset_samples = np.array([int(5 * sfreq), int(7.3 * sfreq)])
    data[0, onset_samples] = 1.0                       # impulse exactly at onset
    data[1, onset_samples + int(round(0.848 * sfreq))] = 2.0
    X, times = pp.epoch(data, sfreq, onset_samples)
    assert times.size == PROTOCOL["preprocessing"]["n_epoch_samples"] == 263
    assert times[0] == pytest.approx(-0.2) and times[-1] == pytest.approx(0.848)
    zero = int(np.argmin(np.abs(times)))
    assert times[zero] == 0.0
    assert np.all(X[:, 0, zero] == 1.0) and X[:, 0].sum() == pytest.approx(2.0)
    assert np.all(X[:, 1, -1] == 2.0)


def test_epoch_out_of_range_raises():
    with pytest.raises(ValueError):
        pp.epoch(np.zeros((1, 100)), 250.0, np.array([10]))
    with pytest.raises(ValueError):
        pp.decimation_factor(300.0, 250.0)


def test_baseline_excludes_time_zero():
    times = np.round(np.arange(263) / 250.0 - 0.2, 6)
    X = np.zeros((1, 1, 263))
    X[..., times >= 0] = 5.0                           # step at 0: baseline must not include the 0-s sample
    Z = pp.baseline_correct(X, times)
    assert np.allclose(Z[..., times < 0], 0.0) and np.allclose(Z[..., times >= 0], 5.0)
    X[..., 0] = 50.0                                   # first baseline sample is inside [-0.2, 0)
    assert pp.baseline_correct(X, times)[..., -1] == pytest.approx(5.0 - 50.0 / 50)


def test_common_average_includes_fpz_and_bad_rows_nan(rng):
    x = rng.normal(size=(3, 11, 20))
    good = [c for c in NAMES if c != "ER3"]
    y = pp.common_average(x, NAMES, good)
    gi = [NAMES.index(c) for c in good]
    np.testing.assert_allclose(y[:, gi], x[:, gi] - x[:, gi].mean(axis=1, keepdims=True), atol=1e-12)
    assert np.isnan(y[:, NAMES.index("ER3")]).all()


def test_150uV_rule_tests_ear_and_mastoid_only_inclusive_threshold():
    T = 50
    X = np.zeros((4, 11, T))
    X[0, NAMES.index("Fpz"), :25] = 400.0              # huge Fpz deflection: NOT tested -> kept
    X[1, NAMES.index("EL1"), :25] = 150.0              # exactly 150: kept (<=)
    X[2, NAMES.index("M2"), :25] = 150.0001            # just above: rejected
    X[3, NAMES.index("ER4"), 3] = np.nan               # non-finite on a tested row: rejected
    keep, ptp = pp.reject_epochs(X, NAMES, NAMES)
    assert keep.tolist() == [True, True, False, False]
    # a bad channel is not tested
    keep2, _ = pp.reject_epochs(X[2:3], NAMES, [c for c in NAMES if c != "M2"])
    assert keep2.tolist() == [True]


def test_channel_rule_and_frontal_flag_are_distinct():
    """An epoch with a 120-uV Fpz-mastoid transient passes the 150-uV channel rule but is flagged at 100 uV."""
    from eareeg_decoder import transients as tr
    times = np.round(np.arange(263) / 250.0 - 0.2, 6)
    x = np.zeros((1, 11, 263))
    x[0, 0] = 120.0 * np.exp(-0.5 * ((times - 0.4) / 0.05) ** 2)
    y = pp.common_average(x, NAMES, NAMES)
    keep, _ = pp.reject_epochs(y, NAMES, NAMES)
    fl = tr.transient_flags(mt.fpz_mastoid(y, NAMES, NAMES), times)
    assert keep[0] and fl.any[0] and fl.post[0]


def test_channel_qc_flags():
    rng = np.random.default_rng(1)
    d = rng.normal(0, 10, size=(11, 5000))
    d[2] *= 0.01                                      # flat
    d[5] *= 30                                        # high SD
    d[7, 10] = np.nan                                 # non-finite
    reasons, sd = pp.channel_qc(d, NAMES)
    bad = pp.bad_channels(reasons)
    assert {"M2", "EL4", "ER1"} <= set(bad)
    assert "flat" in reasons["M2"] and "high_sd" in reasons["EL4"] and "nonfinite" in reasons["ER1"]
    assert "Fpz" not in bad


def test_median_reference_uses_finite_rows_only():
    d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 9.0], [np.nan, 0.0]])
    out = pp.median_reference(d)
    np.testing.assert_allclose(out[:3], d[:3] - np.array([[3.0, 4.0]]))


def test_clip_fraction():
    d = np.array([[0.0, 1.0, 2.0, 3.0], [5.0, 5.0, 5.0, 5.0]])
    np.testing.assert_allclose(pp.clip_fraction(d), [0.5, 1.0])


def test_no_unit_scaling_is_applied(rng):
    """No calibration or unit factor: a pure re-reference/baseline of zero-mean-baseline data is identity."""
    times = np.round(np.arange(263) / 250.0 - 0.2, 6)
    x = rng.normal(size=(2, 11, 263))
    y = pp.common_average(x, NAMES, NAMES)
    z = pp.baseline_correct(y, times)
    expected = y - y[..., times < 0].mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(z, expected, atol=1e-12)
    assert "not independently calibrated" in PROTOCOL["units"]["amplitude"]


needs_mne = pytest.mark.skipif(importlib.util.find_spec("mne") is None,
                               reason="BLOCKED_MISSING_DEPENDENCY: mne (optional 'filter' extra)")


@needs_mne
def test_fir_is_zero_phase_and_band_limited():
    sfreq = 500.0
    t = np.arange(int(60 * sfreq)) / sfreq
    sig = np.vstack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 60 * t), np.exp(-0.5 * ((t - 30) / 0.05) ** 2)])
    out = pp.bandpass_fir(sig, sfreq)
    mid = slice(int(10 * sfreq), int(50 * sfreq))
    assert np.std(out[0, mid]) / np.std(sig[0, mid]) == pytest.approx(1.0, abs=0.02)   # passband
    assert np.std(out[1, mid]) / np.std(sig[1, mid]) < 0.01                             # 60 Hz stopband
    assert abs(t[np.argmax(out[2])] - 30.0) < 1.0 / sfreq + 1e-9                        # no lag (zero phase)


@needs_mne
def test_preprocess_recording_end_to_end():
    from eareeg_decoder.synthetic import synthetic_continuous
    rec_in = synthetic_continuous(n_events=30, sfreq=500.0, seed=3)
    rec = pp.preprocess_recording(rec_in["data"], rec_in["names"], rec_in["sfreq"], rec_in["onsets_s"])
    assert rec.status == "ok" and rec.X.shape == (30, 11, 263)
    assert rec.keep.all()
    # every channel carried the same bump: after the common average it cancels
    assert np.nanmax(np.abs(rec.X.mean(axis=0))) < 3.0
