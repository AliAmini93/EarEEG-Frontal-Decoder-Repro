"""Shrinkage LDA settings and EEGNet architecture / deterministic CPU smoke tests (no claim of trained parameters)."""
import importlib.util

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from eareeg_decoder import config, decoders
from eareeg_decoder.features import apply_scaler, bin_means, fit_scaler, window_columns

needs_torch = pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                                 reason="BLOCKED_MISSING_DEPENDENCY: torch (optional 'eegnet' extra)")


def test_bin_means_layout_and_window_columns():
    times = np.round(np.arange(263) / 250.0 - 0.2, 6)
    X = np.zeros((1, 2, 263))
    X[0, 1, (times >= 0.10) & (times < 0.15)] = 3.0            # channel 1, bin 2
    F = bin_means(X, times)
    assert F.shape == (1, 32)
    assert F[0, 16 + 2] == pytest.approx(3.0) and F[0].sum() == pytest.approx(3.0)
    cols = window_columns(2, (0.2, 0.4))
    assert cols.tolist() == [4, 5, 6, 7, 20, 21, 22, 23]
    with pytest.raises(ValueError, match="empty feature bin"):
        bin_means(X[..., :60], times[:60])


def test_scaler_handles_nan_and_constant_columns():
    F = np.array([[1.0, np.nan, 5.0], [3.0, np.nan, 5.0]])
    mu, sd = fit_scaler(F)
    np.testing.assert_allclose(mu, [2.0, 0.0, 5.0])
    np.testing.assert_allclose(sd, [1.0, 1.0, 1.0])
    Z = apply_scaler(np.array([[2.0, np.nan, 7.0]]), mu, sd)
    np.testing.assert_allclose(Z, [[0.0, 0.0, 2.0]])


@pytest.mark.lda
def test_lda_matches_sklearn_with_equal_priors(rng):
    F = rng.normal(size=(400, 20)) + np.outer((rng.random(400) < 0.8), np.linspace(0, 1, 20))
    y = (F[:, -1] > 0.5).astype(int)
    mu, sd = fit_scaler(F[:300])
    ref = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]).fit(apply_scaler(F[:300], mu, sd), y[:300])
    want = ref.decision_function(apply_scaler(F[300:], mu, sd))
    np.testing.assert_allclose(decoders.lda_decision(F[:300], y[:300], F[300:]), want, rtol=1e-10)
    clf = decoders.ShrinkageLDA().fit(F[:300], y[:300])
    assert np.array_equal(clf.predict(F[300:]), (want > 0).astype(int))
    with pytest.raises(ValueError, match="both classes"):
        decoders.ShrinkageLDA().fit(F[:10], np.ones(10, int))


@pytest.mark.eegnet
def test_eegnet_settings_from_protocol():
    assert decoders.eegnet_config("montage")["dropout"] == 0.25
    assert decoders.eegnet_config("transient")["dropout"] == 0.5
    for stage in ("montage", "transient"):
        c = decoders.eegnet_config(stage)
        assert (c["F1"], c["D"], c["F2"], c["kernel"], c["epochs"], c["lr"], c["batch"]) == (8, 2, 16, 64, 30, 1e-3, 128)
    assert config.eegnet_seed("transient", "loso", 100, 3) == 20261111 + 100 + 3
    assert config.eegnet_seed("montage", "joint", 200, 5) == 20261107 + 200 + 1000 + 5
    assert config.eegnet_seed("montage", "loso", 0, 5) == 20261107 + 5
    with pytest.raises(ValueError):
        config.eegnet_seed("transient", "loso", 7, 0)


@pytest.mark.eegnet
@needs_torch
def test_eegnet_architecture_shapes_and_parameter_count():
    net = decoders.build_eegnet(11, 200, dropout=0.5)
    import torch
    out = net.eval()(torch.zeros(3, 1, 11, 200))
    assert tuple(out.shape) == (3, 2)
    # conv1 8*64 + bn 16 + depthwise 16*11 + bn 32 + sep 16*16 + pointwise 16*16 + bn 32 + head (16*6)*2+2
    expected = 8 * 64 + 16 + 16 * 11 + 32 + 16 * 16 + 16 * 16 + 32 + (16 * 6) * 2 + 2
    assert decoders.n_parameters(net) == expected


@pytest.mark.eegnet
@needs_torch
def test_eegnet_cpu_determinism_and_training_only_standardisation(rng):
    X = rng.normal(size=(64, 11, 200)).astype(np.float32)
    y = np.tile([1, 1, 1, 0], 16)
    X[y == 0, 0, 80:120] += 2.0
    cfg = dict(decoders.eegnet_config("transient"), epochs=2)
    p1 = decoders.eegnet_fit_predict(X[:48], y[:48], X[48:], seed=20261111, cfg=cfg)
    p2 = decoders.eegnet_fit_predict(X[:48], y[:48], X[48:], seed=20261111, cfg=cfg)
    p3 = decoders.eegnet_fit_predict(X[:48], y[:48], X[48:], seed=20261112, cfg=cfg)
    assert np.array_equal(p1, p2) and not np.allclose(p1, p3)
    assert p1.shape == (16,) and np.all((p1 >= 0) & (p1 <= 1))
    Ztr, _ = decoders.standardise_channels(X[:48], X[48:])
    Ztr2, _ = decoders.standardise_channels(X[:48], X[48:] * 1000)
    assert np.array_equal(Ztr, Ztr2)


@pytest.mark.eegnet
@needs_torch
def test_eegnet_rejects_single_class():
    with pytest.raises(ValueError, match="both classes"):
        decoders.eegnet_fit_predict(np.zeros((4, 11, 200)), np.ones(4, int), np.zeros((2, 11, 200)), seed=0)
