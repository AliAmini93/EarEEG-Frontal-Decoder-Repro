"""EEG-Inception ERP (candidate 0.2.0rc1): kernel conversion, closed-form parameter counts, forward/backward
finiteness, eval-mode batch independence, determinism, training-only standardisation, CPU smoke on synthetic data."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eareeg_decoder import inception  # noqa: E402
from eareeg_decoder.metrics import balanced_accuracy  # noqa: E402

pytestmark = pytest.mark.eegnet
FAST = {"dropout": 0.5, "lr": 0.001, "batch": 64, "epochs": 3}


def test_kernel_conversion_physical_scales():
    assert inception.scale_samples(250.0) == [125, 62, 31]
    net = inception.build_inception(11, 200)
    assert net.kernels_block1 == [125, 62, 31] and net.kernels_block2 == [31, 15, 7]
    assert inception.scale_samples(128.0) == [64, 32, 16]


@pytest.mark.parametrize("chans,times,sfreq,expected", [(11, 200, 250.0, 25542), (10, 200, 250.0, 25494),
                                                         (8, 128, 128.0, 14926)])
def test_parameter_counts_match_specification(chans, times, sfreq, expected):
    net = inception.build_inception(chans, times, sfreq=sfreq)
    assert inception.n_trainable(net) == expected
    assert inception.analytic_param_count(chans, times, sfreq=sfreq) == expected


def test_forward_backward_finite_and_eval_batch_independent():
    torch.manual_seed(0)
    net = inception.build_inception(11, 200)
    x = torch.randn(6, 11, 200)
    out = net(x)
    assert out.shape == (6, 2) and torch.isfinite(out).all()
    out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    net.eval()
    with torch.no_grad():
        full = net(x)
        single = torch.cat([net(x[i:i + 1]) for i in range(6)])
    torch.testing.assert_close(full, single, rtol=1e-5, atol=1e-6)


def _data(rng, n=240, signal=True):
    y = np.r_[np.ones(n // 2, int), np.zeros(n - n // 2, int)]
    rng.shuffle(y)
    X = rng.normal(size=(n, 10, 200)).astype(np.float32)
    if signal:
        t = np.arange(200) / 250.0
        X[:, :5] += (np.where(y[:, None, None] == 1, 1.0, -1.0) * 1.2 * np.exp(-0.5 * ((t - 0.3) / 0.05) ** 2))
    return X, y


def test_deterministic_given_seed_and_training_only_standardisation(rng):
    X, y = _data(rng, n=160)
    a = inception.inception_fit_predict(X[:120], y[:120], X[120:], seed=3, cfg=FAST)
    b = inception.inception_fit_predict(X[:120], y[:120], X[120:], seed=3, cfg=FAST)
    np.testing.assert_array_equal(a, b)
    X2 = X.copy()
    X2[150:] = X2[150:] * 50.0                          # changing some test epochs leaves the others unchanged
    c = inception.inception_fit_predict(X2[:120], y[:120], X2[120:], seed=3, cfg=FAST)
    np.testing.assert_allclose(a[:30], c[:30], rtol=1e-5, atol=1e-6)


def test_cpu_smoke_signal_and_chance():
    rng = np.random.default_rng(11)
    X, y = _data(rng, n=320)
    p = inception.inception_fit_predict(X[:240], y[:240], X[240:], seed=1, cfg={**FAST, "epochs": 6})
    assert np.isfinite(p).all() and balanced_accuracy(y[240:], (p > 0.5).astype(int)) > 0.7
    rng = np.random.default_rng(12)
    X0, y0 = _data(rng, n=320, signal=False)
    p0 = inception.inception_fit_predict(X0[:240], y0[:240], X0[240:], seed=1, cfg={**FAST, "epochs": 6})
    assert abs(balanced_accuracy(y0[240:], (p0 > 0.5).astype(int)) - 0.5) < 0.2


def test_bad_channel_nan_is_training_mean(rng):
    X, y = _data(rng, n=120)
    X[:, 3] = np.nan
    p = inception.inception_fit_predict(X[:90], y[:90], X[90:], seed=0, cfg={**FAST, "epochs": 1})
    assert np.isfinite(p).all()
