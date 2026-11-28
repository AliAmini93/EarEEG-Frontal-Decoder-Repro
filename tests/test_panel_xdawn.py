"""xDAWN-TS-LR (candidate 0.2.0rc1): oracles for OAS, the xDAWN eigenproblem, the Riemannian mean and the tangent map;
common-average rank deficiency, duplicate and bad channels; source-only fitting; signal recovery and chance.

All inputs are explicitly synthetic."""
import numpy as np
import pytest
from scipy import linalg
from sklearn.covariance import oas

from eareeg_decoder import xdawn
from eareeg_decoder.metrics import balanced_accuracy


def _erp_data(rng, n=240, c=10, t=200, amp=1.5, car=True, signal=True):
    """Synthetic epochs: noise + a class-dependent bump on half of the channels; optional common-average reference."""
    y = np.r_[np.ones(n // 2, int), np.zeros(n - n // 2, int)]
    rng.shuffle(y)
    X = rng.normal(size=(n, c, t))
    if signal:
        tt = np.arange(t) / 250.0
        bump = amp * np.exp(-0.5 * ((tt - 0.3) / 0.05) ** 2)
        X[:, : c // 2] += np.where(y[:, None, None] == 1, 1.0, -1.0) * bump
    if car:
        X = X - X.mean(axis=1, keepdims=True)
    return X, y


def test_oas_matches_scikit_learn(rng):
    A = rng.normal(size=(5, 6, 120))
    batch = xdawn.oas_batch(A)
    for i in range(5):
        ref, _ = oas(A[i].T, assume_centered=False)
        np.testing.assert_allclose(batch[i], ref, rtol=1e-10, atol=1e-12)


def test_xdawn_generalised_eigenvectors_ordering_and_norm(rng):
    X, y = _erp_data(rng, car=False)
    Z = xdawn.channel_scaler(X)(X)
    xd = xdawn.Xdawn(nfilter=3).fit(Z, y)
    assert xd.filters_.shape == (6, 10)
    np.testing.assert_allclose(np.linalg.norm(xd.filters_, axis=1), 1.0, rtol=1e-12)
    for k, cls in enumerate((0, 1)):
        Ck = xdawn.oas_single(Z[y == cls].mean(axis=0))
        w, _ = linalg.eigh(Ck, xd.Cx_)
        top = np.sort(w)[::-1][:3]
        np.testing.assert_allclose(xd.evals_[3 * k:3 * k + 3], top, rtol=1e-8)
        for j in range(3):
            v = xd.filters_[3 * k + j]
            lam = xd.evals_[3 * k + j]
            np.testing.assert_allclose(Ck @ v, lam * (xd.Cx_ @ v), rtol=1e-6, atol=1e-10)


def test_riemannian_mean_fixed_point_and_tangent_norm(rng):
    B = rng.normal(size=(20, 4, 50))
    covs = xdawn.oas_batch(B)
    M, n_iter, conv = xdawn.mean_riemann(covs)
    assert conv and n_iter <= 50
    T = xdawn.tangent_space(covs, M)
    np.testing.assert_allclose(T.mean(axis=0), 0.0, atol=1e-7)          # mean of tangent vectors at the mean is 0
    Cm12 = xdawn.invsqrtm(M)
    L = xdawn.logm(Cm12 @ covs[0] @ Cm12)
    np.testing.assert_allclose(np.linalg.norm(T[0]), np.linalg.norm(L, "fro"), rtol=1e-10)
    np.testing.assert_allclose(xdawn.tangent_space(M[None], M), 0.0, atol=1e-10)


@pytest.mark.parametrize("variant", ["car", "duplicate", "bad_nan"])
def test_rank_deficient_duplicate_and_bad_channels_stay_finite(rng, variant):
    X, y = _erp_data(rng, n=160, car=(variant == "car"))
    if variant == "duplicate":
        X[:, 1] = X[:, 0]
    if variant == "bad_nan":
        X[:, 2] = np.nan
    prob, diag = xdawn.fit_predict_diag(X[:120], y[:120], X[120:])
    assert diag["gate_ok"], diag
    assert np.isfinite(prob).all() and prob.shape == (40,)
    if variant == "car":
        # documented property of the fixed configuration: a selected filter aligns with the common-mode null direction
        assert diag["filter_null_alignment_max_abs_cos"] > 0.99


def test_source_only_fitting(rng):
    X, y = _erp_data(rng, n=200)
    Xtr, ytr, Xte = X[:150], y[:150], X[150:]
    p1 = xdawn.xdawn_fit_predict(Xtr, ytr, Xte)
    Xte2 = Xte.copy()
    Xte2[:25] = rng.normal(size=Xte2[:25].shape) * 100.0
    p2 = xdawn.xdawn_fit_predict(Xtr, ytr, Xte2)
    np.testing.assert_allclose(p1[25:], p2[25:], rtol=1e-10)             # other test epochs unaffected
    np.testing.assert_allclose(xdawn.xdawn_fit_predict(Xtr, ytr, Xte[25:]), p1[25:], rtol=1e-10)  # inductive


def test_signal_recovered_and_chance_without_signal():
    rng = np.random.default_rng(5)
    X, y = _erp_data(rng, n=300, amp=0.8)
    p = xdawn.xdawn_fit_predict(X[:200], y[:200], X[200:])
    assert balanced_accuracy(y[200:], (p > 0.5).astype(int)) > 0.75
    rng = np.random.default_rng(6)
    X0, y0 = _erp_data(rng, n=300, signal=False)
    p0 = xdawn.xdawn_fit_predict(X0[:200], y0[:200], X0[200:])
    assert abs(balanced_accuracy(y0[200:], (p0 > 0.5).astype(int)) - 0.5) < 0.15


def test_deterministic_and_seed_independent(rng):
    X, y = _erp_data(rng, n=120)
    a = xdawn.xdawn_fit_predict(X[:90], y[:90], X[90:], seed=1)
    b = xdawn.xdawn_fit_predict(X[:90], y[:90], X[90:], seed=999)
    np.testing.assert_array_equal(a, b)


def test_requires_both_classes(rng):
    X, _ = _erp_data(rng, n=40)
    with pytest.raises(ValueError):
        xdawn.xdawn_fit_predict(X, np.ones(40, int), X[:2])
