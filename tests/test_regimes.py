"""Train -> test regimes: matched draws, unchanged application ('transferred'), shared test sets, Eq. (4)."""
import numpy as np
import pytest

from eareeg_decoder import pipeline, regimes
from eareeg_decoder.synthetic import synthetic_epochs

pytestmark = [pytest.mark.lda, pytest.mark.loso]


@pytest.fixture(scope="module")
def stack_blinks():
    es, _ = synthetic_epochs(n_participants=6, n_streams=10, seed=11)
    return pipeline.build_stack(es)


def test_matched_mask_counts_and_signal_independence(stack_blinks):
    S = stack_blinks
    m1 = regimes.matched_random_mask(S.free, S.y, S.g, np.random.default_rng(20261111))
    for p in np.unique(S.g):
        for c in (0, 1):
            sel = (S.g == p) & (S.y == c)
            assert m1[sel].sum() == S.free[sel].sum()
    # the draw depends only on counts and the RNG: identical for any feature values / montage
    m2 = regimes.matched_random_mask(S.free, S.y, S.g, np.random.default_rng(20261111))
    assert np.array_equal(m1, m2)
    m3 = regimes.matched_random_mask(S.free, S.y, S.g, np.random.default_rng(20261112))
    assert not np.array_equal(m1, m3)
    # matched draws keep transient-bearing epochs (selection from ALL retained epochs)
    assert (m1 & ~S.free).any()


def test_transferred_decoder_is_applied_unchanged(stack_blinks):
    """all->free decision values equal all->all decision values on the free epochs (same training sets, LOSO)."""
    S = stack_blinks
    allm = np.ones(len(S.y), bool)
    aa = regimes.cv_scores(S.F["full"], S.y, S.g, allm, allm)
    af = regimes.cv_scores(S.F["full"], S.y, S.g, allm, S.free)
    np.testing.assert_allclose(af[S.free], aa[S.free], rtol=1e-12, atol=1e-12)
    assert np.isnan(af[~S.free]).all()


def test_regime_set_shares_test_epochs_and_decomposes_exactly(stack_blinks):
    S = stack_blinks
    res = regimes.lda_regimes(S.F, S.y, S.g, S.free, n_draws=3, min_face=10, min_tree=3)
    assert res.test_sets_identical
    assert set(res.advantage) == set(regimes.REGIMES)
    dec = res.decomposition
    assert dec["max_abs_identity_error"] < 1e-12
    np.testing.assert_allclose(dec["delta"], res.advantage["free->free"] - res.advantage["all->free"])
    assert res.matched_free_by_draw.shape == (len(res.participants), 3)
    # mutant: composition taken against matched->matched (different test epochs) breaks the identity
    c_bad = res.advantage["free->free"] - res.advantage["matched->matched"]
    assert not np.allclose(dec["delta"], c_bad + dec["s"])


def test_eligibility_rule_sets_nan():
    y = np.array([1] * 50 + [0] * 10 + [1] * 50 + [0] * 20)
    g = np.array([0] * 60 + [1] * 70)
    scores = np.where(y == 1, 1.0, -1.0)
    ba = regimes.ba_by_participant(y, g, scores, min_face=40, min_tree=15)
    assert np.isnan(ba[0]) and ba[1] == pytest.approx(1.0)


def test_crossed_scaler_and_intercept_recentring_use_training_rows_only(stack_blinks):
    S = stack_blinks
    allm = np.ones(len(S.y), bool)
    base = regimes.cv_scores(S.F["full"], S.y, S.g, allm, S.free)
    c3 = regimes.cv_scores(S.F["full"], S.y, S.g, allm, S.free, recal_mask=S.free)
    shift = base - c3
    for p in np.unique(S.g):
        te = (S.g == p) & S.free
        assert np.ptp(shift[te]) < 1e-9                       # one constant shift per fold
    # the shift recentres the training participants' free epochs (check one fold by hand)
    from eareeg_decoder.decoders import ShrinkageLDA
    p = 0
    tr = S.g != p
    clf = ShrinkageLDA().fit(S.F["full"][tr], S.y[tr])
    d = clf.decision_function(S.F["full"][tr & S.free])
    yy = S.y[tr & S.free]
    expected = 0.5 * (d[yy == 1].mean() + d[yy == 0].mean())
    assert shift[(S.g == p) & S.free][0] == pytest.approx(expected)
    # scikit-learn's shrinkage LDA standardises within class before Ledoit-Wolf, so its decisions are invariant to
    # the external z-scaler: the crossed-scaler diagnostic is uninformative by construction unless features are
    # missing (NaN -> training mean, where the scaler's mean matters).
    c1 = regimes.cv_scores(S.F["full"], S.y, S.g, S.free, S.free, scaler_mask=allm)
    ff = regimes.cv_scores(S.F["full"], S.y, S.g, S.free, S.free)
    np.testing.assert_allclose(c1[S.free], ff[S.free], rtol=1e-7, atol=1e-9)
    F = S.F["full"].copy()
    F[S.g == 1, :16] = np.nan                                  # participant 1: Fpz features missing
    c1n = regimes.cv_scores(F, S.y, S.g, S.free, S.free, scaler_mask=allm)
    ffn = regimes.cv_scores(F, S.y, S.g, S.free, S.free)
    assert not np.allclose(c1n[S.free & (S.g == 1)], ffn[S.free & (S.g == 1)])


def test_frontal_advantage_requires_same_participants():
    with pytest.raises(ValueError):
        regimes.frontal_advantage({0: 0.6, 1: 0.6}, {0: 0.5})
    np.testing.assert_allclose(regimes.frontal_advantage({1: 0.7, 0: 0.6}, {0: 0.5, 1: 0.5}), [0.1, 0.2])


@pytest.mark.joint
def test_regimes_under_joint_holdout_and_joint_recentring_base(stack_blinks):
    S = stack_blinks
    res = regimes.lda_regimes(S.F, S.y, S.g, S.free, S.sg, n_draws=2, min_face=5, min_tree=2, keys=S.keys)
    assert res.test_sets_identical and res.decomposition["max_abs_identity_error"] < 1e-12
    allm = np.ones(len(S.y), bool)
    base = regimes.cv_scores(S.F["full"], S.y, S.g, allm, S.free, S.sg)
    c3 = regimes.cv_scores(S.F["full"], S.y, S.g, allm, S.free, S.sg, recal_mask=S.free)
    from eareeg_decoder.decoders import ShrinkageLDA
    p, s = 0, 1
    te = (S.g == p) & (S.sg == s) & S.free
    trbase = (S.g != p) & (S.sg != s) & (S.sg >= 0)                # held-out participant AND group excluded
    clf = ShrinkageLDA().fit(S.F["full"][trbase], S.y[trbase])
    d = clf.decision_function(S.F["full"][trbase & S.free])
    yy = S.y[trbase & S.free]
    assert (base - c3)[te][0] == pytest.approx(0.5 * (d[yy == 1].mean() + d[yy == 0].mean()))
