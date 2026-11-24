"""Rules that match the original analyses and must not drift (added after independent review).

Network cross-validation is checked with a stub ``fit`` that records seeds and indices, so these tests need no
PyTorch. All inputs are synthetic.
"""
import numpy as np
import pytest

from eareeg_decoder import config, montage as mt, pipeline, preprocessing as pp, regimes, splits
from eareeg_decoder.data import EpochSet, validate
from eareeg_decoder.errors import LeakageError, SchemaError
from eareeg_decoder.synthetic import synthetic_epochs
from eareeg_decoder.transients import resolve_theta, transient_flags

NAMES = list(mt.EAR_CHANNELS)


# ------------------------------------------------------------------ full montage needs the stored common average
def test_full_montage_requires_stored_common_average(rng):
    x = rng.normal(size=(4, 11, 30)) + 30 * rng.normal(size=(4, 1, 30))
    good = [c for c in NAMES if c != "Fpz"]
    stored = pp.common_average(x, NAMES, good)
    full = mt.full_montage(stored, NAMES, good)
    # without Fpz, the full montage equals the Fpz-free montage (the stored average excluded Fpz)
    np.testing.assert_allclose(full[:, 1:], mt.fpz_free(stored, NAMES, good), atol=1e-9)
    m1_ref = x - x[:, [NAMES.index("M1")]]                      # a different common reference
    with pytest.raises(ValueError, match="stored common average"):
        mt.full_montage(m1_ref, NAMES, NAMES)
    assert mt.is_common_average(pp.common_average(m1_ref, NAMES, NAMES), NAMES, NAMES)


def test_data_contract_rejects_non_common_average(rng):
    es, _ = synthetic_epochs(n_participants=2, n_streams=2, seed=1)
    validate(es)
    X = es.X.copy()
    X[es.participant == 1] -= X[es.participant == 1][:, [NAMES.index("M1")]]
    bad = EpochSet(X=X, channel_names=es.channel_names, times=es.times, y=es.y, participant=es.participant,
                   stimulus_key=es.stimulus_key, keep=es.keep)
    with pytest.raises(SchemaError, match="stored common average"):
        validate(bad)


# ------------------------------------------------------------------ stimulus-key leakage
def test_masked_folds_check_stimulus_keys_under_joint_holdout():
    g = np.array([0, 0, 1, 1])
    sg = np.array([0, 1, 1, 0])
    keys = ["a", "b", "a", "c"]                                   # key "a" in two groups: inconsistent grouping
    allm = np.ones(4, bool)
    with pytest.raises(LeakageError, match="stimulus-key"):
        list(splits.masked_folds(g, allm, allm, sg, keys))
    assert len(list(splits.masked_folds(g, allm, allm, None, keys))) == 2   # LOSO may share stimuli


def test_pipeline_groups_are_a_function_of_the_stimulus_key():
    es, _ = synthetic_epochs(n_participants=3, n_streams=4, seed=2)
    S = pipeline.build_stack(es)
    for k in np.unique(S.keys):
        assert len(set(S.sg[S.keys == k].tolist())) == 1
    assert sorted(np.bincount(S.sg[S.y == 0]).nonzero()[0].tolist()) == [0, 1, 2]


# ------------------------------------------------------------------ network CV with a stub fit
class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, Xtr, ytr, Xte, seed, cfg, device):
        self.calls.append({"seed": seed, "n_train": len(ytr), "train_sum": float(Xtr.sum()), "n_test": len(Xte)})
        return np.full(len(Xte), 0.75)


def _toy(n_part=4, n=12):
    g = np.repeat(np.arange(n_part), n)
    y = np.tile([1, 1, 1, 0], n_part * n // 4)
    X = g[:, None, None] * 1000.0 + np.arange(len(g))[:, None, None] + np.zeros((1, 2, 3))
    return X, y, g


@pytest.mark.eegnet
@pytest.mark.loso
def test_net_probs_loso_seeds_skips_and_no_leakage():
    X, y, g = _toy()
    test_mask = np.ones(len(y), bool)
    test_mask[(g == 1) & (y == 0)] = False                       # participant 1 has no tree test epochs
    rec = Recorder()
    prob = regimes.net_probs(X, y, g, np.ones(len(y), bool), test_mask, 20261111, {}, min_face=1, min_tree=1, fit=rec)
    assert [c["seed"] for c in rec.calls] == [20261111 + 0, 20261111 + 2, 20261111 + 3]   # index 1 consumed
    assert [c["seed"] for c in rec.calls] == [config.eegnet_seed("transient", "loso", 0, f) for f in (0, 2, 3)]
    for c, p in zip(rec.calls, (0, 2, 3)):
        assert c["n_train"] == 36 and c["train_sum"] == pytest.approx(float(X[g != p].sum()))
    assert np.isnan(prob[g == 1]).all() and np.all(prob[g != 1] == 0.75)


@pytest.mark.eegnet
@pytest.mark.joint
def test_net_probs_joint_enumerates_all_pairs():
    X, y, g = _toy(n_part=3)
    sg = np.tile([0, 1, 2], len(y) // 3)
    test_mask = np.ones(len(y), bool)
    test_mask[(g == 0) & (sg == 1)] = False                      # an empty (participant, group) cell
    rec = Recorder()
    regimes.net_probs(X, y, g, np.ones(len(y), bool), test_mask, 20261111, {}, sg=sg, fit=rec)
    assert [c["seed"] - 20261111 for c in rec.calls] == [0, 2, 3, 4, 5, 6, 7, 8]
    first = rec.calls[0]
    assert first["train_sum"] == pytest.approx(float(X[(g != 0) & (sg != 0)].sum()))


# ------------------------------------------------------------------ regime rules
@pytest.fixture(scope="module")
def stack():
    es, _ = synthetic_epochs(n_participants=5, n_streams=8, seed=21)
    return es, pipeline.build_stack(es)


def test_eligibility_boundary_40_face_15_tree():
    y = np.array([1] * 40 + [0] * 15 + [1] * 39 + [0] * 15 + [1] * 40 + [0] * 14)
    g = np.repeat([0, 1, 2], [55, 54, 54])
    ba = regimes.ba_by_participant(y, g, np.where(y == 1, 1.0, -1.0), 40, 15)
    assert ba[0] == 1.0 and np.isnan(ba[1]) and np.isnan(ba[2])


def test_ba_uses_only_scored_epochs():
    y = np.array([1, 1, 0, 0])
    g = np.zeros(4, int)
    scores = np.array([1.0, np.nan, -1.0, np.nan])               # untested epochs are NaN
    assert regimes.ba_by_participant(y, g, scores)[0] == 1.0


def test_matched_draw_k_uses_seed_base_plus_k_and_draw_average(stack):
    _, S = stack
    F = {"full": S.F["full"], "fpz_free": S.F["fpz_free"]}
    res = regimes.lda_regimes(F, S.y, S.g, S.free, n_draws=2, min_face=10, min_tree=3)
    for k in range(2):
        mask = regimes.matched_random_mask(S.free, S.y, S.g, np.random.default_rng(20261111 + k))
        b = {m: regimes.ba_by_participant(S.y, S.g, regimes.cv_scores(F[m], S.y, S.g, mask, S.free), 10, 3) for m in F}
        np.testing.assert_allclose(res.matched_free_by_draw[:, k], regimes.frontal_advantage(b["full"], b["fpz_free"]))
    mm = []
    for k in range(2):
        mask = regimes.matched_random_mask(S.free, S.y, S.g, np.random.default_rng(20261111 + k))
        mm.append(regimes.vector(regimes.ba_by_participant(S.y, S.g, regimes.cv_scores(F["full"], S.y, S.g, mask, mask), 10, 3)))
    np.testing.assert_allclose(res.ba["matched->matched"]["full"], np.mean(mm, axis=0))
    np.testing.assert_allclose(res.advantage["matched->free"], res.matched_free_by_draw.mean(axis=1))
    assert not np.allclose(res.matched_free_by_draw[:, 0], res.matched_free_by_draw[:, 1])   # averaging matters


def test_relative_threshold_and_rates_use_all_presentations():
    es, _ = synthetic_epochs(n_participants=2, n_streams=5, seed=3, blink_rate=(0.0, 0.0))
    X = es.X.copy()
    p0 = np.flatnonzero(es.participant == 0)
    amp = np.linspace(20.0, 900.0, p0.size)                       # graded frontal deflections (continuous ranges)
    w = amp[:, None] * np.exp(-0.5 * ((es.times - 0.4) / 0.05) ** 2)[None, :]
    X[p0, 0] += w * 10 / 11                                       # add to the Fpz potential, re-average (stays CA)
    X[p0, 1:] -= (w / 11)[:, None, :]
    fm = mt.fpz_mastoid(X[p0], es.channel_names, es.good_channels(0))
    keep = es.keep.copy()
    keep[p0[amp < 120]] = False                                   # pretend the low-range epochs were rejected
    es2 = EpochSet(X=X, channel_names=es.channel_names, times=es.times, y=es.y, participant=es.participant,
                   stimulus_key=es.stimulus_key, keep=keep)
    keep0 = keep[p0]
    theta_all = resolve_theta(fm, "relative")
    theta_kept = resolve_theta(fm[keep0], "relative")
    want = ~transient_flags(fm, es.times, theta_all).any[keep0]
    wrong = ~transient_flags(fm, es.times, theta_kept).any[keep0]
    assert theta_kept > theta_all and not np.array_equal(want, wrong)   # the test can tell the rules apart
    S2 = pipeline.build_stack(es2, "relative")
    np.testing.assert_array_equal(S2.free[S2.g == 0], want)
    r = pipeline.transient_rates(es2)
    tree = es.y[p0] == 0
    post = transient_flags(fm, es.times).post
    assert r["difference"][0] == pytest.approx(post[tree].mean() - post[~tree].mean())


def test_participant_without_fpz_never_counts_as_free():
    es, _ = synthetic_epochs(n_participants=3, n_streams=3, seed=5, bad={1: ["Fpz"]})
    S = pipeline.build_stack(es, require_frontal=False)
    assert not S.free[S.g == 1].any() and np.isnan(S.descriptor[S.g == 1]).all()
