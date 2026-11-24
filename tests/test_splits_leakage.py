"""Joint vs LOSO splits, stimulus-group assignment, training-only scaling and leakage canaries.

Independent oracles:
* participant canary -- a feature that encodes the label ONLY within the held-out participant's own data in a
  participant-specific way; a correct cross-participant pipeline cannot exploit it (BA ~ 0.5), a pipeline that
  leaks test-participant rows into training can (BA ~ 1);
* stimulus canary -- a feature that is a random code per stimulus key; under LOSO the same tree photographs are in
  training, so it is exploitable (stimulus-specific information), under joint hold-out it is not.
"""
import numpy as np
import pytest

from eareeg_decoder import regimes, splits
from eareeg_decoder.decoders import ShrinkageLDA, lda_decision
from eareeg_decoder.errors import LeakageError
from eareeg_decoder.features import apply_scaler, fit_scaler
from eareeg_decoder.metrics import balanced_accuracy

pytestmark = [pytest.mark.lda, pytest.mark.joint, pytest.mark.loso, pytest.mark.oracle]


def design(n_part=8, n_streams=10, seed=0):
    """Synthetic design arrays: 16 faces + 4 trees per stream, 35 identities, 6 tree photographs."""
    rng = np.random.default_rng(seed)
    y, g, keys = [], [], []
    for p in range(n_part):
        tc = 0
        for _ in range(n_streams):
            lab = np.array([1] * 16 + [0] * 4)
            rng.shuffle(lab)
            for v in lab:
                y.append(int(v))
                g.append(p)
                if v:
                    keys.append(f"SYNTH_FACE_{rng.integers(1, 36):02d}")
                else:
                    keys.append(f"SYNTH_TREE_{tc % 6 + 1}")
                    tc += 1
    return np.array(y), np.array(g), np.array(keys)


def test_stimulus_groups_are_class_stratified_12_12_11_and_2_2_2():
    keys = [f"SYNTH_FACE_{i:02d}" for i in range(1, 36)] + [f"SYNTH_TREE_{i}" for i in range(1, 7)]
    is_face = [True] * 35 + [False] * 6
    sg = splits.assign_stimulus_groups(keys, is_face)
    assert sorted(np.bincount(sg[:35]).tolist()) == [11, 12, 12]
    assert np.bincount(sg[35:]).tolist() == [2, 2, 2]
    again = splits.assign_stimulus_groups(list(reversed(keys)), list(reversed(is_face)))
    assert np.array_equal(again[::-1], sg)                         # deterministic, order-independent
    other = splits.assign_stimulus_groups(keys, is_face, salt="another-salt")
    assert not np.array_equal(other, sg)
    assert splits.assign_stimulus_groups(["", None], [True, False]).tolist() == [-1, -1]


def test_joint_folds_share_no_person_group_or_key_and_count():
    y, g, keys = design()
    sg = splits.assign_stimulus_groups(list(keys), y == 1)
    folds = splits.joint_folds(g, sg, keys, y)
    assert len(folds) == 8 * 3
    tested = np.zeros(len(y), int)
    for f in folds:
        assert not set(g[f.train]) & set(g[f.test])
        assert not set(sg[f.train]) & set(sg[f.test])
        assert not set(keys[f.train]) & set(keys[f.test])
        tested[f.test] += 1
    assert np.all(tested == 1)                                     # each epoch tested exactly once
    assert [f.index for f in folds] == list(range(24))


def test_loso_shares_tree_photographs_by_design():
    y, g, keys = design()
    for f in splits.loso_folds(g, y):
        assert not set(g[f.train]) & set(g[f.test])
        test_trees = set(keys[f.test][y[f.test] == 0])
        assert test_trees and test_trees <= set(keys[f.train])     # every test tree photograph seen in training


def test_leakage_assertions_raise():
    g = np.array([0, 0, 1, 1])
    with pytest.raises(LeakageError, match="participant"):
        splits.assert_no_leakage(g, np.array([0, 2]), np.array([1, 3]))
    sg = np.array([0, 1, 0, 1])
    with pytest.raises(LeakageError, match="stimulus-group"):
        splits.assert_no_leakage(g, np.array([0]), np.array([2]), sg)
    with pytest.raises(LeakageError, match="stimulus-key"):
        splits.assert_no_leakage(g, np.array([0]), np.array([3]), None, ["a", "b", "c", "a"])
    with pytest.raises(LeakageError, match="split across"):
        splits.joint_folds(g, np.array([0, 1, 0, 0]), ["k", "k", "m", "n"])


def test_mutant_split_that_ignores_participant_is_caught():
    """Wrong-equation mutation: a 'joint' split that only excludes the stimulus group leaks the participant."""
    y, g, keys = design(n_part=4)
    sg = splits.assign_stimulus_groups(list(keys), y == 1)
    p, s = 0, 0
    te = np.flatnonzero((g == p) & (sg == s))
    tr_mutant = np.flatnonzero((sg != s) & (sg >= 0))
    with pytest.raises(LeakageError):
        splits.assert_no_leakage(g, tr_mutant, te, sg)


def test_training_only_scaling_and_inductive_scoring(rng):
    F = rng.normal(size=(300, 12))
    y = (rng.random(300) < 0.8).astype(int)
    tr, te = np.arange(200), np.arange(200, 300)
    clf = ShrinkageLDA().fit(F[tr], y[tr])
    mu, sd = fit_scaler(F[tr])
    np.testing.assert_allclose(clf.mu_, mu)
    np.testing.assert_allclose(clf.sd_, sd)
    base = clf.decision_function(F[te])
    F2 = F.copy()
    F2[te[1:]] += 1e3 * rng.normal(size=(99, 12))                  # perturb all OTHER test rows massively
    np.testing.assert_allclose(lda_decision(F2[tr], y[tr], F2[te])[0], base[0], rtol=1e-10)
    # a mutant that fits the scaler on train+test differs (would be test contamination)
    mu_bad, sd_bad = fit_scaler(np.vstack([F2[tr], F2[te]]))
    assert not np.allclose(apply_scaler(F2[te[:1]], mu_bad, sd_bad), apply_scaler(F2[te[:1]], mu, sd))


def _participant_canary(rng, n_part=8, n=200):
    """Label visible only through participant-specific class means (independent across participants)."""
    y, g, F = [], [], []
    for p in range(n_part):
        yy = (rng.random(n) < 0.8).astype(int)
        direction = rng.normal(size=5)
        f = rng.normal(size=(n, 5)) + np.outer(2 * yy - 1, direction) * 2.0
        y.append(yy)
        g.append(np.full(n, p))
        F.append(f)
    return np.concatenate(F), np.concatenate(y), np.concatenate(g)


def test_heldout_decisions_do_not_depend_on_heldout_labels_loso_and_joint(rng):
    """Exact oracle: a fold's held-out decisions cannot change when labels of rows excluded from its training
    set change (the held-out participant's own labels; under joint hold-out also the held-out stimulus group)."""
    y, g, keys = design(n_part=6, n_streams=6, seed=2)
    F = rng.normal(size=(len(y), 8)) + 0.5 * y[:, None]
    sg = splits.assign_stimulus_groups(list(keys), y == 1)
    allm = np.ones(len(y), bool)
    for groups in (None, sg):
        base = regimes.cv_scores(F, y, g, allm, allm, groups)
        p, s = 2, 1
        test = (g == p) if groups is None else ((g == p) & (sg == s))
        excluded = (g == p) if groups is None else ((g == p) | (sg == s))
        y2 = y.copy()
        y2[excluded] = 1 - y2[excluded]
        again = regimes.cv_scores(F, y2, g, allm, allm, groups)
        np.testing.assert_array_equal(again[test], base[test])
    # a leaky mutant that trains on everything except nothing changes with the held-out labels
    leaky_a = lda_decision(F, y, F[g == 2])
    y3 = y.copy()
    y3[g == 2] = 1 - y3[g == 2]
    assert not np.allclose(lda_decision(F, y3, F[g == 2]), leaky_a)


def test_participant_canary_leaky_mutant_detected(rng):
    F, y, g = _participant_canary(rng)
    allm = np.ones(len(y), bool)
    sc = regimes.cv_scores(F, y, g, allm, allm)
    ba = np.nanmean(regimes.vector(regimes.ba_by_participant(y, g, sc)))
    leaky = np.full(len(y), np.nan)
    for p in np.unique(g):
        te = g == p
        leaky[te] = lda_decision(F, y, F[te])                  # MUTANT: training includes the test participant
    ba_leaky = np.mean([balanced_accuracy(y[g == p], (leaky[g == p] > 0).astype(int)) for p in np.unique(g)])
    assert ba < 0.7 < 0.85 < ba_leaky
    with pytest.raises(LeakageError):
        splits.assert_no_leakage(g, np.arange(len(y)), np.flatnonzero(g == 0))


def test_stimulus_canary_exploitable_under_loso_not_under_joint(rng):
    y, g, keys = design(n_part=8, n_streams=10, seed=3)
    codes = {k: rng.normal(size=12) for k in np.unique(keys)}
    F = np.array([codes[k] for k in keys]) + 0.05 * rng.normal(size=(len(y), 12))
    sg = splits.assign_stimulus_groups(list(keys), y == 1)
    allm = np.ones(len(y), bool)
    ba_loso = np.nanmean(regimes.vector(regimes.ba_by_participant(y, g, regimes.cv_scores(F, y, g, allm, allm))))
    ba_joint = np.nanmean(regimes.vector(regimes.ba_by_participant(y, g, regimes.cv_scores(F, y, g, allm, allm, sg))))
    assert ba_loso > 0.7            # same tree photographs in training: stimulus identity is decodable
    assert abs(ba_joint - 0.5) < 0.12  # new stimuli: identity codes carry no class information
