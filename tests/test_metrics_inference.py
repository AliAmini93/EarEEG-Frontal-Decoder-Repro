"""Balanced accuracy under 16:4 imbalance and participant-level inference, with brute-force oracles and
deliberate wrong-equation mutants."""
import itertools

import numpy as np
import pytest

from eareeg_decoder import inference as inf
from eareeg_decoder import metrics as me

pytestmark = pytest.mark.oracle


def test_balanced_accuracy_ignores_16_to_4_imbalance():
    y = np.array([1] * 16 + [0] * 4)
    always_face = np.ones(20, int)
    assert (always_face == y).mean() == pytest.approx(0.8)
    assert me.balanced_accuracy(y, always_face) == pytest.approx(0.5)
    pred = y.copy()
    pred[:4] = 0                                  # 12/16 faces, 4/4 trees
    assert me.balanced_accuracy(y, pred) == pytest.approx(0.5 * (12 / 16 + 1.0))
    assert np.isnan(me.balanced_accuracy(np.ones(5, int), np.ones(5, int)))


def test_rank_auc_matches_pairwise_oracle(rng):
    y = (rng.random(60) < 0.7).astype(int)
    s = np.round(rng.normal(size=60), 1)          # ties on purpose
    pos, neg = s[y == 1], s[y == 0]
    oracle = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
    assert me.rank_auc(y, s) == pytest.approx(oracle)


def test_flag_agreement_and_kappa():
    flag = np.array([1, 1, 1, 0, 0, 0, 0, 0], bool)
    ref = np.array([1, 1, 0, 1, 0, 0, 0, 0], bool)
    a = me.flag_agreement(flag, ref)
    assert a["sensitivity"] == pytest.approx(2 / 3) and a["ppv"] == pytest.approx(2 / 3)
    po, pe = 6 / 8, (3 / 8) * (3 / 8) + (5 / 8) * (5 / 8)
    assert a["kappa"] == pytest.approx((po - pe) / (1 - pe))


def brute_signflip(d):
    d = np.asarray(d, float)
    obs = abs(d.mean())
    stats = [abs(np.dot(s, d)) / d.size for s in itertools.product((1, -1), repeat=d.size)]
    return np.mean([v >= obs - 1e-12 for v in stats])


@pytest.mark.parametrize("n", [1, 2, 5, 9, 12])
def test_signflip_exact_matches_brute_force(rng, n):
    d = rng.normal(0.02, 0.05, size=n)
    assert inf.signflip_exact_p(d) == pytest.approx(brute_signflip(d), abs=0)


def test_signflip_known_values_and_mutants():
    assert inf.signflip_exact_p([1, 2, 3, 4, 5]) == pytest.approx(2 / 32)          # only all-+ and all--
    assert inf.signflip_exact_p(np.ones(16)) == pytest.approx(2 / 2 ** 16)
    assert inf.signflip_exact_p([1.0, -1.0]) == pytest.approx(1.0)
    d = np.array([0.03, 0.05, -0.01, 0.04, 0.02, 0.06, -0.02, 0.01])
    p2 = inf.signflip_exact_p(d)
    one_sided_mutant = np.mean([np.dot(s, d) / d.size >= d.mean() - 1e-12 for s in itertools.product((1, -1), repeat=8)])
    assert one_sided_mutant != pytest.approx(p2)                                   # the mutant would be caught
    assert p2 == pytest.approx(2 * one_sided_mutant, abs=1 / 256)
    with pytest.raises(ValueError):
        inf.signflip_exact_p(np.ones(25))


def test_holm_against_hand_calculation_and_bonferroni_mutant():
    p = [0.01, 0.04, 0.03, np.nan]
    adj = inf.holm(p)
    assert adj[0] == pytest.approx(0.03) and adj[2] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06)
    assert np.isnan(adj[3])
    bonferroni = np.minimum(1, np.array(p[:3]) * 3)
    assert not np.allclose(adj[:3], bonferroni)
    assert np.all(np.diff(adj[np.argsort(p[:3])]) >= 0)                            # monotone in p


def test_bootstrap_is_over_participants_seeded_and_matches_manual(rng):
    v = rng.normal(0.03, 0.05, size=16)
    lo, hi = inf.bootstrap_ci(v, 2000, 20261117)
    r = np.random.default_rng(20261117)
    m = v[r.integers(0, 16, size=(2000, 16))].mean(axis=1)
    assert (lo, hi) == (pytest.approx(np.percentile(m, 2.5)), pytest.approx(np.percentile(m, 97.5)))
    assert inf.bootstrap_ci(v, 2000, 20261117) == (lo, hi)
    # mutant: resampling epochs (many correlated rows per participant) gives a falsely narrow interval
    epochs = np.repeat(v, 200) + rng.normal(0, 0.001, size=16 * 200)
    lo_e, hi_e = inf.bootstrap_ci(epochs, 2000, 1)
    assert (hi_e - lo_e) < 0.25 * (hi - lo)
    assert inf.bootstrap_ci([np.nan, 1.0, 2.0], 100, 0)[0] >= 1.0                   # non-finite dropped


def test_inversion_ci_edges_are_test_boundaries(rng):
    d = rng.normal(0.04, 0.05, size=12)
    lo, hi = inf.signflip_inversion_ci(d)
    assert lo < d.mean() < hi
    eps = 1e-6
    assert inf.signflip_exact_p(d - (lo + eps)) >= 0.05 and inf.signflip_exact_p(d - (lo - 1e-4)) < 0.05
    assert inf.signflip_exact_p(d - (hi - eps)) >= 0.05 and inf.signflip_exact_p(d - (hi + 1e-4)) < 0.05


def test_paired_summary_and_decision_labels(rng):
    v = rng.normal(0.05, 0.02, size=16)
    s = inf.paired_summary(v, 1000, 0)
    assert s["n"] == 16 and s["n_above_null"] + s["n_below_null"] <= 16 and s["p_signflip"] < 0.01
    ba = inf.paired_summary(0.5 + v, 1000, 0, null=0.5)
    assert ba["p_signflip"] == s["p_signflip"] and ba["dz"] == pytest.approx(s["dz"])
    big = inf.paired_summary(rng.normal(size=30), 200, 0)
    assert big["p_signflip"] is None and "n <= 24" in big["p_note"]
    assert inf.paired_summary([], 10, 0) == {"n": 0, "null": 0.0}
    assert np.isnan(inf.signflip_inversion_ci([np.nan])[0])
    assert inf.decision_label(s, 0.02) == "SUPPORTED_EXPLORATORY"
    assert inf.decision_label(s, 0.06) == "INCONCLUSIVE"                            # Holm p not below 0.05
    neg = dict(s, mean=-0.05, ci95=[-0.08, -0.02])
    assert inf.decision_label(neg, 0.01) == "CONTRADICTED"
    wide = dict(s, ci95=[-0.001, 0.09])
    assert inf.decision_label(wide, 0.01) == "INCONCLUSIVE"


def test_draw_uncertainty():
    mat = np.array([[0.1, 0.3], [0.2, 0.4], [0.0, 0.2]])
    per_p, u = inf.draw_uncertainty(mat)
    np.testing.assert_allclose(per_p, [0.2, 0.3, 0.1])
    assert u["mc_se_cohort_mean"] == pytest.approx(np.std([0.1, 0.3], ddof=1) / np.sqrt(2))


def test_montage_contrasts_average_in_ear_within_participant():
    from eareeg_decoder.montage import DROP_ONE
    ba = {c: np.array([0.60, 0.58]) for c in DROP_ONE}
    ba["drop_Fpz"] = np.array([0.55, 0.57])
    ba["drop_M1"] = np.array([0.59, 0.56])
    c = inf.montage_contrasts(ba)
    np.testing.assert_allclose(c["d_F"], [0.55 - 0.60, 0.57 - 0.58])
    np.testing.assert_allclose(c["d_M"], [(0.59 + 0.60) / 2 - 0.60, (0.56 + 0.58) / 2 - 0.58])
    np.testing.assert_allclose(c["d_FM"], c["d_F"] - c["d_M"])


def test_twoway_bootstrap_shape_and_seed():
    cells = np.arange(12, dtype=float).reshape(4, 3)
    assert inf.twoway_bootstrap_ci(cells, 200, 5) == inf.twoway_bootstrap_ci(cells, 200, 5)
