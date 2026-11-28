"""Decoder-panel orchestration (candidate 0.2.0) on explicitly synthetic cohorts, with a fast stub decoder.

Covers: raw-window stack consistency; identical free test sets across regimes; Delta = c + s; seed then draw averaging;
seed rules; inclusive vs exclusive thresholds; the finite-prediction (completeness) gate; Holm with a frozen family
size and an absent member; label rules; decoder registry; nested inner splits, tie-break and per-fold selection; the
opt-in 20 -> 50 matched-draw rule."""
import numpy as np
import pytest

from eareeg_decoder import config, inference, nested, panel, pipeline, regimes
from eareeg_decoder.synthetic import synthetic_epochs

SEEN = []


def stub_fit(Xtr, ytr, Xte, seed, cfg, device):
    """Nearest-centroid on the mean of channel 0 over samples 50-150; a tiny seed-dependent offset."""
    SEEN.append(int(seed))
    f_tr = np.nan_to_num(Xtr[:, 0, 50:150].mean(axis=1))
    f_te = np.nan_to_num(Xte[:, 0, 50:150].mean(axis=1))
    mid = 0.5 * (f_tr[ytr == 1].mean() + f_tr[ytr == 0].mean())
    sign = 1.0 if f_tr[ytr == 1].mean() >= f_tr[ytr == 0].mean() else -1.0
    z = sign * (f_te - mid) / (f_tr.std() + 1e-9) + 1e-3 * ((seed % 7) - 3)
    return 1.0 / (1.0 + np.exp(-z))


@pytest.fixture(scope="module")
def raw():
    es, _ = synthetic_epochs(n_participants=6, n_streams=10, seed=21, neural_amp=6.0)
    return es, pipeline.build_raw_stack(es, ("full", "fpz_free"))


def test_raw_stack_matches_feature_stack(raw):
    es, R = raw
    S = pipeline.build_stack(es)
    assert R.X["full"].shape[1:] == (11, 200) and R.X["fpz_free"].shape[1:] == (10, 200)
    for a in ("y", "g", "free", "sg", "participants"):
        np.testing.assert_array_equal(getattr(R, a), getattr(S, a))


def test_gap_panel_semantics_with_stub(raw):
    _, R = raw
    SEEN.clear()
    res = panel.gap_panel(R.X["full"], R.X["fpz_free"], R.y, R.g, R.free, "eegnet_step1", n_draws=3,
                          min_face=10, min_tree=3, fit=stub_fit)
    assert res.complete and res.test_sets_identical
    d = res.decomposition
    np.testing.assert_allclose(d["delta"], d["c"] + d["s"], atol=1e-12)
    np.testing.assert_allclose(d["delta"], res.advantage["free->free"] - res.advantage["all->free"], atol=1e-12)
    assert res.matched_by_draw.shape == (len(R.participants), 3)
    np.testing.assert_allclose(res.advantage["matched->free"], res.matched_by_draw.mean(axis=1))
    assert res.delta_by_seed.shape == (3, len(R.participants))
    np.testing.assert_allclose(res.delta_by_seed.mean(axis=0), d["delta"], atol=1e-12)
    # follow-up seed rule: 20261111 + s + fold (LOSO fold = participant index)
    assert min(SEEN) == 20261111 and set(SEEN) >= {20261111 + 100, 20261111 + 200}


def test_deterministic_decoder_runs_once(raw):
    _, R = raw
    SEEN.clear()
    panel.gap_panel(R.X["full"], R.X["fpz_free"], R.y, R.g, R.free, "xdawn_ts_lr", n_draws=1, min_face=10,
                    min_tree=3, fit=stub_fit)
    assert max(SEEN) < 20261111 + 100            # only offset 0 for the deterministic decoder
    assert panel.offsets_for("xdawn_ts_lr") == (0,) and panel.offsets_for("eeg_inception_erp") == (0, 100, 200)


def test_completeness_gate_detects_missing_predictions(raw):
    _, R = raw

    def nan_fit(Xtr, ytr, Xte, seed, cfg, device):
        p = stub_fit(Xtr, ytr, Xte, seed, cfg, device)
        p[0] = np.nan
        return p
    res = panel.gap_panel(R.X["full"], R.X["fpz_free"], R.y, R.g, R.free, "eegnet_step1", n_draws=1, offsets=(0,),
                          min_face=10, min_tree=3, fit=nan_fit)
    assert not res.complete


def test_drop_one_panel_with_stub_and_inclusive_threshold():
    es, _ = synthetic_epochs(n_participants=5, n_streams=8, seed=3, neural_amp=6.0)
    configs = ("full",) + tuple(__import__("eareeg_decoder.montage", fromlist=["DROP_ONE"]).DROP_ONE)
    R = pipeline.build_raw_stack(es, configs, require_frontal=False)
    SEEN.clear()
    res = panel.drop_one_panel(R.X, R.y, R.g, R.sg, "eegnet_step1", offsets=(0,), fit=stub_fit)
    assert res.complete and set(res.ba) == set(configs)
    np.testing.assert_allclose(res.contrasts["d_F"], res.ba["drop_Fpz"] -
                               np.mean([res.ba[f"drop_{c}"] for c in ("EL1", "EL3", "EL4", "EL5", "ER1", "ER2", "ER3",
                                                                      "ER4")], axis=0))
    assert min(SEEN) == 20261107 + 1000          # montage-stage joint seed rule
    y = np.array([1, 0, 1, 0])
    g = np.zeros(4, int)
    prob = np.array([0.5, 0.5, 0.9, 0.1])
    assert panel.ba_from_probs(y, g, prob, inclusive=True)[0] == pytest.approx(0.75)   # 0.5 counts as face
    assert panel.ba_from_probs(y, g, prob, inclusive=False)[0] == pytest.approx(0.75)  # 0.5 counts as tree
    assert panel.ba_from_probs(np.array([1, 1, 0, 0]), g, prob, inclusive=True)[0] != \
        panel.ba_from_probs(np.array([1, 1, 0, 0]), g, prob, inclusive=False)[0]


def test_holm_frozen_keeps_family_size():
    adj = inference.holm_frozen([0.01, None, 0.04], 3)
    assert np.isnan(adj[1])
    np.testing.assert_allclose(adj[[0, 2]], [0.03, 0.08])
    assert inference.holm([0.01, 0.04]).tolist() == [0.02, 0.04]    # plain Holm would shrink m
    with pytest.raises(ValueError):
        inference.holm_frozen([0.1] * 4, 3)


def test_family_labels_and_incomplete_member(rng):
    good = rng.normal(0.05, 0.02, size=12)
    null = rng.normal(0.0, 0.02, size=12)
    fam = panel.family({"a": good, "b": null, "c": None}, m=3, n_boot=2000, seeds=[1, 2, 3])
    assert fam["c"]["label"] == "INCOMPLETE" and "p_holm" not in fam["c"]
    assert fam["a"]["holm_m"] == 3 and fam["a"]["label"] == "SUPPORTED_EXPLORATORY"
    assert fam["b"]["label"] in ("INCONCLUSIVE", "CONTRADICTED")
    lo, hi = fam["a"]["signflip_inversion_ci95"]
    assert lo < good.mean() < hi
    neg = panel.family({"d_F": -good}, m=3, n_boot=2000, seeds=7, rule="drop_one")
    assert neg["d_F"]["label"] == "FPZ_LOSS_EXCEEDS_IN_EAR_LOSS"


def test_family_size_frozen_when_members_are_missing(rng):
    """Fewer members than the frozen size: Holm still uses m (the missing members count as p = 1)."""
    v = rng.normal(0.05, 0.03, size=12)
    fam = panel.family({"only": v}, m=3, n_boot=500, seeds=1)
    assert fam["only"]["holm_m"] == 3
    assert fam["only"]["p_holm"] == pytest.approx(min(1.0, 3 * fam["only"]["p_signflip"]))


def test_decoder_registry_matches_protocol():
    assert panel.decoder("eegnet_step1")[1]["dropout"] == 0.25
    assert panel.decoder("eegnet_followup")[1]["dropout"] == 0.5
    assert panel.decoder("xdawn_ts_lr")[1] == {"nfilter": 4, "C": 1.0}
    assert panel.decoder("eeg_inception_erp")[1] == {"dropout": 0.5, "lr": 0.001, "batch": 128, "epochs": 30}
    assert config.PROTOCOL["panel"]["families"]["gap_primary_delta"] == 6
    assert config.PROTOCOL["panel"]["matched_draws"] == {"loso": 20, "joint": 5, "lda_joint_reference": 20,
                                                        "seed_base": 20261111}
    with pytest.raises(ValueError):
        panel.decoder("svm")


@pytest.fixture(scope="module")
def nested_raw():
    es, _ = synthetic_epochs(n_participants=10, n_streams=10, seed=33, neural_amp=6.0)
    return pipeline.build_raw_stack(es, ("full",))


def test_nested_inner_splits_disjoint_and_exclude_outer(nested_raw):
    R = nested_raw
    folds = nested.outer_joint_folds(R.g, R.sg)
    assert len(folds) == 30
    for _, p, s in folds[:6]:
        splits = nested.inner_splits(R.g, R.sg, p, s)
        test_parts = []
        for tr, te in splits:
            assert not (set(R.g[tr].tolist()) & set(R.g[te].tolist()))
            assert not (set(R.sg[tr].tolist()) & set(R.sg[te].tolist()))
            assert p not in set(R.g[tr].tolist()) | set(R.g[te].tolist())
            assert s not in set(R.sg[tr].tolist()) | set(R.sg[te].tolist())
            test_parts += sorted(set(R.g[te].tolist()))
            assert nested.split_feasibility(R.y, R.g, tr, te)["valid"]
        assert sorted(test_parts) == sorted(q for q in np.unique(R.g).tolist() if q != p)   # each source once
        again = nested.inner_splits(R.g, R.sg, p, s)
        assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(splits, again))


def test_nested_select_tie_break():
    assert nested.select([0.6, 0.6, 0.59]) == 0
    assert nested.select([0.6, 0.6 + 1e-13, 0.59]) == 0
    assert nested.select([0.6, 0.61, 0.61]) == 1
    with pytest.raises(ValueError):
        nested.select([0.6, float("nan")])


def test_nested_choice_uses_source_only_and_tuned_probs(nested_raw):
    R = nested_raw
    grid = [{"k": 0}, {"k": 1}]
    sub = np.isin(R.g, np.unique(R.g)[:10])
    choice = nested.choose_configurations(R.X["full"][sub], R.y[sub], R.g[sub], R.sg[sub], stub_fit, grid, {})
    assert all(c["index"] in (0, 1) and len(c["scores"]) == 2 for c in choice.values())
    allm = np.ones(sub.sum(), bool)
    prob = nested.tuned_probs(R.X["full"][sub], R.y[sub], R.g[sub], R.sg[sub], allm, allm, stub_fit, grid, {},
                              choice, seed0=20261111)
    assert np.isfinite(prob).all()


def test_adaptive_draw_rule(monkeypatch):
    es, _ = synthetic_epochs(n_participants=6, n_streams=6, seed=41)
    S = pipeline.build_stack(es)
    rule = dict(config.PROTOCOL["regimes"]["draw_extension"])
    monkeypatch.setitem(config.PROTOCOL["regimes"], "draw_extension", {**rule, "mc_se_threshold": 1.0})
    res, rec = regimes.lda_regimes_adaptive(S.F, S.y, S.g, S.free, min_face=10, min_tree=3)
    assert not rec["extended"] and rec["n_draws_used"] == 20
    monkeypatch.setitem(config.PROTOCOL["regimes"], "draw_extension", {**rule, "mc_se_threshold": 0.0})
    res50, rec50 = regimes.lda_regimes_adaptive(S.F, S.y, S.g, S.free, min_face=10, min_tree=3)
    assert rec50["extended"] and rec50["n_draws_used"] == 50
    np.testing.assert_allclose(res50.matched_free_by_draw[:, :20], res.matched_free_by_draw)
    with pytest.raises(ValueError):
        regimes.lda_regimes_adaptive(S.F, S.y, S.g, S.free, n_draws=5)


def _threshold_fixture():
    """Six participants x 40 epochs; channel 0, sample 0 encodes the class so a stub can return exactly 0.5."""
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(6), 40)
    y = np.tile(np.r_[np.ones(30, int), np.zeros(10, int)], 6)
    sg = np.tile(np.arange(3).repeat(1), 80)[:240]
    X = rng.normal(size=(240, 3, 200)).astype(np.float32)
    X[:, 0, 0] = np.where(y == 1, 10.0, -10.0)
    return X, y, g, sg


def half_fit(Xtr, ytr, Xte, seed, cfg, device):
    """Face epochs -> exactly 0.5, tree epochs -> 0.1 (reads the encoded class of the TEST epoch)."""
    return np.where(Xte[:, 0, 0] > 0, 0.5, 0.1)


def test_drop_one_wiring_uses_inclusive_threshold():
    from eareeg_decoder.montage import DROP_ONE
    X, y, g, sg = _threshold_fixture()
    res = panel.drop_one_panel({m: X for m in ("full",) + tuple(DROP_ONE)}, y, g, sg, "eegnet_step1", offsets=(0,),
                               fit=half_fit)
    np.testing.assert_allclose(res.ba["full"], 1.0)          # P = 0.5 counts as face under the montage-stage rule


def montage_fit(Xtr, ytr, Xte, seed, cfg, device):
    """Face epochs -> exactly 0.5 for the 'full' input (channel 0, sample 1 = +1) and 0.9 for the 'Fpz-free' input
    (sample 1 = -1); tree epochs -> 0.1."""
    face = Xte[:, 0, 0] > 0
    return np.where(face, np.where(Xte[:, 0, 1] > 0, 0.5, 0.9), 0.1)


def test_gap_wiring_uses_exclusive_threshold():
    X, y, g, _ = _threshold_fixture()
    Xf, Xn = X.copy(), X.copy()
    Xf[:, 0, 1], Xn[:, 0, 1] = 1.0, -1.0
    free = np.ones(len(y), bool)
    res = panel.gap_panel(Xf, Xn, y, g, free, "eegnet_step1", n_draws=1, offsets=(0,), min_face=10, min_tree=3,
                          fit=montage_fit)
    # follow-up rule: P = 0.5 is NOT face, so the full montage scores BA 0.5 and the Fpz-free montage BA 1.0
    np.testing.assert_allclose(res.advantage["free->free"], -0.5)
    np.testing.assert_allclose(res.advantage["all->free"], -0.5)
    pr = np.where(X[:, 0, 0] > 0, 0.5, 0.1)
    assert panel.ba_from_probs(y, g, pr, inclusive=False)[0] == pytest.approx(0.5)


def test_joint_seed_indexes_distinguish_empty_cells():
    # The middle cell is absent, while both group levels occur elsewhere.
    g = np.array([0, 1, 1, 2, 2])
    sg = np.array([0, 0, 1, 0, 1])
    y = np.array([0, 0, 1, 1, 0])
    X = np.arange(len(y))[:, None]
    mask = np.ones(len(y), bool)
    seen = []

    def fit(Xtr, ytr, Xte, seed, cfg, device):
        seen.append(seed)
        return np.full(len(Xte), 0.5)

    regimes.net_probs(X, y, g, mask, mask, 20, {}, sg=sg, fit=fit)
    assert seen == [20, 22, 23, 24, 25]
    seen.clear()
    regimes.net_probs(X, y, g, mask, mask, 20, {}, sg=sg, fit=fit, compact_joint=True)
    assert seen == [20, 21, 22, 23, 24]


def test_exact_planned_joint_coverage_and_probability_range():
    y, g, sg = np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([1, -1, 0])
    planned = panel.planned_test_mask(y, g, np.ones(3, bool), sg)
    assert planned.tolist() == [True, False, True]
    assert panel.predictions_complete([0.1, np.nan, 0.7], planned)
    assert not panel.predictions_complete([0.1, 0.2, 0.7], planned)
    assert not panel.predictions_complete([0.1, np.nan, 1.1], planned)
    assert not panel.predictions_complete([np.nan], [False])


def test_nested_validation_refuses_nonfinite_predictions(nested_raw):
    R = nested_raw

    def bad_fit(Xtr, ytr, Xte, seed, cfg, device):
        return np.full(len(Xte), np.nan)

    with pytest.raises(FloatingPointError, match="invalid inner"):
        nested.choose_configurations(R.X["full"], R.y, R.g, R.sg, bad_fit, [{}], {})


def test_paired_auc_uses_scores_with_equal_binary_predictions(raw):
    _, R = raw
    # Test scores remain above 0.5. Only the ordering supplied by retrained
    # full-montage predictions changes. BA is unchanged; AUC is not.
    labels = R.y
    ids = np.arange(len(labels))
    X_full = np.column_stack((ids, np.ones(len(ids), dtype=int)))
    X_free = np.column_stack((ids, np.zeros(len(ids), dtype=int)))
    n_source = {int(p): int((R.g != p).sum()) for p in np.unique(R.g)}

    def fit(Xtr, ytr, Xte, seed, cfg, device):
        held = int(R.g[Xte[0, 0]])
        if Xte[0, 1] and len(Xtr) < n_source[held]:
            return 0.6 + 0.3 * labels[Xte[:, 0]]
        return np.full(len(Xte), 0.7)

    result = panel.gap_panel(X_full, X_free, labels, R.g, R.free, "xdawn_ts_lr",
                             n_draws=1, min_face=10, min_tree=3, fit=fit)
    np.testing.assert_allclose(result.decomposition["delta"], 0)
    np.testing.assert_allclose(result.auc_delta, 0.5)


def test_inversion_is_unbounded_when_exact_resolution_is_too_coarse():
    assert inference.signflip_inversion_ci([0.01, 0.02, 0.03, 0.04, 0.05]) == (-np.inf, np.inf)


def test_scalar_model_output_cannot_broadcast_to_fold():
    group = np.repeat(np.arange(3), 2)
    y = np.tile([0, 1], 3)
    use = np.ones(6, bool)
    X = np.arange(6)[:, None]
    with pytest.raises(FloatingPointError, match="number of probabilities"):
        regimes.net_probs(X, y, group, use, use, 0, {}, fit=lambda *_: 0.6)
