import numpy as np
import pytest

from eareeg_decoder import clock, pipeline, regimes, sensitivities as sens, synthetic


def test_photograph_bootstrap_matches_explicit_replication():
    # P0 contributes twice to A and once to B and C. P1 contributes only to B.
    y = np.array([1, 0, 0, 0, 0, 1, 0])
    g = [0, 0, 0, 0, 0, 1, 1]
    key = ["SYNTH_F", "SYNTH_A", "SYNTH_A", "SYNTH_B", "SYNTH_C", "SYNTH_F", "SYNTH_B"]
    changed = [0.2, -0.1, -0.1, 0.1, 0.1, 0.3, -0.1]
    constant = [0.4] * len(y)
    for seed in range(10):
        ans = sens.photograph_weighted_gap(y, g, key, np.ones(len(y), bool), retrained_full=changed,
                                           retrained_free=constant, transferred_full=constant,
                                           transferred_free=constant, n_boot=7, seed=seed)
        rng = np.random.default_rng(seed)
        oracle = []
        for _ in range(7):
            pi, photos = rng.integers(2, size=2), rng.integers(3, size=3)
            p0 = [hit for photo in photos for hit in ([1, 1] if photo == 0 else [0])]
            p1 = [1 for photo in photos if photo == 1]
            vals = [np.mean(p0) / 2, np.mean(p1) / 2 if p1 else 0]
            oracle.append(np.mean([vals[i] for i in pi]))
        assert ans["mean"] == pytest.approx((0.5 + 1) / 4)
        assert ans["leave_one_out_range"] == pytest.approx([1 / 6, 5 / 12])
        assert ans["ci95"] == pytest.approx(np.quantile(oracle, [.025, .975]))


def test_h1_shuffle_refits_source_and_preserves_within_person_label_counts(monkeypatch):
    g = np.repeat(np.arange(3), 6)
    y = np.tile([0, 0, 0, 1, 1, 1], 3)
    free = np.tile([True, False, True, True, False, True], 3)
    witness = []

    def scores(F, labels, groups, tr, te):
        witness.append((labels.copy(), tr.copy(), te.copy()))
        return np.where(te, 0.5, np.nan)

    monkeypatch.setattr(regimes, "cv_scores", scores)
    values = sens.h1_label_permutation(dict(full=None, fpz_free=None), y, g, free, n_permutations=3,
                                        min_face=1, min_tree=1)
    assert len(witness) == 12
    np.testing.assert_array_equal(values, 0)
    rng = np.random.default_rng(20261117)
    for i in range(3):
        expected = np.concatenate([rng.permutation(y[g == p]) for p in np.unique(g)])
        for fold, (labels, tr, te) in enumerate(witness[i * 4: i * 4 + 4]):
            np.testing.assert_array_equal(labels, expected)
            np.testing.assert_array_equal(te, free)
            np.testing.assert_array_equal(tr, free if fold < 2 else np.ones(len(y), bool))


def test_history_uses_rejected_previous_rows_and_rotates_only_retained(monkeypatch):
    y = np.tile([1, 0, 1, 0, 1, 0], 2)
    g = np.repeat([1, 3], 6)
    stream = np.tile([0, 0, 0, 1, 1, 1], 2)
    order = np.tile([1, 2, 3], 4)
    flags = np.tile([False, True, False], 4)
    keep = np.tile([True, False, True, True, True, True], 2)
    witness = []

    def scores(F, labels, groups, tr, te):
        witness.append((F.copy(), labels.copy()))
        return np.where(te, 0.1, np.nan)

    monkeypatch.setattr(regimes, "cv_scores", scores)
    out = sens.stream_history_control(y, g, stream, order, flags, keep, n_rotations=2)
    assert out["n_rotations"] == 2
    F = witness[0][0]
    assert F[2].tolist() == [1, 0, 0, 1]
    assert F[3].tolist() == [0, 0, 1, 0]
    for same_F, labels in witness[1:]:
        np.testing.assert_array_equal(same_F, F)
        np.testing.assert_array_equal(labels[~keep], y[~keep])
        for p in np.unique(g):
            for s in np.unique(stream):
                rows = (g == p) & (stream == s) & keep
                assert sorted(labels[rows]) == sorted(y[rows])


def test_propagation_centering_sign_qc_and_floor():
    times = [0, .15, .3]
    a = np.tile([10, 150, 0], (10, 1))
    b = -0.25 * a + np.arange(len(a))[:, None]
    assert sens.propagation_slope(a, b, times) == pytest.approx(-0.25)
    assert np.isnan(sens.propagation_slope(a[:-1], b[:-1], times))
    assert np.isnan(sens.propagation_slope(a, b * np.nan, times))
    assert np.isnan(sens.propagation_slope(a * 0, b, times, theta=0))


def test_cross_device_explicit_permuted_row_map():
    es, _ = synthetic.synthetic_epochs(n_participants=3, n_streams=2, seed=12)
    data = pipeline.build_stack(es)
    y = data.y
    frontal = np.zeros((len(y), len(es.times)))
    frontal[y == 0, np.searchsorted(es.times, 0.3)] = 130
    reverse = np.arange(len(y))[::-1]
    eog = np.repeat((2 * y[:, None, None] - 1).astype(float), len(es.times), axis=2)
    eog += np.random.default_rng(11).normal(0, 0.1, eog.shape)
    kw = dict(ear_index=np.arange(len(y)), scalp_index=reverse, alignment_provenance="synthetic known map",
              scale_provenance="synthetic scale", eog_placement="synthetic channel")
    out = sens.cross_device_control(frontal, frontal[reverse], eog[reverse], es.times, y, data.g,
                                    np.ones(len(y), bool), **kw)
    assert out["ear_rate_gap_mean"] == out["scalp_rate_gap_mean"] == out["eog_ba"]["all"] == 1
    assert out["flag_kappa_median"] == 1
    with pytest.raises(ValueError, match="one-to-one"):
        sens.cross_device_control(frontal, frontal, eog, es.times, y, data.g, np.ones(len(y), bool),
                                  **{**kw, "scalp_index": np.zeros(len(y), int)})


def test_truncation_scenario_crosses_threshold_without_neural_category_effect():
    _, a = synthetic.truncation_scenario(n_participants=2, n_streams=15, seed=14)
    _, b = synthetic.truncation_scenario(n_participants=2, n_streams=15, seed=14)
    amps = np.concatenate([row.blink_amp[row.blink] for row in a])
    assert 0 < (amps > 100).sum() < len(amps)
    np.testing.assert_array_equal(a[0].blink_amp, b[0].blink_amp)


def test_trigger_fit_and_blink_unknown_states():
    source = np.array([0., 1.3, 3.9, 7.1, 13.2])
    tracker = 4.0 + 1.0002 * source
    kw = dict(coarse_tol=0.05, residual_tol=0.005, alternative_offset=.1,
              minimum_margin=.15, min_fraction=1, event_provenance="synthetic pulses")
    mapping = clock.fit_trigger_map(source, tracker, **kw)
    assert mapping.offset == pytest.approx(4)
    assert mapping.slope == pytest.approx(1.0002)
    assert mapping.matched_fraction == 1
    sample_times = np.arange(3., 19., .002)
    pupil = np.where(abs(sample_times - tracker[2]) < .5, 0., 1.)
    intervals = np.array([[tracker[0], tracker[0] + .1]])
    state = clock.classify_eye_epochs([*source, source[-1] + 1], mapping, sample_times, pupil, intervals)
    assert state.tolist() == [1, 0, -1, 0, 0, -1]
    with pytest.raises(ValueError, match="uniqueness"):
        clock.fit_trigger_map(np.arange(5.), np.arange(6.), **kw)
