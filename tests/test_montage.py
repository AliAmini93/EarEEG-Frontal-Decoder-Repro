"""Re-reference algebra (Eq. 1) against an independent oracle built from hardware-referenced potentials.

Oracle: draw potentials x against an unknown reference, form the stored signal y = x - mean(x over good channels),
then the correct montage row must equal x_i - mean_{j in R_g} x_j, computed directly from x. These tests are
written so that the naive "delete the Fpz row" shortcut FAILS them (red tests).
"""
import numpy as np
import pytest

from eareeg_decoder import montage as mt

pytestmark = pytest.mark.oracle

RNG = np.random.default_rng(20261123)
NAMES = list(mt.EAR_CHANNELS)


def stored_from_potentials(x, good):
    """Independent oracle of the stored reference: subtract the mean over good channels; bad rows NaN."""
    gi = [NAMES.index(c) for c in good]
    y = x - x[:, gi, :].mean(axis=1, keepdims=True)
    for i, c in enumerate(NAMES):
        if c not in good:
            y[:, i, :] = np.nan
    return y


def oracle_rows(x, good, channels):
    gi = [NAMES.index(c) for c in channels if c in good]
    ref = x[:, gi, :].mean(axis=1)
    out = np.full((x.shape[0], len(channels), x.shape[2]), np.nan)
    for j, c in enumerate(channels):
        if c in good:
            out[:, j] = x[:, NAMES.index(c)] - ref
    return out


@pytest.fixture
def potentials():
    x = RNG.normal(size=(7, 11, 50))
    x += 40.0 * RNG.normal(size=(7, 1, 50))      # common hardware-reference signal
    x[:, 0] += 120.0 * np.hanning(50)             # large frontal deflection on Fpz
    return x


@pytest.mark.parametrize("bad", [(), ("EL3",), ("M2", "ER4"), ("Fpz",)])
def test_drop_one_matches_oracle_for_every_electrode(potentials, bad):
    good = [c for c in NAMES if c not in bad]
    y = stored_from_potentials(potentials, good)
    for cfg in mt.DROP_ONE:
        got = mt.drop_one(y, NAMES, good, cfg)
        want = oracle_rows(potentials, good, mt.retained_channels(cfg))
        np.testing.assert_allclose(got, want, atol=1e-9, equal_nan=True)


def test_fpz_free_rows_do_not_depend_on_fpz(potentials):
    good = NAMES
    a = mt.fpz_free(stored_from_potentials(potentials, good), NAMES, good)
    x2 = potentials.copy()
    x2[:, 0] += 1000.0 * RNG.normal(size=x2[:, 0].shape)   # change ONLY Fpz
    b = mt.fpz_free(stored_from_potentials(x2, good), NAMES, good)
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_red_naive_fpz_row_deletion_fails_the_oracle(potentials):
    """RED TEST: deleting the stored Fpz row without re-referencing leaves Fpz in every retained row."""
    good = NAMES
    y = stored_from_potentials(potentials, good)
    naive = mt.naive_drop(y, NAMES, "drop_Fpz")
    want = oracle_rows(potentials, good, mt.retained_channels("drop_Fpz"))
    assert not np.allclose(naive, want, atol=1e-6)
    # the residual is exactly -Fpz/11 + mean-of-others/11, i.e. it carries the frontal deflection
    resid = naive - want
    expected = -(potentials[:, :1] - potentials[:, 1:].mean(axis=1, keepdims=True)) / 11.0
    np.testing.assert_allclose(resid, np.broadcast_to(expected, resid.shape), atol=1e-9)
    # and changing only Fpz changes the naive rows (leak), but not the correct rows
    x2 = potentials.copy()
    x2[:, 0] += 500.0
    y2 = stored_from_potentials(x2, good)
    assert not np.allclose(mt.naive_drop(y2, NAMES, "drop_Fpz"), naive)
    np.testing.assert_allclose(mt.drop_one(y2, NAMES, good, "drop_Fpz"), mt.drop_one(y, NAMES, good, "drop_Fpz"),
                               atol=1e-9)


def test_drop_fpz_equals_fpz_free(potentials):
    y = stored_from_potentials(potentials, NAMES)
    np.testing.assert_array_equal(mt.drop_one(y, NAMES, NAMES, "drop_Fpz"), mt.fpz_free(y, NAMES, NAMES))


def test_full_montage_is_invariant_under_own_mean_rereference(potentials):
    good = [c for c in NAMES if c != "ER2"]
    y = stored_from_potentials(potentials, good)
    full = mt.full_montage(y, NAMES, good)
    again = mt.own_mean_reference(y, NAMES, good, NAMES, min_good=1)
    np.testing.assert_allclose(full, again, atol=1e-9, equal_nan=True)
    assert np.isnan(full[:, NAMES.index("ER2")]).all()


def test_fpz_mastoid_is_reference_free(potentials):
    good = NAMES
    a = mt.fpz_mastoid(stored_from_potentials(potentials, good), NAMES, good)
    want = potentials[:, 0] - potentials[:, 1:3].mean(axis=1)
    np.testing.assert_allclose(a, want, atol=1e-9)
    # one mastoid bad -> the other alone; both bad or Fpz bad -> NaN
    g1 = [c for c in NAMES if c != "M1"]
    np.testing.assert_allclose(mt.fpz_mastoid(stored_from_potentials(potentials, g1), NAMES, g1),
                               potentials[:, 0] - potentials[:, 2], atol=1e-9)
    g2 = [c for c in NAMES if c not in ("M1", "M2")]
    assert np.isnan(mt.fpz_mastoid(stored_from_potentials(potentials, g2), NAMES, g2)).all()
    g3 = NAMES[1:]
    assert np.isnan(mt.fpz_mastoid(stored_from_potentials(potentials, g3), NAMES, g3)).all()


def test_in_ear_and_min_good(potentials):
    good = NAMES
    y = stored_from_potentials(potentials, good)
    np.testing.assert_allclose(mt.in_ear(y, NAMES, good), oracle_rows(potentials, good, list(mt.IN_EAR)), atol=1e-9)
    only_one = ["Fpz", "M1", "M2", "EL1"]
    assert np.isnan(mt.in_ear(stored_from_potentials(potentials, only_one), NAMES, only_one)).all()


def test_rows_sum_to_zero_and_rank(potentials):
    y = stored_from_potentials(potentials, NAMES)
    z = mt.drop_one(y, NAMES, NAMES, "drop_EL5")
    np.testing.assert_allclose(z.sum(axis=1), 0.0, atol=1e-9)
    M = np.transpose(z, (1, 0, 2)).reshape(10, -1)
    assert np.linalg.matrix_rank(M, tol=1e-8) == 9


def test_span_identical_full_spans_same_subspace(potentials):
    y = stored_from_potentials(potentials, NAMES)
    a = np.transpose(mt.full_montage(y, NAMES, NAMES), (1, 0, 2)).reshape(11, -1)
    b = np.transpose(mt.span_identical_full(y, NAMES, NAMES), (1, 0, 2)).reshape(11, -1)
    assert np.linalg.matrix_rank(np.vstack([a, b]), tol=1e-8) == np.linalg.matrix_rank(a, tol=1e-8) == 10
    assert not np.allclose(a, b)


def test_missing_channel_names_raise():
    with pytest.raises(ValueError, match="missing ear-system channels"):
        mt.fpz_free(np.zeros((1, 3, 5)), ["Fpz", "M1", "M2"], ["Fpz", "M1", "M2"])
    with pytest.raises(ValueError):
        mt.drop_one(np.zeros((1, 11, 5)), NAMES, NAMES, "drop_XX")


def test_channel_order_of_input_does_not_matter(potentials):
    y = stored_from_potentials(potentials, NAMES)
    perm = RNG.permutation(11)
    yp = y[:, perm]
    names_p = [NAMES[i] for i in perm]
    for m in ("full", "fpz_free", "in_ear", "fpz_mastoid", "drop_M1"):
        np.testing.assert_allclose(mt.make_montage(m, yp, names_p, NAMES), mt.make_montage(m, y, NAMES, NAMES),
                                   atol=1e-12)
