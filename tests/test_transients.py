"""Large-transient flag, post-onset rule, descriptor, selection screens and tracker-interval overlap."""
import numpy as np
import pytest

from eareeg_decoder import transients as tr

pytestmark = pytest.mark.sensitivity

TIMES = np.round(np.arange(263) / 250.0 - 0.2, 6)


def bump(amp, centre, width=0.05):
    return amp * np.exp(-0.5 * ((TIMES - centre) / width) ** 2)


def test_threshold_is_inclusive_at_100():
    fm = np.zeros((3, TIMES.size))
    fm[0, 100] = 100.0          # ptp exactly 100 -> flagged (>=)
    fm[1, 100] = 99.999         # just below -> free
    fm[2, 100], fm[2, 101] = 60.0, -40.0   # ptp 100 from both signs -> flagged
    f = tr.transient_flags(fm, TIMES)
    assert f.any.tolist() == [True, False, True]
    assert f.free.tolist() == [False, True, False]


@pytest.mark.parametrize("peak, post", [(0.148, False), (0.152, True), (-0.1, False), (0.848, True)])
def test_post_onset_rule_uses_peak_at_or_after_150ms(peak, post):
    fm = np.zeros((1, TIMES.size))
    fm[0, int(np.argmin(np.abs(TIMES - peak)))] = 150.0
    f = tr.transient_flags(fm, TIMES)
    assert f.any[0] and bool(f.post[0]) is post and bool(f.pre[0]) is (not post)


def test_post_onset_boundary_is_inclusive_on_any_grid():
    times = np.array([-0.2, 0.0, 0.1, 0.15, 0.2])
    fm = np.array([[0.0, 0.0, 0.0, 120.0, 0.0], [0.0, 0.0, 120.0, 0.0, 0.0]])
    f = tr.transient_flags(fm, times)
    assert f.post.tolist() == [True, False] and f.pre.tolist() == [False, True]


def test_peak_is_signed_maximum_not_absolute():
    fm = np.zeros((1, TIMES.size))
    fm[0, 20] = 30.0             # positive max at t = -0.12 s (pre-onset)
    fm[0, 150] = -200.0          # larger negative excursion at 0.4 s
    f = tr.transient_flags(fm, TIMES)
    assert f.any[0] and f.peak_time[0] == pytest.approx(TIMES[20]) and f.pre[0]


def test_relative_and_sensitivity_thresholds():
    rng = np.random.default_rng(0)
    fm = rng.normal(0, 10, size=(100, TIMES.size))
    theta = tr.resolve_theta(fm, "relative")
    assert theta == pytest.approx(2.5 * np.percentile(np.ptp(fm, axis=1), 25))
    assert tr.resolve_theta(fm, 70) == 70.0


def test_descriptor_has_32_strict_threshold_features():
    fm = np.vstack([bump(51.0, 0.125, 0.2), bump(-50.0, 0.125, 0.2) * 0 + 0.0, -bump(80.0, 0.625, 0.02)])
    d = tr.thresholded_descriptor(fm, TIMES)
    assert d.shape == (3, 32) and set(np.unique(d)) <= {0.0, 1.0}
    assert d[0, :16].sum() > 0 and d[0, 16:].sum() == 0
    assert d[1].sum() == 0
    exact = np.full((1, TIMES.size), 50.0)       # bin mean exactly +50 is NOT above threshold
    assert tr.thresholded_descriptor(exact, TIMES).sum() == 0
    assert d[2, 16 + 12] == 1.0                   # bin 600-650 ms below -50


def test_preonset_and_late_flank_screens():
    fm = np.vstack([bump(60, -0.1, 0.02), bump(60, 0.4, 0.02), bump(60, 0.8, 0.02), np.zeros(TIMES.size)])
    assert tr.preonset_clean(fm, TIMES).tolist() == [False, True, True, True]
    assert tr.late_flank_clean(fm, TIMES).tolist() == [True, True, False, True]
    edge = np.zeros((1, TIMES.size))
    edge[0, -1] = 50.0                             # 0.848 s is inside the closed late window, limit is strict
    assert not tr.late_flank_clean(edge, TIMES)[0]


def test_adjacent_and_next_baseline():
    order = np.array([3, 0, 1, 2, 4])
    stream = np.array([1, 0, 0, 1, 1])
    prev, nxt = tr.adjacent(order, stream)
    assert prev.tolist() == [3, -1, 1, -1, 0] and nxt.tolist() == [4, 2, -1, 0, -1]
    onsets = np.array([3.0, 0.0, 1.0, 2.0, 4.2])
    fm = np.zeros((5, TIMES.size))
    fm[4, 10] = 80.0                               # presentation 4 has a pre-onset deflection
    flag, mappable = tr.next_baseline_transient(fm, TIMES, onsets, nxt)
    # 0 -> next is 4 but IOI 1.2 s > 1.048 s: not mappable, not flagged
    assert mappable.tolist() == [False, True, False, True, False] and not flag.any()
    onsets[4] = 4.0
    flag, mappable = tr.next_baseline_transient(fm, TIMES, onsets, nxt)
    assert flag.tolist() == [True, False, False, False, False]


def test_interval_overlap_with_clock_tolerance():
    onsets = np.array([10.0, 20.0, np.nan])
    blinks = np.array([[9.70, 9.796], [21.0, 21.2]])      # first ends 4 ms before the window opens (9.8 s)
    assert tr.interval_overlap(onsets, blinks, tol=0.0).tolist() == [False, False, False]
    assert tr.interval_overlap(onsets, blinks, tol=0.004 + 1e-9).tolist() == [True, False, False]
    assert tr.interval_overlap(onsets, blinks, tol=0.15).tolist() == [True, False, False]
    assert tr.interval_overlap(onsets, np.array([[20.8, 20.9]])).tolist() == [False, True, False]


def test_rate_difference():
    flag = np.array([1, 1, 0, 0, 0, 1, 0, 0], bool)
    is_tree = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)
    assert tr.rate_difference(flag, is_tree) == pytest.approx(0.5 - 0.25)
