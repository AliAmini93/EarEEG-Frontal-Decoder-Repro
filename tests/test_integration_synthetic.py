"""End-to-end synthetic scenarios (qualitative, generator-calibrated; NOT the study's values).

Checks that the full chain -- stored reference -> montages -> transient flags -> LDA regimes -> Eq. (4) -- behaves
as the algebra predicts on data whose generating mechanism is known:

* no class signal sharing time bins with the transients  -> no paired gap;
* a sub-threshold blink-shaped deflection on trees, or a frontal neural class signal in the transients' time
  window -> a large gap (the gap does not identify whether the information is ocular or neural);
* a frontal class signal outside the transients' window -> the transferred decoder keeps its advantage;
* a frontal class signal -> removing Fpz costs more than removing an in-ear electrode (d_F < 0).
"""
import numpy as np
import pytest

from eareeg_decoder import inference, pipeline, regimes
from eareeg_decoder.synthetic import synthetic_epochs

pytestmark = [pytest.mark.lda, pytest.mark.loso, pytest.mark.joint]

SCEN = {
    "null": dict(blink_rate=(0.2, 0.2)),
    "contingent_blinks": dict(blink_rate=(0.35, 0.10)),
    "subthreshold_ocular": dict(blink_rate=(0.35, 0.10), subthreshold_ocular=20.0),
    "frontal_neural_in_transient_window": dict(blink_rate=(0.35, 0.10), neural_amp=3.0, neural_latency=0.30),
    "frontal_neural_early": dict(blink_rate=(0.35, 0.10), neural_amp=3.0, neural_latency=0.17),
}


def run(name, seed=1):
    es, _ = synthetic_epochs(n_participants=6, n_streams=10, seed=seed, **SCEN[name])
    S = pipeline.build_stack(es)
    return regimes.lda_regimes(S.F, S.y, S.g, S.free, n_draws=2, min_face=20, min_tree=5)


@pytest.mark.parametrize("name", ["null", "contingent_blinks"])
def test_no_gap_without_shared_time_bin_signal(name):
    r = run(name)
    assert abs(np.nanmean(r.decomposition["delta"])) < 0.06


@pytest.mark.parametrize("name", ["subthreshold_ocular", "frontal_neural_in_transient_window"])
def test_large_gap_when_signal_shares_transient_time_bins(name):
    r = run(name)
    assert np.nanmean(r.decomposition["delta"]) > 0.25
    assert np.nanmean(r.advantage["all->free"]) < 0.1 < np.nanmean(r.advantage["free->free"])


def test_transferred_advantage_kept_when_signal_outside_transient_window():
    r = run("frontal_neural_early")
    assert np.nanmean(r.advantage["all->free"]) > 0.25
    assert np.nanmean(r.decomposition["delta"]) < 0.15


def test_contingent_blinks_give_rate_difference_and_fpz_dependence():
    es, truth = synthetic_epochs(n_participants=6, n_streams=10, seed=3, blink_rate=(0.35, 0.10), neural_amp=4.0)
    rates = pipeline.transient_rates(es)
    assert rates["difference"].mean() > 0.15 and np.all(rates["difference"] > 0)
    con = inference.montage_contrasts(pipeline.drop_one_bas(es))
    assert con["d_F"].mean() < -0.1 and abs(con["d_M"].mean()) < 0.05
    # the 100-uV flag recovers the injected large blinks (synthetic ground truth)
    from eareeg_decoder import montage as mt
    from eareeg_decoder.transients import transient_flags
    idx = es.subset_participant(0)
    fm = mt.fpz_mastoid(es.X[idx], es.channel_names, es.good_channels(0))
    flags = transient_flags(fm, es.times)
    assert (flags.any == truth[0].blink).mean() > 0.97


def test_participant_without_fpz_is_excluded_from_transient_steps_only():
    es, _ = synthetic_epochs(n_participants=4, n_streams=6, seed=4, bad={0: ["Fpz"]})
    assert len(pipeline.build_stack(es).participants) == 3
    assert len(pipeline.build_stack(es, require_frontal=False).participants) == 4
    assert len(pipeline.drop_one_bas(es)["full"]) == 4
    assert len(pipeline.transient_rates(es)["difference"]) == 3
