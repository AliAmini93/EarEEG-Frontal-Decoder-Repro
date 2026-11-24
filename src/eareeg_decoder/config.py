"""Protocol constants of the analyses described in the manuscript (in preparation).

These are *method* constants (thresholds, windows, seeds, architecture settings). They contain no data and no
results. ``configs/protocol.json`` in the repository root is a byte-for-byte JSON rendering of :data:`PROTOCOL`
(checked by ``tests/test_data_adapter_config_cli.py``) so that readers can inspect the values without importing Python.

Amplitude constants are on the recording's *nominal* microvolt scale. The original amplifiers were not
independently calibrated, so a threshold such as 100 uV is a value on that nominal scale, not a calibrated
physical quantity.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

PROTOCOL: dict[str, Any] = {
    "schema": "eareeg_decoder.protocol/1",
    "description": (
        "Method constants for the ear-centred EEG face-versus-tree analyses of the manuscript (in preparation). "
        "No data and no results are contained here."
    ),
    "units": {
        "amplitude": "nominal microvolt scale of the recording (not independently calibrated)",
        "time": "seconds relative to the event marker of image onset (display latency not measured)",
    },
    "channels": {
        "order": ["Fpz", "M1", "M2", "EL1", "EL3", "EL4", "EL5", "ER1", "ER2", "ER3", "ER4"],
        "frontal": "Fpz",
        "mastoids": ["M1", "M2"],
        "in_ear": ["EL1", "EL3", "EL4", "EL5", "ER1", "ER2", "ER3", "ER4"],
        "labels": {"face": 1, "tree": 0},
        "label_meaning": "the displayed image type, not the participant's emotion or cognitive state",
    },
    "preprocessing": {
        "filter": {"l_freq": 0.1, "h_freq": 30.0, "method": "fir", "fir_design": "firwin",
                   "fir_window": "hamming", "phase": "zero"},
        "filter_padding_s": 60.0,
        "qc_span_margin_s": 1.0,
        "target_sfreq": 250.0,
        "tmin": -0.2,
        "tmax": 0.85,
        "n_epoch_samples": 263,
        "last_sample_s": 0.848,
        "baseline": [-0.2, 0.0],
        "onset_rounding": "floor(onset * sfreq + 0.5)",
        "stored_reference": "mean of the good ear-system channels (includes Fpz when Fpz is good)",
        "epoch_rejection_ptp_uv": 150.0,
        "epoch_rejection_channels": "good mastoid and in-ear channels after the stored common average (Fpz not tested)",
    },
    "channel_qc": {
        "reference_for_qc": "per-sample median of the finite neural channels",
        "flat_sd_uv": 0.5,
        "max_sd_uv": 100.0,
        "robust_z_log10_sd": 4.0,
        "robust_z_mad_floor_log10": 0.05,
        "clip_fraction_max": 0.01,
        "max_bad_channel_fraction": 0.5,
    },
    "montage": {
        "min_good_for_own_mean": 2,
        "full": "all 11 stored rows (own-mean re-reference leaves them unchanged)",
        "fpz_free": "M1, M2 and the eight in-ear rows re-referenced to their own good mean",
        "in_ear": "the eight in-ear rows re-referenced to their own good mean",
        "fpz_mastoid": "Fpz minus the mean of the good mastoids (one row)",
        "drop_one": "delete one electrode and re-reference the other ten to their own good mean",
    },
    "transients": {
        "derivation": "Fpz minus mean of good mastoids",
        "theta_uv": 100.0,
        "theta_sensitivity_uv": [70.0, 150.0],
        "theta_relative_factor_of_p25": 2.5,
        "ptp_window": "whole stored epoch [-0.2, 0.848] s",
        "post_peak_min_s": 0.15,
        "peak_definition": "time of the signed maximum of the derivation",
        "descriptor_bin_threshold_uv": 50.0,
        "late_flank_window_s": [0.748, 0.848],
        "late_flank_limit_uv": 50.0,
        "preonset_window_s": [-0.2, 0.0],
        "preonset_limit_uv": 50.0,
        "next_baseline_max_ioi_s": 1.048,
        "tracker_blink_tolerance": "in-sample clock residual plus one 250-Hz sample",
    },
    "features": {"window_s": [0.0, 0.8], "bin_s": 0.05, "layout": "channel-major (channel, bin)"},
    "lda": {"solver": "lsqr", "shrinkage": "auto (Ledoit-Wolf)", "priors": [0.5, 0.5],
            "scaling": "z-score fitted on the training rows only; NaN -> training mean (0)",
            "decision": "decision value > 0 -> face"},
    "eegnet_montage_stage": {"F1": 8, "D": 2, "F2": 16, "kernel": 64, "dropout": 0.25, "epochs": 30,
                             "lr": 0.001, "batch": 128, "loss": "class-weighted cross-entropy N/(2 n_c)",
                             "input_window_s": [0.0, 0.8],
                             "provenance": "frozen preregistration network settings and network-control contract of the montage stage",
                             "seed_rule": "joint: 20261107 + s + 1000 + fold; LOSO: 20261107 + s + fold; s in {0, 100, 200}",
                             "seed_base": 20261107, "seed_offsets": [0, 100, 200],
                             "seed_split_extra": {"joint": 1000, "loso": 0}},
    "eegnet_transient_stage": {"F1": 8, "D": 2, "F2": 16, "kernel": 64, "dropout": 0.5, "epochs": 30,
                               "lr": 0.001, "batch": 128, "loss": "class-weighted cross-entropy N/(2 n_c)",
                               "input_window_s": [0.0, 0.8],
                               "provenance": "frozen transient-analysis contract (network section)",
                               "seed_rule": ("20261111 + s + fold, s in {0, 100, 200}; LOSO fold = participant index "
                                             "(ineligible participants still consume an index); joint fold = index of "
                                             "the (participant, stimulus group) pair in sorted order"),
                               "seed_base": 20261111, "seed_offsets": [0, 100, 200],
                               "seed_split_extra": {"joint": 0, "loso": 0}},
    "splits": {"stimulus_groups_k": 3, "stimulus_group_salt": "phase2-stimgroup-v1",
               "salt_note": "opaque constant of the original analyses; changing it changes fold membership",
               "stimulus_group_rule": "class-stratified; keys ordered by SHA-256(salt + '|' + key); group = rank mod k",
               "joint": "train: other participants AND other stimulus groups; test: one participant AND one group",
               "loso": "train: other participants (all stimuli, so tree photographs are shared)"},
    "regimes": {"names": ["all->all", "free->free", "matched->matched", "all->free", "matched->free"],
                "matched_draws_lda": 20, "matched_seed_base": 20261111,
                "matched_draws_network": {"matched->matched": 1, "matched->free (LOSO)": 20, "matched->free (joint)": 5},
                "draw_extension_rule": "extend LDA draws to 50 if the Monte Carlo SE of the composition contrast exceeds 0.0025 (not automated here)",
                "eligibility_min_face": 40, "eligibility_min_tree": 15},
    "inference": {"unit": "participant", "bootstrap_ci": "95% percentile over participants",
                  "bootstrap_montage": {"n_boot": 10000, "seed_per_montage_base": 20261111,
                                        "seeds": {"d_F": 20261121, "d_M": 20261122, "d_FM": 20261123,
                                                  "d_F_two_way": 20261131}},
                  "bootstrap_transient": {"n_boot": 10000, "seed": 20261111},
                  "bootstrap_followup": {"n_boot": 20000, "seed": 20261117},
                  "two_way_bootstrap": ("participants x stimulus groups (d_F) or participants x tree photographs "
                                        "(transient rate): mean over resampled cells; the follow-up's photograph-"
                                        "weighted two-way bootstrap of the paired gap is not implemented"),
                  "signflip": "exact two-sided over all 2^n sign patterns",
                  "holm": "within each named family only", "alpha": 0.05,
                  "inversion_ci": "set of mean shifts not rejected by the exact sign-flip test at 0.05"},
}


def protocol() -> dict[str, Any]:
    """Return a deep copy of :data:`PROTOCOL` (callers may modify their copy freely)."""
    return copy.deepcopy(PROTOCOL)


def protocol_json() -> str:
    """Canonical JSON text of :data:`PROTOCOL` as stored in ``configs/protocol.json``."""
    return json.dumps(PROTOCOL, indent=1, ensure_ascii=True) + "\n"


def load_protocol_file(path: str | Path) -> dict[str, Any]:
    """Read a protocol JSON file (for example ``configs/protocol.json``)."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def eegnet_seed(stage: str, split: str, seed_offset: int, fold: int) -> int:
    """Per-fit seed of the original network runs.

    ``stage`` is ``"montage"`` (montage analyses; seed = 20261107 + s + 1000 + fold under joint hold-out and
    20261107 + s + fold under LOSO) or ``"transient"`` (transient and follow-up analyses; seed = 20261111 + s + fold).
    ``s`` is one of 0, 100, 200 and ``fold`` is the enumeration index of the held-out unit.
    """
    key = {"montage": "eegnet_montage_stage", "transient": "eegnet_transient_stage"}[stage]
    cfg = PROTOCOL[key]
    if seed_offset not in cfg["seed_offsets"]:
        raise ValueError(f"seed_offset must be one of {cfg['seed_offsets']}")
    if split not in cfg["seed_split_extra"]:
        raise ValueError("split must be 'joint' or 'loso'")
    if fold < 0:
        raise ValueError("fold index must be non-negative")
    return int(cfg["seed_base"] + seed_offset + cfg["seed_split_extra"][split] + fold)
