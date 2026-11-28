"""End-to-end helpers that chain the modules for the three analysis steps of the paper.

Step 1 -- montage dependence (:func:`drop_one_bas`): 11 drop-one montages x shrinkage LDA x joint folds, then
          d_F, d_M, d_FM (Eq. 2) per participant.
Step 2 -- transients (:func:`transient_rates`, descriptor features in :func:`build_stack`).
Step 3 -- regimes on transient-free epochs (:func:`eareeg_decoder.regimes.lda_regimes` on a :class:`Stack`).

These helpers return PARTICIPANT-LEVEL values. Summaries for reporting are computed with
:mod:`eareeg_decoder.inference`. Running them on authorized data is an independent re-analysis, not a
reproduction of the values reported in the manuscript (in preparation), unless the data, folds and software versions match exactly.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import montage as mt
from .data import EpochSet
from .features import bin_means, raw_window
from .regimes import ba_by_participant, cv_scores
from .splits import assign_stimulus_groups
from .transients import resolve_theta, thresholded_descriptor, transient_flags

DEFAULT_MONTAGES = ("full", "fpz_free", "in_ear", "fpz_mastoid")


@dataclass
class Stack:
    """Retained-epoch arrays across participants (participant-level data: keep private)."""

    F: dict[str, np.ndarray]     # montage -> binned features (retained epochs)
    y: np.ndarray
    g: np.ndarray                # participant codes of retained epochs
    free: np.ndarray             # transient-free flag of retained epochs
    descriptor: np.ndarray       # 32 thresholded frontal-deflection features
    keys: np.ndarray             # stimulus keys of retained epochs
    sg: np.ndarray               # stimulus group (-1 unknown)
    participants: np.ndarray     # participants included, sorted


def frontal_usable(es: EpochSet, p: int) -> bool:
    """Fpz and at least one mastoid good for participant ``p`` (the Fpz-mastoid derivation exists)."""
    good = set(es.good_channels(p))
    return mt.FPZ in good and any(m in good for m in mt.MASTOIDS)


def build_stack(es: EpochSet, theta: float | str = 100.0, montages: Sequence[str] = DEFAULT_MONTAGES,
                require_frontal: bool = True) -> Stack:
    """Features, labels, participant codes, free flags and descriptor for the retained epochs.

    ``require_frontal=True`` keeps only participants whose Fpz-mastoid derivation exists (the transient and
    follow-up analyses); the montage analyses used every participant (``require_frontal=False``). The transient
    flag uses the whole stored epoch; ``theta='relative'`` scales it per participant over ALL presentations.
    """
    parts = [int(p) for p in es.participants if not require_frontal or frontal_usable(es, int(p))]
    F = {m: [] for m in montages}
    y, g, free, desc, keys = [], [], [], [], []
    for p in parts:
        idx = es.subset_participant(p)
        Xp = np.asarray(es.X)[idx]
        good = es.good_channels(p)
        keep = np.asarray(es.keep)[idx]
        fm = mt.fpz_mastoid(Xp, es.channel_names, good)
        if np.isnan(fm).all():
            # no usable Fpz-mastoid derivation: the transient flag is undefined, so no epoch counts as free and the
            # descriptor is missing (NaN -> training mean); only all-epoch analyses use such participants
            fl_any = np.ones(len(idx), bool)
            d = np.full((int(keep.sum()), 32), np.nan)
        else:
            fl_any = transient_flags(fm, es.times, resolve_theta(fm, theta)).any
            d = thresholded_descriptor(fm[keep], es.times)
        for m in montages:
            F[m].append(bin_means(mt.make_montage(m, Xp[keep], es.channel_names, good), es.times))
        y.append(np.asarray(es.y)[idx][keep])
        g.append(np.full(int(keep.sum()), p))
        free.append(~fl_any[keep])
        desc.append(d)
        keys.append(np.asarray(es.stimulus_key)[idx][keep])
    y_all = np.concatenate(y).astype(int)
    keys_all = np.concatenate(keys)
    sg = assign_stimulus_groups(list(keys_all), y_all == 1)
    for k in set(keys_all.tolist()) - {""}:
        if len(set(sg[keys_all == k].tolist())) != 1:
            raise AssertionError("a stimulus key is split across stimulus groups")
    return Stack(F={m: np.concatenate(v) for m, v in F.items()}, y=y_all, g=np.concatenate(g),
                 free=np.concatenate(free), descriptor=np.concatenate(desc), keys=keys_all, sg=sg,
                 participants=np.array(parts))


def drop_one_bas(es: EpochSet, joint: bool = True) -> dict[str, np.ndarray]:
    """Per-participant LDA BA of the full montage and of the eleven drop-one montages (sorted participants)."""
    configs = ("full",) + mt.DROP_ONE
    S = build_stack(es, montages=configs, require_frontal=False)
    allm = np.ones(len(S.y), bool)
    out = {}
    for m in configs:
        sc = cv_scores(S.F[m], S.y, S.g, allm, allm, S.sg if joint else None, keys=S.keys)
        b = ba_by_participant(S.y, S.g, sc)
        out[m] = np.array([b[k] for k in sorted(b)])
    return out


def transient_rates(es: EpochSet, theta: float | str = 100.0, which: str = "post") -> dict[str, np.ndarray]:
    """Per participant (frontal-usable, sorted): tree rate, face rate and their difference of the chosen flag over
    ALL presentations (not only retained epochs)."""
    rt, rf = [], []
    for p in es.participants:
        p = int(p)
        if not frontal_usable(es, p):
            continue
        idx = es.subset_participant(p)
        fm = mt.fpz_mastoid(np.asarray(es.X)[idx], es.channel_names, es.good_channels(p))
        fl = getattr(transient_flags(fm, es.times, resolve_theta(fm, theta)), which)
        tree = np.asarray(es.y)[idx] == 0
        rt.append(float(fl[tree].mean()))
        rf.append(float(fl[~tree].mean()))
    rt_a, rf_a = np.array(rt), np.array(rf)
    return {"rate_tree": rt_a, "rate_face": rf_a, "difference": rt_a - rf_a}


@dataclass
class RawStack:
    """Raw-window network inputs of the retained epochs (participant-level data: keep private)."""

    X: dict[str, np.ndarray]     # montage -> epochs x channels x samples, window [0, 0.8) s
    y: np.ndarray
    g: np.ndarray
    free: np.ndarray             # transient-free flag (False for every epoch of a participant without Fpz-mastoid)
    keys: np.ndarray
    sg: np.ndarray
    participants: np.ndarray


def build_raw_stack(es: EpochSet, montages: Sequence[str], theta: float | str = 100.0,
                    require_frontal: bool = True, dtype: type = np.float32) -> RawStack:
    """Network / xDAWN inputs: the samples in [0, 0.8) s of each montage for the retained epochs (the decoder panel's
    input; 200 samples at 250 Hz), with labels, participant codes, transient-free flags and stimulus groups exactly as
    :func:`build_stack` (``require_frontal=False`` for the drop-one panel, ``True`` for the transfer-gap panel)."""
    parts = [int(p) for p in es.participants if not require_frontal or frontal_usable(es, int(p))]
    X = {m: [] for m in montages}
    y, g, free, keys = [], [], [], []
    for p in parts:
        idx = es.subset_participant(p)
        Xp = np.asarray(es.X)[idx]
        good = es.good_channels(p)
        keep = np.asarray(es.keep)[idx]
        fm = mt.fpz_mastoid(Xp, es.channel_names, good)
        fl_any = (np.ones(len(idx), bool) if np.isnan(fm).all()
                  else transient_flags(fm, es.times, resolve_theta(fm, theta)).any)
        for m in montages:
            X[m].append(raw_window(mt.make_montage(m, Xp[keep], es.channel_names, good), es.times).astype(dtype))
        y.append(np.asarray(es.y)[idx][keep])
        g.append(np.full(int(keep.sum()), p))
        free.append(np.logical_not(fl_any[keep]))
        keys.append(np.asarray(es.stimulus_key)[idx][keep])
    y_all = np.concatenate(y).astype(int)
    keys_all = np.concatenate(keys)
    sg = assign_stimulus_groups(list(keys_all), is_face=y_all == 1)
    return RawStack(X={m: np.concatenate(v) for m, v in X.items()}, y=y_all, g=np.concatenate(g),
                    free=np.concatenate(free), keys=keys_all, sg=sg, participants=np.array(parts))
