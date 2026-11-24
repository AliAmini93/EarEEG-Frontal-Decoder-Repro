"""Cross-validation splits: joint participant x stimulus-group hold-out and leave-one-participant-out (LOSO).

Joint hold-out (``joint``)
    The stimuli are split once into ``k = 3`` groups (class-stratified; for the original design 12, 12 and 11 face
    identity codes and two of the six tree photographs per group). A fold tests one participant on one stimulus
    group after training on the OTHER participants' epochs from the OTHER groups. Test faces and trees are new to
    the decoder. Epochs without a stimulus group (group < 0) are never used.

Leave-one-participant-out (``loso``)
    Holds out participants only. Every tree photograph is also in the training set, so LOSO results may include
    stimulus-specific information. The two schemes answer different questions and are never pooled.

Participant codes are integers used only to define splits; they never enter a model. Leakage is checked with
assertions that raise :class:`~eareeg_decoder.errors.LeakageError` (never warnings).
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from .config import PROTOCOL
from .errors import LeakageError

SALT: str = PROTOCOL["splits"]["stimulus_group_salt"]
K_GROUPS: int = PROTOCOL["splits"]["stimulus_groups_k"]


@dataclass(frozen=True)
class Fold:
    """One held-out unit. ``group`` is ``None`` for LOSO. ``train``/``test`` are integer index arrays."""

    index: int
    participant: int
    group: int | None
    train: np.ndarray
    test: np.ndarray


def assign_stimulus_groups(keys: Sequence[str | None], is_face: Sequence[bool], k: int = K_GROUPS,
                           salt: str = SALT) -> np.ndarray:
    """Deterministic class-stratified assignment of stimulus keys to ``k`` groups (-1 for missing keys).

    Within each class, the distinct keys are ordered by ``sha256(salt + '|' + key)`` and the i-th key goes to group
    ``i mod k``. Exact agreement with the original groups requires the same key strings (part of the data
    contract); keys must identify a stimulus (face identity code or tree photograph), never a participant.
    """
    keys = list(keys)
    is_face = [bool(f) for f in is_face]
    if len(keys) != len(is_face):
        raise ValueError("keys and is_face differ in length")
    mapping: dict[str, int] = {}
    for cls in (False, True):
        uniq = sorted({kk for kk, f in zip(keys, is_face) if f == cls and kk is not None and kk != ""},
                      key=lambda s: hashlib.sha256((salt + "|" + s).encode()).hexdigest())
        for i, key in enumerate(uniq):
            if key in mapping:
                raise ValueError(f"stimulus key {key!r} occurs in both classes")
            mapping[key] = i % k
    return np.array([mapping.get(kk, -1) if kk not in (None, "") else -1 for kk in keys], dtype=int)


def assert_no_leakage(groups: np.ndarray, train: np.ndarray, test: np.ndarray, stim_groups: np.ndarray | None = None,
                      keys: Sequence[str | None] | None = None) -> None:
    """Raise :class:`LeakageError` if train and test share a participant (always), or, when given, a stimulus
    group or a stimulus key."""
    groups = np.asarray(groups)
    if set(groups[train].tolist()) & set(groups[test].tolist()):
        raise LeakageError("participant leakage across split")
    if stim_groups is not None:
        sg = np.asarray(stim_groups)
        if set(sg[train].tolist()) & set(sg[test].tolist()):
            raise LeakageError("stimulus-group leakage across split")
    if keys is not None:
        kk = np.asarray(keys, dtype=object)
        shared = (set(kk[train].tolist()) & set(kk[test].tolist())) - {None, ""}
        if shared:
            raise LeakageError("stimulus-key leakage across split")


def _has_both_classes(y: np.ndarray | None, idx: np.ndarray) -> bool:
    return y is None or set(np.asarray(y)[idx].tolist()) >= {0, 1}


def loso_folds(groups: np.ndarray, y: np.ndarray | None = None) -> list[Fold]:
    """One fold per participant (sorted codes): test = that participant, train = all others."""
    groups = np.asarray(groups)
    folds = []
    for k, p in enumerate(np.unique(groups)):
        te = np.flatnonzero(groups == p)
        tr = np.flatnonzero(groups != p)
        assert_no_leakage(groups, tr, te)
        if not _has_both_classes(y, tr):
            raise LeakageError("training fold lacks a class")
        folds.append(Fold(k, int(p), None, tr, te))
    return folds


def joint_folds(groups: np.ndarray, stim_groups: np.ndarray, keys: Sequence[str | None] | None = None,
                y: np.ndarray | None = None) -> list[Fold]:
    """Participant x stimulus-group folds in sorted (participant, group) order; empty test cells are skipped.

    ``Fold.index`` counts every (participant, group) pair in that order, including skipped ones (the enumeration
    of the transient-stage network seed rule). The montage-stage code numbered only non-empty folds; the two
    coincide when every participant has test epochs in every group.
    """
    groups, sg = np.asarray(groups), np.asarray(stim_groups)
    if groups.shape != sg.shape:
        raise ValueError("groups and stim_groups differ in shape")
    if keys is not None:
        kk = np.asarray(keys, dtype=object)
        for key in set(kk[sg >= 0].tolist()) - {None, ""}:
            if len(set(sg[kk == key].tolist())) != 1:
                raise LeakageError("a stimulus key is split across stimulus groups")
    folds, idx = [], 0
    for p in np.unique(groups):
        for s in sorted(set(sg[sg >= 0].tolist())):
            te = np.flatnonzero((groups == p) & (sg == s))
            tr = np.flatnonzero((groups != p) & (sg != s) & (sg >= 0))
            if te.size:
                assert_no_leakage(groups, tr, te, sg, keys)
                if not _has_both_classes(y, tr):
                    raise LeakageError("training fold lacks a class")
                folds.append(Fold(idx, int(p), int(s), tr, te))
            idx += 1
    return folds


def masked_folds(groups: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray,
                 stim_groups: np.ndarray | None = None,
                 keys: Sequence[str | None] | None = None) -> Iterator[tuple[int, int | None, np.ndarray, np.ndarray]]:
    """Folds whose training rows are restricted to ``train_mask`` and test rows to ``test_mask``.

    Yields ``(participant, group, train_bool, test_bool)``; LOSO if ``stim_groups`` is None, joint otherwise. This
    is how a decoder is trained on one epoch set and scored on another (e.g. all -> transient-free). With
    ``stim_groups`` and ``keys`` given, joint folds are also checked for shared stimulus keys.
    """
    g = np.asarray(groups)
    train_mask, test_mask = np.asarray(train_mask, bool), np.asarray(test_mask, bool)
    sg = None if stim_groups is None else np.asarray(stim_groups)
    levels: list[int | None] = [None] if sg is None else sorted(set(sg[sg >= 0].tolist()))
    for p in np.unique(g):
        for s in levels:
            te = (g == p) & test_mask
            tr = (g != p) & train_mask
            if s is not None:
                te = te & (sg == s)
                tr = tr & (sg != s) & (sg >= 0)
            if not te.any():
                continue
            assert_no_leakage(g, np.flatnonzero(tr), np.flatnonzero(te), sg, keys if sg is not None else None)
            yield int(p), s, tr, te
