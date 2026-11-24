"""Train -> test regimes, the frontal advantage A_F (Eq. 3) and the paired gap decomposition (Eq. 4).

A decoder for montage ``m`` is trained on the training participants' epochs in a set ``T`` and scored on the
held-out participant's epochs in a set ``S``. Each set is ``all`` retained epochs, the transient-``free`` ones, or
``matched``: for each participant and class, as many epochs drawn at random from all its retained epochs as it has
free epochs, with ONE draw shared by both montages. The five comparisons are

=====================  ==========  ===========  ==================================================================
name                   train T     test S       question
=====================  ==========  ===========  ==================================================================
``all->all``           all         all          all-epoch baseline
``free->free``         free        free         retrained: is frontal information learnable without transients?
``matched->matched``   matched     matched      same size and class counts, random selection
``all->free``          all         free         transferred: the usual decoder applied UNCHANGED to free epochs
``matched->free``      matched     free         transferred at the retrained decoder's size and class counts
=====================  ==========  ===========  ==================================================================

"Transferred" means the decoder trained on all epochs is applied unchanged; no transfer learning is involved.
``free->free``, ``all->free`` and ``matched->free`` are scored on the identical held-out free epochs, so their
participant-level differences decompose exactly (Eq. 4)::

    Delta_p = A_F,p(free->free) - A_F,p(all->free) = c_p + s_p
    c_p = A_F,p(free->free) - A_F,p(matched->free)        (composition at equal size and class mix)
    s_p = A_F,p(matched->free) - A_F,p(all->free)          (training size at all-epoch composition)

Because free epochs are selected on a quantity measured mostly after onset, subset results describe those epochs
and support no causal statement.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import PROTOCOL
from .decoders import ShrinkageLDA, eegnet_fit_predict, lda_decision
from .metrics import balanced_accuracy, rank_auc
from .splits import assert_no_leakage, masked_folds

_R = PROTOCOL["regimes"]
REGIMES: tuple[str, ...] = tuple(_R["names"])
MIN_FACE: int = _R["eligibility_min_face"]
MIN_TREE: int = _R["eligibility_min_tree"]
N_DRAWS: int = _R["matched_draws_lda"]
SEED_BASE: int = _R["matched_seed_base"]


# ------------------------------------------------------------------------------------------------ scoring
def cv_scores(F: np.ndarray, y: np.ndarray, g: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray,
              sg: np.ndarray | None = None, scaler_mask: np.ndarray | None = None,
              recal_mask: np.ndarray | None = None, keys: Sequence[str | None] | None = None) -> np.ndarray:
    """Held-out shrinkage-LDA decision values (NaN where an epoch was not tested).

    LOSO if ``sg`` is None, joint participant x stimulus-group hold-out otherwise. Diagnostics (training data only):
    ``scaler_mask`` fits the z-scaler on the fold's training participants (and groups) restricted to this mask
    instead of on the LDA training rows; ``recal_mask`` shifts the intercept so that the class-balanced mean
    decision over the fold's training participants restricted to this mask is zero. ``keys`` (stimulus keys)
    adds a stimulus-key leakage check to joint folds.
    """
    y, g = np.asarray(y), np.asarray(g)
    sgv = None if sg is None else np.asarray(sg)
    scores = np.full(len(y), np.nan)
    for p, s, tr, te in masked_folds(g, train_mask, test_mask, sgv, keys):
        if scaler_mask is None and recal_mask is None:
            scores[te] = lda_decision(F[tr], y[tr], F[te])
            continue
        base = (g != p) if s is None else ((g != p) & (sgv != s) & (sgv >= 0))
        clf = ShrinkageLDA().fit(F[tr], y[tr], scaler_rows=None if scaler_mask is None else F[base & scaler_mask])
        dec = clf.decision_function(F[te])
        if recal_mask is not None:
            rm = base & recal_mask
            d_tr = clf.decision_function(F[rm])
            dec = dec - 0.5 * (d_tr[y[rm] == 1].mean() + d_tr[y[rm] == 0].mean())
        scores[te] = dec
    return scores


def _eligible(y: np.ndarray, m: np.ndarray, min_face: int, min_tree: int) -> bool:
    return (y[m] == 1).sum() >= max(1, min_face) and (y[m] == 0).sum() >= max(1, min_tree)


def ba_by_participant(y, g, scores, min_face: int = 0, min_tree: int = 0, subset=None) -> dict[int, float]:
    """Per-participant BA of held-out decision values (threshold 0), pooled over all of the participant's tested
    epochs; NaN unless at least ``min_face`` face and ``min_tree`` tree epochs were scored (at least one each)."""
    y, g, scores = np.asarray(y), np.asarray(g), np.asarray(scores, dtype=float)
    out = {}
    for p in np.unique(g):
        m = (g == p) & np.isfinite(scores)
        if subset is not None:
            m = m & np.asarray(subset, bool)
        out[int(p)] = (balanced_accuracy(y[m], (scores[m] > 0).astype(int))
                       if _eligible(y, m, min_face, min_tree) else float("nan"))
    return out


def recalls_by_participant(y, g, scores, min_face: int = 0, min_tree: int = 0, subset=None):
    """Per-participant (face recall, tree recall) dictionaries with the same eligibility rule as BA."""
    y, g, scores = np.asarray(y), np.asarray(g), np.asarray(scores, dtype=float)
    face, tree = {}, {}
    for p in np.unique(g):
        m = (g == p) & np.isfinite(scores)
        if subset is not None:
            m = m & np.asarray(subset, bool)
        ok = _eligible(y, m, min_face, min_tree)
        face[int(p)] = float((scores[m & (y == 1)] > 0).mean()) if ok else float("nan")
        tree[int(p)] = float((scores[m & (y == 0)] <= 0).mean()) if ok else float("nan")
    return face, tree


def auc_by_participant(y, g, scores, min_face: int = 0, min_tree: int = 0, subset=None) -> dict[int, float]:
    y, g, scores = np.asarray(y), np.asarray(g), np.asarray(scores, dtype=float)
    out = {}
    for p in np.unique(g):
        m = (g == p) & np.isfinite(scores)
        if subset is not None:
            m = m & np.asarray(subset, bool)
        out[int(p)] = rank_auc(y[m], scores[m]) if _eligible(y, m, min_face, min_tree) else float("nan")
    return out


def vector(d: Mapping[int, float]) -> np.ndarray:
    """Values of a per-participant dict in sorted participant order."""
    return np.array([d[k] for k in sorted(d)], dtype=float)


def frontal_advantage(ba_full: Mapping[int, float], ba_fpz_free: Mapping[int, float]) -> np.ndarray:
    """Eq. (3), participant level: ``A_F,p = BA_p(full) - BA_p(Fpz-free)`` (same participants required)."""
    if sorted(ba_full) != sorted(ba_fpz_free):
        raise ValueError("frontal advantage needs the same participants for both montages")
    return vector(ba_full) - vector(ba_fpz_free)


# ------------------------------------------------------------------------------------------------ draws
def matched_random_mask(free: np.ndarray, y: np.ndarray, g: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Matched draw: per participant (sorted) and class (tree 0, then face 1), as many epochs drawn without
    replacement from ALL that participant's retained epochs of that class as it has free epochs.

    The selection uses only class counts and the random stream, never signal values or outcomes. Use the same mask
    for both montages. Draw ``k`` of the original analyses used ``np.random.default_rng(20261111 + k)``.
    """
    free, y, g = np.asarray(free, bool), np.asarray(y), np.asarray(g)
    out = np.zeros(len(y), bool)
    for p in np.unique(g):
        for c in (0, 1):
            pool = np.flatnonzero((g == p) & (y == c))
            n = int(free[pool].sum())
            if n:
                out[rng.choice(pool, size=n, replace=False)] = True
    return out


def gap_decomposition(a_free_free: np.ndarray, a_matched_free: np.ndarray, a_all_free: np.ndarray) -> dict[str, Any]:
    """Eq. (4): ``Delta = c + s`` per participant, with the maximum absolute identity error (should be ~1e-16)."""
    a_ff, a_mf, a_af = (np.asarray(a, dtype=float) for a in (a_free_free, a_matched_free, a_all_free))
    delta, c, s = a_ff - a_af, a_ff - a_mf, a_mf - a_af
    err = np.abs(delta - (c + s))
    return {"delta": delta, "c": c, "s": s, "max_abs_identity_error": float(np.nanmax(err)) if err.size else 0.0}


# ------------------------------------------------------------------------------------------------ LDA regime set
@dataclass
class RegimeResult:
    """Per-participant results (sorted participant order) of :func:`lda_regimes`. Participant-level: keep private."""

    participants: np.ndarray
    ba: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)          # regime -> montage -> BA
    advantage: dict[str, np.ndarray] = field(default_factory=dict)              # regime -> A_F
    matched_free_by_draw: np.ndarray | None = None                              # participants x draws of A_F
    decomposition: dict[str, Any] = field(default_factory=dict)                 # delta, c, s
    test_sets_identical: bool = False


def lda_regimes(F: Mapping[str, np.ndarray], y: np.ndarray, g: np.ndarray, free: np.ndarray, sg: np.ndarray | None = None,
                *, montages: Sequence[str] = ("full", "fpz_free"), n_draws: int = N_DRAWS, seed_base: int = SEED_BASE,
                min_face: int = MIN_FACE, min_tree: int = MIN_TREE,
                keys: Sequence[str | None] | None = None) -> RegimeResult:
    """Run the five regimes with shrinkage LDA for the two montages and decompose the paired gap.

    ``F[montage]`` are binned features of the retained epochs; ``free`` flags transient-free retained epochs.
    Eligibility (>= ``min_face`` face and >= ``min_tree`` tree test epochs) applies to every regime scored on
    free or matched epochs; ``all->all`` only requires both classes. Matched draws ``k = 0..n_draws-1`` use
    ``default_rng(seed_base + k)``; ``matched->matched`` BA is averaged over draws within participant.
    Matched test sets have the same per-class counts as the free ones, so eligibility is identical to the original
    rule (eligibility of the free full-montage set). The original's optional extension to 50 draws when the Monte
    Carlo SE of the composition contrast exceeds 0.0025 is not automated: check ``inference.draw_uncertainty``.
    """
    full, nofpz = montages
    y, g, free = np.asarray(y), np.asarray(g), np.asarray(free, bool)
    allm = np.ones(len(y), bool)
    res = RegimeResult(participants=np.unique(g))

    def run(train, test, mf, mt):
        sc = {m: cv_scores(F[m], y, g, train, test, sg, keys=keys) for m in (full, nofpz)}
        return sc, {m: ba_by_participant(y, g, sc[m], mf, mt) for m in (full, nofpz)}

    sc_aa, b_aa = run(allm, allm, 0, 0)
    sc_ff, b_ff = run(free, free, min_face, min_tree)
    sc_af, b_af = run(allm, free, min_face, min_tree)
    mm_draws, mf_draws, mm_ba = [], [], {full: [], nofpz: []}
    same_test = all(np.array_equal(np.isfinite(sc_ff[m]), np.isfinite(sc_af[m])) for m in (full, nofpz))
    for k in range(n_draws):
        mask = matched_random_mask(free, y, g, np.random.default_rng(seed_base + k))
        sc_mf, b_mf = run(mask, free, min_face, min_tree)
        same_test &= all(np.array_equal(np.isfinite(sc_mf[m]), np.isfinite(sc_ff[m])) for m in (full, nofpz))
        mf_draws.append(frontal_advantage(b_mf[full], b_mf[nofpz]))
        _, b_mm = run(mask, mask, min_face, min_tree)
        for m in (full, nofpz):
            mm_ba[m].append(vector(b_mm[m]))
        mm_draws.append(frontal_advantage(b_mm[full], b_mm[nofpz]))
    for name, b in (("all->all", b_aa), ("free->free", b_ff), ("all->free", b_af)):
        res.ba[name] = {m: vector(b[m]) for m in (full, nofpz)}
        res.advantage[name] = frontal_advantage(b[full], b[nofpz])
    if n_draws:
        with np.errstate(invalid="ignore"):
            res.ba["matched->matched"] = {m: np.nanmean(np.array(mm_ba[m]), axis=0) for m in (full, nofpz)}
        res.advantage["matched->matched"] = res.ba["matched->matched"][full] - res.ba["matched->matched"][nofpz]
        res.matched_free_by_draw = np.array(mf_draws).T
        res.advantage["matched->free"] = res.matched_free_by_draw.mean(axis=1)
        res.decomposition = gap_decomposition(res.advantage["free->free"], res.advantage["matched->free"],
                                              res.advantage["all->free"])
    res.test_sets_identical = bool(same_test)
    return res


# ------------------------------------------------------------------------------------------------ networks
FitFn = Callable[[np.ndarray, np.ndarray, np.ndarray, int, dict, str], np.ndarray]


def net_probs(X: np.ndarray, y: np.ndarray, g: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray, seed0: int,
              cfg: dict, *, sg: np.ndarray | None = None, device: str = "cpu", min_face: int = 0, min_tree: int = 0,
              fit: FitFn = eegnet_fit_predict) -> np.ndarray:
    """Held-out P(face) of the network (NaN where untested), seed per fit = ``seed0 + fold index``.

    LOSO (``sg`` None): fold index = position of the participant in sorted order; a participant with fewer than
    ``min_face``/``min_tree`` test epochs is skipped but still consumes its index. Joint: the index counts every
    (participant, group) pair in sorted order, including pairs with no test epochs.
    """
    y, g = np.asarray(y), np.asarray(g)
    train_mask, test_mask = np.asarray(train_mask, bool), np.asarray(test_mask, bool)
    prob = np.full(len(y), np.nan)
    if sg is None:
        for fold, p in enumerate(np.unique(g)):
            te = (g == p) & test_mask
            tr = (g != p) & train_mask
            if (y[te] == 1).sum() < min_face or (y[te] == 0).sum() < min_tree or not te.any():
                continue
            assert_no_leakage(g, np.flatnonzero(tr), np.flatnonzero(te))
            prob[te] = fit(X[tr], y[tr], X[te], seed0 + fold, cfg, device)
        return prob
    sgv = np.asarray(sg)
    fold = 0
    for p in np.unique(g):
        for s in sorted(set(sgv[sgv >= 0].tolist())):
            te = (g == p) & (sgv == s) & test_mask
            tr = (g != p) & (sgv != s) & (sgv >= 0) & train_mask
            if te.any():
                assert_no_leakage(g, np.flatnonzero(tr), np.flatnonzero(te), sgv)
                prob[te] = fit(X[tr], y[tr], X[te], seed0 + fold, cfg, device)
            fold += 1
    return prob


def probs_to_scores(prob: np.ndarray) -> np.ndarray:
    """P(face) -> decision value with the 0.5 threshold at 0, so BA/AUC helpers can be shared with LDA."""
    return np.asarray(prob, dtype=float) - 0.5
