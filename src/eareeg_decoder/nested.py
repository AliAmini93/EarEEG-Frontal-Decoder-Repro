"""Source-only nested tuning sensitivity of the transfer gap (candidate v0.2.0).

For every outer joint fold (held-out participant ``p``, held-out stimulus group ``s``) and montage, one configuration
per decoder is chosen with OUTER-SOURCE data only (participants != p, stimulus groups != s) and ALL retained source
epochs, then used unchanged for both outer training sets (free -> free and all -> free) on the identical held-out free
epochs. The outer participant and stimulus group never enter selection; selection does see transient-bearing source
epochs even for free-epoch training, so this is a sensitivity analysis, not a free-only pipeline.

Inner splits (fixed before any outcome; ``configs/protocol.json`` -> ``panel.nested``):

* the outer-source participants are ordered by ``SHA-256(salt + '|' + <participant code>)`` (label- and outcome-blind)
  and dealt round-robin into three subsets H_0, H_1, H_2;
* with the two remaining stimulus groups ``R = sorted({0, 1, 2} - {s})``, split k holds out ``H_k`` AND group
  ``R[k mod 2]`` and trains on the other source participants AND group ``R[1 - k mod 2]``;
* a held-out inner participant is scored with >= 10 face and >= 5 tree inner-test epochs; a split is valid with >= 50
  training epochs per class and >= 3 scored participants; an invalid split blocks the analysis (no fallback);
* inner score = mean over the three splits of the mean participant BA (P(face) > 0.5); one fixed inner seed;
* selection = highest inner score; exact ties (|difference| <= 1e-12) go to the earlier configuration of the grid,
  which lists the default first, then lower complexity / stronger regularisation.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .config import PROTOCOL
from .errors import LeakageError
from .regimes import check_fold_probabilities

_N = PROTOCOL["panel"]["nested"]
INNER_SEED: int = _N["inner_seed"]
SALT: str = _N["salt"]
MIN_FACE_INNER: int = _N["min_face_inner"]
MIN_TREE_INNER: int = _N["min_tree_inner"]
MIN_TRAIN_PER_CLASS: int = _N["min_train_per_class"]
MIN_SCORED: int = _N["min_scored"]
TIE: float = _N["tie_tolerance"]
GRIDS: dict[str, list[dict[str, Any]]] = {k: [dict(c) for c in v] for k, v in _N["grids"].items()}


class InfeasibleSplitError(RuntimeError):
    """The frozen inner-split eligibility rules cannot be met."""


def participant_subsets(source: Sequence[int], salt: str = SALT) -> list[list[int]]:
    order = sorted((int(q) for q in source), key=lambda q: hashlib.sha256(f"{salt}|{q}".encode()).hexdigest())
    return [sorted(order[k::3]) for k in range(3)]


def inner_splits(g, sg, p: int, s: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Three (train mask, test mask) pairs for outer joint fold (p, s); masks over all epochs."""
    g, sg = np.asarray(g), np.asarray(sg)
    groups = sorted(set(sg[sg >= 0].tolist()))
    if len(groups) != 3:
        raise ValueError("the nested design needs exactly three stimulus groups")
    Q = [int(q) for q in np.unique(g) if q != p]
    R = sorted(set(groups) - {int(s)})
    out = []
    for k, H in enumerate(participant_subsets(Q)):
        h, other = R[k % 2], R[1 - (k % 2)]
        inH = np.isin(g, H)
        src = (g != p) & np.isin(sg, R)
        tr = src & ~inH & (sg == other)
        te = src & inH & (sg == h)
        if (set(g[tr].tolist()) & set(g[te].tolist())) or (set(sg[tr].tolist()) & set(sg[te].tolist())) \
                or p in set(g[tr].tolist()) | set(g[te].tolist()) or s in set(sg[tr].tolist()) | set(sg[te].tolist()):
            raise LeakageError("inner split leaks the outer or inner held-out unit")
        out.append((tr, te))
    return out


def split_feasibility(y, g, tr, te) -> dict[str, Any]:
    y, g = np.asarray(y), np.asarray(g)
    ntr = [int(((y == c) & tr).sum()) for c in (0, 1)]
    scored = [q for q in np.unique(g[te]) if ((g == q) & te & (y == 1)).sum() >= MIN_FACE_INNER
              and ((g == q) & te & (y == 0)).sum() >= MIN_TREE_INNER]
    return {"n_train_tree": ntr[0], "n_train_face": ntr[1], "n_scored": len(scored),
            "valid": min(ntr) >= MIN_TRAIN_PER_CLASS and len(scored) >= MIN_SCORED}


def inner_score(y, g, te, prob) -> float:
    """Mean participant BA over scored inner-test participants (face if P > 0.5)."""
    y, g, prob = np.asarray(y), np.asarray(g), np.asarray(prob, dtype=float)
    bas = []
    for q in np.unique(g[te]):
        m = (g == q) & te
        if (y[m] == 1).sum() < MIN_FACE_INNER or (y[m] == 0).sum() < MIN_TREE_INNER:
            continue
        pr = prob[m] > 0.5
        bas.append(0.5 * (pr[y[m] == 1].mean() + (~pr[y[m] == 0]).mean()))
    if not bas:
        raise ValueError("no scored inner participant")
    return float(np.mean(bas))


def select(scores: Sequence[float], tie: float = TIE) -> int:
    """Index of the highest inner score; ties within ``tie`` go to the earliest configuration."""
    if len(scores) == 0 or any(not np.isfinite(x) for x in scores):
        raise ValueError("non-finite inner score")
    best = 0
    for i in range(1, len(scores)):
        if scores[i] > scores[best] + tie:
            best = i
    return best


def outer_joint_folds(g, sg) -> list[tuple[int, int, int]]:
    """(fold index, participant, group) in the joint enumeration (index advanced for every pair)."""
    g, sg = np.asarray(g), np.asarray(sg)
    out, fold = [], 0
    for p in np.unique(g):
        for s in sorted(set(sg[sg >= 0].tolist())):
            if ((g == p) & (sg == s)).any():
                out.append((fold, int(p), int(s)))
            fold += 1
    return out


def choose_configurations(X, y, g, sg, fit: Callable, grid: Sequence[Mapping[str, Any]], base_cfg: Mapping[str, Any],
                          *, device: str = "cpu", seed: int = INNER_SEED) -> dict[tuple[int, int], dict[str, Any]]:
    """Per outer joint fold: the chosen configuration index and the inner scores (training epochs only)."""
    y, g, sg = np.asarray(y), np.asarray(g), np.asarray(sg)
    out = {}
    for fold, p, s in outer_joint_folds(g, sg):
        splits = inner_splits(g, sg, p, s)
        feas = [split_feasibility(y, g, tr, te) for tr, te in splits]
        if not all(f["valid"] for f in feas):
            raise InfeasibleSplitError(f"BLOCKED: infeasible inner split in outer fold {fold}")
        scores = []
        for c in grid:
            cfg = {**base_cfg, **c}
            per = []
            for tr, te in splits:
                prob = np.full(len(y), np.nan)
                pred = check_fold_probabilities(fit(X[tr], y[tr], X[te], seed, cfg, device), int(te.sum()))
                if not ((pred >= 0) & (pred <= 1)).all():
                    raise FloatingPointError("invalid inner validation probabilities")
                prob[te] = pred
                per.append(inner_score(y, g, te, prob))
            scores.append(float(np.mean(per)))
        out[(p, s)] = {"fold": fold, "index": select(scores), "scores": scores}
    return out


def tuned_probs(X, y, g, sg, train_mask, test_mask, fit: Callable, grid, base_cfg, choice, seed0: int,
                *, device: str = "cpu") -> np.ndarray:
    """Outer joint predictions with the configuration chosen for each outer fold (seed = ``seed0 + fold index``)."""
    y, g, sg = np.asarray(y), np.asarray(g), np.asarray(sg)
    train_mask, test_mask = np.asarray(train_mask, bool), np.asarray(test_mask, bool)
    prob = np.full(len(y), np.nan)
    for fold, p, s in outer_joint_folds(g, sg):
        te = (g == p) & (sg == s) & test_mask
        tr = (g != p) & (sg != s) & (sg >= 0) & train_mask
        if not te.any():
            continue
        cfg = {**base_cfg, **grid[choice[(p, s)]["index"]]}
        prob[te] = check_fold_probabilities(fit(X[tr], y[tr], X[te], seed0 + fold, cfg, device), int(te.sum()))
    return prob
