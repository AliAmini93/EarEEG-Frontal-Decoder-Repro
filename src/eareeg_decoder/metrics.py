"""Metrics: balanced accuracy (primary), class recalls, rank AUC, flag agreement (sensitivity, PPV, Cohen's kappa).

Balanced accuracy (BA) is the mean of face recall and tree recall, so chance is 0.5 despite the 16:4 face:tree
ratio of each stream (always answering "face" gives accuracy 0.8 but BA 0.5). A participant's BA is computed over
ALL of that participant's held-out predictions -- pooled over the joint folds that test the participant and, for
networks, averaged over seeds afterwards -- and cohort values are means over participants.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def recalls(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """(face recall, tree recall); NaN for a class that is absent."""
    y, pred = np.asarray(y), np.asarray(pred)
    r1 = float((pred[y == 1] == 1).mean()) if (y == 1).any() else float("nan")
    r0 = float((pred[y == 0] == 0).mean()) if (y == 0).any() else float("nan")
    return r1, r0


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    """Mean of face recall and tree recall (NaN if either class is absent)."""
    r1, r0 = recalls(y, pred)
    return 0.5 * (r1 + r0)


def rank_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve of decision values for class 1 vs class 0 (ties count one half)."""
    y = np.asarray(y)
    s = np.asarray(scores, dtype=float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s, method="average")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def flag_agreement(flag: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Agreement of a binary flag with a binary reference over the same epochs.

    Returns sensitivity (share of reference-positive epochs flagged), positive predictive value (share of flagged
    epochs that are reference-positive) and Cohen's kappa. Used per participant, then averaged.
    """
    a = np.asarray(flag, bool)
    b = np.asarray(reference, bool)
    if a.shape != b.shape:
        raise ValueError("flag and reference differ in shape")
    sens = float(a[b].mean()) if b.any() else float("nan")
    ppv = float(b[a].mean()) if a.any() else float("nan")
    return {"sensitivity": sens, "ppv": ppv, "kappa": cohen_kappa(a, b)}


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa of two binary vectors; NaN when chance agreement is 1."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")
