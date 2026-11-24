"""Participant-level inference: percentile bootstrap, exact sign-flip test, Holm, test-inversion interval.

Participants are the unit of inference. Every tested contrast is a mean over participants of within-participant
values; matched draws and network seeds are averaged within participant BEFORE any inference, never treated as
extra observations. Intervals of a paired contrast are computed from the paired per-participant values, never by
combining two separate intervals.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .montage import FPZ, IN_EAR, MASTOIDS

MAX_EXACT_N = 24


def finite(values: Sequence[float] | np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float).ravel()
    return v[np.isfinite(v)]


def bootstrap_ci(values: Sequence[float] | np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling participants with replacement (non-finite values dropped).

    Draws ``default_rng(seed).integers(0, n, size=(n_boot, n))`` -- the same stream as the original code.
    """
    v = finite(values)
    if v.size == 0:
        raise ValueError("no finite values")
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(np.percentile(m, 100 * alpha / 2)), float(np.percentile(m, 100 * (1 - alpha / 2)))


def twoway_bootstrap_ci(cells: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """Sensitivity bootstrap over participants (rows) AND stimulus groups (columns), resampled independently."""
    C = np.asarray(cells, dtype=float)
    P, G = C.shape
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        r = rng.integers(0, P, P)
        c = rng.integers(0, G, G)
        out[b] = C[np.ix_(r, c)].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def signflip_exact_p(values: Sequence[float] | np.ndarray, *, atol: float = 1e-12, rtol: float = 0.0,
                     max_n: int = MAX_EXACT_N, chunk: int = 1 << 15) -> float:
    """Exact two-sided sign-flip p of ``mean(values) = 0`` over all ``2^n`` sign patterns.

    p = share of patterns with ``|mean(signs * v)| >= |mean(v)| - atol - rtol * max(|mean(v)|, 1e-12)``.
    The transient / follow-up code used ``atol=1e-12, rtol=0`` (default); the montage code used ``atol=0,
    rtol=1e-9``. The tolerance only decides floating-point ties. Non-finite values are dropped.
    """
    d = finite(values)
    n = d.size
    if n == 0:
        return float("nan")
    if n > max_n:
        raise ValueError(f"exact enumeration limited to n <= {max_n}")
    obs = abs(d.mean())
    thr = obs - atol - rtol * max(obs, 1e-12)
    total = 1 << n
    bits = np.arange(n, dtype=np.int64)
    count = 0
    for start in range(0, total, chunk):
        idx = np.arange(start, min(total, start + chunk), dtype=np.int64)
        signs = 1.0 - 2.0 * ((idx[:, None] >> bits) & 1)
        count += int(np.sum(np.abs(signs @ d) / n >= thr))
    return count / total


def signflip_inversion_ci(values: Sequence[float] | np.ndarray, alpha: float = 0.05,
                          iterations: int = 60) -> tuple[float, float]:
    """Exact sign-flip test-inversion interval: the shifts ``mu`` with ``p(values - mu) >= alpha``.

    Each edge is found by bisection between the mean (accepted) and the mean +/- the range of the values
    (rejected), ``iterations`` halvings.
    """
    d = finite(values)
    if d.size == 0:
        return float("nan"), float("nan")
    if d.size > MAX_EXACT_N:
        raise ValueError(f"exact sign-flip inversion limited to n <= {MAX_EXACT_N}")
    m = d.mean()
    span = np.ptp(d) + 1e-9

    def edge(direction: float) -> float:
        lo, hi = m, m + direction * span
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            if signflip_exact_p(d - mid) >= alpha:
                lo = mid
            else:
                hi = mid
        return float(lo)

    return edge(-1.0), edge(1.0)


def holm(pvalues: Sequence[float] | np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p values in input order; NaN entries stay NaN and are not counted in m."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    idx = [i for i in np.argsort(p) if np.isfinite(p[i])]
    m = len(idx)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def paired_summary(values: Sequence[float] | np.ndarray, n_boot: int, seed: int, test: bool = True,
                   null: float = 0.0) -> dict[str, Any]:
    """Mean and percentile bootstrap CI of participant values, plus counts above/below ``null``, SD,
    ``d_z = (mean - null) / SD`` and (optionally) the exact sign-flip p of ``values - null``.

    Use ``null=0`` for paired contrasts and ``null=0.5`` for balanced accuracies. The exact test enumerates
    ``2^n`` patterns and is limited to ``n <= 24``; above that ``p_signflip`` is ``None`` with a note.
    """
    v = finite(values)
    if v.size == 0:
        return {"n": 0, "null": null}
    lo, hi = bootstrap_ci(v, n_boot, seed)
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    out: dict[str, Any] = {"n": int(v.size), "mean": float(v.mean()), "ci95": [lo, hi], "null": null,
                           "n_above_null": int((v > null).sum()), "n_below_null": int((v < null).sum()), "sd": sd,
                           "dz": float((v.mean() - null) / sd) if np.isfinite(sd) and sd > 0 else None}
    if test:
        if v.size <= MAX_EXACT_N:
            out["p_signflip"] = signflip_exact_p(v - null)
        else:
            out["p_signflip"] = None
            out["p_note"] = f"exact sign-flip enumeration limited to n <= {MAX_EXACT_N}"
    return out


def decision_label(summary: Mapping[str, Any], p_holm: float | None, alpha: float = 0.05) -> str:
    """Per-test label rule of the follow-up plan.

    ``SUPPORTED_EXPLORATORY`` if the mean is positive, the bootstrap CI excludes 0 and the Holm-adjusted exact p is
    below ``alpha``; ``CONTRADICTED`` if the CI lies wholly below 0; otherwise ``INCONCLUSIVE`` (a non-detection
    is not equivalence).
    """
    lo, hi = summary["ci95"]
    if summary["mean"] > 0 and lo > 0 and p_holm is not None and p_holm < alpha:
        return "SUPPORTED_EXPLORATORY"
    if summary["mean"] < 0 and hi < 0:
        return "CONTRADICTED"
    return "INCONCLUSIVE"


def draw_uncertainty(mat: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """``mat``: participants x draws of a per-participant contrast.

    Returns the per-participant mean over draws (the inference unit) and the Monte Carlo SE of the cohort mean
    (SD over draws of the per-draw cohort mean / sqrt(number of draws)).
    """
    mat = np.asarray(mat, dtype=float)
    per_p = np.nanmean(mat, axis=1)
    per_draw = np.nanmean(mat, axis=0)
    D = mat.shape[1]
    mcse = float(np.std(per_draw, ddof=1) / np.sqrt(D)) if D > 1 else float("nan")
    return per_p, {"n_draws": int(D), "mc_se_cohort_mean": mcse}


def montage_contrasts(ba: Mapping[str, Sequence[float]]) -> dict[str, np.ndarray]:
    """Eq. (2) and its companions from per-participant drop-one BAs (same participant order in every entry).

    ``d_F  = BA(-Fpz) - mean_e BA(-e)``  over the eight in-ear electrodes e;
    ``d_M  = mean_m BA(-m) - mean_e BA(-e)``  over the two mastoids m;
    ``d_FM = BA(-Fpz) - mean_m BA(-m)``.
    Negative values mean that removing the first-named site costs more. The eight in-ear removals are averaged
    within each participant and are never treated as separate tests.
    """
    f = np.asarray(ba[f"drop_{FPZ}"], dtype=float)
    m = np.mean([np.asarray(ba[f"drop_{c}"], dtype=float) for c in MASTOIDS], axis=0)
    e = np.mean([np.asarray(ba[f"drop_{c}"], dtype=float) for c in IN_EAR], axis=0)
    return {"d_F": f - e, "d_M": m - e, "d_FM": f - m}
