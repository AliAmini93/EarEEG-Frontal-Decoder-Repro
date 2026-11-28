"""xDAWN-augmented covariance + Riemannian tangent space + logistic regression (xDAWN-TS-LR).

Implemented from the published method descriptions (Rivet et al., 2009; Barachant et al., 2012; Barachant & Congedo,
2014) and the documented semantics of pyRiemann's ``XdawnCovariances(nfilter, applyfilters=True, classes=[0, 1],
estimator='oas', xdawn_estimator='oas')`` followed by ``TangentSpace(metric='riemann', tsupdate=False)``, using only
NumPy, SciPy and scikit-learn. pyRiemann is not a dependency and no third-party code is copied. Everything below is
fitted on TRAINING epochs only; test epochs are only transformed.

1. Per-channel standardisation with mean/SD over training epochs and samples; NaN (bad channels) -> 0 (= training
   mean).
2. Signal covariance ``C_x`` = OAS(all training epochs concatenated in time), C x C.
3. Per class k in (0, 1): evoked ``P_k`` = mean of the class's training epochs; ``C_k`` = OAS(P_k); generalised
   eigenproblem ``C_k v = lambda C_x v`` (``scipy.linalg.eigh``); eigenvectors sorted by decreasing ``lambda``,
   scaled to unit Euclidean norm; filters ``W_k`` = first ``nfilter``; prototype ``E_k = W_k P_k``.
4. Every epoch Z -> augmented signal ``[E_0; E_1; W_0 Z; W_1 Z]`` (4 nfilter x T) -> ``Sigma`` = OAS(augmented).
5. Reference = affine-invariant Riemannian mean of the TRAINING ``Sigma`` (arithmetic initialisation, step size
   adaptation 0.95 / 0.5, tolerance 1e-8, at most 50 iterations); tangent vector ``upper(logm(R^-1/2 Sigma R^-1/2))``
   with weight 1 on the diagonal and sqrt(2) off the diagonal; no reference update on test data.
6. Standardisation of the tangent vectors (training statistics) and
   ``LogisticRegression(C, class_weight='balanced', solver='lbfgs', max_iter=2000)``; output P(face).

Known property of the fixed configuration (reported, never corrected): when a montage is referenced to its own mean,
``C_x`` has a near-null common-mode direction and the leading filter of each class aligns with it, leaving in effect
``nfilter - 1`` informative filters per class. :func:`fit_predict_diag` reports this alignment and numerical gates
(SPD covariances, finite features and probabilities, convergence) for every fit.

The function :func:`xdawn_fit_predict` has the network signature ``(Xtr, ytr, Xte, seed, cfg, device)`` so it can be
passed to :func:`eareeg_decoder.regimes.net_probs`; ``seed`` and ``device`` are ignored (the method is
deterministic and runs on the CPU).
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy import linalg
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import PROTOCOL

DEFAULT: dict[str, Any] = {k: PROTOCOL["panel"]["xdawn_ts_lr"][k] for k in ("nfilter", "C")}


# ------------------------------------------------------------------------------------------------ covariance
def oas_batch(A: np.ndarray) -> np.ndarray:
    """OAS covariance of each ``rows x samples`` matrix in ``A`` (n x C x T); the formula of
    ``sklearn.covariance.oas`` (rows centred over samples, shrinkage towards trace/C * I)."""
    A = np.asarray(A, np.float64)
    _, c, t = A.shape
    Ac = A - A.mean(axis=2, keepdims=True)
    S = np.einsum("nct,ndt->ncd", Ac, Ac) / t
    mu = np.trace(S, axis1=1, axis2=2) / c
    alpha = np.mean(S ** 2, axis=(1, 2))
    num = alpha + mu ** 2
    den = (t + 1.0) * (alpha - mu ** 2 / c)
    with np.errstate(divide="ignore", invalid="ignore"):
        shrink = np.where(den == 0, 1.0, np.minimum(num / np.where(den == 0, 1.0, den), 1.0))
    out = (1.0 - shrink)[:, None, None] * S
    idx = np.arange(c)
    out[:, idx, idx] += (shrink * mu)[:, None]
    return out


def oas_single(M: np.ndarray) -> np.ndarray:
    return oas_batch(np.asarray(M, np.float64)[None])[0]


# ------------------------------------------------------------------------------------------------ SPD functions
def _eig_fun(C: np.ndarray, fun) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    return np.einsum("...ij,...j,...kj->...ik", V, fun(w), V)


def sqrtm(C):
    return _eig_fun(C, np.sqrt)


def invsqrtm(C):
    return _eig_fun(C, lambda w: 1.0 / np.sqrt(w))


def logm(C):
    return _eig_fun(C, np.log)


def expm_sym(C):
    return _eig_fun(C, np.exp)


def mean_riemann(covs: np.ndarray, tol: float = 10e-9, maxiter: int = 50) -> tuple[np.ndarray, int, bool]:
    """Affine-invariant Riemannian mean of SPD matrices. Returns ``(mean, n_iter, converged)``."""
    C = covs.mean(axis=0)
    k, nu = 0, 1.0
    tau = np.finfo(np.float64).max
    crit = np.finfo(np.float64).max
    while crit > tol and k < maxiter and nu > tol:
        k += 1
        C12, Cm12 = sqrtm(C), invsqrtm(C)
        J = logm(Cm12 @ covs @ Cm12).mean(axis=0)
        crit = float(np.linalg.norm(J, ord="fro"))
        h = nu * crit
        C = C12 @ expm_sym(nu * J) @ C12
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu
    return C, k, bool(crit <= tol or nu <= tol)


def upper(M: np.ndarray) -> np.ndarray:
    """Upper triangle with weight 1 on the diagonal and sqrt(2) off it (so the Euclidean norm equals Frobenius)."""
    c = M.shape[-1]
    iu = np.triu_indices(c)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return M[..., iu[0], iu[1]] * coef


def tangent_space(covs: np.ndarray, Cref: np.ndarray) -> np.ndarray:
    Cm12 = invsqrtm(Cref)
    return upper(logm(Cm12 @ covs @ Cm12))


def min_eig(C: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(C).min(axis=-1)


# ------------------------------------------------------------------------------------------------ scaler
def channel_scaler(Xtr: np.ndarray):
    """Per-channel z-score fitted on training epochs (mean/SD over epochs and samples); NaN -> 0."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(Xtr, axis=(0, 2), keepdims=True)
        sd = np.nanstd(Xtr, axis=(0, 2), keepdims=True)
    mu[~np.isfinite(mu)] = 0.0
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0

    def apply(X):
        return np.nan_to_num((np.asarray(X, np.float64) - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    return apply


# ------------------------------------------------------------------------------------------------ xDAWN
class Xdawn:
    """Class-wise xDAWN spatial filters fitted on training epochs (``Z``: epochs x channels x samples)."""

    def __init__(self, nfilter: int = 4, classes: tuple[int, ...] = (0, 1)) -> None:
        self.nfilter = int(nfilter)
        self.classes = tuple(classes)

    def fit(self, Z: np.ndarray, y: np.ndarray) -> Xdawn:
        n, c, t = Z.shape
        if self.nfilter > c:
            raise ValueError("nfilter exceeds the channel count")
        y = np.asarray(y)
        self.Cx_ = oas_single(np.transpose(Z, (1, 2, 0)).reshape(c, t * n))
        filters, evokeds, evals = [], [], []
        for k in self.classes:
            if not np.any(y == k):
                raise ValueError("a class is absent from the training data")
            P = Z[y == k].mean(axis=0)
            w, V = linalg.eigh(oas_single(P), self.Cx_)
            order = np.argsort(w)[::-1]
            V = V[:, order]
            V = V / np.linalg.norm(V, axis=0, keepdims=True)
            W = V[:, :self.nfilter].T
            filters.append(W)
            evokeds.append(W @ P)
            evals.append(w[order][:self.nfilter])
        self.filters_ = np.concatenate(filters, axis=0)
        self.evokeds_ = np.concatenate(evokeds, axis=0)
        self.evals_ = np.concatenate(evals)
        return self

    def augmented_cov(self, Z: np.ndarray) -> np.ndarray:
        F = np.einsum("fc,nct->nft", self.filters_, Z)
        P = np.broadcast_to(self.evokeds_, (Z.shape[0],) + self.evokeds_.shape)
        return oas_batch(np.concatenate([P, F], axis=1))

    def null_alignment(self) -> tuple[float, float]:
        """Largest |cosine| between a selected filter and the eigenvector of the smallest eigenvalue of ``C_x``,
        and the ratio of smallest to largest eigenvalue (descriptive diagnostics)."""
        w, V = np.linalg.eigh(self.Cx_)
        Wn = self.filters_ / np.linalg.norm(self.filters_, axis=1, keepdims=True)
        return float(np.max(np.abs(Wn @ V[:, 0]))), float(w[0] / w[-1])


# ------------------------------------------------------------------------------------------------ pipeline
def fit_predict_diag(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
                     cfg: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """P(face) for ``Xte`` and a diagnostics dict (numerical gates; ``gate_ok`` summarises them)."""
    cfg = dict(DEFAULT if cfg is None else cfg)
    ytr = np.asarray(ytr, np.int64)
    if set(np.unique(ytr).tolist()) != {0, 1}:
        raise ValueError("training labels must contain both classes 0 (tree) and 1 (face)")
    scale = channel_scaler(Xtr)
    Ztr, Zte = scale(Xtr), scale(Xte)
    xd = Xdawn(nfilter=int(cfg["nfilter"])).fit(Ztr, ytr)
    Str, Ste = xd.augmented_cov(Ztr), xd.augmented_cov(Zte)
    Cref, n_iter, conv = mean_riemann(Str)
    Ttr, Tte = tangent_space(Str, Cref), tangent_space(Ste, Cref)
    sc = StandardScaler().fit(Ttr)
    clf = LogisticRegression(C=float(cfg["C"]), class_weight="balanced", solver="lbfgs", max_iter=2000)
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(sc.transform(Ttr), ytr)
    lr_conv = not any(issubclass(w.category, ConvergenceWarning) for w in wlist) and int(np.max(clf.n_iter_)) < 2000
    prob = clf.predict_proba(sc.transform(Tte))[:, 1] if len(Xte) else np.zeros(0)
    cx_min = float(np.linalg.eigvalsh(xd.Cx_).min())
    align, ratio = xd.null_alignment()
    diag: dict[str, Any] = {
        "cx_spd": bool(np.isfinite(xd.Cx_).all() and cx_min > 0), "cx_min_over_max_eig": ratio,
        "aug_spd": bool(np.isfinite(Str).all() and np.isfinite(Ste).all() and min_eig(Str).min() > 0
                        and (len(Ste) == 0 or min_eig(Ste).min() > 0)),
        "riemann_mean_iter": int(n_iter), "riemann_mean_converged": bool(conv),
        "features_finite": bool(np.isfinite(Ttr).all() and np.isfinite(Tte).all()),
        "lr_converged": bool(lr_conv), "prob_finite": bool(np.isfinite(prob).all()),
        "filter_null_alignment_max_abs_cos": align, "n_features": int(Ttr.shape[1])}
    diag["gate_ok"] = bool(diag["cx_spd"] and diag["aug_spd"] and diag["riemann_mean_converged"]
                           and diag["features_finite"] and diag["lr_converged"] and diag["prob_finite"])
    return prob, diag


def xdawn_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, seed: int = 0,
                      cfg: dict[str, Any] | None = None, device: str = "cpu") -> np.ndarray:
    """``FitFn``-compatible wrapper: P(face) for ``Xte``. Raises ``FloatingPointError`` if a numerical gate fails
    (the affected estimand must then be reported incomplete; nothing is substituted)."""
    prob, diag = fit_predict_diag(Xtr, ytr, Xte, cfg)
    if not diag["gate_ok"]:
        raise FloatingPointError(f"xDAWN-TS-LR numerical gate failed: {diag}")
    return prob
