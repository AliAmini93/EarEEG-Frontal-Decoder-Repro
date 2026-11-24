"""Binned-amplitude features and training-only standardisation (the linear decoder's input).

The linear decoder takes, per channel, the mean amplitude in sixteen consecutive 50-ms bins over [0, 0.8) s.
Features are laid out channel-major: column ``c * n_bins + b`` is channel ``c``, bin ``b``.
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from .config import PROTOCOL

_F = PROTOCOL["features"]
WINDOW: tuple[float, float] = tuple(_F["window_s"])  # type: ignore[assignment]
BIN_S: float = _F["bin_s"]


def bin_edges(window: Sequence[float] = WINDOW, width: float = BIN_S) -> np.ndarray:
    return np.arange(window[0], window[1] + 1e-9, width)


def bin_means(X: np.ndarray, times: np.ndarray, window: Sequence[float] = WINDOW, width: float = BIN_S) -> np.ndarray:
    """Mean amplitude in consecutive ``[a, a + width)`` bins; ``X`` epochs x channels x times -> epochs x (ch*bins).

    A bin containing no sample raises ``ValueError`` (the epoch grid does not cover the window).
    NaN rows (bad channels) give NaN features, imputed later with the training mean.
    """
    X = np.asarray(X)
    times = np.asarray(times, dtype=float)
    edges = bin_edges(window, width)
    cols = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (times >= a - 1e-9) & (times < b - 1e-9)
        if not m.any():
            raise ValueError("empty feature bin")
        cols.append(X[:, :, m].mean(axis=2))
    return np.stack(cols, axis=2).reshape(X.shape[0], -1)


def window_columns(n_channels: int, window: Sequence[float], full_window: Sequence[float] = WINDOW,
                   width: float = BIN_S) -> np.ndarray:
    """Columns of channel-major binned features whose bin lies inside ``window`` (e.g. the 200-ms windows)."""
    edges = bin_edges(full_window, width)
    sel = [j for j, (a, b) in enumerate(zip(edges[:-1], edges[1:]))
           if a >= window[0] - 1e-9 and b <= window[1] + 1e-9]
    nb = len(edges) - 1
    return np.array([c * nb + j for c in range(n_channels) for j in sel], dtype=int)


def fit_scaler(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column mean and SD (ddof 0) ignoring NaN. Constant or all-NaN columns get SD 1; all-NaN columns mean 0.

    Fit on TRAINING rows only.
    """
    F = np.asarray(F, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(F, axis=0)
        sd = np.nanstd(F, axis=0)
    mu[~np.isfinite(mu)] = 0.0
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return mu, sd


def apply_scaler(F: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """z-score with a fitted scaler; non-finite results (bad channels) become 0, i.e. the training mean."""
    Z = (np.asarray(F, dtype=float) - mu) / sd
    Z[~np.isfinite(Z)] = 0.0
    return Z


def raw_window(X: np.ndarray, times: np.ndarray, window: Sequence[float] = WINDOW) -> np.ndarray:
    """Samples with ``window[0] <= t < window[1]`` (the network input; 200 samples at 250 Hz for [0, 0.8))."""
    times = np.asarray(times, dtype=float)
    m = (times >= window[0] - 1e-9) & (times < window[1] - 1e-9)
    return np.asarray(X)[..., m]
