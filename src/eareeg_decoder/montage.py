"""Ear-centred montages and the re-reference algebra of Eq. (1).

Notation. Let ``x_i(t)`` be the potential of channel ``i`` against the unknown hardware reference. The stored
epochs are ``y_i(t) = x_i(t) - r(t)``, where ``r`` is the mean of the *good* ear-system channels at extraction; it
contains Fpz whenever Fpz passed quality control. For a montage with channel set ``R`` and good members ``R_g``::

    z_i = y_i - mean_{j in R_g} y_j = x_i - mean_{j in R_g} x_j        (i in R_g)        Eq. (1)

because every common term ``r(t)`` cancels. A deleted electrode therefore enters neither the retained rows nor
their new reference. Deleting the Fpz row of ``y`` *without* this step (:func:`naive_drop`) leaves Fpz inside every
retained row through ``r`` -- the pitfall this module exists to avoid.

This is digital re-referencing. It does not establish the physical hardware reference, calibration, electrode
placement, or whether information at a site is ocular or cortical.

Array convention: ``X`` has shape ``(..., n_channels, n_times)``; channel rows are addressed by name through
``names``. Channels that failed quality control ("bad") are returned as NaN rows; downstream decoders impute
them with the training mean, so they carry no information.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

FPZ: str = "Fpz"
MASTOIDS: tuple[str, ...] = ("M1", "M2")
IN_EAR: tuple[str, ...] = ("EL1", "EL3", "EL4", "EL5", "ER1", "ER2", "ER3", "ER4")
EAR_CHANNELS: tuple[str, ...] = (FPZ,) + MASTOIDS + IN_EAR
ROLES: dict[str, str] = {FPZ: "frontal", **{c: "mastoid" for c in MASTOIDS}, **{c: "in_ear" for c in IN_EAR}}
DROP_ONE: tuple[str, ...] = tuple(f"drop_{c}" for c in EAR_CHANNELS)
MONTAGES: tuple[str, ...] = ("full", "fpz_free", "in_ear", "fpz_mastoid") + DROP_ONE
MIN_GOOD_FOR_OWN_MEAN = 2


def _names(names: Iterable[str]) -> list[str]:
    out = [str(n) for n in names]
    if len(set(out)) != len(out):
        raise ValueError("duplicate channel names")
    return out


def _require(names: Sequence[str], channels: Iterable[str]) -> None:
    missing = [c for c in channels if c not in names]
    if missing:
        raise ValueError(f"missing ear-system channels: {missing}")


def good_from_nan(X: np.ndarray, names: Sequence[str]) -> list[str]:
    """Channels without any NaN sample (bad channels are stored as all-NaN rows)."""
    X = np.asarray(X)
    ok = ~np.isnan(X).any(axis=tuple(i for i in range(X.ndim) if i != X.ndim - 2))
    return [n for n, g in zip(_names(names), ok) if g]


def own_mean_reference(X: np.ndarray, names: Sequence[str], good: Iterable[str], channels: Sequence[str], *,
                       min_good: int = MIN_GOOD_FOR_OWN_MEAN, dtype: type = np.float64) -> np.ndarray:
    """Eq. (1): rows ``channels`` minus the mean of their good members; non-good rows NaN.

    Parameters
    ----------
    X : array (..., n_channels, n_times) in any common reference (the result does not depend on it).
    names : channel names of the rows of ``X``.
    good : channels that passed quality control.
    channels : the montage's channel set ``R`` (output row order).
    min_good : if fewer than this many members of ``R`` are good, every output row is NaN. The montage
        analyses required at least two (a one-channel own mean is identically zero).
    dtype : output dtype (the original montage encodings were float32; transient analyses used float64).
    """
    names = _names(names)
    channels = list(channels)
    _require(names, channels)
    gs = set(good)
    X = np.asarray(X)
    out = np.full(X.shape[:-2] + (len(channels), X.shape[-1]), np.nan, dtype=dtype)
    ref = [c for c in channels if c in gs]
    if len(ref) < min_good:
        return out
    r = X[..., [names.index(c) for c in ref], :].mean(axis=-2)
    for j, c in enumerate(channels):
        if c in gs:
            out[..., j, :] = X[..., names.index(c), :] - r
    return out


def is_common_average(X: np.ndarray, names: Sequence[str], good: Iterable[str], rtol: float = 1e-4) -> bool:
    """True if the good ear-system rows of ``X`` sum to zero at every sample (the stored common average).

    The tolerance is relative to the RMS amplitude of the good rows, so float32 storage passes.
    """
    names = _names(names)
    gs = [c for c in EAR_CHANNELS if c in set(good)]
    if not gs:
        return False
    rows = np.asarray(X, dtype=np.float64)[..., [names.index(c) for c in gs], :]
    if not np.isfinite(rows).all():
        return False
    scale = max(float(np.sqrt(np.mean(rows ** 2))), 1e-12)
    return float(np.max(np.abs(rows.sum(axis=-2)))) <= rtol * scale * len(gs)


def full_montage(X: np.ndarray, names: Sequence[str], good: Iterable[str], *, dtype: type = np.float64,
                 check_reference: bool = True) -> np.ndarray:
    """All 11 stored rows in canonical order (bad rows NaN).

    The full montage is the common average of the good ear-system channels. Stored data already carry it, and
    Eq. (1) over all 11 channels leaves them unchanged, so the rows are returned as they are (bit-identical to the
    original encoding). ``X`` MUST therefore be in that stored common average: with ``check_reference=True`` (the
    default) any other reference raises ``ValueError`` -- convert with
    :func:`eareeg_decoder.preprocessing.common_average` first. When Fpz is bad, the stored average excludes it and
    the full montage equals the Fpz-free montage.
    """
    names = _names(names)
    _require(names, EAR_CHANNELS)
    gs = set(good)
    X = np.asarray(X)
    if check_reference and not is_common_average(X, names, gs):
        raise ValueError("full montage needs data in the stored common average of the good ear-system channels; "
                         "apply preprocessing.common_average(X, names, good) first")
    out = np.full(X.shape[:-2] + (len(EAR_CHANNELS), X.shape[-1]), np.nan, dtype=dtype)
    for j, c in enumerate(EAR_CHANNELS):
        if c in gs:
            out[..., j, :] = X[..., names.index(c), :]
    return out


def fpz_free(X, names, good, *, min_good: int = MIN_GOOD_FOR_OWN_MEAN, dtype: type = np.float64) -> np.ndarray:
    """M1, M2 and the eight in-ear rows re-referenced to their own good mean (Fpz enters nowhere)."""
    return own_mean_reference(X, names, good, MASTOIDS + IN_EAR, min_good=min_good, dtype=dtype)


def in_ear(X, names, good, *, min_good: int = MIN_GOOD_FOR_OWN_MEAN, dtype: type = np.float64) -> np.ndarray:
    """The eight in-ear rows re-referenced to their own good mean."""
    return own_mean_reference(X, names, good, IN_EAR, min_good=min_good, dtype=dtype)


def fpz_mastoid(X, names, good, *, dtype: type = np.float64) -> np.ndarray:
    """Fpz minus the mean of the good mastoids, shape (..., n_times). NaN if Fpz or both mastoids are bad.

    The stored reference ``r`` cancels, so the derivation does not depend on the hardware or stored reference.
    """
    names = _names(names)
    _require(names, (FPZ,) + MASTOIDS)
    gs = set(good)
    X = np.asarray(X)
    res = np.full(X.shape[:-2] + (X.shape[-1],), np.nan, dtype=dtype)
    mast = [c for c in MASTOIDS if c in gs]
    if FPZ in gs and mast:
        res[...] = X[..., names.index(FPZ), :] - X[..., [names.index(c) for c in mast], :].mean(axis=-2)
    return res


def dropped_channel(config: str) -> str:
    if config not in DROP_ONE:
        raise ValueError(f"unknown drop-one montage {config!r}; expected one of {DROP_ONE}")
    return config[len("drop_"):]


def retained_channels(config: str) -> list[str]:
    """The ten channels kept by a drop-one montage, in canonical order."""
    d = dropped_channel(config)
    return [c for c in EAR_CHANNELS if c != d]


def drop_one(X, names, good, config: str, *, min_good: int = MIN_GOOD_FOR_OWN_MEAN,
             dtype: type = np.float64) -> np.ndarray:
    """Delete one electrode and re-reference the other ten to their own good mean (Eq. 1).

    ``drop_one(..., "drop_Fpz")`` equals :func:`fpz_free`.
    """
    return own_mean_reference(X, names, good, retained_channels(config), min_good=min_good, dtype=dtype)


def naive_drop(X, names, config: str) -> np.ndarray:
    """THE PITFALL (for tests and teaching only): select the retained stored rows without re-referencing.

    Every returned row still contains the deleted channel through the stored common average. Do not use for
    analysis.
    """
    names = _names(names)
    _require(names, EAR_CHANNELS)
    return np.asarray(X)[..., [names.index(c) for c in retained_channels(config)], :]


def span_identical_full(X, names, good, *, min_good: int = MIN_GOOD_FOR_OWN_MEAN,
                        dtype: type = np.float64) -> np.ndarray:
    """Review-check montage: the ten Fpz-free own-mean rows plus one row Fpz - mean(good non-Fpz rows).

    It spans the same sum-zero subspace as the full montage but without mixing -Fpz/11 into every row; shrinkage
    LDA is not invariant to such mixing, which is what this montage probes.
    """
    names = _names(names)
    _require(names, EAR_CHANNELS)
    gs = set(good)
    X = np.asarray(X)
    rows = fpz_free(X, names, good, min_good=min_good, dtype=dtype)
    others = [c for c in EAR_CHANNELS[1:] if c in gs]
    extra = np.full(X.shape[:-2] + (1, X.shape[-1]), np.nan, dtype=dtype)
    if FPZ in gs and others:
        extra[..., 0, :] = X[..., names.index(FPZ), :] - X[..., [names.index(c) for c in others], :].mean(axis=-2)
    return np.concatenate([rows, extra], axis=-2)


def make_montage(name: str, X, names, good, *, dtype: type = np.float64) -> np.ndarray:
    """Dispatch by montage name: ``full``, ``fpz_free``, ``in_ear``, ``fpz_mastoid`` (1 row) or ``drop_<ch>``.

    ``full`` requires the stored common average (see :func:`full_montage`); all other montages are reference-free.
    """
    if name == "full":
        return full_montage(X, names, good, dtype=dtype)
    if name == "fpz_free":
        return fpz_free(X, names, good, dtype=dtype)
    if name == "in_ear":
        return in_ear(X, names, good, dtype=dtype)
    if name == "fpz_mastoid":
        return fpz_mastoid(X, names, good, dtype=dtype)[..., None, :]
    if name in DROP_ONE:
        return drop_one(X, names, good, name, dtype=dtype)
    raise ValueError(f"unknown montage {name!r}; expected one of {MONTAGES}")


def montage_channels(name: str) -> list[str]:
    """Row labels of a montage."""
    if name == "full":
        return list(EAR_CHANNELS)
    if name == "fpz_free":
        return list(MASTOIDS + IN_EAR)
    if name == "in_ear":
        return list(IN_EAR)
    if name == "fpz_mastoid":
        return ["Fpz-M"]
    return retained_channels(name)
