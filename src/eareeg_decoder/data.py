"""Fail-closed adapter for legitimately obtained, already-epoched ear-system data.

This package ships NO data and contains NO downloader. :func:`load_authorized_epochs` refuses to read anything
unless the caller (1) supplies a path and (2) explicitly attests that they are authorized to use the data;
otherwise it raises :class:`~eareeg_decoder.errors.DataNotProvidedError` whose message starts with
``DATA_NOT_PROVIDED``. The adapter never uploads, caches or copies data.

Data contract (one ``.npz`` file, read with ``allow_pickle=False``; see docs/DATA_ACCESS.md):

==================  ==========================  =====================================================================
key                 shape / dtype               meaning
==================  ==========================  =====================================================================
``X``               (n, C, T) float             epochs on the recording's nominal uV scale in the STORED COMMON
                                                AVERAGE of the participant's good ear-system channels (convert with
                                                ``preprocessing.common_average``); a channel that failed QC for a
                                                participant is an all-NaN row in all of that participant's epochs
``channel_names``   (C,) str                    must include Fpz, M1, M2, EL1, EL3, EL4, EL5, ER1, ER2, ER3, ER4
``times``           (T,) float                  seconds; the paper protocol uses 263 samples, -0.2 ... 0.848 s
``y``               (n,) int                    1 = face shown, 0 = tree shown (the image, not a mental state)
``participant``     (n,) int                    pseudonymous integer codes (never names or source IDs)
``stimulus_key``    (n,) str                    stimulus identity (face identity code / tree photograph), '' if unknown
``keep``            (n,) bool                   epoch passed the 150-uV rule (decoders use kept epochs only)
``stream``          (n,) int, optional          image-stream (block) index
``position``        (n,) int, optional          presentation order within the recording
``onset_s``         (n,) float, optional        onset time (s) on the EEG clock, for adjacent-epoch screens
==================  ==========================  =====================================================================
"""
from __future__ import annotations

import os
from zipfile import BadZipFile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import montage as mt
from .config import PROTOCOL
from .errors import DataNotProvidedError, SchemaError

REQUIRED_KEYS = ("X", "channel_names", "times", "y", "participant", "stimulus_key", "keep")
OPTIONAL_KEYS = ("stream", "position", "onset_s")


@dataclass
class EpochSet:
    """In-memory epochs obeying the data contract. Participant-level data: keep private."""

    X: np.ndarray
    channel_names: list[str]
    times: np.ndarray
    y: np.ndarray
    participant: np.ndarray
    stimulus_key: np.ndarray
    keep: np.ndarray
    stream: np.ndarray | None = None
    position: np.ndarray | None = None
    onset_s: np.ndarray | None = None
    synthetic: bool = False

    @property
    def participants(self) -> np.ndarray:
        return np.unique(self.participant)

    def subset_participant(self, p: int) -> np.ndarray:
        return np.flatnonzero(self.participant == p)

    def good_channels(self, p: int) -> list[str]:
        """Ear-system channels without NaN in any of participant ``p``'s epochs."""
        idx = self.subset_participant(p)
        return mt.good_from_nan(self.X[idx], self.channel_names)

    def validate(self, strict_grid: bool = True) -> None:
        validate(self, strict_grid=strict_grid)


def validate(es: EpochSet, strict_grid: bool = True) -> None:
    """Raise :class:`SchemaError` with an explicit message if ``es`` violates the data contract."""
    X = np.asarray(es.X)
    if X.ndim != 3:
        raise SchemaError("X must have shape (epochs, channels, times)")
    n, C, T = X.shape
    names = [str(c) for c in es.channel_names]
    if len(names) != C:
        raise SchemaError("channel_names length differs from X.shape[1]")
    missing = [c for c in mt.EAR_CHANNELS if c not in names]
    if missing:
        raise SchemaError(f"missing ear-system channels: {missing}")
    times = np.asarray(es.times, dtype=float)
    if times.shape != (T,):
        raise SchemaError("times length differs from X.shape[2]")
    if strict_grid:
        pre = PROTOCOL["preprocessing"]
        grid = np.arange(pre["n_epoch_samples"]) / pre["target_sfreq"] + pre["tmin"]
        if T != pre["n_epoch_samples"] or not np.allclose(times, grid, atol=1e-6):
            raise SchemaError("times must be the 250-Hz grid -0.2 ... 0.848 s (263 samples); "
                              "pass strict_grid=False only for deliberate sensitivity work")
    for key in ("y", "participant", "stimulus_key", "keep"):
        if np.asarray(getattr(es, key)).shape != (n,):
            raise SchemaError(f"{key} must have one entry per epoch")
    for key in ("stream", "position", "onset_s"):
        v = getattr(es, key)
        if v is not None and np.asarray(v).shape != (n,):
            raise SchemaError(f"{key} must have one entry per epoch")
    y = np.asarray(es.y)
    if not np.issubdtype(y.dtype, np.integer) or not set(np.unique(y).tolist()) <= {0, 1}:
        raise SchemaError("y must be integers 1 (face) / 0 (tree)")
    keys = np.asarray(es.stimulus_key).astype(str)
    both = set(keys[(y == 1) & (keys != "")].tolist()) & set(keys[(y == 0) & (keys != "")].tolist())
    if both:
        raise SchemaError(f"{len(both)} stimulus key(s) occur with both labels; a key must identify one stimulus")
    if not np.issubdtype(np.asarray(es.participant).dtype, np.integer):
        raise SchemaError("participant must be pseudonymous integer codes (no names or source identifiers)")
    if np.asarray(es.keep).dtype != bool:
        raise SchemaError("keep must be boolean")
    for p in np.unique(es.participant):
        Xp = X[np.asarray(es.participant) == p]
        nan_rows = np.isnan(Xp).any(axis=(0, 2))
        all_nan_rows = np.isnan(Xp).all(axis=(0, 2))
        if np.any(nan_rows & ~all_nan_rows):
            raise SchemaError("a channel is partially NaN for a participant; bad channels must be all-NaN rows")
        good = mt.good_from_nan(Xp[:, [names.index(c) for c in mt.EAR_CHANNELS]], mt.EAR_CHANNELS)
        if not mt.is_common_average(Xp, names, good):
            raise SchemaError("X must be in the stored common average of the participant's good ear-system channels "
                              "(the good rows must sum to zero); convert with preprocessing.common_average")


def load_authorized_epochs(path: str | os.PathLike | None = None, *, attest_authorized: bool = False,
                           strict_grid: bool = True) -> EpochSet:
    """Load a user-supplied ``.npz`` obeying the data contract, failing closed.

    Raises :class:`DataNotProvidedError` (``DATA_NOT_PROVIDED: ...``) if no path is given, the file does not
    exist, or ``attest_authorized`` is not True. The attestation is the caller's statement that they obtained the
    data legitimately and may analyse it; this package cannot verify it.
    """
    if path is None or str(path).strip() == "":
        raise DataNotProvidedError("no authorized data source was supplied. This package ships no data and has no "
                                   "downloader; see docs/DATA_ACCESS.md for the data contract.")
    if not attest_authorized:
        raise DataNotProvidedError("a data path was given but authorization was not attested "
                                   "(attest_authorized=True / --i-am-authorized). Nothing was read.")
    p = Path(path)
    if not p.is_file():
        raise DataNotProvidedError("the supplied data path is not a readable file. Nothing was read.")
    try:
        loaded = np.load(p, allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise SchemaError("expected a NumPy NPZ archive")
        with loaded as z:
            absent = [k for k in REQUIRED_KEYS if k not in z.files]
            if absent:
                raise SchemaError(f"data file lacks required keys: {absent}")
            kw = {k: z[k] for k in REQUIRED_KEYS}
            for k in OPTIONAL_KEYS:
                kw[k] = z[k] if k in z.files else None
        es = EpochSet(X=kw["X"].astype(float), channel_names=[str(c) for c in kw["channel_names"]],
                      times=kw["times"].astype(float), y=kw["y"], participant=kw["participant"],
                      stimulus_key=kw["stimulus_key"].astype(str), keep=kw["keep"], stream=kw["stream"],
                      position=kw["position"], onset_s=kw["onset_s"])
    except SchemaError:
        raise
    except (ValueError, TypeError, OSError, EOFError, BadZipFile) as exc:
        raise SchemaError("invalid NPZ epoch archive or array dtype") from exc
    validate(es, strict_grid=strict_grid)
    return es


def save_epochs(es: EpochSet, path: str | os.PathLike) -> None:
    """Write an :class:`EpochSet` in the contract format (used for synthetic examples and tests)."""
    arrays = {"X": np.asarray(es.X, dtype=np.float64), "channel_names": np.asarray(es.channel_names, dtype=str),
              "times": np.asarray(es.times, dtype=float), "y": np.asarray(es.y, dtype=np.int64),
              "participant": np.asarray(es.participant, dtype=np.int64),
              "stimulus_key": np.asarray(es.stimulus_key, dtype=str), "keep": np.asarray(es.keep, dtype=bool)}
    for k in OPTIONAL_KEYS:
        v = getattr(es, k)
        if v is not None:
            arrays[k] = np.asarray(v)
    np.savez(path, **arrays)
