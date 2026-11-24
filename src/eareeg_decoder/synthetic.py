"""Explicitly SYNTHETIC ear-system data for tests, demos and teaching. No human data are involved.

The generator mimics only the *design* described in the manuscript (in preparation): streams of 20 images with 16 faces and 4 trees, 35 face
identity codes, six repeatedly shown tree photographs, 11 ear-system channels at 250 Hz from -0.2 to 0.848 s.
Magnitudes are arbitrary and NOT calibrated to any recording; synthetic outputs say nothing about the study's
results. All identifiers are synthetic (``SYNTH_...``).

Signal model per epoch (potentials ``x`` against an unknown "hardware" reference):
``x = noise + common(t) + [blink] + [neural] + [subthreshold ocular]`` where ``common`` is a large signal shared by
all channels (it must cancel in every re-referenced montage), a *blink* is an Fpz-dominant Gaussian deflection
(5 % at the mastoids, 2 % in the ears) occurring at class-dependent rates, *neural* is a weak class-dependent
deflection on all channels with a frontal gain, and *subthreshold ocular* is a small blink-shaped deflection on
every tree epoch. Stored data are ``y = x - mean(x over good channels)``, as in the original recordings.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from . import montage as mt
from .data import EpochSet
from .preprocessing import reject_epochs

SFREQ = 250.0
TIMES = np.round(np.arange(263) / SFREQ - 0.2, 6)
N_FACE_IDS = 35
N_TREE_PHOTOS = 6


@dataclass
class SyntheticTruth:
    x: np.ndarray            # potentials vs. the synthetic hardware reference (n x 11 x T)
    blink: np.ndarray        # True where a blink was injected
    blink_amp: np.ndarray    # injected blink amplitude (0 if none)


def _stream_design(rng: np.random.Generator, n_streams: int) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    y, keys, stream, pos = [], [], [], []
    tree_cycle = 0
    for s in range(n_streams):
        labels = np.array([1] * 16 + [0] * 4)
        rng.shuffle(labels)
        faces = rng.choice(N_FACE_IDS, size=16, replace=True)
        fi = 0
        for j, lab in enumerate(labels):
            if lab == 1:
                keys.append(f"SYNTH_FACE_{faces[fi] + 1:02d}")
                fi += 1
            else:
                keys.append(f"SYNTH_TREE_{tree_cycle % N_TREE_PHOTOS + 1}")
                tree_cycle += 1
            y.append(int(lab))
            stream.append(s)
            pos.append(s * 20 + j)
    return np.array(y), keys, np.array(stream), np.array(pos)


def make_participant(rng: np.random.Generator, n_streams: int = 20, *, blink_rate: tuple[float, float] = (0.30, 0.15),
                     blink_amp: float = 200.0, blink_amp_sigma_log: float = 0.0,
                     blink_latency: tuple[float, float] = (0.25, 0.6), neural_amp: float = 0.0,
                     neural_latency: float = 0.17, frontal_gain: float = 3.0, subthreshold_ocular: float = 0.0,
                     subthreshold_latency: float = 0.30, noise: float = 8.0, common_amp: float = 50.0,
                     artifact_rate: float = 0.0, bad: Sequence[str] = ()) -> dict:
    """One synthetic participant. ``blink_rate`` = (tree rate, face rate). ``artifact_rate`` adds a 400-uV step to
    a random in-ear channel in that share of epochs, so the 150-uV rule rejects them (label-independent)."""
    y, keys, stream, pos = _stream_design(rng, n_streams)
    n, t = y.size, TIMES
    x = rng.normal(0.0, noise, size=(n, 11, t.size))
    x += common_amp * rng.normal(size=(n, 1, 1)) * np.sin(2 * np.pi * rng.uniform(1, 4, size=(n, 1, 1)) * t)
    blink = np.zeros(n, bool)
    amps = np.zeros(n)
    for i in range(n):
        if rng.random() < (blink_rate[0] if y[i] == 0 else blink_rate[1]):
            blink[i] = True
            c = rng.uniform(*blink_latency)
            a = blink_amp * (np.exp(rng.normal(0.0, blink_amp_sigma_log)) if blink_amp_sigma_log else 1.0)
            amps[i] = a
            w = a * np.exp(-0.5 * ((t - c) / 0.06) ** 2)
            x[i, 0] += w
            x[i, 1:3] += 0.05 * w
            x[i, 3:] += 0.02 * w
        if subthreshold_ocular and y[i] == 0:
            w = subthreshold_ocular * np.exp(-0.5 * ((t - subthreshold_latency) / 0.06) ** 2)
            x[i, 0] += w
            x[i, 1:3] += 0.05 * w
            x[i, 3:] += 0.02 * w
        if neural_amp:
            s = (1.0 if y[i] == 1 else -1.0) * neural_amp * np.exp(-0.5 * ((t - neural_latency) / 0.04) ** 2)
            x[i] += s
            x[i, 0] += (frontal_gain - 1.0) * s
    for i in np.flatnonzero(rng.random(n) < artifact_rate):
        ch = 3 + int(rng.integers(0, 8))
        x[i, ch, t >= rng.uniform(0.0, 0.6)] += 400.0
    good = [c for c in mt.EAR_CHANNELS if c not in set(bad)]
    stored = mt.own_mean_reference(x, mt.EAR_CHANNELS, good, mt.EAR_CHANNELS, min_good=1)
    stored = stored - stored[:, :, (t >= -0.2 - 1e-9) & (t < 0 - 1e-9)].mean(axis=2, keepdims=True)
    onset = pos * 1.0 + rng.uniform(-0.01, 0.01, size=n) + stream * 5.0
    keep, _ = reject_epochs(stored, mt.EAR_CHANNELS, good)
    return {"X": stored, "y": y, "keys": keys, "stream": stream, "position": pos, "onset_s": onset, "keep": keep,
            "truth": SyntheticTruth(x=x, blink=blink, blink_amp=amps)}


def synthetic_epochs(n_participants: int = 8, n_streams: int = 20, seed: int = 0,
                     bad: Mapping[int, Sequence[str]] | None = None, **kwargs) -> tuple[EpochSet, list[SyntheticTruth]]:
    """A synthetic cohort as an :class:`~eareeg_decoder.data.EpochSet` plus per-participant ground truth.

    Extra keyword arguments go to :func:`make_participant`. ``bad`` maps participant index to channels that
    "failed QC" (all-NaN rows), e.g. ``{0: ["Fpz"]}`` for a participant without a usable frontal channel.
    """
    rng = np.random.default_rng(seed)
    bad = dict(bad or {})
    parts = [make_participant(rng, n_streams, bad=bad.get(i, ()), **kwargs) for i in range(n_participants)]
    es = EpochSet(
        X=np.concatenate([p["X"] for p in parts]), channel_names=list(mt.EAR_CHANNELS), times=TIMES.copy(),
        y=np.concatenate([p["y"] for p in parts]).astype(np.int64),
        participant=np.concatenate([np.full(p["y"].size, i, dtype=np.int64) for i, p in enumerate(parts)]),
        stimulus_key=np.array(sum((p["keys"] for p in parts), []), dtype=str),
        keep=np.concatenate([p["keep"] for p in parts]),
        stream=np.concatenate([p["stream"] for p in parts]),
        position=np.concatenate([p["position"] for p in parts]),
        onset_s=np.concatenate([p["onset_s"] for p in parts]), synthetic=True)
    return es, [p["truth"] for p in parts]


def synthetic_continuous(n_events: int = 40, sfreq: float = 500.0, seed: int = 0, ioi: float = 1.0,
                         lead_s: float = 5.0, tail_s: float = 5.0, noise: float = 2.0) -> dict:
    """Continuous 11-channel synthetic recording with known onsets (for preprocessing tests).

    Each event adds a 10-uV Gaussian bump peaking 0.3 s after onset on every channel, so the epoch average has a
    known time course. Returns ``{"data", "sfreq", "onsets_s", "names"}``.
    """
    rng = np.random.default_rng(seed)
    n = int(round((lead_s + n_events * ioi + tail_s) * sfreq))
    t = np.arange(n) / sfreq
    data = rng.normal(0.0, noise, size=(11, n))
    onsets = lead_s + np.arange(n_events) * ioi
    for o in onsets:
        data += 10.0 * np.exp(-0.5 * ((t - o - 0.3) / 0.03) ** 2)
    return {"data": data, "sfreq": sfreq, "onsets_s": onsets, "names": list(mt.EAR_CHANNELS)}
