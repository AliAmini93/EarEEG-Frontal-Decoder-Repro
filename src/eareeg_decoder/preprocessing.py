"""Label-blind preprocessing: filtering, epoching, baseline, stored common average and epoch rejection.

Order used by the original pipeline for one recording (all steps label-blind):

1. zero-phase FIR band-pass 0.1-30 Hz (firwin design, Hamming window) on the continuous data
   (:func:`bandpass_fir`; requires the optional dependency MNE-Python);
2. channel quality control on the filtered task span, referenced to the per-sample median of the finite channels
   (:func:`channel_qc`), plus a saturation check on the unfiltered span (:func:`clip_fraction`);
3. onsets mapped to samples with ``floor(onset * sfreq + 0.5)`` (:func:`map_onsets`); epochs from -0.2 to 0.85 s
   cut at the native rate and decimated by an integer factor to 250 Hz, keeping the sample at time 0
   (:func:`epoch`); the last retained sample is at 0.848 s, giving 263 samples;
4. baseline correction over [-0.2, 0) s (:func:`baseline_correct`);
5. common average of the good ear-system channels -- the *stored* reference ``r`` of Eq. (1), which includes Fpz
   whenever Fpz is good -- with bad rows set to NaN (:func:`common_average`);
6. rejection of epochs whose peak-to-peak range exceeds 150 (nominal uV) on any good mastoid or in-ear channel,
   tested after step 5; Fpz itself is not tested (:func:`reject_epochs`).

Amplitudes are on the recording's nominal microvolt scale. Nothing here calibrates them, and no unit scale factor
is applied: callers must pass data already expressed on that nominal scale.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from . import montage as mt
from .config import PROTOCOL
from .errors import MissingDependencyError

_PRE = PROTOCOL["preprocessing"]
_QC = PROTOCOL["channel_qc"]


# ------------------------------------------------------------------------------------------------ filtering
def _mne():
    try:
        import mne  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only when MNE is absent
        raise MissingDependencyError("mne", "the 0.1-30 Hz zero-phase FIR filter", "filter") from exc
    return mne


def bandpass_fir(data: np.ndarray, sfreq: float, l_freq: float = _PRE["filter"]["l_freq"],
                 h_freq: float = _PRE["filter"]["h_freq"]) -> np.ndarray:
    """Zero-phase FIR band-pass of continuous data (channels x samples), as MNE ``Raw.filter`` did originally.

    Uses ``mne.filter.filter_data`` with ``method='fir'``, ``fir_design='firwin'``, ``fir_window='hamming'``,
    ``phase='zero'`` and MNE's automatic filter length and transition bandwidths. Filter whole continuous
    segments (the original filtered the task span plus up to 60 s of surrounding recording) before epoching.
    """
    mne = _mne()
    x = np.asarray(data, dtype=np.float64)
    return mne.filter.filter_data(x, float(sfreq), l_freq, h_freq, method="fir", fir_design="firwin",
                                  fir_window="hamming", phase="zero", verbose="ERROR")


# ------------------------------------------------------------------------------------------------ timing
def onset_to_sample(onset_s: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample index ``floor(onset * sfreq + 0.5)`` (round half up) and a finiteness mask; NaN onsets -> -1."""
    onset = np.asarray(onset_s, dtype=float)
    out = np.full(onset.shape, -1, dtype=np.int64)
    finite = np.isfinite(onset)
    out[finite] = np.floor(onset[finite] * float(sfreq) + 0.5).astype(np.int64)
    return out, finite


def map_onsets(onset_s: np.ndarray, sfreq: float, n_times: int, tmin: float = _PRE["tmin"],
               tmax: float = _PRE["tmax"]) -> tuple[np.ndarray, np.ndarray]:
    """Map onsets (s, relative to sample 0) to samples with a per-event reason.

    Reasons: ``ok``, ``nonfinite``, ``epoch_before_start``, ``epoch_after_end``, ``duplicate_sample``. Nothing is
    clipped: an epoch that does not fit is reported, not shortened.
    """
    if sfreq <= 0 or n_times <= 0:
        raise ValueError("invalid sampling description")
    idx, finite = onset_to_sample(onset_s, sfreq)
    reasons = np.array(["ok"] * idx.size, dtype=object)
    reasons[~finite] = "nonfinite"
    first = idx + int(np.round(tmin * sfreq))
    last = idx + int(np.round(tmax * sfreq))
    reasons[finite & (first < 0)] = "epoch_before_start"
    reasons[finite & (first >= 0) & (last > n_times - 1)] = "epoch_after_end"
    ok = reasons == "ok"
    if ok.any():
        values, counts = np.unique(idx[ok], return_counts=True)
        reasons[np.isin(idx, values[counts > 1]) & ok] = "duplicate_sample"
    return idx, reasons


def decimation_factor(sfreq: float, target: float = _PRE["target_sfreq"]) -> int:
    ratio = float(sfreq) / float(target)
    if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 1:
        raise ValueError("native rate is not an integer multiple of the target rate")
    return int(round(ratio))


def epoch(data: np.ndarray, sfreq: float, samples: np.ndarray, tmin: float = _PRE["tmin"],
          tmax: float = _PRE["tmax"], target_sfreq: float = _PRE["target_sfreq"]) -> tuple[np.ndarray, np.ndarray]:
    """Cut epochs at the native rate and decimate to ``target_sfreq``.

    Native sample offsets run from ``round(tmin*sfreq)`` to ``round(tmax*sfreq)`` inclusive; decimation keeps
    the offsets that are multiples of the decimation factor, so the sample at time 0 is always kept. The data
    must already be low-pass filtered (step 1) -- decimation applies no further anti-alias filter.

    Returns ``X`` (epochs x channels x times) and ``times`` (s).
    """
    data = np.asarray(data)
    samples = np.asarray(samples, dtype=np.int64)
    decim = decimation_factor(sfreq, target_sfreq)
    offsets = np.arange(int(np.round(tmin * sfreq)), int(np.round(tmax * sfreq)) + 1)
    offsets = offsets[(offsets % decim) == 0]
    idx = samples[:, None] + offsets[None, :]
    if idx.size and (idx.min() < 0 or idx.max() >= data.shape[-1]):
        raise ValueError("an epoch extends beyond the data; use map_onsets to exclude it")
    X = np.transpose(data[:, idx], (1, 0, 2))
    return X, offsets / float(sfreq)


def baseline_correct(X: np.ndarray, times: np.ndarray, window: Sequence[float] = tuple(_PRE["baseline"])) -> np.ndarray:
    """Subtract the per-epoch, per-channel mean over ``[window[0], window[1])`` (time 0 excluded)."""
    times = np.asarray(times, dtype=float)
    m = (times >= window[0] - 1e-9) & (times < window[1] - 1e-9)
    if not m.any():
        raise ValueError("empty baseline window")
    X = np.asarray(X, dtype=float)
    return X - X[..., m].mean(axis=-1, keepdims=True)


def common_average(X: np.ndarray, names: Sequence[str], good: Iterable[str],
                   channels: Sequence[str] = mt.EAR_CHANNELS) -> np.ndarray:
    """The stored reference: ``channels`` minus the mean of their good members (bad rows NaN).

    With the default channel set this is the common average of the good ear-system channels, which includes Fpz
    when Fpz is good. Returns rows in the order of ``channels``.
    """
    return mt.own_mean_reference(X, names, good, channels, min_good=1)


def reject_epochs(X: np.ndarray, names: Sequence[str], good: Iterable[str],
                  threshold_uv: float = _PRE["epoch_rejection_ptp_uv"],
                  channels: Sequence[str] = mt.MASTOIDS + mt.IN_EAR) -> tuple[np.ndarray, np.ndarray]:
    """Keep mask: peak-to-peak <= ``threshold_uv`` on every good channel in ``channels`` (Fpz excluded by default).

    Epochs containing non-finite samples on a tested channel are rejected. Returns ``(keep, max_ptp)``.
    The 150-uV channel rule and the 100-uV Fpz-mastoid transient flag (:mod:`eareeg_decoder.transients`) are
    different rules on different signals; an epoch can pass this rule and still carry a large frontal transient.
    """
    names = list(names)
    gs = set(good)
    rows = [c for c in channels if c in gs]
    if not rows:
        raise ValueError("no good channels available for epoch rejection")
    sub = np.asarray(X, dtype=float)[:, [names.index(c) for c in rows], :]
    ptp = sub.max(axis=2) - sub.min(axis=2)
    finite = np.all(np.isfinite(sub), axis=(1, 2))
    return finite & np.all(ptp <= threshold_uv, axis=1), ptp.max(axis=1)


# ------------------------------------------------------------------------------------------------ channel QC
def robust_z(values: np.ndarray, mad_floor: float = 0.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    med = np.median(values)
    mad = max(np.median(np.abs(values - med)) * 1.4826, mad_floor)
    if mad <= 0:
        return np.zeros_like(values)
    return (values - med) / mad


def median_reference(data: np.ndarray) -> np.ndarray:
    """Subtract the per-sample median of the rows that are finite everywhere (QC reference only)."""
    data = np.asarray(data, dtype=float)
    finite_rows = np.all(np.isfinite(data), axis=1)
    if not finite_rows.any():
        return data
    return data - np.median(data[finite_rows], axis=0, keepdims=True)


def channel_qc(data_uv: np.ndarray, names: Sequence[str], flat_sd_uv: float = _QC["flat_sd_uv"],
               max_sd_uv: float = _QC["max_sd_uv"], robust_z_limit: float = _QC["robust_z_log10_sd"],
               mad_floor: float = _QC["robust_z_mad_floor_log10"]) -> tuple[dict[str, list[str]], np.ndarray]:
    """Label-blind channel QC of continuous data (channels x samples), already median-referenced.

    A channel is flagged ``nonfinite``, ``flat`` (SD < flat_sd_uv), ``high_sd`` (SD > max_sd_uv) or
    ``robust_z_log_sd`` (robust z of log10 SD across channels above the limit; needs >= 3 usable channels).
    Returns ``({name: [reasons]}, sd)``.
    """
    data_uv = np.asarray(data_uv, dtype=float)
    names = list(names)
    reasons: dict[str, list[str]] = {n: [] for n in names}
    finite = np.all(np.isfinite(data_uv), axis=1)
    sd = np.array([np.std(r[np.isfinite(r)]) if np.isfinite(r).any() else 0.0 for r in data_uv])
    for i, n in enumerate(names):
        if not finite[i]:
            reasons[n].append("nonfinite")
        if sd[i] < flat_sd_uv:
            reasons[n].append("flat")
        if sd[i] > max_sd_uv:
            reasons[n].append("high_sd")
    usable = sd > 0
    if usable.sum() >= 3:
        z = np.zeros(len(names))
        z[usable] = robust_z(np.log10(sd[usable]), mad_floor)
        for i, n in enumerate(names):
            if usable[i] and z[i] > robust_z_limit:
                reasons[n].append("robust_z_log_sd")
    return reasons, sd


def clip_fraction(data: np.ndarray) -> np.ndarray:
    """Per channel, the fraction of finite samples equal to the channel's own minimum or maximum (saturation)."""
    out = []
    for row in np.asarray(data, dtype=float):
        f = row[np.isfinite(row)]
        out.append(float(np.mean((f == f.max()) | (f == f.min()))) if f.size else float("nan"))
    return np.array(out)


def bad_channels(reasons: dict[str, list[str]]) -> list[str]:
    return sorted(n for n, r in reasons.items() if r)


# ------------------------------------------------------------------------------------------------ orchestration
@dataclass
class PreprocessedRecording:
    """Output of :func:`preprocess_recording` for one participant (keep it private: it is participant data)."""

    status: str
    X: np.ndarray | None = None          # epochs x 11 x times, stored common average, bad rows NaN
    times: np.ndarray | None = None
    keep: np.ndarray | None = None       # passes the 150-uV rule
    event_index: np.ndarray | None = None  # index into the supplied onsets for each epoch
    good: list[str] = field(default_factory=list)
    bad: list[str] = field(default_factory=list)
    reasons: dict[str, list[str]] = field(default_factory=dict)
    event_reasons: np.ndarray | None = None


def preprocess_recording(data_uv: np.ndarray, names: Sequence[str], sfreq: float, onsets_s: np.ndarray, *,
                         filtered: bool = False) -> PreprocessedRecording:
    """Run steps 1-6 on one continuous ear-system recording (channels x samples, nominal uV).

    ``onsets_s`` are image onsets in seconds relative to sample 0 of ``data_uv``. Set ``filtered=True`` if the
    data were already band-pass filtered (then MNE is not needed). Returns a :class:`PreprocessedRecording`;
    ``status`` is ``ok`` or ``recording_unusable_bad_channels``.
    """
    names = list(names)
    mt._require(names, mt.EAR_CHANNELS)
    raw = np.asarray(data_uv, dtype=float)
    onsets = np.asarray(onsets_s, dtype=float)
    fin = onsets[np.isfinite(onsets)]
    if fin.size == 0:
        raise ValueError("no finite onsets")
    margin = PROTOCOL["preprocessing"]["qc_span_margin_s"]
    lo = max(0, int(np.floor((fin.min() - margin) * sfreq)))
    hi = min(raw.shape[1], int(np.floor((fin.max() + margin) * sfreq)))
    rows = [names.index(c) for c in mt.EAR_CHANNELS]
    clip = clip_fraction(raw[rows, lo:hi])
    filt = raw if filtered else bandpass_fir(raw, sfreq)
    reasons, _ = channel_qc(median_reference(filt[rows, lo:hi]), mt.EAR_CHANNELS)
    for c, frac in zip(mt.EAR_CHANNELS, clip):
        if np.isfinite(frac) and frac > _QC["clip_fraction_max"]:
            reasons[c].append("clipped")
    bad = bad_channels(reasons)
    good = [c for c in mt.EAR_CHANNELS if c not in bad]
    rec = PreprocessedRecording(status="ok", good=good, bad=bad, reasons={k: v for k, v in reasons.items() if v})
    if len(bad) > _QC["max_bad_channel_fraction"] * len(mt.EAR_CHANNELS) or not good:
        rec.status = "recording_unusable_bad_channels"
        return rec
    samples, ev_reasons = map_onsets(onsets, sfreq, raw.shape[1])
    valid = np.flatnonzero(ev_reasons == "ok")
    X, times = epoch(filt[rows], sfreq, samples[valid])
    X = baseline_correct(X, times)
    Xref = common_average(X, mt.EAR_CHANNELS, good)
    keep, _ = reject_epochs(Xref, mt.EAR_CHANNELS, good)
    rec.X, rec.times, rec.keep, rec.event_index, rec.event_reasons = Xref, times, keep, valid, ev_reasons
    return rec
