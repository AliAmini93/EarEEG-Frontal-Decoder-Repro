"""Large blink-like frontal transients and related label-free epoch screens.

All measures use the Fpz-mastoid derivation ``F(t)`` = Fpz minus the mean of the good mastoids
(:func:`eareeg_decoder.montage.fpz_mastoid`), which does not depend on the stored or hardware reference.

* An epoch contains a **large transient** when the peak-to-peak range of ``F`` over the whole stored epoch
  (-0.2 to 0.848 s) is at least ``theta`` (100 on the nominal uV scale; sensitivities 70, 150 and 2.5 x the
  participant's 25th percentile of the range).
* The transient is **post-onset** when the signed maximum of ``F`` falls at or after 0.15 s (later events are not
  seen inside the epoch).
* **Transient-free** ("free") epochs have a range below ``theta``. This means free of LARGE transients, not free of
  blinks or artefacts; the flag is not an eye measurement and uses mostly post-onset samples.
* The **thresholded frontal-deflection descriptor** has 32 binary features: for each 50-ms bin in [0, 0.8) s,
  whether the bin mean of ``F`` is above +50 and whether it is below -50.

The 100-uV frontal flag and the 150-uV channel rejection rule (:func:`eareeg_decoder.preprocessing.reject_epochs`)
are different rules on different signals and must not be conflated.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import PROTOCOL
from .features import bin_means

_T = PROTOCOL["transients"]
THETA_UV: float = _T["theta_uv"]
POST_PEAK_MIN_S: float = _T["post_peak_min_s"]


@dataclass(frozen=True)
class TransientFlags:
    ptp: np.ndarray        # peak-to-peak range of F over the whole epoch
    peak_time: np.ndarray  # time of the signed maximum of F
    any: np.ndarray        # ptp >= theta
    post: np.ndarray       # any & peak_time >= post_min
    pre: np.ndarray        # any & peak_time < post_min

    @property
    def free(self) -> np.ndarray:
        """Transient-free: range below theta (NOT 'blink-free')."""
        return ~self.any


def relative_theta(fm: np.ndarray, factor: float = _T["theta_relative_factor_of_p25"], percentile: float = 25.0) -> float:
    """Participant-scaled threshold: ``factor`` x the participant's ``percentile`` of the peak-to-peak range
    (computed over all of that participant's presentations; label-blind)."""
    return float(factor * np.percentile(np.ptp(np.asarray(fm, dtype=float), axis=1), percentile))


def resolve_theta(fm: np.ndarray, spec: float | str = THETA_UV) -> float:
    """A numeric threshold, or ``"relative"`` for :func:`relative_theta` of this participant's ``fm``."""
    return relative_theta(fm) if spec == "relative" else float(spec)


def transient_flags(fm: np.ndarray, times: np.ndarray, theta: float = THETA_UV,
                    post_min: float = POST_PEAK_MIN_S) -> TransientFlags:
    """Label-blind flags for epochs x times of the Fpz-mastoid derivation."""
    fm = np.asarray(fm, dtype=float)
    times = np.asarray(times, dtype=float)
    ptp = np.ptp(fm, axis=1)
    peak_t = times[np.argmax(fm, axis=1)]
    any_ = ptp >= theta
    return TransientFlags(ptp, peak_t, any_, any_ & (peak_t >= post_min), any_ & (peak_t < post_min))


def thresholded_descriptor(fm: np.ndarray, times: np.ndarray, threshold: float = _T["descriptor_bin_threshold_uv"],
                           window: Sequence[float] = tuple(PROTOCOL["features"]["window_s"]),
                           width: float = PROTOCOL["features"]["bin_s"]) -> np.ndarray:
    """32 binary features: per 50-ms bin, ``bin mean > +threshold`` (first 16) and ``bin mean < -threshold`` (last 16).

    Not a blink sensor: any large frontal deflection can cross the thresholds, and because epochs are baseline
    corrected a transient just before onset can push post-onset bins below -threshold.
    """
    b = bin_means(np.asarray(fm, dtype=float)[:, None, :], times, window, width)
    return np.concatenate([(b > threshold).astype(float), (b < -threshold).astype(float)], axis=1)


def rate_difference(flag: np.ndarray, is_tree: np.ndarray) -> float:
    """Tree rate minus face rate of a flag over one participant's presentations."""
    flag = np.asarray(flag, bool)
    is_tree = np.asarray(is_tree, bool)
    return float(flag[is_tree].mean() - flag[~is_tree].mean())


def preonset_clean(fm: np.ndarray, times: np.ndarray, window: Sequence[float] = tuple(_T["preonset_window_s"]),
                   limit: float = _T["preonset_limit_uv"]) -> np.ndarray:
    """Range of F over [-0.2, 0) s below ``limit`` (a selection on pre-onset data only)."""
    t = np.asarray(times, dtype=float)
    m = (t >= window[0] - 1e-9) & (t < window[1] - 1e-9)
    return np.ptp(np.asarray(fm, dtype=float)[:, m], axis=1) < limit


def late_flank_clean(fm: np.ndarray, times: np.ndarray, window: Sequence[float] = tuple(_T["late_flank_window_s"]),
                     limit: float = _T["late_flank_limit_uv"]) -> np.ndarray:
    """``max |F|`` over the closed window [0.748, 0.848] s below ``limit`` (no rising flank at the epoch end)."""
    t = np.asarray(times, dtype=float)
    m = (t >= window[0] - 1e-9) & (t <= window[1] + 1e-9)
    return np.nanmax(np.abs(np.asarray(fm, dtype=float)[:, m]), axis=1) < limit


def adjacent(order: np.ndarray, stream: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the previous and next presentation in the same stream (-1 if none).

    ``order`` is the presentation order (any sortable key); ``stream`` the stream/block id of each presentation.
    """
    order = np.asarray(order)
    stream = np.asarray(stream)
    n = order.size
    prev = np.full(n, -1)
    nxt = np.full(n, -1)
    srt = np.argsort(order, kind="stable")
    for a, b in zip(srt[:-1], srt[1:]):
        if stream[a] == stream[b]:
            prev[b] = a
            nxt[a] = b
    return prev, nxt


def next_baseline_transient(fm: np.ndarray, times: np.ndarray, onsets_s: np.ndarray, nxt: np.ndarray,
                            max_ioi: float = _T["next_baseline_max_ioi_s"],
                            limit: float = _T["preonset_limit_uv"]) -> tuple[np.ndarray, np.ndarray]:
    """Flag presentations followed within ``max_ioi`` s by a presentation whose [-0.2, 0) range of F >= ``limit``.

    Returns ``(flag, mappable)``; presentations without a mappable next presentation are not flagged (and are
    reported through ``mappable``).
    """
    onsets = np.asarray(onsets_s, dtype=float)
    nxt = np.asarray(nxt)
    has = nxt >= 0
    ioi = np.where(has, onsets[np.maximum(nxt, 0)] - onsets, np.nan)
    mappable = has & (ioi <= max_ioi + 1e-9)
    pre_bad = ~preonset_clean(fm, times, limit=limit)
    flag = np.zeros(onsets.size, bool)
    flag[mappable] = pre_bad[nxt[mappable]]
    return flag, mappable


def interval_overlap(onsets_s: np.ndarray, intervals: np.ndarray, tmin: float = -0.2, tmax: float = 0.848,
                     tol: float = 0.0) -> np.ndarray:
    """Whether any interval ``(start, end)`` overlaps the epoch window ``[onset + tmin - tol, onset + tmax + tol]``.

    Used for tracker-parsed blinks once both are on the same clock. ``tol`` was the in-sample clock residual plus
    one 250-Hz sample. Onsets that could not be mapped should be passed as NaN; they return False and must be
    excluded by the caller (unknown is not blink-free). A parsed blink is an interval of pupil loss, not a direct
    measure of lid closure.
    """
    on = np.asarray(onsets_s, dtype=float)
    iv = np.asarray(intervals, dtype=float).reshape(-1, 2)
    iv = iv[np.all(np.isfinite(iv), axis=1)]
    out = np.zeros(on.size, bool)
    for i, t in enumerate(on):
        if not np.isfinite(t) or iv.size == 0:
            continue
        lo, hi = t + tmin - tol, t + tmax + tol
        out[i] = bool(np.any((iv[:, 0] <= hi) & (iv[:, 1] >= lo)))
    return out
