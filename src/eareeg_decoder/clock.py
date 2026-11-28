"""Clock mapping and eye-epoch quality from normalized, authorized inputs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _timestamps(values, *, minimum):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or len(values) < minimum or not np.isfinite(values).all()
            or not (np.diff(values) > 0).all()):
        raise ValueError("timestamps must be finite, strictly increasing and sufficiently numerous")
    return values


def _nearest(sorted_times, predictions):
    right = np.clip(np.searchsorted(sorted_times, predictions), 1, len(sorted_times) - 1)
    left = right - 1
    return np.where(abs(sorted_times[right] - predictions) < abs(sorted_times[left] - predictions), right, left)


@dataclass(frozen=True)
class TriggerMap:
    offset: float
    slope: float
    matched_source: np.ndarray
    matched_fraction: float
    maximum_residual: float
    uniqueness_margin: float


def fit_trigger_map(source, tracker, *, coarse_tol, residual_tol, alternative_offset, minimum_margin,
                    min_fraction, event_provenance, max_anchor=60):
    """Fit tracker time = offset + slope × source time from pulse-time patterns.

    All tolerances are caller-specified in seconds except the fractions.
    Candidate offsets use anchor pairs and are ranked by coarse match count.
    The best is refined iteratively with nearest pulses. An affine fit is
    accepted only with one-to-one final matches and a unique coarse solution
    at the requested resolution. Evidence of a shared clock depends on the
    provenance of independently measured marks supplied by the caller.
    """
    source, tracker = _timestamps(source, minimum=3), _timestamps(tracker, minimum=3)
    values = (coarse_tol, residual_tol, alternative_offset, minimum_margin, min_fraction)
    if (not np.isfinite(values).all() or not 0 < residual_tol <= coarse_tol
            or alternative_offset <= 0 or not 0 < minimum_margin <= 1 or not 0 < min_fraction <= 1
            or max_anchor < 1 or not isinstance(event_provenance, str) or not event_provenance.strip()):
        raise ValueError("invalid alignment thresholds or missing event provenance")

    def match(offset, slope):
        pred = offset + slope * source
        idx = _nearest(tracker, pred)
        return idx, tracker[idx] - pred

    candidates = []
    for x in source[:max_anchor]:
        for z in tracker[:3 * max_anchor]:
            offset = z - x
            _, residual = match(offset, 1.0)
            candidates.append((int((abs(residual) < coarse_tol).sum()), offset))
    candidates.sort(key=lambda pair: -pair[0])
    count, offset = candidates[0]
    rival = next((n for n, alt in candidates if abs(alt - offset) > alternative_offset), 0)
    margin = (count - rival) / len(source)
    slope = 1.0
    for tol in (coarse_tol, np.sqrt(coarse_tol * residual_tol), residual_tol, residual_tol):
        idx, residual = match(offset, slope)
        take = abs(residual) < tol
        if take.sum() < 3:
            raise ValueError("insufficient matched pulses for an affine fit")
        slope, offset = np.polyfit(source[take], tracker[idx[take]], 1)
    idx, residual = match(offset, slope)
    good = abs(residual) < residual_tol
    if (slope <= 0 or good.mean() < min_fraction or good.sum() < 3 or margin < minimum_margin
            or len(np.unique(idx[good])) != good.sum()):
        raise ValueError("pulse map failed match coverage, uniqueness or one-to-one verification")
    return TriggerMap(float(offset), float(slope), source[good].copy(), float(good.mean()),
                      float(abs(residual[good]).max()), float(margin))


def classify_eye_epochs(onsets, mapping: TriggerMap, sample_times, pupil, blink_intervals, *,
                        tmin=-0.2, tmax=0.848, padding_s=0.0, min_coverage=0.9, max_missing=0.1):
    """Return 1 for overlapping blink, 0 for clean, and -1 for unknown.

    Eye samples and blink endpoints must already be on the tracker pulse
    clock in seconds. Onsets must be the same source-clock values submitted
    to the fit. Use padding from an independently justified timing budget.
    Only matched onsets can be called clean or blink-bearing. Low coverage or
    unexplained pupil loss stays unknown. Bounds of blink intervals are inclusive.
    """
    onsets = _timestamps(onsets, minimum=1)
    t = _timestamps(sample_times, minimum=2)
    pupil, blinks = np.asarray(pupil, dtype=float), np.asarray(blink_intervals, dtype=float)
    if (pupil.shape != t.shape or blinks.ndim != 2 or blinks.shape[1] != 2
            or not np.isfinite(blinks).all() or (blinks[:, 0] > blinks[:, 1]).any()
            or not np.isfinite([tmin, tmax, padding_s, min_coverage, max_missing]).all()
            or tmin >= tmax or padding_s < 0 or not 0 < min_coverage <= 1 or not 0 <= max_missing <= 1
            or not np.isfinite([mapping.slope, mapping.offset]).all() or mapping.slope <= 0):
        raise ValueError("invalid sample, blink interval, map or epoch contract")
    states = np.full(len(onsets), -1, dtype=int)
    dt = np.median(np.diff(t))
    missing = ~np.isfinite(pupil) | (pupil <= 0)
    for i in np.flatnonzero(np.isin(onsets, mapping.matched_source)):
        lo = mapping.offset + mapping.slope * (onsets[i] + tmin - padding_s)
        hi = mapping.offset + mapping.slope * (onsets[i] + tmax + padding_s)
        if ((blinks[:, 0] <= hi) & (blinks[:, 1] >= lo)).any():
            states[i] = 1
            continue
        start = np.searchsorted(t, lo)
        end = np.searchsorted(t, hi, side="right")
        if end > start and (end - start) * dt >= min_coverage * (hi - lo) and missing[start:end].mean() <= max_missing:
            states[i] = 0
    return states
