"""Supplementary analyses for caller-authorized, aligned inputs.

Vectors and intermediate scores can describe individuals. Callers are responsible
for their access and disclosure terms. The functions do not read files.
"""
from __future__ import annotations

import numpy as np

from . import features, metrics, regimes, transients


def _aligned(y, participant, *vectors):
    y, g = np.asarray(y), np.asarray(participant)
    if y.ndim != 1 or y.size == 0 or g.shape != y.shape or not np.isin(y, (0, 1)).all():
        raise ValueError("expected aligned nonempty binary labels and participant codes")
    if any(np.shape(v) != y.shape for v in vectors):
        raise ValueError("each metadata vector needs one value per presentation")
    return y, g


def photograph_weighted_gap(y, participant, stimulus_key, free, *, retrained_full, retrained_free,
                            transferred_full, transferred_free, n_boot=20_000, seed=20261117):
    """Two-way bootstrap and leave-one-photograph-out paired BA gap.

    The four score vectors share one free-epoch mask. Score > 0 predicts
    class 1. The face recall stays fixed under photo resampling. For each
    participant, tree recall is weighted correct counts divided by weighted
    epoch counts. Zero weighted tree count contributes zero tree recall,
    matching the denominator-floor convention of this sensitivity analysis.
    """
    scores = np.asarray([retrained_full, retrained_free, transferred_full, transferred_free], dtype=float)
    y, g = _aligned(y, participant, stimulus_key, free, *scores)
    use = np.asarray(free)
    if use.dtype != bool or not np.isfinite(scores[:, use]).all() or n_boot < 1:
        raise ValueError("expected finite scores on a shared boolean mask and positive n_boot")
    y, g, score = y[use], g[use], scores[:, use]
    key = np.asarray(stimulus_key, dtype=str)[use]
    if (key[y == 0] == "").any():
        raise ValueError("every scored tree epoch needs a photograph key")
    parts, photos = np.unique(g), np.unique(key[y == 0])
    if not parts.size or not photos.size:
        raise ValueError("the selected epochs must include participants and tree photographs")
    contrast = np.array([1, -1, -1, 1])
    face, tree = y == 1, y == 0
    fp, count, correct = [], [], []
    for p in parts:
        fr = face & (g == p)
        if not fr.any() or not (tree & (g == p)).any():
            raise ValueError("each selected participant needs both labels")
        fp.append(contrast @ (score[:, fr] > 0).mean(axis=1))
        cells = tree & (g == p) & (key == photos[:, None])
        count.append(cells.sum(axis=1))
        correct.append(contrast @ ((score <= 0).astype(int) @ cells.T))
    fp, count, correct = np.asarray(fp), np.asarray(count), np.asarray(correct)

    def gap(pi, weights):
        tr = (correct[pi] @ weights) / np.maximum(count[pi] @ weights, 1e-12)
        return float(np.mean((fp[pi] + tr) / 2))

    pi = np.arange(len(parts))
    unit = np.ones(len(photos), int)
    mean = gap(pi, unit)
    leave_one_out = [gap(pi, unit - (np.arange(len(photos)) == j)) for j in range(len(photos))]
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_boot)
    for b in range(n_boot):
        p = rng.integers(len(parts), size=len(parts))
        j = rng.integers(len(photos), size=len(photos))
        bootstrap[b] = gap(p, np.bincount(j, minlength=len(photos)))
    return {"mean": mean, "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "leave_one_out_range": [min(leave_one_out), max(leave_one_out)]}


def h1_label_permutation(F, y, participant, free, *, n_permutations=20, seed=20261117,
                         min_face=regimes.MIN_FACE, min_tree=regimes.MIN_TREE):
    """Label-permutation procedure check for the LOSO paired LDA gap.

    Shuffle within each participant. Keep the label-blind transient mask and
    refit both montages for both training sets on every repetition.
    """
    y, g = _aligned(y, participant, free)
    free = np.asarray(free)
    if free.dtype != bool or n_permutations < 1:
        raise ValueError("expected a boolean transient mask and positive n_permutations")
    all_rows = np.ones(len(y), bool)
    parts = [np.flatnonzero(g == p) for p in np.unique(g)]

    def gap(labels):
        advantage = []
        for train in (free, all_rows):
            ba = {}
            for montage in ("full", "fpz_free"):
                score = regimes.cv_scores(F[montage], labels, g, train, free)
                ba[montage] = regimes.ba_by_participant(labels, g, score, min_face, min_tree)
            advantage.append(regimes.frontal_advantage(ba["full"], ba["fpz_free"]))
        d = advantage[0] - advantage[1]
        if not np.isfinite(d).any():
            raise ValueError("no eligible paired scores in a label permutation")
        return float(np.nanmean(d))

    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_permutations):
        permuted = y.copy()
        for rows in parts:
            permuted[rows] = y[rng.permutation(rows)]
        null.append(gap(permuted))
    return np.asarray(null)


def stream_history_control(y, participant, stream, order, flagged, keep, *, n_rotations=1000, seed=20261117):
    """Previous-presentation LDA and refitted within-stream cyclic null.

    Supply all ordered presentations, including rejected epochs. Previous
    features are built before applying keep. Rotations act only on retained
    targets and leave those features fixed. Each stream belongs to one
    participant and labels rotate by a uniformly chosen nonzero shift.
    """
    y, g = _aligned(y, participant, stream, order, flagged, keep)
    stream, order, flag, keep = map(np.asarray, (stream, order, flagged, keep))
    if flag.dtype != bool or keep.dtype != bool or n_rotations < 1:
        raise ValueError("expected boolean flags, boolean keep and positive n_rotations")
    prev_label = np.full(len(y), -1)
    prev_flag = np.zeros(len(y), bool)
    groups = []
    for p in np.unique(g):
        for stream_id in np.unique(stream[g == p]):
            rows = np.flatnonzero((g == p) & (stream == stream_id))
            if len(np.unique(order[rows])) != len(rows):
                raise ValueError("presentation order must be unique within a stream")
            rows = rows[np.argsort(order[rows], kind="stable")]
            prev_label[rows[1:]], prev_flag[rows[1:]] = y[rows[:-1]], flag[rows[:-1]]
            retained = rows[keep[rows]]
            if len(retained) > 1:
                groups.append(retained)
    F = np.column_stack((prev_label == 0, prev_label == 1, prev_label < 0, prev_flag)).astype(float)
    if not groups:
        raise ValueError("no stream contains at least two retained presentations")

    def cohort_ba(labels):
        scores = regimes.cv_scores(F, labels, g, keep, keep)
        ba = regimes.vector(regimes.ba_by_participant(labels, g, scores))
        if not np.isfinite(ba).any():
            raise ValueError("no eligible participant for the stream-history control")
        return float(np.nanmean(ba))

    observed = cohort_ba(y)
    rng = np.random.default_rng(seed)
    null = np.empty(n_rotations)
    for i in range(n_rotations):
        target = y.copy()
        for rows in groups:
            target[rows] = np.roll(y[rows], rng.integers(1, len(rows)))
        null[i] = cohort_ba(target)
    return {"observed_ba": observed, "null_mean": float(null.mean()), "null_p95": float(np.quantile(null, 0.95)),
            "p_rotation": float((1 + (null >= observed).sum()) / (1 + n_rotations)), "n_rotations": n_rotations}


def propagation_slope(frontal, in_ear, times, *, theta=transients.THETA_UV, min_transients=10):
    """Median centered-waveform slope for one participant on flagged epochs.

    Inputs are Fpz minus good mastoids and mean good strict in-ear channels
    minus the same mastoid mean. Missing derivations or a zero denominator
    give NaN. The sign refers to the supplied derivations.
    """
    a, b, times = (np.asarray(item, dtype=float) for item in (frontal, in_ear, times))
    if a.ndim != 2 or a.shape != b.shape or times.shape != (a.shape[1],) or min_transients < 1:
        raise ValueError("expected equal epoch-by-time derivations and matching times")
    if not np.isfinite(a).all():
        return float("nan")
    flags = transients.transient_flags(a, times, theta).any
    a, b = a[flags], b[flags]
    if len(a) < min_transients or not np.isfinite(b).all():
        return float("nan")
    a, b = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
    den = np.einsum("ij,ij->i", a, a)
    if (den <= 0).any():
        return float("nan")
    return float(np.median(np.einsum("ij,ij->i", a, b) / den))


def cross_device_control(ear_frontal, scalp_frontal, scalp_eog, times, y, participant, scalp_keep, *,
                         ear_index, scalp_index, alignment_provenance, scale_provenance,
                         eog_placement, theta=transients.THETA_UV):
    """Paired frontal-detector and scalp EOG LOSO controls.

    Sources must be independently mapped to presentations by the caller.
    The arrays represent preprocessed nominal-microvolt derivations and
    a common epoch grid. Aligned y and participant label the mapping rows;
    scalp_keep labels scalp source rows. Metadata is required and stays with
    the caller. The result does not establish hardware calibration or timing.
    """
    y, g = _aligned(y, participant, ear_index, scalp_index)
    ear, scalp, eog, times, keep = map(np.asarray, (ear_frontal, scalp_frontal, scalp_eog, times, scalp_keep))
    if (not all(isinstance(text, str) and text.strip() for text in
                (alignment_provenance, scale_provenance, eog_placement))
            or times.ndim != 1 or not np.isfinite(times).all() or not (np.diff(times) > 0).all()
            or ear.ndim != 2 or scalp.ndim != 2 or eog.ndim != 3 or not eog.shape[1]
            or ear.shape[-1] != len(times) or scalp.shape[-1] != len(times)
            or eog.shape[0] != len(scalp) or eog.shape[-1] != len(times)
            or keep.dtype != bool or keep.shape != (len(scalp),)):
        raise ValueError("incompatible epoch arrays, times or missing provenance")

    def mapped(rows, length):
        rows = np.asarray(rows)
        if (not np.issubdtype(rows.dtype, np.integer) or (rows < 0).any() or (rows >= length).any()
                or len(np.unique(rows)) != len(rows)):
            raise ValueError("event map must be one-to-one and in bounds")
        return rows

    ei, si = mapped(ear_index, len(ear)), mapped(scalp_index, len(scalp))
    if not all(np.isfinite(x).all() for x in (ear[ei], scalp[si], eog[si])):
        raise ValueError("mapped channels must have finite samples")
    ear_flags = transients.transient_flags(ear[ei], times, theta)
    scalp_flags = transients.transient_flags(scalp[si], times, theta)
    detector_rows = []
    for p in np.unique(g):
        test = g == p
        tree = y[test] == 0
        if not tree.any() or tree.all():
            raise ValueError("each participant needs both classes")
        detector_rows.append([
            transients.rate_difference(ear_flags.post[test], tree),
            transients.rate_difference(scalp_flags.post[test], tree),
            metrics.flag_agreement(ear_flags.any[test], scalp_flags.any[test])["kappa"],
        ])
    F = features.bin_means(eog[si], times)
    eog_ba = {}
    for name, mask in (("retained", keep[si]), ("all", np.ones(len(si), bool))):
        score = regimes.cv_scores(F, y, g, mask, mask)
        eog_ba[name] = regimes.vector(regimes.ba_by_participant(y, g, score))
    rates = np.asarray(detector_rows)
    finite = rates[:, 2][np.isfinite(rates[:, 2])]
    return {"ear_rate_gap_mean": float(rates[:, 0].mean()), "scalp_rate_gap_mean": float(rates[:, 1].mean()),
            "flag_kappa_median": float(np.median(finite)) if finite.size else None,
            "eog_ba": {name: float(np.nanmean(ba)) for name, ba in eog_ba.items()}}
