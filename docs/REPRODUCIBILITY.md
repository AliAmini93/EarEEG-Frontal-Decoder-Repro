# Reproducibility

This package provides runnable estimators and synthetic tests. To repeat an
analysis, record the software commit, Python and package versions, device,
time grid, eligibility rules, stimulus grouping, seed offsets, matched draw
count and input provenance. Use `configs/protocol.json` for the method defaults.
Run `python -m pytest -q` for the local synthetic suite.

Electrode ablation enumerates nonempty joint test cells for seed indexing
and uses float32 windows. Transfer and nested comparisons use float64 windows
before model-specific conversion. Transfer joint indexing advances over every
participant/group cell, including empty cells. In LOSO, an ineligible
participant still consumes its index. The stimulus keys must identify whole
stimulus units consistently, and grouping must precede fitting. The two
hold-out schemes answer different generalization questions.

Preprocessing of the supplied epochs and deterministic montage algebra are
kept separate from model fittings. Fitted scalers, xDAWN filters, covariance
geometry, model parameters and inner selection use source partitions only.
The nested sensitivity tunes on all retained source epochs in each outer fold,
then uses that setting for both training conditions on the same target epochs.

Neural seeds index folds and offsets, and balanced accuracies are computed
per seed before averaging. Matched-training draws use the same masks for
both montages. The LOSO and joint fixed panels use 20 and 5 draws. The joint
LDA reference uses 20. LDA's opt-in adaptive rule extends 20 to 50 when the
composition Monte Carlo standard error exceeds 0.0025. The follow-up network
provides the descriptive one-draw LOSO matched-to-matched comparison.

The audited validation uses synthetic data. Identical empirical values depend
on original authorized inputs, original partitions, and a compatible numerical
runtime. CPU and GPU floating-point and training behavior can differ. This
package stores no per-fit checkpoints and supplies no study-data acquisition
or proprietary event adapter.
