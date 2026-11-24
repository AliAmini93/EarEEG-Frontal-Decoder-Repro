# Reproducibility: what can and cannot be reproduced

## Level A — synthetic code-path reproducibility (available)

Every algorithm in this repository runs on synthetic inputs with fixed seeds:

| Item | How to check |
|---|---|
| Unit, oracle, leakage and integration tests | `python -m pytest -q` |
| Mutation check (deliberately broken copies must fail the tests) | `python tools/mutation_check.py` |

Determinism:

* LDA, splits, stimulus groups, matched draws (`default_rng(20261111 + k)`), bootstrap
  (`default_rng(seed).integers(...)`), sign-flip enumeration and Holm are deterministic given the inputs.
* The network is deterministic on CPU for a given PyTorch build and seed. GPU results can differ from CPU results;
  the original network runs used a GPU with deterministic cuDNN settings, which this package sets when
  `device="cuda"` but has not exercised.
* Results can differ at floating-point level across numpy/scipy/scikit-learn/BLAS versions and thread counts.

**Equivalence to the original analysis code.** Before this release candidate, the maintainers compared each
re-implemented function with the authors' original (private) analysis code on synthetic inputs only: re-reference
encodings, features, scaler, LDA decisions, regime scores, matched masks, transient flags and descriptor, folds and
stimulus groups, BA/AUC, bootstrap, sign-flip, Holm, inversion intervals, epoching/filtering and a short CPU
network fit. Differences that exist are listed in [VALIDATION.md](VALIDATION.md). This shows algorithmic fidelity on
synthetic inputs; it is **not** an independent verification of the original experiment.

## Level B — the paper's figures and tables (not in scope)

This is an experimental-code repository. It ships no aggregate values, figures, tables or figure-drawing code; the
analysis scripts write numerical summaries only.

## Level C — end-to-end replication (not performed)

Replicating balanced accuracies, intervals and p values requires (1) legitimately obtained recordings from the
dataset controller, (2) the same preprocessing, quality control, stimulus keys and folds, (3) the same library
versions and, for networks, comparable hardware, and (4) an independently repeated analysis. None of this was done
while packaging. `scripts/eareeg_analyze.py` can run the LDA analyses on authorized epochs, but its output is an
**independent re-analysis**, not a reproduction, unless all four conditions are met and documented.

Specific obstacles to exact replication even with data:

* stimulus-group membership depends on the exact stimulus key strings (salted hash order);
* the original analyses fixed participant order, eligibility and seeds per fold (see `configs/protocol.json`);
* channel QC and epoch retention depend on recording-level details outside this package (file import, units);
* the eye-tracker analyses need a trigger-based clock map that is not included (see METHODS_TO_CODE_MAP).

## Versions tested

Python 3.11.15; numpy 1.26.4; scipy 1.16.0; scikit-learn 1.6.1; torch 2.9.1 (CPU execution only); mne 1.11.0;
pytest 9.1.1; Linux x86-64 (reported for `v0.1.0rc1`; not re-run for `v0.1.0`).
The optional CI workflow pins Python 3.10/3.11 with numpy 1.26.4, scipy 1.15.3, scikit-learn 1.6.1 and pytest
8.3.5; it is manual-dispatch only and has not been run. numpy 2 is not supported yet (untested).
