# Changelog

## 0.1.0 — first public release (2026-11-24)

Experimental code for the methods of the manuscript (in preparation). Same algorithm code as the private release
candidate 0.1.0rc1 (2026-11-23), published from a fresh repository.

* `eareeg_decoder` package: montage algebra and drop-one montages; label-blind preprocessing (FIR filter via
  optional MNE, epoching, baseline, stored common average, 150-µV rule, channel QC); binned features and
  training-only scaling; shrinkage LDA and EEGNet-type network (optional PyTorch); joint and LOSO splits with
  leakage assertions; large-transient flags, descriptor and selection screens; train→test regimes, matched draws
  and Δ = c + s; balanced accuracy, AUC, flag agreement; participant bootstrap, exact sign-flip, Holm and
  test-inversion intervals; fail-closed data adapter; synthetic generators for tests.
* Scripts: data-contract check and LDA analyses of authorized epochs (`drop_one`, `transients`, `regimes`); both
  fail closed without data.
* Synthetic tests (unit, oracle, leakage canaries, integration) and a mutation check.
* Added `LICENSE` (MIT) and `CITATION.cff`.
* Removed relative to 0.1.0rc1: the synthetic demo script, synthetic examples, the optional plotting helper and its
  `figures` extra, figure-provenance and release-process documents, and the 7 tests that exercised the demo and
  examples. No algorithm code changed. Tests were not re-run for this release (see `docs/VALIDATION.md`).
