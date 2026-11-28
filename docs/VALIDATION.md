# Validation

The validation uses synthetic arrays and the supplied synthetic generators.
Models, regressions and CLI paths are tested without study data.

| Check | Result |
|---|---|
| Inherited v0.1.0 suite, run unchanged at the baseline | 116 passed |
| Candidate synthetic suite | 172 passed |
| Deliberately broken variants | 63 detected by targeted synthetic tests, including all 56 inherited mutations |
| Same-input algorithm comparisons | 15 passed across xDAWN probabilities and diagnostics, EEG-Inception parameter counts and probabilities, nested partitions and selection, frozen Holm and decision rules |
| CPU command-line smoke | All three comparative modes, follow-up EEGNet train–test comparisons, photograph-weighted bootstrap, stream history and label-permutation procedure check |

Tests cover montage reference and missing-channel handling, fold/seed
indexing, source-only fitting and nested selection, identical test masks,
threshold ties, rank AUC, seed and draw means, frozen families, malformed
predictions, authorization and output handling. Synthetic model tests
exercise both neural architectures. Numerical comparison on synthetic inputs
addresses selected algorithms and fixtures. Reproducing manuscript values
requires the original authorized data and acquisition context.

The local verification used Python 3.11, NumPy 1.26, SciPy 1.16, scikit-learn
1.6, PyTorch 2.9, MNE 1.11 and pytest 9.1. The installed PyTorch build could
run on CPU in the verification environment. CUDA execution was unavailable.
The repository includes a manual, synthetic-only GitHub Actions workflow
with its own dependency matrix.

```bash
python -m pytest -q
python tools/mutation_check.py
```
