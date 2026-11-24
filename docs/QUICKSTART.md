# Quick start

Nothing below installs, downloads or uploads anything. No data are shipped: the analysis scripts need your own
authorized epoch file and otherwise stop with `DATA_NOT_PROVIDED`.

## 1. Environment

* Python ≥ 3.10 (tested: 3.11.15).
* Required: `numpy` (tested 1.26.4), `scipy` (1.16.0), `scikit-learn` (1.6.1).
* Optional: `torch` (2.9.1; EEGNet), `mne` (1.11.0; FIR filter), `pytest` (9.1.1).

The repository never installs packages. If something is missing, the code raises
`BLOCKED_MISSING_DEPENDENCY: '<package>' is required for ...` and the corresponding tests are skipped with the same
reason. To add a dependency yourself, use your own environment manager, e.g. `pip install "torch>=2.0"`
(or install the package with an extra: `pip install -e ".[eegnet,filter,test]"`).

Multithreaded BLAS is much slower than one thread for the small solves inside shrinkage LDA on many-core
machines; the LDA fit therefore limits BLAS to one thread internally (via `threadpoolctl`, which scikit-learn
installs), and the scripts and tests also set `OPENBLAS_NUM_THREADS=1` when it is unset.

## 2. Run the tests (synthetic, optional)

```bash
cd EarEEG-Frontal-Decoder-Repro
python -m pytest -q
```

`pytest` finds the package through `pythonpath = ["src"]` in `pyproject.toml`. Select groups with markers, e.g.
`python -m pytest -q -m "lda and joint"` (markers: `lda`, `eegnet`, `joint`, `loso`, `sensitivity`, `oracle`).
The tests were not re-run for `v0.1.0` (see [VALIDATION.md](VALIDATION.md)).

## 3. Data check and analyses

```bash
python scripts/eareeg_check_data.py ; echo "exit=$?"
```

prints `DATA_NOT_PROVIDED` and `exit=3`. With your own authorized file, kept **outside** the repository:

```bash
python scripts/eareeg_check_data.py --data ../my_data/epochs.npz --i-am-authorized
python scripts/eareeg_analyze.py --data ../my_data/epochs.npz --i-am-authorized --analysis drop_one   --out ../my_results/drop_one.json
python scripts/eareeg_analyze.py --data ../my_data/epochs.npz --i-am-authorized --analysis transients --out ../my_results/transients.json
python scripts/eareeg_analyze.py --data ../my_data/epochs.npz --i-am-authorized --analysis regimes    --out ../my_results/regimes_loso.json
python scripts/eareeg_analyze.py --data ../my_data/epochs.npz --i-am-authorized --analysis regimes --joint --out ../my_results/regimes_joint.json
python scripts/eareeg_analyze.py --help
```

Exit codes: 0 success, 2 argument error, 3 `DATA_NOT_PROVIDED`, 4 schema error, 6 refused output path (inside the
repository, missing directory, or existing file without `--overwrite`).

See [DATA_ACCESS.md](DATA_ACCESS.md) for the file format.

## 4. Using the library

Analyses without a command-line entry point (network, follow-up variants; see the README) are called from your own
script. Without installing, run from the repository root with the source on the path, e.g.
`PYTHONPATH=src python my_script.py`. The pattern below uses a synthetic cohort only so that it runs without data:

```python
from eareeg_decoder import inference, pipeline, regimes, synthetic

es, _ = synthetic.synthetic_epochs(n_participants=8, seed=0)        # SYNTHETIC; use data.load_authorized_epochs(...) for real data
S = pipeline.build_stack(es)                                         # features, labels, free flags
res = regimes.lda_regimes(S.F, S.y, S.g, S.free, n_draws=20)         # LOSO; pass S.sg for joint hold-out
d = res.decomposition                                                # delta = c + s per participant
print(inference.paired_summary(d["delta"], n_boot=20000, seed=20261117))
print(inference.signflip_inversion_ci(d["delta"]))
```
