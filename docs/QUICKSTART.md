# Quickstart

Run these commands from the repository root with an existing Python
environment that has the dependencies in `pyproject.toml`.

## Generate a synthetic epoch file

Choose an existing output directory outside this checkout. Keep the file
there with the analysis outputs.

```bash
export EAREEG_SCRATCH="$(mktemp -d)"
PYTHONPATH=src python - <<'PY'
import os
from pathlib import Path
from eareeg_decoder.data import save_epochs
from eareeg_decoder.synthetic import synthetic_epochs

epochs, _ = synthetic_epochs(n_participants=10, n_streams=10, seed=33, neural_amp=6.0)
save_epochs(epochs, Path(os.environ["EAREEG_SCRATCH"]) / "synthetic.npz")
PY
python scripts/eareeg_check_data.py --data "$EAREEG_SCRATCH/synthetic.npz" --i-am-authorized
```

## Run the comparative decoder methods

These commands use the deterministic xDAWN-TS-LR decoder. The shortened draw
and bootstrap settings are for exercising the code. Omit them to use the
frozen defaults. The nested command fits all four settings in all eligible
inner folds for both montages and takes longer.

```bash
python scripts/eareeg_analyze.py --data "$EAREEG_SCRATCH/synthetic.npz" --i-am-authorized \
  --analysis panel_drop_one --decoder xdawn_ts_lr --n-boot 100 \
  --out "$EAREEG_SCRATCH/ablation.json"
python scripts/eareeg_analyze.py --data "$EAREEG_SCRATCH/synthetic.npz" --i-am-authorized \
  --analysis panel_regimes --decoder xdawn_ts_lr --draws 1 --n-boot 100 --min-face 10 --min-tree 3 \
  --out "$EAREEG_SCRATCH/transfer.json"
python scripts/eareeg_analyze.py --data "$EAREEG_SCRATCH/synthetic.npz" --i-am-authorized \
  --analysis panel_nested --decoder xdawn_ts_lr --n-boot 100 --min-face 10 --min-tree 3 \
  --out "$EAREEG_SCRATCH/nested.json"
```

The decoder choices also include `eegnet_step1`, `eeg_inception_erp` and `all`
(the default). The network default device is CPU. Select a supported device
with `--device`. The full neural suite can require substantial computation.
A partial model run preserves every family position and marks omitted
members `INCOMPLETE`.

## Run other analyses

Change `--analysis` and choose a new output filename for each command.

- `regimes --joint` gives the joint LDA reference. It defaults to 20 draws.
  Use `--adaptive-draws` for the optional Monte Carlo extension.
- `network_regimes --device cpu --offsets 0 --draws 1` exercises the follow-up
  EEGNet configuration. The subset of offsets and draws is for smoke testing.
- `weighted_gap --n-boot 100` uses the tree-photograph key and a common free test mask.
- `stream_history --repetitions 3` requires stream and position.
- `permutation_check --repetitions 3 --min-face 10 --min-tree 3` refits the LDA
  procedure on synthetic permutations.

JSON contains cohort summaries by default. Undefined statistics are `null`.
Output files are owner-only and the path must be outside the checkout. For
authorized study input, follow the [data contract](DATA_ACCESS.md).
