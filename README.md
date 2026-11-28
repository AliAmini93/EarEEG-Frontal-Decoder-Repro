# EarEEG-Frontal-Decoder-Repro

MIT-licensed Python software for the experimental methods of *Frontal
Electrode Contributions and Blink-Like Transients in Ear-Centered EEG
Decoding*, a manuscript in preparation. The repository distributes code and
synthetic tests. Data, figures, trained models and study results are held separately.

## Analyses

| Command | Analysis |
|---|---|
| `drop_one` | Shrinkage LDA electrode ablation under joint participant and stimulus-group hold-out |
| `transients` | Category differences in post-onset frontal transient rates |
| `regimes` | Five shrinkage LDA train–test conditions, matched draws and paired gap decomposition |
| `panel_drop_one` | Electrode ablation with EEGNet, xDAWN-TS-LR and EEG-Inception ERP |
| `panel_regimes` | Fixed-configuration decoder panel under participant-only and joint hold-out, including a paired contrast against LDA |
| `panel_nested` | Separate joint-hold-out sensitivity with source-only hyperparameter selection |
| `network_regimes` | Follow-up EEGNet configuration for the train–test comparisons |
| `weighted_gap`, `stream_history`, `permutation_check` | Photograph-weighted bootstrap, previous-presentation control and within-participant label-permutation procedure check |

The decoders predict the displayed image category. The analyses are
exploratory and provide no source localization. The software supports
re-analysis of compatible authorized epochs and synthetic verification of
the implemented methods. See [privacy and limitations](docs/PRIVACY_AND_LIMITATIONS.md).

## Getting started

Use Python 3.10 or newer with NumPy 1.x, SciPy and scikit-learn. The neural
decoders use PyTorch, the preprocessing FIR filter uses MNE, and the tests use
pytest. The xDAWN-TS-LR implementation uses NumPy, SciPy and scikit-learn.

```bash
python scripts/eareeg_analyze.py --help
python -m pytest -q
```

[Quickstart and synthetic examples](docs/QUICKSTART.md) ·
[Authorized input schema](docs/DATA_ACCESS.md) ·
[Methods and implementation](docs/METHODS_TO_CODE_MAP.md) ·
[Reproducibility](docs/REPRODUCIBILITY.md) ·
[Validation](docs/VALIDATION.md)

## Licence and citation

MIT licence. Cite the software using [CITATION.cff](CITATION.cff) and the
specific version used. The architecture citations are in the
[methods map](docs/METHODS_TO_CODE_MAP.md).
