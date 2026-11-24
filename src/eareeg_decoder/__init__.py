"""eareeg_decoder: montage, transient and train->test regime code for ear-centred EEG face-versus-tree decoding.

A clean re-implementation of the methods described in the manuscript "A Frontal Electrode in Ear-Centered EEG
Decoding: Retraining Without Large Blink-Like Transients Shows What Remains, Not What the Decoder Used"
(in preparation). It ships no data and no results. See README.md and docs/ for scope and limitations.

Modules
-------
montage        Eq. (1) own-mean re-reference; full, Fpz-free, in-ear, Fpz-mastoid and drop-one montages
preprocessing  FIR filter (optional MNE), epoching, baseline, stored common average, 150-uV rejection, channel QC
features       50-ms bin means over [0, 0.8) s; training-only scaling
decoders       shrinkage LDA; EEGNet-type network (optional PyTorch)
splits         joint participant x stimulus-group folds, LOSO, leakage assertions
transients     Fpz-mastoid large-transient flags, descriptor, selection screens, interval overlap
regimes        train->test regimes, frontal advantage (Eq. 3), Delta = c + s (Eq. 4), network CV
metrics        balanced accuracy, recalls, AUC, flag agreement
inference      participant bootstrap, exact sign-flip, Holm, test-inversion CI, d_F contrasts (Eq. 2)
data           fail-closed adapter for authorized data (DATA_NOT_PROVIDED otherwise)
pipeline       helpers chaining the steps
synthetic      explicitly synthetic cohorts
"""
from .config import PROTOCOL, protocol
from .errors import DataNotProvidedError, LeakageError, MissingDependencyError, SchemaError

__version__ = "0.1.0"

__all__ = ["PROTOCOL", "protocol", "DataNotProvidedError", "LeakageError", "MissingDependencyError", "SchemaError",
           "__version__"]
