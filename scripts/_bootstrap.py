"""Shared start-up for the command-line scripts (no installation required).

* Limits BLAS/OpenMP threads to one unless the caller set them (speed only: multithreaded LAPACK is slow for the
  small solves inside shrinkage LDA on many-core machines).
* Makes ``src/`` importable when the package is not installed.
"""
import os
import sys
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
