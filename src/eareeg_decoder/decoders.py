"""Decoders: shrinkage LDA on binned features, and an EEGNet-type network (optional PyTorch).

Shrinkage LDA (Blankertz et al., 2011; Ledoit & Wolf, 2004)
    scikit-learn ``LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=[0.5, 0.5])`` on features
    z-scored with statistics of the training rows only (NaN -> training mean). Decision value > 0 -> face (1).
    Deterministic; no hyper-parameter search.

EEGNet-type network (architecture of Lawhern et al., 2018, implemented here from the published description)
    temporal conv (F1 filters of length ``kernel``) -> BN -> depthwise spatial conv (depth multiplier D) -> BN ->
    ELU -> avg-pool 4 -> dropout -> separable conv (16-sample depthwise + 1x1 to F2) -> BN -> ELU -> avg-pool 8 ->
    dropout -> linear (2 classes). Training: per-channel z-score from training epochs (NaN -> 0), class-weighted
    cross-entropy with weights N / (2 n_c), Adam, fixed number of epochs, no early stopping, nothing fitted on test
    data. Two configurations were used: the montage analyses (dropout 0.25) and the transient / follow-up analyses
    (dropout 0.5); all other settings are equal (see ``configs/protocol.json``).

    Re-running this code does NOT reproduce the original trained parameters: those depend on the authorized data,
    hardware and library versions. Synthetic smoke tests only check determinism and shapes.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .config import PROTOCOL
from .errors import MissingDependencyError
from .features import apply_scaler, fit_scaler

try:  # threadpoolctl ships with scikit-learn; without it the fit simply runs with the default thread count
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:  # pragma: no cover
    _threadpool_limits = None


def _single_thread_blas():
    """Context manager limiting BLAS to one thread (speed only: the small least-squares solve inside shrinkage LDA
    is many times slower with multithreaded BLAS on many-core machines). Results are unaffected in our tests."""
    import contextlib  # noqa: PLC0415
    return _threadpool_limits(limits=1, user_api="blas") if _threadpool_limits else contextlib.nullcontext()


# ------------------------------------------------------------------------------------------------ LDA
def _lda() -> LinearDiscriminantAnalysis:
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])


class ShrinkageLDA:
    """Training-scaled shrinkage LDA. ``fit`` sees training rows only; ``decision_function`` is inductive
    (each test row is scored independently, with no test statistics)."""

    def fit(self, F: np.ndarray, y: np.ndarray, scaler_rows: np.ndarray | None = None) -> ShrinkageLDA:
        """Fit; ``scaler_rows`` (optional, training data only) are used for the z-scaler instead of ``F``."""
        y = np.asarray(y)
        if set(np.unique(y).tolist()) != {0, 1}:
            raise ValueError("training labels must contain both classes 0 (tree) and 1 (face)")
        self.mu_, self.sd_ = fit_scaler(F if scaler_rows is None else scaler_rows)
        with _single_thread_blas():
            self.clf_ = _lda().fit(apply_scaler(F, self.mu_, self.sd_), y)
        return self

    def decision_function(self, F: np.ndarray) -> np.ndarray:
        return self.clf_.decision_function(apply_scaler(F, self.mu_, self.sd_))

    def predict(self, F: np.ndarray) -> np.ndarray:
        return (self.decision_function(F) > 0).astype(int)


def lda_decision(Ftr: np.ndarray, ytr: np.ndarray, Fte: np.ndarray) -> np.ndarray:
    """Held-out LDA decision values for ``Fte`` after fitting scaler and LDA on ``(Ftr, ytr)``."""
    return ShrinkageLDA().fit(Ftr, ytr).decision_function(Fte)


# ------------------------------------------------------------------------------------------------ EEGNet
def _torch():
    try:
        import torch  # noqa: PLC0415
        from torch import nn  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only when torch is absent
        raise MissingDependencyError("torch", "the EEGNet-type network", "eegnet") from exc
    return torch, nn


def torch_available() -> bool:
    try:
        _torch()
    except MissingDependencyError:
        return False
    return True


def eegnet_config(stage: str = "transient") -> dict[str, Any]:
    """Network settings of the original runs: ``stage`` is ``"montage"`` (dropout 0.25) or ``"transient"`` (0.5)."""
    key = {"montage": "eegnet_montage_stage", "transient": "eegnet_transient_stage"}[stage]
    c = PROTOCOL[key]
    return {k: c[k] for k in ("F1", "D", "F2", "kernel", "dropout", "epochs", "lr", "batch")}


def build_eegnet(n_channels: int, n_times: int, F1: int = 8, D: int = 2, F2: int = 16, kernel: int = 64,
                 dropout: float = 0.5):
    """Construct the network (input shape: batch x 1 x channels x times). Requires PyTorch."""
    torch, nn = _torch()

    class EEGNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, kernel), padding=(0, kernel // 2), bias=False), nn.BatchNorm2d(F1),
                nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(),
                nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
            self.block2 = nn.Sequential(
                nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
                nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)),
                nn.Dropout(dropout))
            # Shape probe in training mode, exactly as in the original code (it also touches BatchNorm buffers).
            with torch.no_grad():
                n_flat = self.block2(self.block1(torch.zeros(1, 1, n_channels, n_times))).numel()
            self.head = nn.Linear(n_flat, 2)

        def forward(self, x):
            return self.head(self.block2(self.block1(x)).flatten(1))

    return EEGNet()


def standardise_channels(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel z-score with mean/SD over training epochs and time; NaN -> 0 (training mean); float32."""
    Xtr = np.asarray(Xtr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(Xtr, axis=(0, 2), keepdims=True)
        sd = np.nanstd(Xtr, axis=(0, 2), keepdims=True)
    mu[~np.isfinite(mu)] = 0.0
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0

    def f(X):
        return np.nan_to_num((np.asarray(X) - mu) / sd, nan=0.0).astype(np.float32)

    return f(Xtr), f(Xte)


def eegnet_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, seed: int, cfg: dict[str, Any] | None = None,
                       device: str = "cpu") -> np.ndarray:
    """Train on ``(Xtr, ytr)`` and return P(face) for ``Xte``. ``X``: epochs x channels x times.

    The random-number order follows the original trainer: seed torch and numpy, standardise, build the network,
    then draw mini-batch permutations from a CPU generator seeded with ``seed``. On CPU the result is
    deterministic for a given library build. GPU use is possible but was not exercised in this package.
    """
    torch, nn = _torch()
    cfg = dict(eegnet_config("transient") if cfg is None else cfg)
    ytr = np.asarray(ytr, dtype=np.int64)
    if set(np.unique(ytr).tolist()) != {0, 1}:
        raise ValueError("training labels must contain both classes 0 (tree) and 1 (face)")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    Ztr, Zte = standardise_channels(Xtr, Xte)
    net = build_eegnet(Ztr.shape[1], Ztr.shape[2], cfg["F1"], cfg["D"], cfg["F2"], cfg["kernel"],
                       cfg["dropout"]).to(device)
    xt = torch.from_numpy(Ztr[:, None]).to(device)
    yt = torch.from_numpy(ytr).to(device)
    counts = np.bincount(ytr, minlength=2).astype(np.float32)
    weight = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)), device=device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = xt.shape[0]
    net.train()
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n, cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            if idx.numel() < 2:
                continue
            opt.zero_grad()
            loss_fn(net(xt[idx]), yt[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        prob = torch.softmax(net(torch.from_numpy(Zte[:, None]).to(device)), dim=1)[:, 1]
    return prob.cpu().numpy()


def n_parameters(net) -> int:
    return int(sum(p.numel() for p in net.parameters()))
