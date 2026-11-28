"""EEG-Inception for ERP decoding (Santamaria-Vazquez et al., 2020), re-implemented in PyTorch from the published
architectural specification and adapted to this package's input and training protocol (optional PyTorch).

No third-party model code is used or copied (the original TensorFlow implementation is GPL-3.0). Adaptation choices,
fixed before any use on study data and documented in ``configs/protocol.json`` (``panel.eeg_inception_erp``):

* Input: epochs x channels x samples (200 samples at 250 Hz for [0, 0.8) s). The temporal scales are kept in
  physical time: (0.5, 0.25, 0.125) s -> ``int(s * sfreq)`` = (125, 62, 31) samples; block-2 kernels are those // 4.
* Block 1 (per scale): temporal Conv2d(1 -> 8, (1, k), 'same', bias) -> BatchNorm -> ELU -> Dropout -> depthwise
  spatial Conv2d(8 -> 16, (channels, 1), groups=8, no bias) -> BatchNorm -> ELU -> Dropout; concatenation -> AvgPool 4.
* Block 2 (per scale): Conv2d(48 -> 8, (1, k // 4), 'same', no bias) -> BatchNorm -> ELU -> Dropout; concatenation ->
  AvgPool 2.
* Output: Conv2d(24 -> 12, (1, 8)) -> BN -> ELU -> AvgPool 2 -> Dropout -> Conv2d(12 -> 6, (1, 4)) -> BN -> ELU ->
  AvgPool 2 -> Dropout -> Linear(6 * floor(T / 32), 2).
* BatchNorm momentum 0.01, eps 1e-5; PyTorch default initialisation; no max-norm constraint; dropout 0.5.
* Training: the protocol of this package's EEGNet-type networks (per-channel z-score from training epochs, NaN -> 0;
  class-weighted cross-entropy N / (2 n_c); Adam, learning rate 0.001; batch 128; 30 epochs; no early stopping;
  P(face) from the softmax in eval mode). This is NOT the original study's calibration/transfer protocol.

Trainable parameters: 25,542 with 11 channels and 200 samples, 25,494 with 10; with the original configuration (8
channels, 128 samples at 128 Hz) the same code gives 14,926, the closed-form count of the specification
(:func:`analytic_param_count`; checked by the tests).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .config import PROTOCOL
from .decoders import _torch, standardise_channels

_P = PROTOCOL["panel"]["eeg_inception_erp"]
SCALES_S: tuple[float, ...] = tuple(_P["scales_s"])
POOLS: tuple[int, ...] = tuple(_P["pools"])
DEFAULT: dict[str, Any] = {k: _P[k] for k in ("dropout", "lr", "batch", "epochs")}


def scale_samples(sfreq: float, scales_s=SCALES_S) -> list[int]:
    return [int(s * sfreq) for s in scales_s]


def build_inception(n_chans: int, n_times: int, sfreq: float = 250.0, n_classes: int = 2, n_filters: int = 8,
                    depth_multiplier: int = 2, dropout: float = 0.5, scales_s=SCALES_S, pools=POOLS,
                    bn_momentum: float = 0.01):
    """Construct the network (input: batch x channels x times). Requires PyTorch."""
    torch, nn = _torch()
    ks = scale_samples(sfreq, scales_s)
    if min(k // 4 for k in ks) < 1:
        raise ValueError("temporal scale shorter than 4 samples")

    class EEGInceptionERP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kernels_block1 = ks
            self.kernels_block2 = [k // 4 for k in ks]
            nb = len(ks)
            self.block1 = nn.ModuleList([nn.Sequential(
                nn.Conv2d(1, n_filters, (1, k), padding="same", bias=True),
                nn.BatchNorm2d(n_filters, momentum=bn_momentum), nn.ELU(), nn.Dropout(dropout),
                nn.Conv2d(n_filters, n_filters * depth_multiplier, (n_chans, 1), groups=n_filters, bias=False),
                nn.BatchNorm2d(n_filters * depth_multiplier, momentum=bn_momentum), nn.ELU(), nn.Dropout(dropout))
                for k in ks])
            self.pool1 = nn.AvgPool2d((1, pools[0]))
            c1 = nb * n_filters * depth_multiplier
            self.block2 = nn.ModuleList([nn.Sequential(
                nn.Conv2d(c1, n_filters, (1, k // 4), padding="same", bias=False),
                nn.BatchNorm2d(n_filters, momentum=bn_momentum), nn.ELU(), nn.Dropout(dropout)) for k in ks])
            self.pool2 = nn.AvgPool2d((1, pools[1]))
            c2 = nb * n_filters
            self.out1 = nn.Sequential(nn.Conv2d(c2, c2 // 2, (1, 8), padding="same", bias=False),
                                      nn.BatchNorm2d(c2 // 2, momentum=bn_momentum), nn.ELU(),
                                      nn.AvgPool2d((1, pools[2])), nn.Dropout(dropout))
            self.out2 = nn.Sequential(nn.Conv2d(c2 // 2, c2 // 4, (1, 4), padding="same", bias=False),
                                      nn.BatchNorm2d(c2 // 4, momentum=bn_momentum), nn.ELU(),
                                      nn.AvgPool2d((1, pools[3])), nn.Dropout(dropout))
            with torch.no_grad():
                self.n_flat = self.features(torch.zeros(1, n_chans, n_times)).shape[1]
            self.head = nn.Linear(self.n_flat, n_classes)

        def features(self, x):
            x = x.unsqueeze(1)
            x = self.pool1(torch.cat([b(x) for b in self.block1], dim=1))
            x = self.pool2(torch.cat([b(x) for b in self.block2], dim=1))
            return self.out2(self.out1(x)).flatten(1)

        def forward(self, x):
            return self.head(self.features(x))

    return EEGInceptionERP()


def n_trainable(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def analytic_param_count(n_chans: int, n_times: int, sfreq: float = 250.0, n_filters: int = 8,
                         depth_multiplier: int = 2, n_classes: int = 2, scales_s=SCALES_S, pools=POOLS) -> int:
    """Closed-form trainable-parameter count of the specification above."""
    ks = scale_samples(sfreq, scales_s)
    nb, F, D = len(ks), n_filters, depth_multiplier
    b1 = sum(F * k + F for k in ks) + nb * 2 * F + nb * F * D * n_chans + nb * 2 * F * D
    c1 = nb * F * D
    b2 = sum(c1 * F * (k // 4) for k in ks) + nb * 2 * F
    c2 = nb * F
    o1 = c2 * (c2 // 2) * 8 + 2 * (c2 // 2)
    o2 = (c2 // 2) * (c2 // 4) * 4 + 2 * (c2 // 4)
    t = n_times
    for p in pools:
        t //= p
    return b1 + b2 + o1 + o2 + (c2 // 4) * t * n_classes + n_classes


def inception_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, seed: int,
                          cfg: dict[str, Any] | None = None, device: str = "cpu", sfreq: float = 250.0) -> np.ndarray:
    """Train on ``(Xtr, ytr)`` and return P(face) for ``Xte`` (``FitFn`` signature). Raises ``FloatingPointError``
    on a non-finite training loss. Deterministic on CPU for a given library build."""
    torch, nn = _torch()
    cfg = dict(DEFAULT if cfg is None else cfg)
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
    net = build_inception(Ztr.shape[1], Ztr.shape[2], sfreq=sfreq, dropout=cfg["dropout"]).to(device)
    xt = torch.from_numpy(Ztr).to(device)
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
            loss = loss_fn(net(xt[idx]), yt[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            opt.step()
    net.eval()
    out = []
    with torch.no_grad():
        xe = torch.from_numpy(Zte)
        for i in range(0, xe.shape[0], 1024):
            out.append(torch.softmax(net(xe[i:i + 1024].to(device)), dim=1)[:, 1].cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, np.float32)
