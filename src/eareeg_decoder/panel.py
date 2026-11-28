"""Post hoc decoder panel (candidate v0.2.0): drop-one and transfer-gap analyses for several decoders.

Decoders (``configs/protocol.json`` -> ``panel``): ``eegnet_step1`` (montage-stage EEGNet, dropout 0.25),
``eegnet_followup`` (transient/follow-up EEGNet, dropout 0.5), ``xdawn_ts_lr`` (:mod:`eareeg_decoder.xdawn`) and
``eeg_inception_erp`` (:mod:`eareeg_decoder.inception`); the shrinkage LDA of :mod:`eareeg_decoder.regimes` is the
reference decoder. All decoders share the network call signature ``fit(Xtr, ytr, Xte, seed, cfg, device) -> P(face)``
and are run through :func:`eareeg_decoder.regimes.net_probs`, so every fit sees only training epochs.

* :func:`drop_one_panel` -- the twelve montages (full + eleven drop-one) under joint participant x stimulus-group
  hold-out; per-participant BA (P(face) >= 0.5, the montage-stage rule) averaged over seeds; contrasts d_F, d_M, d_FM.
* :func:`gap_panel` -- full vs Fpz-free montage on all / free / matched training sets, scored on identical held-out
  free epochs (P(face) > 0.5, the follow-up rule); seeds averaged within participant, then matched draws; frontal
  advantage per comparison, Delta = c + s, per-draw values for the Monte Carlo SE and a completeness gate (a finite
  probability for every planned test epoch; otherwise the result is marked INCOMPLETE).
* :func:`family` -- summaries of one multiplicity family with the FROZEN family size (absent members count as p = 1
  and are labelled INCOMPLETE), bootstrap and exact sign-flip inversion CIs and the label rule.

Running these on authorized data is an independent re-analysis: fold enumeration, seeds, hardware and library versions
determine the exact values, so the manuscript's numbers are not expected to be reproduced bit for bit. Outputs are
participant level; summarise them with :mod:`eareeg_decoder.inference` before reporting.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import inference, regimes
from .config import PROTOCOL, eegnet_seed
from .decoders import eegnet_config, eegnet_fit_predict
from .inception import DEFAULT as INCEPTION_DEFAULT
from .inception import inception_fit_predict
from .metrics import balanced_accuracy
from .montage import DROP_ONE
from .xdawn import DEFAULT as XDAWN_DEFAULT
from .xdawn import xdawn_fit_predict

PANEL = PROTOCOL["panel"]
PANEL_DECODERS: tuple[str, ...] = tuple(PANEL["decoders"])
OFFSETS: tuple[int, ...] = tuple(PANEL["seeds"]["offsets"])
DETERMINISTIC = {"xdawn_ts_lr"}


def decoder(name: str) -> tuple[Callable, dict[str, Any]]:
    """``(fit, default cfg)`` of a panel or reference network decoder."""
    if name == "eegnet_step1":
        return eegnet_fit_predict, eegnet_config("montage")
    if name == "eegnet_followup":
        return eegnet_fit_predict, eegnet_config("transient")
    if name == "xdawn_ts_lr":
        return xdawn_fit_predict, dict(XDAWN_DEFAULT)
    if name == "eeg_inception_erp":
        return inception_fit_predict, dict(INCEPTION_DEFAULT)
    raise ValueError(f"unknown panel decoder {name!r}")


def offsets_for(name: str, offsets: Sequence[int] = OFFSETS) -> tuple[int, ...]:
    """Seed offsets actually run (one run for the deterministic xDAWN-TS-LR)."""
    return (0,) if name in DETERMINISTIC else tuple(offsets)


# ------------------------------------------------------------------------------------------------ scoring
def ba_from_probs(y, g, prob, *, inclusive: bool, min_face: int = 0, min_tree: int = 0,
                  subset=None) -> dict[int, float]:
    """Per-participant BA of held-out P(face) (NaN where untested); ``inclusive`` -> face if P >= 0.5 (montage
    stage), otherwise face if P > 0.5 (follow-up). NaN unless at least ``min_face``/``min_tree`` epochs (>= 1 each)."""
    y, g, prob = np.asarray(y), np.asarray(g), np.asarray(prob, dtype=float)
    out: dict[int, float] = {}
    for p in np.unique(g):
        m = (g == p) & np.isfinite(prob)
        if subset is not None:
            m = m & np.asarray(subset, bool)
        nf, nt = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        if nf < max(1, min_face) or nt < max(1, min_tree):
            out[int(p)] = float("nan")
            continue
        pred = (prob[m] >= 0.5) if inclusive else (prob[m] > 0.5)
        out[int(p)] = balanced_accuracy(y[m], pred.astype(int))
    return out


def planned_test_mask(y, g, test_mask, sg=None, min_face: int = 0, min_tree: int = 0) -> np.ndarray:
    """Epochs that :func:`eareeg_decoder.regimes.net_probs` must predict (eligible LOSO participants; every joint
    participant x group cell with test epochs)."""
    y, g, test_mask = np.asarray(y), np.asarray(g), np.asarray(test_mask, bool)
    if sg is not None:
        return test_mask & (np.asarray(sg) >= 0)
    out = np.zeros(len(y), bool)
    for p in np.unique(g):
        te = (g == p) & test_mask
        if (y[te] == 1).sum() < min_face or (y[te] == 0).sum() < min_tree or not te.any():
            continue
        out |= te
    return out


def predictions_complete(prob, planned) -> bool:
    """Completeness gate: a finite probability for EVERY planned test epoch."""
    prob = np.asarray(prob, dtype=float)
    planned = np.asarray(planned, bool)
    if not planned.any() or prob.shape != planned.shape:
        return False
    return bool(np.isnan(prob[~planned]).all() and ((prob[planned] >= 0) & (prob[planned] <= 1)).all())


# ------------------------------------------------------------------------------------------------ drop-one panel
@dataclass
class DropOneResult:
    participants: np.ndarray
    ba: dict[str, np.ndarray]                  # montage -> seed-mean BA per participant
    ba_by_seed: dict[str, np.ndarray]          # montage -> seeds x participants
    contrasts: dict[str, np.ndarray]           # d_F, d_M, d_FM per participant
    complete: bool


def drop_one_panel(X_by_montage: Mapping[str, np.ndarray], y, g, sg, name: str, *, offsets: Sequence[int] = OFFSETS,
                   device: str = "cpu", cfg: dict[str, Any] | None = None,
                   fit: Callable | None = None) -> DropOneResult:
    """Drop-one panel for one decoder. ``X_by_montage``: ``"full"`` and every ``"drop_<electrode>"`` raw window
    (epochs x channels x samples) of ALL retained epochs; joint folds over ``sg`` (stimulus groups, -1 unknown).

    Seed per fit = montage-stage joint rule ``20261107 + s + 1000 + fold index`` (index among nonempty joint test cells)."""
    fit0, cfg0 = decoder(name)
    fit = fit or fit0
    cfg = dict(cfg0 if cfg is None else cfg)
    y, g, sg = np.asarray(y), np.asarray(g), np.asarray(sg)
    allm = np.ones(len(y), bool)
    planned = planned_test_mask(y, g, allm, sg)
    configs = ("full",) + tuple(DROP_ONE)
    parts = np.unique(g)
    ba, by_seed, complete = {}, {}, True
    for m in configs:
        rows = []
        for s in offsets_for(name, offsets):
            seed0 = eegnet_seed("montage", "joint", s, 0)
            prob = regimes.net_probs(X_by_montage[m], y, g, allm, allm, seed0, cfg, sg=sg, device=device, fit=fit,
                                     compact_joint=True)
            complete &= predictions_complete(prob, planned)
            b = ba_from_probs(y, g, prob, inclusive=True)
            rows.append([b[int(p)] for p in parts])
        by_seed[m] = np.array(rows)
        ba[m] = by_seed[m].mean(axis=0)
    con = inference.montage_contrasts(ba)
    return DropOneResult(participants=parts, ba=ba, ba_by_seed=by_seed, contrasts=con, complete=bool(complete and np.isfinite(con["d_F"]).all()))


# ------------------------------------------------------------------------------------------------ transfer-gap panel
@dataclass
class GapResult:
    participants: np.ndarray
    advantage: dict[str, np.ndarray] = field(default_factory=dict)       # aa, ff, af, matched (draw mean) -> A_F
    matched_by_draw: np.ndarray | None = None                            # participants x draws of A_F(matched->free)
    decomposition: dict[str, Any] = field(default_factory=dict)          # delta, c, s
    delta_by_seed: np.ndarray | None = None                              # seeds x participants
    test_sets_identical: bool = False
    complete: bool = True
    auc_delta: np.ndarray | None = None
    ba: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)


def gap_panel(X_full, X_fpz_free, y, g, free, name: str, *, sg=None, n_draws: int | None = None,
              offsets: Sequence[int] = OFFSETS, device: str = "cpu", min_face: int = regimes.MIN_FACE,
              min_tree: int = regimes.MIN_TREE, seed_base: int = regimes.SEED_BASE,
              cfg: dict[str, Any] | None = None, fit: Callable | None = None,
              matched_to_matched: bool = False) -> GapResult:
    """Transfer-gap panel for one decoder: regimes all->all (context), free->free, all->free and matched_k->free,
    full vs Fpz-free montage, LOSO (``sg`` None) or joint. Draws default to the panel's 20 (LOSO) / 5 (joint).

    Seed per fit = follow-up rule ``20261111 + s + fold index`` (the fold index of
    :func:`eareeg_decoder.regimes.net_probs`; ineligible LOSO participants still consume an index)."""
    fit0, cfg0 = decoder(name)
    fit = fit or fit0
    cfg = dict(cfg0 if cfg is None else cfg)
    y, g, free = np.asarray(y), np.asarray(g), np.asarray(free, bool)
    design = "loso" if sg is None else "joint"
    n_draws = PANEL["matched_draws"][design] if n_draws is None else n_draws
    allm = np.ones(len(y), bool)
    parts = np.unique(g)
    res = GapResult(participants=parts)
    seeds = offsets_for(name, offsets)

    def run(train, test, mf, mt, s):
        seed0 = eegnet_seed("transient", design, s, 0)
        planned = planned_test_mask(y, g, test, sg, mf, mt)
        out, ok = {}, True
        for m, X in (("full", X_full), ("fpz_free", X_fpz_free)):
            prob = regimes.net_probs(X, y, g, train, test, seed0, cfg, sg=sg, device=device, min_face=mf,
                                     min_tree=mt, fit=fit)
            ok &= predictions_complete(prob, planned)
            out[m] = (prob, ba_from_probs(y, g, prob, inclusive=False, min_face=mf, min_tree=mt))
        return out, ok

    def af_mean(spec, *, auc=False):
        """A_F per participant for regime ``spec`` = (train, test, mf, mt), averaged over seeds."""
        rows, tests, auc_rows, ba_rows = [], [], [], []
        for s in seeds:
            out, ok = run(*spec, s)
            res.complete &= ok
            rows.append(regimes.frontal_advantage(out["full"][1], out["fpz_free"][1]))
            tests.append({m: np.isfinite(out[m][0]) for m in out})
            ba_rows.append({m: regimes.vector(out[m][1]) for m in out})
            if auc:
                scores = {m: regimes.auc_by_participant(y, g, out[m][0], spec[2], spec[3]) for m in out}
                auc_rows.append(regimes.frontal_advantage(scores["full"], scores["fpz_free"]))
        ba = {m: np.mean([b[m] for b in ba_rows], axis=0) for m in ("full", "fpz_free")}
        return np.mean(rows, axis=0), np.array(rows), tests, ba, auc_rows

    res.advantage["all->all"], _, _, res.ba["all->all"], _ = af_mean((allm, allm, 0, 0))
    ff, ff_seeds, ff_tests, res.ba["free->free"], ff_auc = af_mean((free, free, min_face, min_tree), auc=True)
    af, af_seeds, af_tests, res.ba["all->free"], af_auc = af_mean((allm, free, min_face, min_tree), auc=True)
    res.advantage.update({"free->free": ff, "all->free": af})
    res.auc_delta = np.mean(ff_auc, axis=0) - np.mean(af_auc, axis=0)
    same = all(np.array_equal(a[m], b[m]) for a, b in zip(ff_tests, af_tests) for m in a)
    draws, matched_ba = [], []
    for k in range(n_draws):
        mask = regimes.matched_random_mask(free, y, g, np.random.default_rng(seed_base + k))
        a, _, mtests, ba, _ = af_mean((mask, free, min_face, min_tree))
        matched_ba.append(ba)
        if matched_to_matched and k == 0:
            mm, _, _, res.ba["matched->matched"], _ = af_mean((mask, mask, min_face, min_tree))
            res.advantage["matched->matched"] = mm
        same &= all(np.array_equal(t[m], f[m]) for t, f in zip(mtests, ff_tests) for m in t)
        draws.append(a)
    if n_draws:
        res.ba["matched->free"] = {m: np.mean([b[m] for b in matched_ba], axis=0)
                                    for m in ("full", "fpz_free")}
        res.matched_by_draw = np.array(draws).T
        res.advantage["matched->free"] = res.matched_by_draw.mean(axis=1)
        res.decomposition = regimes.gap_decomposition(res.advantage["free->free"], res.advantage["matched->free"],
                                                      res.advantage["all->free"])
    res.delta_by_seed = ff_seeds - af_seeds
    res.test_sets_identical = bool(same)
    res.complete = bool(res.complete and same and np.isfinite(res.delta_by_seed).any())
    return res


# ------------------------------------------------------------------------------------------------ families
def drop_one_label(summary: Mapping[str, Any], p_holm: float | None, alpha: float = 0.05) -> str:
    """Montage-stage rule for d_F (hypothesised negative)."""
    lo, hi = summary["ci95"]
    if p_holm is not None and summary["mean"] < 0 and p_holm < alpha and hi < 0:
        return "FPZ_LOSS_EXCEEDS_IN_EAR_LOSS"
    if p_holm is not None and summary["mean"] > 0 and p_holm < alpha and lo > 0:
        return "IN_EAR_LOSS_EXCEEDS_FPZ_LOSS"
    return "NO_DETECTABLE_DIFFERENCE"


def family(members: Mapping[str, np.ndarray | None], m: int, n_boot: int, seeds: Sequence[int] | int, *,
           rule: str = "gap", inversion: bool = True, montage_ties: bool = False) -> dict[str, dict[str, Any]]:
    """Summaries of one Holm family with the frozen size ``m`` (``members``: name -> per-participant values, or None
    for an absent/incomplete member). ``rule``: ``"gap"`` (follow-up label), ``"drop_one"`` or ``"none"``."""
    names = list(members)
    if len(names) > m:
        raise ValueError("more members than the frozen family size")
    seeds = [seeds] * len(names) if isinstance(seeds, int) else list(seeds)
    if len(seeds) != len(names):
        raise ValueError("each family member needs a bootstrap seed")
    summ, pv = {}, []
    for name, seed in zip(names, seeds):
        v = members[name]
        if v is None or inference.finite(v).size == 0:
            summ[name] = {"status": "INCOMPLETE"}
            pv.append(np.nan)
            continue
        s = inference.paired_summary(v, n_boot, seed, test=not montage_ties)
        if montage_ties:
            s["p_signflip"] = (inference.signflip_exact_p(v, atol=0, rtol=1e-9)
                               if s["n"] <= inference.MAX_EXACT_N else None)
        if inversion and s["n"] <= inference.MAX_EXACT_N:
            interval = inference.signflip_inversion_ci(v)
            s["signflip_inversion_ci95"] = list(interval)
            if np.isinf(interval).any():
                s["inversion_note"] = "Unbounded: exact sign-flip resolution cannot reject at alpha 0.05."
        summ[name] = s
        pv.append(s.get("p_signflip"))
    adj = inference.holm_frozen(pv, m)
    for name, ph in zip(names, adj):
        s = summ[name]
        if s.get("status") == "INCOMPLETE":
            s["label"] = "INCOMPLETE"
            continue
        s["p_holm"] = float(ph)
        s["holm_m"] = m
        if rule == "gap":
            s["label"] = inference.decision_label(s, s["p_holm"])
        elif rule == "drop_one":
            s["label"] = drop_one_label(s, s["p_holm"])
    return summ
