#!/usr/bin/env python3
"""Run the analyses on AUTHORIZED epochs (level C: an independent re-analysis, not a reproduction).

Fails closed with ``DATA_NOT_PROVIDED`` (exit 3) unless ``--data`` and ``--i-am-authorized`` are both given.
Writes cohort-level summaries to ``--out`` (a JSON file you choose, outside this repository). Participant-level values
are written only with ``--write-participant-values`` and must then be handled under your data agreement. Nothing is
uploaded.

LDA analyses (release 0.1.0): ``drop_one`` (Step 1; joint folds; all participants), ``transients`` (Step 2;
usable-Fpz participants), ``regimes`` (Step 3; LOSO by default, ``--joint`` for participant x stimulus-group folds;
``--adaptive-draws`` applies the 20 -> 50 matched-draw rule).

Decoder panel (candidate 0.2.0; post hoc; ``--decoder`` one of eegnet_step1, xdawn_ts_lr, eeg_inception_erp, or
``all``): ``panel_drop_one`` (twelve montages, joint folds, d_F / d_M / d_FM families), ``panel_regimes`` (transfer
gap under LOSO and joint hold-out, primary and secondary families against the LDA reference), ``panel_nested``
(source-only nested tuning sensitivity, joint folds, separate family). Families keep their frozen sizes: a decoder
not run is reported INCOMPLETE and counts as p = 1. The networks need PyTorch; runs can take many hours on real data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

EXIT_DATA_NOT_PROVIDED = 3
EXIT_SCHEMA = 4
EXIT_REFUSED = 6
PANEL_ANALYSES = ("panel_drop_one", "panel_regimes", "panel_nested", "network_regimes")
SUPPLEMENT_ANALYSES = ("weighted_gap", "permutation_check", "stream_history")


def _positive_int(text: str) -> int:
    v = int(text)
    if v < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return v


def _theta(text: str):
    if text == "relative":
        return text
    try:
        v = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("a number in nominal uV or 'relative'") from exc
    if v <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return v


def _offsets(text: str) -> tuple[int, ...]:
    try:
        vals = tuple(int(x) for x in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("comma-separated seed offsets, e.g. 0,100,200") from exc
    if not vals or len(set(vals)) != len(vals) or any(v not in (0, 100, 200) for v in vals):
        raise argparse.ArgumentTypeError("seed offsets must be distinct and among 0, 100, 200")
    return vals


def _inversion(values):
    """Exact sign-flip inversion CI, or None (with the reason in the summary's p_note) when n > 24."""
    from eareeg_decoder import inference
    n = int(np.isfinite(np.asarray(values, dtype=float)).sum())
    return list(inference.signflip_inversion_ci(values)) if n <= inference.MAX_EXACT_N else None


def _check_out(a, out_path: Path) -> int:
    """Reject unsafe output locations before opening input."""
    data_path = Path(a.data).resolve()
    if out_path == data_path or (out_path.exists() and data_path.exists() and out_path.samefile(data_path)):
        print("REFUSED: --out must not replace the input", file=sys.stderr)
        return EXIT_REFUSED
    repo_root = Path(__file__).resolve().parents[1]
    if out_path == repo_root or repo_root in out_path.parents:
        print("REFUSED: --out must be outside this repository (results derived from data must never be committed)",
              file=sys.stderr)
        return EXIT_REFUSED
    if not out_path.parent.is_dir():
        print("REFUSED: the directory of --out does not exist", file=sys.stderr)
        return EXIT_REFUSED
    if out_path.exists() and not a.overwrite:
        print("REFUSED: --out exists; pass --overwrite to replace it", file=sys.stderr)
        return EXIT_REFUSED
    return 0


def _panel(a, es, theta, n_boot, out, private):
    """Decoder-panel analyses (candidate 0.2.0)."""
    from eareeg_decoder import inference, nested, panel, pipeline, regimes
    from eareeg_decoder import montage as mt
    from eareeg_decoder.config import PROTOCOL
    P = PROTOCOL["panel"]
    names = panel.PANEL_DECODERS if a.decoder == "all" else (a.decoder,)
    seed = P["bootstrap"]["gap"]["seed"] if a.seed is None else a.seed
    summ = lambda v, test=True, null=0.0: inference.paired_summary(v, n_boot, seed, test, null)  # noqa: E731
    out["post_hoc"] = "decoder panel: post hoc, outcome-informed; not an independent replication"
    out["decoders_run"] = list(names)
    if a.analysis == "panel_drop_one":
        configs = ("full",) + tuple(mt.DROP_ONE)
        R = pipeline.build_raw_stack(es, configs, theta, require_frontal=False)
        con, out["decoders"] = {}, {}
        for name in names:
            try:
                r = panel.drop_one_panel(R.X, R.y, R.g, R.sg, name, offsets=a.offsets, device=a.device)
            except FloatingPointError:
                out["decoders"][name] = {"complete": False, "status": "NUMERICAL_GATE_FAILED"}
                continue
            model_index = panel.PANEL_DECODERS.index(name) + 1
            ba = {m: inference.paired_summary(v, n_boot, a.seed if a.seed is not None else 20261270 + 10 * model_index + i,
                                              test=False, null=0.5) for i, (m, v) in enumerate(r.ba.items())}
            out["decoders"][name] = {"complete": r.complete, "ba": ba}
            if r.complete:
                con[name] = r.contrasts
            private[name] = {k: v.tolist() for k, v in {**r.ba, **r.contrasts}.items()}
        pb = P["bootstrap"]["drop_one"]
        out["family_primary_d_F"] = panel.family(
            {f"{n}.d_F": con[n]["d_F"] if n in con else None for n in panel.PANEL_DECODERS},
            m=P["families"]["drop_one_primary_d_F"], n_boot=n_boot, seeds=pb["seeds_primary"], rule="drop_one", montage_ties=True)
        sec = {}
        for n in panel.PANEL_DECODERS:
            for k in ("d_M", "d_FM"):
                sec[f"{n}.{k}"] = con[n][k] if n in con else None
        out["family_secondary_mastoid"] = panel.family(sec, m=P["families"]["drop_one_secondary_d_M_d_FM"],
                                                       n_boot=n_boot, seeds=pb["seeds_secondary"], rule="none",
                                                       inversion=False, montage_ties=True)
        return
    R = pipeline.build_raw_stack(es, ("full", "fpz_free"), theta, require_frontal=True, dtype=np.float64)
    designs = (("loso", None), ("joint", R.sg))
    if a.analysis == "network_regimes":
        design, sg = designs[int(a.joint)]
        r = panel.gap_panel(R.X["full"], R.X["fpz_free"], R.y, R.g, R.free, "eegnet_followup",
                            sg=sg, n_draws=a.draws, offsets=a.offsets, device=a.device, min_face=a.min_face,
                            min_tree=a.min_tree, matched_to_matched=not a.joint)
        out["decoders_run"] = ["eegnet_followup"]
        out["design"] = design
        out.update(_describe_gap(r, summ))
        out["gap"] = summ(r.decomposition["delta"]) if r.complete else {"status": "INCOMPLETE"}
        out["gap"]["note"] = "Single comparison; no Holm adjustment across the original follow-up family."
        private.update({k: v.tolist() for k, v in r.decomposition.items() if isinstance(v, np.ndarray)})
        return
    if a.analysis == "panel_regimes":
        S = pipeline.build_stack(es, theta)
        for field in ("participants", "y", "g", "free", "sg", "keys"):
            if not np.array_equal(getattr(S, field), getattr(R, field)):
                raise AssertionError("feature and raw stacks disagree on retained epoch order")
        d_lda = {}
        for design, sg in designs:
            lres = regimes.lda_regimes(S.F, S.y, S.g, S.free, sg, n_draws=0, keys=S.keys, min_face=a.min_face,
                                       min_tree=a.min_tree)
            d_lda[design] = lres.advantage["free->free"] - lres.advantage["all->free"]
        fam = {"delta": {}, "c": {}, "s": {}, "vs_lda": {}}
        out["decoders"] = {}
        for name in names:
            for design, sg in designs:
                key = f"{name}.{design}"
                try:
                    r = panel.gap_panel(R.X["full"], R.X["fpz_free"], R.y, R.g, R.free, name, sg=sg, n_draws=a.draws,
                                        offsets=a.offsets, device=a.device, min_face=a.min_face, min_tree=a.min_tree)
                except FloatingPointError:
                    out["decoders"][key] = {"complete": False, "status": "NUMERICAL_GATE_FAILED"}
                    continue
                d = r.decomposition
                out["decoders"][key] = _describe_gap(r, summ)
                ok = r.complete
                fam["delta"][key] = d["delta"] if ok else None
                fam["c"][key] = d["c"] if ok else None
                fam["s"][key] = d["s"] if ok else None
                fam["vs_lda"][key] = (d["delta"] - d_lda[design]) if ok else None
                private[key] = {k: v.tolist() for k, v in d.items() if isinstance(v, np.ndarray)}
        if any(v.get("status") == "NUMERICAL_GATE_FAILED" for k, v in out["decoders"].items()
               if k.startswith("xdawn_ts_lr.")):
            for values in fam.values():
                for design, _ in designs:
                    values.pop(f"xdawn_ts_lr.{design}", None)
        F = P["families"]
        keys = [f"{n}.{dz}" for n in panel.PANEL_DECODERS for dz, _ in designs]
        for fname, rule, m in (("delta", "gap", F["gap_primary_delta"]), ("c", "none", F["gap_secondary_c"]),
                               ("s", "none", F["gap_secondary_s"]), ("vs_lda", "none", F["gap_secondary_vs_lda"])):
            out[f"family_{fname}"] = panel.family({k: fam[fname].get(k) for k in keys}, m=m, n_boot=n_boot,
                                                  seeds=seed, rule=rule, inversion=(fname == "delta"))
        return
    # panel_nested: joint outer folds only
    sg = R.sg
    allm = np.ones(len(R.y), bool)
    compact_g = np.searchsorted(R.participants, R.g)
    tuned, out["decoders"] = {}, {}
    for name in names:
        fit, base = panel.decoder(name)
        grid = nested.GRIDS[name]
        try:
            choice = {m: nested.choose_configurations(R.X[m], R.y, compact_g, sg, fit, grid, base, device=a.device)
                      for m in ("full", "fpz_free")}
            adv, complete = _tuned_advantage(a, R, compact_g, fit, grid, base, choice, name, allm)
        except (FloatingPointError, nested.InfeasibleSplitError):
            out["decoders"][name] = {"complete": False, "status": "SELECTION_OR_NUMERICAL_GATE_FAILED"}
            continue
        if not complete:
            out["decoders"][name] = {"complete": False, "status": "INCOMPLETE"}
            continue
        try:
            fixed = panel.gap_panel(R.X["full"], R.X["fpz_free"], R.y, R.g, R.free, name, sg=sg, n_draws=0,
                                    offsets=a.offsets, device=a.device, min_face=a.min_face, min_tree=a.min_tree)
        except FloatingPointError:
            fixed = None
        tuned[name] = adv["free->free"] - adv["all->free"]
        counts = np.bincount([c["index"] for m in choice for c in choice[m].values()], minlength=len(grid))
        out["decoders"][name] = {"complete": True, "A_F_tuned": {k: summ(v, test=False) for k, v in adv.items()},
                                 "selection_counts_in_grid_order": counts.tolist(), "grid": grid}
        private[name] = {"delta_tuned": tuned[name].tolist()}
        if fixed is not None and fixed.complete:
            d_fixed = fixed.advantage["free->free"] - fixed.advantage["all->free"]
            out["decoders"][name]["tuned_minus_fixed"] = summ(tuned[name] - d_fixed, test=False)
            private[name]["delta_fixed"] = d_fixed.tolist()
        else:
            out["decoders"][name]["tuned_minus_fixed"] = {"status": "FIXED_COMPARISON_INCOMPLETE"}
    out["family_delta_tuned"] = panel.family({f"{n}.joint.Delta_tuned": tuned.get(n) for n in panel.PANEL_DECODERS},
                                             m=P["families"]["nested_tuned_delta"], n_boot=n_boot, seeds=seed)
    out["caveat"] = "Hyperparameter selection uses all retained outer-source epochs for both training conditions."


def _describe_gap(result, summ):
    from eareeg_decoder import inference

    out = {"complete": result.complete, "test_sets_identical": result.test_sets_identical}
    if not result.complete:
        return out
    c_by_draw = result.advantage["free->free"][:, None] - result.matched_by_draw
    return {
        **out,
        "A_F": {k: summ(v, test=False) for k, v in result.advantage.items()},
        "ba": {r: {m: summ(v, test=False, null=0.5) for m, v in by_montage.items()}
               for r, by_montage in result.ba.items()},
        "auc_delta": summ(result.auc_delta, test=False),
        "c_monte_carlo": inference.draw_uncertainty(c_by_draw)[1],
        "delta_per_seed_cohort_mean": np.nanmean(result.delta_by_seed, axis=1).tolist(),
    }


def _tuned_advantage(a, R, compact_g, fit, grid, base, choice, name, allm):
    from eareeg_decoder import nested, panel, regimes
    from eareeg_decoder.config import eegnet_seed

    sg = R.sg
    planned = panel.planned_test_mask(R.y, R.g, R.free, sg)
    adv, complete = {}, True
    for reg, train in (("free->free", R.free), ("all->free", allm)):
        rows = []
        for offset in panel.offsets_for(name, a.offsets):
            ba = {}
            for m in ("full", "fpz_free"):
                prob = nested.tuned_probs(R.X[m], R.y, compact_g, sg, train, R.free, fit, grid, base, choice[m],
                                          eegnet_seed("transient", "joint", offset, 0), device=a.device)
                complete &= panel.predictions_complete(prob, planned)
                ba[m] = panel.ba_from_probs(R.y, R.g, prob, inclusive=False, min_face=a.min_face,
                                            min_tree=a.min_tree)
            rows.append(regimes.frontal_advantage(ba["full"], ba["fpz_free"]))
        adv[reg] = np.mean(rows, axis=0)
    return adv, bool(complete and np.isfinite(adv["free->free"] - adv["all->free"]).any())


def _supplement(a, es, n_boot, seed, out):
    from eareeg_decoder import montage, pipeline, regimes, sensitivities, transients

    if a.analysis == "stream_history":
        if es.stream is None or es.position is None:
            raise ValueError("stream_history needs stream and position for all presentations")
        valid = np.zeros(len(es.y), bool)
        flagged = np.zeros(len(es.y), bool)
        for p in es.participants:
            if not pipeline.frontal_usable(es, int(p)):
                continue
            rows = es.subset_participant(p)
            frontal = montage.fpz_mastoid(es.X[rows], es.channel_names, es.good_channels(p))
            theta = transients.resolve_theta(frontal, a.theta)
            valid[rows] = True
            flagged[rows] = transients.transient_flags(frontal, es.times, theta).any
        out["history"] = sensitivities.stream_history_control(
            es.y[valid], es.participant[valid], es.stream[valid], es.position[valid], flagged[valid], es.keep[valid],
            n_rotations=a.repetitions or 1000, seed=seed)
        return

    data = pipeline.build_stack(es, a.theta, montages=("full", "fpz_free"))
    if a.analysis == "permutation_check":
        null = sensitivities.h1_label_permutation(
            data.F, data.y, data.g, data.free, n_permutations=a.repetitions or 20,
            min_face=a.min_face, min_tree=a.min_tree, seed=seed)
        out["procedure_check"] = {"n_permutations": len(null), "mean": float(null.mean()),
                                  "sd": float(null.std(ddof=1)) if len(null) > 1 else None,
                                  "max_abs": float(np.abs(null).max())}
        return

    scores = {}
    for name, train in (("retrained", data.free), ("transferred", np.ones(len(data.y), bool))):
        for montage, key in (("full", "full"), ("fpz_free", "free")):
            scores[f"{name}_{key}"] = regimes.cv_scores(data.F[montage], data.y, data.g, train, data.free)
    out["weighted_gap"] = sensitivities.photograph_weighted_gap(
        data.y, data.g, data.keys, data.free, n_boot=n_boot, seed=seed, **scores)


def _finite_json(value):
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_result(path: Path, result: dict, *, overwrite: bool) -> None:
    """Owner-only atomic write, preserving a concurrently created output."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as tmp:
        pending = Path(tmp.name)
        try:
            json.dump(_finite_json(result), tmp, indent=1, allow_nan=False)
            tmp.write("\n")
            tmp.close()
            if overwrite:
                os.replace(pending, path)
            else:
                os.link(pending, path)
        finally:
            pending.unlink(missing_ok=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="path to a .npz file obeying the data contract")
    ap.add_argument("--i-am-authorized", action="store_true", help="attest legitimate access to the data")
    ap.add_argument("--analysis", choices=("drop_one", "transients", "regimes") + PANEL_ANALYSES + SUPPLEMENT_ANALYSES, required=True)
    ap.add_argument("--out", required=True, help="output JSON path (cohort-level summaries)")
    ap.add_argument("--joint", action="store_true", help="regimes: joint participant x stimulus-group folds")
    ap.add_argument("--theta", type=_theta, default=100.0, help="transient threshold in nominal uV, or 'relative'")
    ap.add_argument("--draws", type=_positive_int, default=None,
                    help="matched draws (paper LDA: 20; panel: 20 LOSO / 5 joint by default)")
    ap.add_argument("--adaptive-draws", action="store_true",
                    help="regimes: 20 draws, extended to 50 if the Monte Carlo SE of c exceeds 0.0025")
    ap.add_argument("--min-face", type=int, default=40, help="eligibility: minimum face test epochs (paper: 40)")
    ap.add_argument("--min-tree", type=int, default=15, help="eligibility: minimum tree test epochs (paper: 15)")
    ap.add_argument("--decoder", default="all",
                    choices=("all", "eegnet_step1", "xdawn_ts_lr", "eeg_inception_erp"),
                    help="panel analyses: decoder(s) to run (default: all three)")
    ap.add_argument("--offsets", type=_offsets, default=(0, 100, 200), help="panel: network seed offsets")
    ap.add_argument("--device", default="cpu", help="panel: torch device for the networks (default cpu)")
    ap.add_argument("--no-strict-grid", action="store_true", help="allow a time grid other than the paper's")
    ap.add_argument("--n-boot", type=int, default=None,
                    help="bootstrap resamples (default: 10000 for drop_one/transients/panel_drop_one, else 20000)")
    ap.add_argument("--repetitions", type=_positive_int, help="permutation or rotation repetitions")
    ap.add_argument("--seed", type=int, default=None,
                    help="bootstrap seed (default per analysis, as in configs/protocol.json)")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing --out file")
    ap.add_argument("--write-participant-values", action="store_true",
                    help="also store participant-level vectors in the output (PRIVATE: handle under your agreement)")
    a = ap.parse_args(argv)
    from eareeg_decoder.config import PROTOCOL
    boot = PROTOCOL["inference"]
    followup_boot = boot["bootstrap_followup"]["n_boot"], boot["bootstrap_followup"]["seed"]
    defaults = {"drop_one": (boot["bootstrap_montage"]["n_boot"], boot["bootstrap_montage"]["seeds"]["d_F"]),
                "transients": (boot["bootstrap_transient"]["n_boot"], boot["bootstrap_transient"]["seed"]),
                "regimes": (boot["bootstrap_followup"]["n_boot"], boot["bootstrap_followup"]["seed"]),
                "panel_drop_one": (PROTOCOL["panel"]["bootstrap"]["drop_one"]["n_boot"], None),
                "panel_regimes": (PROTOCOL["panel"]["bootstrap"]["gap"]["n_boot"], None),
                "panel_nested": (PROTOCOL["panel"]["bootstrap"]["gap"]["n_boot"], None),
                "network_regimes": followup_boot,
                **dict.fromkeys(SUPPLEMENT_ANALYSES, followup_boot)}[a.analysis]
    n_boot = a.n_boot if a.n_boot is not None else defaults[0]
    seed = a.seed if a.seed is not None else defaults[1]
    if a.analysis in PANEL_ANALYSES and a.no_strict_grid:
        print("REFUSED: network and decoder-panel modes require the frozen 250-Hz grid", file=sys.stderr)
        return EXIT_REFUSED
    if a.adaptive_draws and (a.analysis != "regimes" or a.draws is not None):
        print("REFUSED: --adaptive-draws applies to --analysis regimes without --draws", file=sys.stderr)
        return EXIT_REFUSED
    from eareeg_decoder.data import load_authorized_epochs
    from eareeg_decoder.errors import DataNotProvidedError, SchemaError
    out_path = None
    if a.data and a.i_am_authorized:
        out_path = Path(a.out).resolve()
        refused = _check_out(a, out_path)
        if refused:
            return refused
    try:
        es = load_authorized_epochs(a.data, attest_authorized=a.i_am_authorized, strict_grid=not a.no_strict_grid)
    except DataNotProvidedError as exc:
        print(str(exc), file=sys.stderr)
        print("DATA_NOT_PROVIDED")
        return EXIT_DATA_NOT_PROVIDED
    except SchemaError as exc:
        print(f"SCHEMA_ERROR: {exc}", file=sys.stderr)
        return EXIT_SCHEMA
    from eareeg_decoder import inference, pipeline, regimes
    theta = a.theta
    summ = lambda v, test=True, s=seed, null=0.0: inference.paired_summary(v, n_boot, s, test, null)  # noqa: E731
    out: dict = {"analysis": a.analysis, "note": "independent re-analysis; not a reproduction of the manuscript's reported values"}
    private: dict = {}
    if a.analysis in PANEL_ANALYSES:
        _panel(a, es, theta, n_boot, out, private)
    elif a.analysis in SUPPLEMENT_ANALYSES:
        try:
            _supplement(a, es, n_boot, seed, out)
        except ValueError as exc:
            print(f"SCHEMA_ERROR: {exc}", file=sys.stderr)
            return EXIT_SCHEMA
    elif a.analysis == "drop_one":
        bas = pipeline.drop_one_bas(es, joint=True)
        con = inference.montage_contrasts(bas)
        base = boot["bootstrap_montage"]["seed_per_montage_base"]
        out["ba"] = {k: summ(v, s=(base + i) if a.seed is None else seed, null=0.5) for i, (k, v) in enumerate(bas.items())}
        sd = boot["bootstrap_montage"]["seeds"]
        out["contrasts"] = {k: summ(v, s=sd[k] if a.seed is None else seed) for k, v in con.items()}
        p = [out["contrasts"][k]["p_signflip"] for k in ("d_F", "d_M", "d_FM")]
        out["holm_family_dF_dM_dFM"] = dict(zip(("d_F", "d_M", "d_FM"), inference.holm(p).tolist()))
        private = {k: v.tolist() for k, v in {**bas, **con}.items()}
    elif a.analysis == "transients":
        r = pipeline.transient_rates(es, theta)
        out["post_onset_rate"] = {"tree": summ(r["rate_tree"], False), "face": summ(r["rate_face"], False),
                                  "tree_minus_face": summ(r["difference"])}
        private = {k: v.tolist() for k, v in r.items()}
    else:
        S = pipeline.build_stack(es, theta)
        kw = dict(keys=S.keys, min_face=a.min_face, min_tree=a.min_tree)
        sg = S.sg if a.joint else None
        if a.adaptive_draws:
            res, out["draw_rule"] = regimes.lda_regimes_adaptive(S.F, S.y, S.g, S.free, sg, **kw)
        else:
            res = regimes.lda_regimes(S.F, S.y, S.g, S.free, sg, n_draws=a.draws or 20, **kw)
        dec = res.decomposition
        out["scheme"] = "joint" if a.joint else "loso"
        out["fraction_free"] = float(S.free.mean())
        out["A_F"] = {k: summ(v) for k, v in res.advantage.items()}
        out["delta"] = dict(summ(dec["delta"]), inversion_ci=_inversion(dec["delta"]))
        out["c"] = dict(summ(dec["c"]), inversion_ci=_inversion(dec["c"]))
        out["s"] = summ(dec["s"])
        c_by_draw = res.advantage["free->free"][:, None] - res.matched_free_by_draw
        _, out["c_monte_carlo"] = inference.draw_uncertainty(c_by_draw)
        out["identity_max_abs_error"] = dec["max_abs_identity_error"]
        out["test_sets_identical"] = res.test_sets_identical
        private = {"A_F": {k: v.tolist() for k, v in res.advantage.items()},
                   "delta": dec["delta"].tolist(), "c": dec["c"].tolist(), "s": dec["s"].tolist()}
    if a.write_participant_values:
        out["PRIVATE_participant_values"] = private
    out["bootstrap"] = {"n_boot": n_boot, "seed": seed}
    try:
        _write_result(out_path, out, overwrite=a.overwrite)
    except FileExistsError:
        print("REFUSED: --out was created during analysis", file=sys.stderr)
        return EXIT_REFUSED
    print(json.dumps({"status": "WROTE", "out": str(a.out), "participant_values_written": a.write_participant_values}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
