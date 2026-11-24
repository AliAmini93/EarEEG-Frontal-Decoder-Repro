#!/usr/bin/env python3
"""Run the LDA analyses on AUTHORIZED epochs (level C: an independent re-analysis, not a reproduction).

Fails closed with ``DATA_NOT_PROVIDED`` (exit 3) unless ``--data`` and ``--i-am-authorized`` are both given.
Writes cohort-level summaries to ``--out`` (a JSON file you choose). Participant-level values are written only
with ``--write-participant-values`` and must then be handled under your data agreement. Nothing is uploaded.

Analyses: ``drop_one`` (Step 1; joint folds; all participants), ``transients`` (Step 2; usable-Fpz participants),
``regimes`` (Step 3; LOSO by default, ``--joint`` for participant x stimulus-group folds).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

EXIT_DATA_NOT_PROVIDED = 3
EXIT_SCHEMA = 4
EXIT_REFUSED = 6


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


def _inversion(values):
    """Exact sign-flip inversion CI, or None (with the reason in the summary's p_note) when n > 24."""
    from eareeg_decoder import inference
    n = int(np.isfinite(np.asarray(values, dtype=float)).sum())
    return list(inference.signflip_inversion_ci(values)) if n <= inference.MAX_EXACT_N else None


def _check_out(a) -> int:
    """Return EXIT_REFUSED (after a message) if --out is unacceptable, else 0."""
    out_path = Path(a.out).resolve()
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="path to a .npz file obeying the data contract")
    ap.add_argument("--i-am-authorized", action="store_true", help="attest legitimate access to the data")
    ap.add_argument("--analysis", choices=("drop_one", "transients", "regimes"), required=True)
    ap.add_argument("--out", required=True, help="output JSON path (cohort-level summaries)")
    ap.add_argument("--joint", action="store_true", help="regimes: joint participant x stimulus-group folds")
    ap.add_argument("--theta", type=_theta, default=100.0, help="transient threshold in nominal uV, or 'relative'")
    ap.add_argument("--draws", type=_positive_int, default=20, help="matched draws (paper LDA: 20)")
    ap.add_argument("--min-face", type=int, default=40, help="eligibility: minimum face test epochs (paper: 40)")
    ap.add_argument("--min-tree", type=int, default=15, help="eligibility: minimum tree test epochs (paper: 15)")
    ap.add_argument("--no-strict-grid", action="store_true", help="allow a time grid other than the paper's")
    ap.add_argument("--n-boot", type=int, default=None,
                    help="bootstrap resamples (default: 10000 for drop_one/transients, 20000 for regimes)")
    ap.add_argument("--seed", type=int, default=None,
                    help="bootstrap seed (default per analysis, as in configs/protocol.json)")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing --out file")
    ap.add_argument("--write-participant-values", action="store_true",
                    help="also store participant-level vectors in the output (PRIVATE: handle under your agreement)")
    a = ap.parse_args(argv)
    from eareeg_decoder.config import PROTOCOL
    boot = PROTOCOL["inference"]
    defaults = {"drop_one": (boot["bootstrap_montage"]["n_boot"], boot["bootstrap_montage"]["seeds"]["d_F"]),
                "transients": (boot["bootstrap_transient"]["n_boot"], boot["bootstrap_transient"]["seed"]),
                "regimes": (boot["bootstrap_followup"]["n_boot"], boot["bootstrap_followup"]["seed"])}[a.analysis]
    n_boot = a.n_boot if a.n_boot is not None else defaults[0]
    seed = a.seed if a.seed is not None else defaults[1]
    from eareeg_decoder.data import load_authorized_epochs
    from eareeg_decoder.errors import DataNotProvidedError, SchemaError
    if a.data and a.i_am_authorized:          # check the output path before reading any data
        refused = _check_out(a)
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
    out_path = Path(a.out).resolve()          # already checked by _check_out before the data were read
    import numpy as np

    from eareeg_decoder import inference, pipeline, regimes
    theta = a.theta
    summ = lambda v, test=True, s=seed, null=0.0: inference.paired_summary(v, n_boot, s, test, null)  # noqa: E731
    out: dict = {"analysis": a.analysis, "note": "independent re-analysis; not a reproduction of the manuscript's reported values"}
    private: dict = {}
    if a.analysis == "drop_one":
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
        res = regimes.lda_regimes(S.F, S.y, S.g, S.free, S.sg if a.joint else None, n_draws=a.draws, keys=S.keys,
                                  min_face=a.min_face, min_tree=a.min_tree)
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
    out_path.write_text(json.dumps(out, indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o)) + "\n",
                        encoding="utf-8")
    print(json.dumps({"status": "WROTE", "out": str(a.out), "participant_values_written": a.write_participant_values}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
