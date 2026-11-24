#!/usr/bin/env python3
"""Mutation check: re-run the test suite against deliberately broken copies of the package.

Each mutant applies one or more exact text replacements to a temporary copy of ``src/`` (the working tree is never
modified) and runs ``pytest -x`` against that copy. A mutant is KILLED if the suite fails, SURVIVED otherwise.
Every mutant encodes a plausible scientific or engineering error (wrong re-reference, leakage, wrong threshold
direction, wrong statistic). Exit code 0 only if every mutant is killed. Synthetic and offline.

    python tools/mutation_check.py            # all mutants (a few minutes)
    python tools/mutation_check.py --list     # show the mutants
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# name -> (file relative to src/eareeg_decoder, [(old, new), ...])
MUTANTS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "montage: no re-reference (naive row deletion)": (
        "montage.py", [("out[..., j, :] = X[..., names.index(c), :] - r", "out[..., j, :] = X[..., names.index(c), :]")]),
    "montage: reference over all good channels incl. the deleted one": (
        "montage.py", [("ref = [c for c in channels if c in gs]", "ref = [c for c in names if c in gs]")]),
    "montage: Fpz plus (not minus) mastoid mean": (
        "montage.py", [("res[...] = X[..., names.index(FPZ), :] - X[", "res[...] = X[..., names.index(FPZ), :] + X[")]),
    "transients: threshold exclusive (> theta)": (
        "transients.py", [("any_ = ptp >= theta", "any_ = ptp > theta")]),
    "transients: post-onset rule exclusive (> 150 ms)": (
        "transients.py", [("any_ & (peak_t >= post_min)", "any_ & (peak_t > post_min)")]),
    "transients: peak of |F| instead of signed maximum": (
        "transients.py", [("peak_t = times[np.argmax(fm, axis=1)]", "peak_t = times[np.argmax(np.abs(fm), axis=1)]")]),
    "transients: descriptor uses >= threshold": (
        "transients.py", [("(b > threshold).astype(float)", "(b >= threshold).astype(float)")]),
    "preprocessing: 150-uV rule exclusive": (
        "preprocessing.py", [("np.all(ptp <= threshold_uv, axis=1)", "np.all(ptp < threshold_uv, axis=1)")]),
    "preprocessing: 150-uV rule also tests Fpz": (
        "preprocessing.py", [("channels: Sequence[str] = mt.MASTOIDS + mt.IN_EAR) -> tuple[np.ndarray, np.ndarray]:",
                              "channels: Sequence[str] = mt.EAR_CHANNELS) -> tuple[np.ndarray, np.ndarray]:")]),
    "preprocessing: baseline includes time 0": (
        "preprocessing.py", [("m = (times >= window[0] - 1e-9) & (times < window[1] - 1e-9)\n    if not m.any():\n        raise ValueError(\"empty baseline window\")",
                              "m = (times >= window[0] - 1e-9) & (times <= window[1] + 1e-9)\n    if not m.any():\n        raise ValueError(\"empty baseline window\")")]),
    "preprocessing: banker's rounding of onsets": (
        "preprocessing.py", [("np.floor(onset[finite] * float(sfreq) + 0.5)", "np.round(onset[finite] * float(sfreq))")]),
    "splits: training includes the held-out participant (assertion removed)": (
        "splits.py", [("            tr = (g != p) & train_mask\n", "            tr = train_mask.copy()\n"),
                      ("            assert_no_leakage(g, np.flatnonzero(tr), np.flatnonzero(te), sg, keys if sg is not None else None)\n            yield",
                       "            yield")]),
    "splits: joint training includes the held-out stimulus group (assertion removed)": (
        "splits.py", [("                tr = tr & (sg != s) & (sg >= 0)", "                tr = tr & (sg >= 0)"),
                      ("            assert_no_leakage(g, np.flatnonzero(tr), np.flatnonzero(te), sg, keys if sg is not None else None)\n            yield",
                       "            yield")]),
    "splits: stimulus groups by blocks instead of hash rank mod k": (
        "splits.py", [("mapping[key] = i % k", "mapping[key] = min(i // 12, k - 1)")]),
    "decoders: scaler fitted on test rows": (
        "decoders.py", [("return self.clf_.decision_function(apply_scaler(F, self.mu_, self.sd_))",
                         "return self.clf_.decision_function(apply_scaler(F, *fit_scaler(F)))")]),
    "decoders: LDA with empirical class priors": (
        "decoders.py", [('LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])',
                         'LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")')]),
    "metrics: accuracy instead of balanced accuracy": (
        "metrics.py", [("    r1, r0 = recalls(y, pred)\n    return 0.5 * (r1 + r0)",
                        "    return float((np.asarray(pred) == np.asarray(y)).mean())")]),
    "inference: one-sided sign-flip": (
        "inference.py", [("count += int(np.sum(np.abs(signs @ d) / n >= thr))", "count += int(np.sum((signs @ d) / n >= thr))")]),
    "inference: Bonferroni instead of Holm": (
        "inference.py", [("running = max(running, min(1.0, (m - rank) * p[i]))", "running = min(1.0, m * p[i])")]),
    "inference: 90% instead of 95% bootstrap interval": (
        "inference.py", [("np.percentile(m, 100 * alpha / 2)), float(np.percentile(m, 100 * (1 - alpha / 2)))",
                          "np.percentile(m, 100 * alpha)), float(np.percentile(m, 100 * (1 - alpha)))")]),
    "regimes: composition measured against all->free": (
        "regimes.py", [("delta, c, s = a_ff - a_af, a_ff - a_mf, a_mf - a_af", "delta, c, s = a_ff - a_af, a_ff - a_af, a_mf - a_af")]),
    "regimes: matched draws taken from free epochs only": (
        "regimes.py", [("pool = np.flatnonzero((g == p) & (y == c))\n            n = int(free[pool].sum())",
                        "pool = np.flatnonzero((g == p) & (y == c) & free)\n            n = int(free[pool].sum())")]),
    "regimes: 'transferred' decoder retrained on free epochs": (
        "regimes.py", [("sc_af, b_af = run(allm, free, min_face, min_tree)", "sc_af, b_af = run(free, free, min_face, min_tree)")]),
    "regimes: intercept recentred with the wrong sign": (
        "regimes.py", [("dec = dec - 0.5 * (d_tr[y[rm] == 1].mean()", "dec = dec + 0.5 * (d_tr[y[rm] == 1].mean()")]),
    "pipeline: free flag inverted": (
        "pipeline.py", [("free.append(~fl_any[keep])", "free.append(fl_any[keep])")]),
    "montage: full montage accepts any reference": (
        "montage.py", [("if check_reference and not is_common_average(X, names, gs):", "if False:")]),
    "data: common-average check removed from the contract": (
        "data.py", [("        if not mt.is_common_average(Xp, names, good):", "        if False:")]),
    "network CV: LOSO training includes the held-out participant (assertion removed)": (
        "regimes.py", [("            tr = (g != p) & train_mask\n            if (y[te] == 1)", "            tr = train_mask.copy()\n            if (y[te] == 1)"),
                       ("            assert_no_leakage(g, np.flatnonzero(tr), np.flatnonzero(te))\n", "")]),
    "network CV: seed ignores the fold index": (
        "regimes.py", [("prob[te] = fit(X[tr], y[tr], X[te], seed0 + fold, cfg, device)\n        return prob",
                        "prob[te] = fit(X[tr], y[tr], X[te], seed0, cfg, device)\n        return prob")]),
    "network CV: joint index counts only non-empty cells": (
        "regimes.py", [("                prob[te] = fit(X[tr], y[tr], X[te], seed0 + fold, cfg, device)\n            fold += 1",
                        "                prob[te] = fit(X[tr], y[tr], X[te], seed0 + fold, cfg, device)\n                fold += 1")]),
    "splits: stimulus keys not checked in joint folds": (
        "splits.py", [("sg, keys if sg is not None else None)", "sg, None)")]),
    "pipeline: stimulus groups not a function of the key (check removed)": (
        "pipeline.py", [("    sg = assign_stimulus_groups(list(keys_all), y_all == 1)\n", "    sg = np.arange(len(keys_all)) % 3\n"),
                        ("            raise AssertionError(\"a stimulus key is split across stimulus groups\")", "            pass")]),
    "regimes: eligibility boundary exclusive": (
        "regimes.py", [("return (y[m] == 1).sum() >= max(1, min_face) and (y[m] == 0).sum() >= max(1, min_tree)",
                        "return (y[m] == 1).sum() > max(1, min_face) and (y[m] == 0).sum() > max(1, min_tree)")]),
    "regimes: joint recentring base keeps the held-out stimulus group": (
        "regimes.py", [("base = (g != p) if s is None else ((g != p) & (sgv != s) & (sgv >= 0))", "base = (g != p)")]),
    "regimes: matched draw k seeded with seed_base + k + 1": (
        "regimes.py", [("mask = matched_random_mask(free, y, g, np.random.default_rng(seed_base + k))",
                        "mask = matched_random_mask(free, y, g, np.random.default_rng(seed_base + k + 1))")]),
    "regimes: matched->free advantage from the first draw only": (
        "regimes.py", [('res.advantage["matched->free"] = res.matched_free_by_draw.mean(axis=1)',
                        'res.advantage["matched->free"] = res.matched_free_by_draw[:, 0]')]),
    "pipeline: relative threshold from retained epochs only": (
        "pipeline.py", [("fl_any = transient_flags(fm, es.times, resolve_theta(fm, theta)).any",
                         "fl_any = transient_flags(fm, es.times, resolve_theta(fm[keep], theta)).any")]),
}


def apply(tmp_src: Path, file: str, edits: list[tuple[str, str]]) -> str | None:
    path = tmp_src / "eareeg_decoder" / file
    text = path.read_text(encoding="utf-8")
    for old, new in edits:
        if text.count(old) != 1:
            return f"pattern occurs {text.count(old)} times in {file}"
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    return None


def run_suite(src: Path, timeout: int) -> tuple[bool, str, str]:
    env = dict(os.environ, PYTHONPATH=str(src), PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    probe = subprocess.run([sys.executable, "-c", "import eareeg_decoder; print(eareeg_decoder.__file__)"],
                           capture_output=True, text=True, env=env, cwd=str(ROOT / "tests"))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-o", "pythonpath=", "-p", "no:cacheprovider",
                        str(ROOT / "tests")], capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=timeout)
    failed = [ln for ln in r.stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))]
    return r.returncode == 0, (failed[0] if failed else r.stdout.strip().splitlines()[-1]), probe.stdout.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--json", help="write the result table to this JSON file")
    a = ap.parse_args(argv)
    if a.list:
        for name in MUTANTS:
            print(name)
        return 0
    results = []
    with tempfile.TemporaryDirectory(prefix="eareeg_mut_") as tmp:
        base = Path(tmp) / "clean"
        shutil.copytree(ROOT / "src", base, ignore=shutil.ignore_patterns("__pycache__"))
        ok, info, where = run_suite(base, a.timeout)
        if not ok or not where.startswith(str(base)):
            print(json.dumps({"status": "BASELINE_FAILED", "detail": info, "imported_from_temp_copy": where.startswith(str(base))}))
            return 2
        for i, (name, (file, edits)) in enumerate(MUTANTS.items()):
            src = Path(tmp) / f"m{i:02d}"
            shutil.copytree(base, src)
            err = apply(src, file, edits)
            if err:
                results.append({"mutant": name, "status": "NOT_APPLIED", "detail": err})
                continue
            passed, info, where = run_suite(src, a.timeout)
            if not where.startswith(str(src)):
                results.append({"mutant": name, "status": "IMPORT_ESCAPED", "detail": where})
                continue
            results.append({"mutant": name, "status": "SURVIVED" if passed else "KILLED", "detail": info})
            print(f"{results[-1]['status']:9s} {name}", flush=True)
    summary = {"n": len(results), "killed": sum(r["status"] == "KILLED" for r in results),
               "survived": [r["mutant"] for r in results if r["status"] != "KILLED"], "results": results}
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n", "killed", "survived")}, indent=1))
    return 0 if not summary["survived"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
