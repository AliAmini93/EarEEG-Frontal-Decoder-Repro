# Validation

Everything here is validation **of the software on synthetic inputs**. It demonstrates algorithmic fidelity, not
the correctness of the original experiment, and it does not show that the balanced accuracies,
intervals or p values can be obtained from these inputs.

> **Previously reported results.** All results below were reported for the private release candidate `v0.1.0rc1`
> (2026-11-23). None was re-run for `v0.1.0`. `v0.1.0` has the same algorithm code but removes the synthetic demo
> script, the two synthetic examples and the optional plotting helper, together with the 7 test items that
> exercised them (demo `--help`, demo run, two missing-dependency demo checks, two example runs, demo output-path
> refusal); 116 test items remain (static count). The mutation result was obtained with the full rc1 suite.

## 1. Test suite (`python -m pytest -q`)

Result for `v0.1.0rc1` (Python 3.11, numpy 1.26, scipy 1.16, scikit-learn 1.6, torch 2.9 CPU, MNE 1.11,
matplotlib 3.10): **123 passed, 0 skipped, 0 failed** (~23 s, CPU only). Without torch / MNE the corresponding
tests are skipped with the reason `BLOCKED_MISSING_DEPENDENCY` (checked by hiding them: 118 passed, 5 skipped).

| File | Tests | What is checked |
|---|---|---|
| `test_montage.py` | 14 | Eq. (1) against a hardware-reference oracle for every drop-one montage and several bad-channel patterns; red test for naïve Fpz deletion; Fpz-free rows unchanged when only Fpz changes; full-montage invariance; Fpz–mastoid reference-freeness; rank; span-identical montage; input-order invariance |
| `test_preprocessing.py` | 16 | half-up onset rounding; epoch grid (263 samples, −0.2…0.848 s) and time-zero indexing at 250/500/1000 Hz; baseline excludes t = 0; common average includes Fpz; 150-µV rule inclusive and limited to in-ear + mastoid rows; 150-µV rule vs 100-µV frontal flag; channel QC; no unit scaling; FIR passband, stopband and zero phase; whole-recording chain |
| `test_transients.py` | 13 | θ inclusive at 100; post-onset rule at ≥ 150 ms on any grid; signed (not absolute) maximum; relative threshold; 32-feature descriptor with strict ±50 thresholds; pre-onset and late-flank screens; adjacency and next-baseline screen; tracker-interval overlap with clock tolerance |
| `test_splits_leakage.py` | 9 | 12/12/11 and 2/2/2 stimulus groups; joint folds share no participant, group or key; LOSO shares tree photographs; leakage assertions; mutant split; training-only scaling; label-invariance oracle (LOSO and joint); participant and stimulus canaries |
| `test_metrics_inference.py` | 16 | BA under 16:4 imbalance; AUC; κ; sign-flip vs brute force; Holm; bootstrap replay; inversion interval; label rule; summaries against 0 and 0.5; n > 24 handling; MC SE; d_F/d_M/d_FM; mutants |
| `test_regimes.py` | 7 | matched draws; unchanged application of the all-epoch decoder; shared test epochs; Δ = c + s (LOSO and joint); eligibility; crossed scaler (uninformative by construction) and intercept recentring (LOSO and joint base) |
| `test_rules_pinned.py` | 11 | full montage requires the stored common average (also in the data contract); stimulus-key leakage in joint folds; pipeline groups are a function of the key; network CV fold/seed rules and leakage with a stub fit (LOSO and joint); eligibility boundary 40/15; BA on scored epochs only; matched draw seeds and draw averaging (matched→matched and matched→free); relative θ and transient rates over all presentations; no-Fpz participants never free |
| `test_decoders.py` | 7 | bin layout; NaN-safe scaler; LDA = scikit-learn with equal priors; network settings and seed rules; architecture and parameter count (1,474 for 11 × 200 input); CPU determinism |
| `test_integration_synthetic.py` | 7 | end-to-end synthetic mechanisms (below); participants without Fpz excluded from transient steps only |
| `test_data_adapter_config_cli.py` | 20 | `DATA_NOT_PROVIDED` paths; pickle refused; schema errors (incl. key in both classes); protocol JSON = module; script `--help`; fail-closed scripts; output refused inside the repository; schema exit 4; argument validation; missing optional dependency exit 5 without traceback; synthetic analysis incl. n > 24 participants; demo (also refuses figures inside the repository); examples; synthetic rejection |
| `test_repo_hygiene.py` | 3 | tracked files contain no data/binary files, absolute paths, e-mail addresses, tokens or network imports |

Markers (`python -m pytest -m <marker>`): `lda`, `eegnet` (network settings, architecture and the network
cross-validation fold/seed tests with a stub fit), `joint`, `loso`, `sensitivity`, `oracle`; a test can carry
several. Current counts: see `python -m pytest --collect-only -q -m <marker>`.

Which tests concern which decoder and scheme:

| Concern | Tests |
|---|---|
| LDA | `test_decoders.py::test_lda_*`, `test_regimes.py`, `test_splits_leakage.py` (canaries), `test_integration_synthetic.py` |
| EEGNet-type network | `test_decoders.py::test_eegnet_*` (architecture, parameter count, CPU determinism; skipped without torch) |
| Joint participant × stimulus-group folds | `test_splits_leakage.py::test_joint_*`, `test_stimulus_canary_*`, `test_heldout_decisions_*`; `pipeline.drop_one_bas` in integration tests |
| LOSO | `test_splits_leakage.py::test_loso_*`, `test_regimes.py`, integration tests |
| Computational sensitivities (thresholds, windows, selections, crossed scaler, intercept) | `test_transients.py`, `test_decoders.py::test_bin_means_layout_and_window_columns`, `test_regimes.py::test_crossed_scaler_*` |

### Independent oracles (not derived from the implementation)

* **Re-reference**: potentials `x` against a synthetic hardware reference plus a large common signal; the stored
  data `y` are formed independently; correct montage rows must equal `x_i − mean_{R_g} x_j`. The naïve Fpz-row
  deletion is shown to FAIL this oracle and to change when only Fpz changes (red test).
* **Leakage**: (i) a fold's held-out decisions must be bit-identical when the labels of every row excluded from
  its training set are flipped (LOSO and joint); a leaky mutant changes; (ii) a participant-specific-direction
  canary that only a leaky pipeline can exploit; (iii) a stimulus-identity canary that is exploitable under LOSO
  (shared tree photographs) but not under joint hold-out.
* **Training-only scaling**: perturbing other test rows does not change a test row's decision; a scaler fitted on
  train + test would.
* **Statistics**: exact sign-flip p against brute-force enumeration; Holm against hand calculation; bootstrap
  against a manual NumPy replay of the same random stream; inversion-interval edges checked as test boundaries;
  AUC against pairwise counting with ties.
* **Regimes**: `all→free` decisions equal `all→all` decisions on free epochs (decoder applied unchanged);
  matched draws reproduce per-participant × class counts and include transient-bearing epochs; Δ = c + s exactly.
* **Synthetic mechanisms**: no gap without a class signal sharing time bins with the transients; a large gap for a
  sub-threshold ocular deflection *and* for a frontal neural signal in the same window (the gap does not identify
  the source); the transferred advantage is kept when the class signal lies outside that window.

### Deliberate wrong-equation mutants inside the tests

one-sided instead of two-sided sign-flip; Bonferroni instead of Holm; bootstrap over epochs instead of
participants; a "joint" split that only excludes the stimulus group; a scaler fitted on test rows; composition
measured against `matched→matched` (different test epochs); naïve Fpz-row deletion.

## 2. Mutation check (`python tools/mutation_check.py`)

The whole suite is re-run against copies of the package with one deliberate error each. A mutant is *killed*
when at least one test fails.

Result for `v0.1.0rc1`: **37 of 37 mutants killed** (after review; an intermediate run left one survivor, which
led to a stronger test — see CHANGELOG).

| Area | Mutants |
|---|---|
| montage | no re-reference; reference over all good channels including the deleted one; Fpz plus (not minus) mastoid mean; full montage accepts any reference; common-average check removed from the data contract |
| transients | threshold exclusive; post-onset rule exclusive; peak of \|F\|; descriptor `>=` |
| preprocessing | 150-µV rule exclusive; 150-µV rule also tests Fpz; baseline includes t = 0; banker's rounding of onsets |
| splits | training includes the held-out participant; joint training includes the held-out stimulus group; stimulus groups by blocks; stimulus keys not checked in joint folds; pipeline groups not a function of the key |
| decoders / metrics | scaler fitted on test rows; LDA with empirical priors; accuracy instead of BA |
| inference | one-sided sign-flip; Bonferroni instead of Holm; 90 % instead of 95 % interval |
| regimes / pipeline | composition against all→free; matched draws from free epochs only; "transferred" decoder retrained; intercept shift sign; joint recentring base keeps the held-out group; eligibility boundary exclusive; draw seed off by one; matched→free advantage from the first draw only; free flag inverted; relative threshold from retained epochs only |
| network cross-validation | LOSO training includes the held-out participant; seed ignores the fold index; joint index counts only non-empty cells |

## 3. Equivalence with the original analysis code (maintainers, private)

The original analysis code is not part of this repository. During packaging, the repository owner's AI coding
assistant ran an equivalence harness that imports both the original (private) modules and this package and
compares them **on synthetic inputs only** (no recording was opened). The harness and its outputs are kept in the
authors' private project records; readers cannot re-run it, so treat this section as a statement by the authors,
not as independently verifiable evidence.

Result for `v0.1.0rc1`: **288 of 288 checks passed**; 284 bit-exact, the remaining 4 (FIR filter and epochs
through MNE, which stores volts) within 2.1 × 10⁻¹⁴.

| Area | Checks | Bit-exact | Max abs. difference |
|---|---|---|---|
| Montage algebra (11 drop-one + naïve deletion × 5 bad-channel patterns; full, Fpz-free, in-ear, Fpz–mastoid; transient-analysis variants) | 140 | 140 | 0 |
| Features, scaler, LDA decisions and predictions | 6 | 6 | 0 |
| Transient flags (θ = 100, 70, 150, relative), descriptor, full feature stack, stimulus groups | 13 | 13 | 0 |
| Regime decision values, BA (also vs the original cross-validation helper), AUC, recalls — LOSO and joint; crossed-scaler and intercept diagnostics | 63 | 63 | 0 |
| Matched draws, retrained / matched / matched→free sets, Δ_p and c_p | 11 | 11 | 0 |
| Joint and LOSO folds | 2 | 2 | 0 |
| Sign-flip (both tolerance conventions, ties), bootstrap (three original variants), Holm, inversion CI, two-way bootstrap, MC SE, label rule, d_F/d_M/d_FM, AUC, BA | 41 | 41 | 0 |
| Onset mapping, channel QC, saturation, 150-µV rule, FIR filter and epoching (500 and 1000 Hz) | 8 | 4 | 2.1 × 10⁻¹⁴ |
| Network: both stages (2-epoch CPU fits), LOSO and joint fold/seed rules | 4 | 4 | 0 |

The harness was itself checked against two deliberately broken copies of this package (no re-reference:
106 checks failed; transient threshold raised by 1 µV: 1 check failed).

Known, intentional differences:

| Topic | Original | This package | Effect |
|---|---|---|---|
| Minimum good channels for an own-mean reference | montage analyses: ≥ 2; transient analyses: ≥ 1 | parameter `min_good` (default 2) | identical when set as in each analysis; differs only if exactly one channel of a set is good |
| Sign-flip floating-point tie tolerance | montage: relative 1e-9; transient/follow-up: absolute 1e-12 | parameters `rtol`, `atol` (default absolute 1e-12) | decides only exact ties |
| Network threshold | montage stage `P ≥ 0.5`, transient stage `P > 0.5` | `probs_to_scores` uses `P − 0.5 > 0` | differs only when a probability is exactly 0.5 |
| Stimulus key occurring in both classes | silently overwritten | `ValueError` | stricter input validation |
| Channel QC span | task span via MNE time indexing | `preprocess_recording` uses floor-indexed span | orchestration helper only; the QC rule itself is identical |
| Eye-tracker clock map and coverage rule | implemented for vendor records | not included | see METHODS_TO_CODE_MAP |
| Full-montage input | stored rows used as they are | stored rows used as they are, after checking they are in the stored common average | identical on stored data; other references are refused instead of silently changing the montage |
| Participants without a usable Fpz in transient regimes | excluded before the stack is built | `build_stack(require_frontal=True)` excludes them; with `False` none of their epochs is "free" | identical for the transient analyses |
| Photograph-weighted two-way bootstrap of the paired gap; 50-draw extension rule | implemented in the follow-up runners | not implemented / not automated | documented in METHODS_TO_CODE_MAP |

## 4. What is not validated

* Agreement with the values reported in the manuscript (in preparation; requires authorized data; level C in REPRODUCIBILITY.md).
* GPU execution of the network, and equality with the original trained parameters.
* The trigger-based eye-tracker alignment (not included).
* Performance on recordings from other devices, montages or tasks.
* AI-assisted reviews were carried out; an independent **human** review has not yet been done.
