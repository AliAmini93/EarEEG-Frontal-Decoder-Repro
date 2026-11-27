# EarEEG-Frontal-Decoder-Repro

> **Release `v0.1.0` — experimental code only. MIT licence. No data, no study results and no figures are included.**

Code for the experimental methods of the manuscript *"A Frontal Electrode in Ear-Centered EEG Decoding: Retraining
Without Large Blink-Like Transients Shows What Remains, Not What the Decoder Used"* (in preparation, not yet
published). It lets a researcher who has legitimately obtained compatible ear-centred EEG epochs re-run the
montage, transient, decoding and train→test analyses described in that manuscript and inspect exactly how montages
were re-referenced, how large blink-like frontal transients were flagged, how participants and stimuli were held
out, and how decoders trained on one epoch set were scored on another.

The package `eareeg_decoder` is a clean re-implementation of the original (private) analysis code; the original
analysis project, its data and its results are not part of this repository.

## What this repository is — and is not

| It is | It is not |
|---|---|
| An implementation of the paper's montage algebra (Eq. 1), drop-one contrasts (Eq. 2), frontal advantage (Eq. 3) and paired-gap decomposition Δ = c + s (Eq. 4) | A general-purpose emotion, attention or "brain state" detector — labels are the **displayed image type** (face vs tree) |
| Shrinkage LDA and an EEGNet-type network with the paper's fixed settings | A new decoder or validation scheme, or "transfer learning" |
| Joint participant × stimulus-group hold-out and LOSO splits with leakage assertions | An in-ear-only decoder: the system includes a frontal scalp electrode (Fpz) and two mastoids |
| Participant-level bootstrap, exact sign-flip test, Holm, test-inversion intervals | A replication: all analyses in the paper are post hoc on one cohort |
| A fail-closed adapter for authorized epoch files, synthetic test fixtures and synthetic tests | A source-localisation or ocular-vs-cortical attribution tool — these analyses cannot identify the source |

Running this code on authorized data is an **independent re-analysis**; matching the manuscript's reported values
additionally requires identical data, folds, stimulus keys, library versions and (for networks) hardware.

## Requirements

Python ≥ 3.10 with `numpy` (< 2), `scipy`, `scikit-learn` (tested with Python 3.11, numpy 1.26, scipy 1.16,
scikit-learn 1.6). Optional: `torch` (EEGNet-type network), `mne` (FIR band-pass filter), `pytest` (tests).
This repository never installs anything; install dependencies yourself if needed (e.g.
`pip install -e ".[eegnet,filter,test]"`). Missing optional dependencies are reported as
`BLOCKED_MISSING_DEPENDENCY`.

```bash
python scripts/eareeg_check_data.py                    # prints DATA_NOT_PROVIDED (exit 3): no data are shipped
python scripts/eareeg_check_data.py --data ../my_data/epochs.npz --i-am-authorized
python scripts/eareeg_analyze.py --data ../my_data/epochs.npz --i-am-authorized --analysis regimes --out ../my_results/regimes.json
python -m pytest -q                                    # synthetic tests (optional)
```

Scripts add `src/` to the path themselves; for your own code run from the repository root with `PYTHONPATH=src`.
Write every output **outside** the repository; `eareeg_analyze.py` refuses output paths inside it. Exit codes:
0 success, 2 argument error, 3 `DATA_NOT_PROVIDED`, 4 schema error, 6 refused output path. The input format (one
`.npz` of epoched, stored-common-average data with labels, participant codes, stimulus keys and retention flags)
is specified in [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md).

## Experiments → entry points

"CLI" = runnable with `scripts/eareeg_analyze.py` on an authorized `.npz`; output is one JSON file of cohort-level
summaries at `--out` (participant-level values only with `--write-participant-values`). "Library" = implemented and
tested on synthetic data, but must be called from your own Python script (no command-line wrapper). Constants for
every row are in [`configs/protocol.json`](configs/protocol.json); the full mapping is in
[`docs/METHODS_TO_CODE_MAP.md`](docs/METHODS_TO_CODE_MAP.md).

| Paper experiment | Code | How to run |
|---|---|---|
| Preprocessing of a recording (FIR 0.1–30 Hz, channel QC, epoching, baseline, stored common average, 150-µV rejection) | `preprocessing.preprocess_recording` | Library (needs `mne`; reading raw files is acquisition-specific and not included) |
| Step 1: LDA on 11 drop-one montages, joint hold-out; d_F, d_M, d_FM with Holm | `pipeline.drop_one_bas`, `inference.montage_contrasts` | **CLI** `--analysis drop_one` |
| Step 1: EEGNet-type network on full / Fpz-free / in-ear / Fpz–mastoid and drop-one checks (montage-stage settings) | `decoders.eegnet_fit_predict(cfg=eegnet_config("montage"))`, `config.eegnet_seed`, `regimes.net_probs` | **Library only** (needs `torch`) |
| Step 1: pre-onset control (50-ms bins in [−0.2, 0) s) | `features.bin_means(window=...)`, `regimes.cv_scores` | Library |
| Step 2: post-onset large-transient rate, tree − face (θ = 100, 70, 150 or relative) | `pipeline.transient_rates`, `transients.transient_flags` | **CLI** `--analysis transients [--theta ...]` |
| Step 2: 32-feature thresholded frontal-deflection descriptor decoder; peak latencies; two-way bootstrap over photographs | `pipeline.build_stack(...).descriptor`, `TransientFlags.peak_time`, `inference.twoway_bootstrap_ci` | Library |
| Step 3: five LDA train→test regimes, A_F, Δ = c + s (LOSO or joint), matched draws | `regimes.lda_regimes`, `regimes.gap_decomposition` | **CLI** `--analysis regimes [--joint] [--theta ...] [--draws ...]` |
| Step 3: EEGNet-type network regimes (transient-stage settings) | `regimes.net_probs`, `regimes.probs_to_scores`, `features.raw_window`, `montage.make_montage` | **Library only** (needs `torch`) |
| Follow-up: single 200-ms windows; post-onset and pre-onset selections; AUC version; intercept re-centring; crossed scaler | `features.window_columns`, `transients.adjacent` / `late_flank_clean` / `next_baseline_transient` / `preonset_clean`, `regimes.auc_by_participant`, `regimes.cv_scores(recal_mask=..., scaler_mask=...)` | Library |
| Flag agreement with tracker blinks; tracker-blink-free epochs | `metrics.flag_agreement`, `transients.interval_overlap` | Library; needs blink intervals **already on the EEG clock** (see below) |
| Synthetic replays (sub-threshold ocular, frontal neural, contingent blinks, null) | `synthetic.synthetic_epochs`; scenarios in `tests/test_integration_synthetic.py` | Library / tests |

### Known incomplete components (not provided)

* **Eye-tracker clock alignment** (trigger-pattern clock map, sample-coverage / pupil-loss rule, vendor event
  parsing): not included; it is specific to the tracker's proprietary records. Only the overlap rule for intervals
  already mapped to the EEG clock is provided.
* **Scalp-device (cross-device) checks** and the scalp EOG-channel decoder: not included.
* **No command-line entry point for the EEGNet-type network** (Steps 1 and 3) or for the follow-up variants listed
  as "Library" above; the functions exist and are tested on synthetic data, but you must assemble per-montage
  inputs and call them yourself.
* **Not implemented:** the photograph-weighted two-way bootstrap of the paired gap; the stream-history control
  decoder with within-stream label rotations; the in-ear-minus-mastoid propagation ratio of large transients; a
  dedicated "blink amplitudes truncated at the threshold" synthetic scenario. The rule extending LDA matched draws
  from 20 to 50 is not automated (`inference.draw_uncertainty` reports the Monte Carlo SE).
* Reading raw recordings, behavioural tables and unit conversions (acquisition-specific); exact original fold
  membership depends on the dataset's stimulus key strings.

This is therefore **not** an end-to-end reproduction package for every number in the manuscript.

## Methods at a glance

| Paper element | Code |
|---|---|
| 11-channel ear-centred system: Fpz, M1, M2, EL1, EL3, EL4, EL5, ER1–ER4 | `montage.EAR_CHANNELS` |
| Eq. (1) own-mean re-reference; full, Fpz-free, in-ear, Fpz–mastoid, eleven drop-one montages | `montage.own_mean_reference`, `fpz_free`, `in_ear`, `fpz_mastoid`, `drop_one` |
| The pitfall: deleting the stored Fpz row without re-referencing | `montage.naive_drop` (tests show it fails) |
| 0.1–30 Hz zero-phase FIR, −200…848 ms epochs at 250 Hz, baseline [−200, 0) ms | `preprocessing.bandpass_fir`, `epoch`, `baseline_correct` |
| Stored common average of good channels; 150 µV (nominal) rejection on in-ear + mastoid rows | `preprocessing.common_average`, `reject_epochs` |
| Large transient: Fpz–mastoid range ≥ 100 µV (nominal) over the whole epoch; post-onset if the maximum is at ≥ 150 ms | `transients.transient_flags` |
| 32-feature thresholded frontal-deflection descriptor (±50 µV, 50-ms bins) | `transients.thresholded_descriptor` |
| Sixteen 50-ms bin means over 0–800 ms; training-only z-scoring; shrinkage LDA (equal priors) | `features.bin_means`, `decoders.ShrinkageLDA` |
| EEGNet-type network (F1 8, D 2, F2 16, kernel 64, dropout 0.25 / 0.5, Adam, 30 epochs, class weights) | `decoders.eegnet_fit_predict` |
| Joint participant × stimulus-group hold-out; LOSO | `splits.joint_folds`, `loso_folds`, `regimes.cv_scores` |
| Five train → test regimes; matched draws; A_F; Δ = c + s | `regimes.lda_regimes`, `matched_random_mask`, `gap_decomposition` |
| Balanced accuracy pooled within participant; AUC; flag agreement (κ) | `metrics` |
| d_F, d_M, d_FM; bootstrap CI; exact sign-flip; Holm; test-inversion CI | `inference` |

All protocol constants are in [`configs/protocol.json`](configs/protocol.json) (identical to
`eareeg_decoder.config.PROTOCOL`).

## Validation (previously reported, not re-run for this release)

For the private release candidate `v0.1.0rc1` (2026-11-23) the maintainers reported 123 passing synthetic tests,
37 of 37 deliberate mutants killed, and 288 of 288 equivalence checks against the original private analysis code on
synthetic inputs. `v0.1.0` contains the same algorithm code; it removes the synthetic demo, examples and plotting
helper together with the 7 tests that exercised them (116 test items remain). No test, mutation check or
equivalence check was re-run for `v0.1.0`. Details and caveats: [`docs/VALIDATION.md`](docs/VALIDATION.md).

## Not supplied

Raw or epoched EEG, eye-tracking records, stimuli, behavioural tables, participant identifiers, trained model
weights, per-participant or cohort results, the manuscript, its figures or tables, figure-drawing code, a
downloader, or the eye-tracker clock-alignment code. Data access is governed by the dataset controller; see
[`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md).

## Repository layout

```
src/eareeg_decoder/   montage, preprocessing, features, decoders, splits, transients, regimes, metrics,
                      inference, data (fail-closed adapter), pipeline, synthetic, config, errors
scripts/              eareeg_check_data.py, eareeg_analyze.py
configs/protocol.json protocol constants (no data)
tests/                synthetic unit, oracle, leakage, mutation-style and integration tests
docs/                 QUICKSTART, DATA_ACCESS, METHODS_TO_CODE_MAP, REPRODUCIBILITY, VALIDATION,
                      PRIVACY_AND_LIMITATIONS
tools/mutation_check.py  re-runs the tests against deliberately broken copies of the package
```
## Licence and citation

MIT licence ([`LICENSE`](LICENSE)). Please cite the software using [`CITATION.cff`](CITATION.cff); the manuscript is
in preparation. Methods references: Lawhern et al. 2018 (EEGNet); Blankertz et al. 2011 and Ledoit & Wolf 2004
(shrinkage LDA).
