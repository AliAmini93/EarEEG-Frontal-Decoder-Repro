# Methods-to-code map

Every element of the manuscript's *Data and Methods* section, linked to the code that implements it. No numeric
result of the study appears here. "LDA" = shrinkage linear discriminant analysis; "network" = EEGNet-type network.
**Joint** = participant × stimulus-group hold-out; **LOSO** = leave-one-participant-out.

Cohort note: the montage analyses used all analysed participants, the transient, follow-up and eye-tracker analyses
the subset whose frontal (Fpz) channel passed quality control, and the cross-device check the subset with usable
frontal derivations on both devices. These are nested subsets of one cohort, not separate cohorts (sizes are
dataset-derived and are not given here). In code: `pipeline.build_stack(require_frontal=False)` (montage analyses) vs `True` (transient analyses).

## Equations

| Eq. | Meaning | Code | Tests |
|---|---|---|---|
| (1) | `z_i = y_i − mean_{j∈R_g} y_j = x_i − mean_{j∈R_g} x_j`; the stored reference `r` cancels | `montage.own_mean_reference`; `fpz_free`, `in_ear`, `drop_one`, `full_montage`, `fpz_mastoid` | `test_montage.py` (oracle from hardware-referenced potentials; red test for naïve deletion; invariance to Fpz content) |
| (2) | `d_F = mean_p [BA_p(−Fpz) − mean_e BA_p(−e)]`, plus `d_FM`, `d_M` | `inference.montage_contrasts`; BAs from `pipeline.drop_one_bas` (LDA, joint) | `test_metrics_inference.py::test_montage_contrasts_*`, `test_integration_synthetic.py` |
| (3) | `A_F,p^(T→S) = BA_p^(T→S)(full) − BA_p^(T→S)(Fpz-free)`; cohort mean over participants | `regimes.frontal_advantage`, `regimes.lda_regimes` | `test_regimes.py` |
| (4) | `Δ_p = c_p + s_p` (free→free, matched→free, all→free on identical free test epochs) | `regimes.gap_decomposition` | `test_regimes.py::test_regime_set_shares_test_epochs_and_decomposes_exactly` |
| — | Large-transient rule (text, not numbered): range of `F(t)` = Fpz − mean(good mastoids) ≥ θ over −0.2…0.848 s; post-onset if the signed maximum is at ≥ 0.15 s | `transients.transient_flags`, `resolve_theta` | `test_transients.py` |

Equation numbers follow the internal manuscript draft and may change.

## Preprocessing (label-blind, before any analysis)

| Step | Protocol constant (`configs/protocol.json`) | Code |
|---|---|---|
| Zero-phase FIR band-pass 0.1–30 Hz, firwin, Hamming | `preprocessing.filter` | `preprocessing.bandpass_fir` (MNE `filter_data`) |
| Channel QC on median-referenced task span: flat SD < 0.5, SD > 100, robust z of log10 SD > 4 (MAD floor 0.05), saturation > 1 % | `channel_qc` | `preprocessing.channel_qc`, `median_reference`, `clip_fraction` |
| Onset → sample `floor(onset·fs + 0.5)`; epochs −0.2…0.85 s at the native rate, integer decimation to 250 Hz keeping t = 0 (263 samples, last at 0.848 s) | `preprocessing.tmin/tmax/target_sfreq` | `preprocessing.map_onsets`, `epoch` |
| Baseline [−0.2, 0) s | `preprocessing.baseline` | `preprocessing.baseline_correct` |
| Stored reference: common average of the good ear-system channels (includes Fpz when good); bad rows NaN | `preprocessing.stored_reference` | `preprocessing.common_average` |
| Reject epochs with > 150 peak-to-peak on any good in-ear or mastoid row (after the common average; Fpz not tested) | `preprocessing.epoch_rejection_*` | `preprocessing.reject_epochs` |
| Whole chain for one recording | — | `preprocessing.preprocess_recording` |

Amplitudes are on the recording's nominal µV scale; no calibration or unit factor is applied.

## Montages compared

| Montage | Rows | Code |
|---|---|---|
| full | all 11 stored rows (Eq. 1 leaves them unchanged) | `montage.full_montage` |
| Fpz-free | M1, M2 + 8 in-ear, own mean | `montage.fpz_free` |
| in-ear | 8 in-ear, own mean | `montage.in_ear` |
| Fpz–mastoid | Fpz − mean(good mastoids), one row | `montage.fpz_mastoid` |
| drop-one (× 11) | delete one electrode, re-reference the other ten | `montage.drop_one`, `DROP_ONE` |
| span-identical full (review check) | Fpz-free rows + Fpz − mean(good others) | `montage.span_identical_full` |

## Decoders

| Decoder | Settings | Used in | Code |
|---|---|---|---|
| LDA | 16 × 50-ms bin means over [0, 0.8) s per channel; z-scaler fitted on training rows (NaN → training mean); lsqr, Ledoit–Wolf `auto`, priors 0.5/0.5; decision > 0 → face | drop-one montages (joint); all transient and follow-up regimes (LOSO and joint); descriptor decoder | `features.bin_means`, `decoders.ShrinkageLDA`, `lda_decision` |
| Network, montage stage | F1 8, D 2, F2 16, kernel 64, dropout **0.25**, Adam 1e-3, batch 128, 30 epochs, class-weighted CE, per-channel training z-score, samples in [0, 0.8) s; 3 seeds: 20261107 + {0,100,200} + 1000 + fold (joint) | full / Fpz-free / in-ear / Fpz–M and two drop-one checks (joint) | `decoders.eegnet_fit_predict(cfg=eegnet_config("montage"))`, `config.eegnet_seed("montage", ...)` |
| Network, transient stage | same, dropout **0.5**; seeds 20261111 + {0,100,200} + fold | retrained / transferred / matched regimes (LOSO and joint) | `decoders.eegnet_fit_predict(cfg=eegnet_config("transient"))`, `regimes.net_probs` |

A channel that failed QC is a NaN row and enters both decoders as the training mean (no information). The full
montage requires input in the stored common average (`montage.full_montage` checks it); all other montages are
reference-free. Montage-stage network settings come from the frozen preregistration and the network-control
contract of the montage stage (not from the manuscript text alone).
Network accuracies are averaged over seeds within participant before inference. This package does not reproduce
the original trained network parameters.

## Validation schemes

| Scheme | Code | Notes |
|---|---|---|
| Joint hold-out: three class-stratified stimulus groups (salted SHA-256 rank mod 3); test one participant on one group after training on the other participants' epochs from the other groups | `splits.assign_stimulus_groups`, `splits.joint_folds`, `splits.masked_folds(stim_groups=...)` | test stimuli are new to the decoder; BA_p pooled over the participant's three folds |
| LOSO | `splits.loso_folds`, `splits.masked_folds(stim_groups=None)` | every tree photograph is also in training: may include stimulus-specific information |
| Leakage assertions | `splits.assert_no_leakage` (participant, stimulus group, stimulus key) | raise `LeakageError` |

## Train → test regimes (Step 3)

| Regime | Train | Test | Code |
|---|---|---|---|
| all-epoch | all | all | `regimes.lda_regimes` (`"all->all"`) |
| retrained | free | free | `"free->free"` |
| matched | matched | matched | `"matched->matched"` (LDA: mean over 20 draws) |
| transferred | all | free | `"all->free"` (decoder applied unchanged; no transfer learning) |
| matched-transferred | matched | free | `"matched->free"` |

Matched draw *k*: per participant and class, as many epochs drawn from all retained epochs as the participant has
free epochs, `np.random.default_rng(20261111 + k)`, one mask for both montages (`regimes.matched_random_mask`).
LDA used 20 draws; the network used 1 (matched→matched) and 20 / 5 (matched→free, LOSO / joint). The rule to extend
LDA draws to 50 when the Monte Carlo SE of c exceeds 0.0025 is not automated (`inference.draw_uncertainty`).
Eligibility: a participant is scored on free/matched epochs only with ≥ 40 face and ≥ 15 tree test epochs
(matched test sets have the free sets' class counts, so eligibility coincides). Participants without a usable Fpz
have no free epochs in this package.
Train-fitted diagnostics: crossed scaler (`cv_scores(scaler_mask=...)`) and intercept re-centring
(`cv_scores(recal_mask=...)`). Note: scikit-learn's shrinkage LDA standardises within class internally, so the
crossed-scaler diagnostic is uninformative by construction unless features are missing (see `test_regimes.py`).
Network regimes: `regimes.net_probs` + `regimes.probs_to_scores`.

## Follow-up variants

| Variant | Code |
|---|---|
| Thresholds 70, 150, 2.5 × participant 25th percentile | `pipeline.build_stack(theta=70 / 150 / "relative")` |
| Single 200-ms windows | `features.window_columns(n_channels, (a, b))` → restrict `F` columns, then `regimes` |
| Post-onset selections: previous presentation without transient / a face; no late flank; no next-baseline transient | `transients.adjacent`, `late_flank_clean`, `next_baseline_transient`; pass the restricted free mask as `subset` to `regimes.ba_by_participant` |
| Pre-onset selection (range of F over [−0.2, 0) < 50) | `transients.preonset_clean` |
| AUC version of the gap | `regimes.auc_by_participant` |
| Tracker-blink-free epochs | `transients.interval_overlap` on blink intervals already mapped to the EEG clock (see below) |
| Label-permutation and synthetic replays | `synthetic.synthetic_epochs`; scenarios in `tests/test_integration_synthetic.py` |

## Statistics

| Element | Code |
|---|---|
| 95 % percentile bootstrap over participants (10 000 montage/transient; 20 000 follow-up; seeds in `configs/protocol.json`) | `inference.bootstrap_ci`, `paired_summary(null=0 / 0.5)` |
| Two-way bootstrap: participants × stimulus groups (d_F) or participants × tree photographs (transient rate; pass cells = per-photograph rate − face rate) | `inference.twoway_bootstrap_ci` |
| Photograph-weighted two-way bootstrap of the paired gap (follow-up conditional sensitivity) | **not implemented** |
| Exact two-sided sign-flip test over all 2^n patterns | `inference.signflip_exact_p` |
| Test-inversion interval | `inference.signflip_inversion_ci` |
| Holm within each named family only | `inference.holm` |
| Monte Carlo SE over matched draws | `inference.draw_uncertainty` |
| Follow-up per-test label rule | `inference.decision_label` |
| Flag agreement (sensitivity, PPV, Cohen's κ, per participant then averaged) | `metrics.flag_agreement` |

## Not implemented (and why)

| Element | Reason |
|---|---|
| Eye-tracker trigger-pattern clock map, sample-coverage/pupil-loss rule, vendor event parsing | specific to proprietary tracker records; alignment could be circular; not needed for code-path reproducibility. `transients.interval_overlap` implements the overlap rule once intervals are on the EEG clock. |
| Scalp-device detector and scalp EOG-channel decoder | need scalp-device data and an undocumented EOG placement; the flag/κ code is reusable |
| Reading raw recordings, behavioural tables, unit conversions | acquisition-specific; not authorized for release |
| Exact original fold/stimulus-group membership | depends on the dataset's stimulus key strings (data contract) |
