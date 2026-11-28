# Methods and implementation

| Method | Entry point and validation |
|---|---|
| Label-blind filtering, epoch QC and stored common average | `preprocessing.preprocess_recording`, synthetic continuous-input tests |
| Montage-specific own mean, good-channel selection, full, Fpz-free, in-ear, bipolar Fpz–mastoid and drop-one | `montage`, reference and missing-channel oracles |
| Binned shrinkage LDA and EEGNet montage comparison | `features.bin_means`, `decoders`, `pipeline.drop_one_bas` and independent model tests |
| Participant and stimulus-group hold-out, LOSO, seed indexing | `splits`, `regimes.net_probs`, `nested.outer_joint_folds` and leakage/index tests |
| Electrode ablation panel and paired frontal/mastoid contrasts | `panel_drop_one`, `panel.drop_one_panel`, `inference.montage_contrasts` and the synthetic CLI and tie-rule tests |
| xDAWN-TS-LR and EEG-Inception ERP | `xdawn`, `inception`, numerical gate, rank, probability and architecture tests |
| Blink-like detector and frontal descriptors | `transients`, `transients.thresholded_descriptor`, `pipeline.transient_rates`, threshold and timing tests |
| Scalp detector agreement and scalp EOG LOSO | `sensitivities.cross_device_control`, synthetic mapped-row test with documented caller inputs |
| Clock fitting, blink overlap and unexplained pupil loss | `clock`, `metrics.flag_agreement`, `transients.interval_overlap`, synthetic drift, ambiguity and unknown-state tests |
| Paired transfer gap, training size and composition | `regimes`, `panel_regimes`, `network_regimes`, mask, averaging and matched-draw tests |
| AUC, selected windows and descriptive controls | `metrics.rank_auc`, `features.window_columns`, `transients.adjacent`, `transients.preonset_clean`, `transients.late_flank_clean`, `transients.next_baseline_transient`, `regimes.cv_scores` |
| Photograph-weighted two-way gap bootstrap | `weighted_gap`, `sensitivities.photograph_weighted_gap`, explicit resampling oracle |
| Previous-presentation control and rotations | `stream_history`, `sensitivities.stream_history_control`, rejected-predecessor and boundary checks |
| In-ear waveform propagation | `sensitivities.propagation_slope`, signed-slope and missingness tests |
| Permutation procedure check and synthetic truncation | `permutation_check`, `sensitivities.h1_label_permutation`, `synthetic.truncation_scenario` |
| Source-only nested sensitivity | `panel_nested`, `nested`, disjoint inner folds, finite probabilities and tie tests |
| Participant and two-way bootstrap, exact sign-flip, intervals, Holm | `inference`, `panel.family`, brute-force and frozen-family checks |

The comparative ablation families have sizes three (frontal) and six
(mastoid and frontal–mastoid). The fixed transfer families have six positions
each for gap, composition, size and paired gap difference against LDA.
The separately interpreted nested-gap family has three positions. The
ablation label includes probability ties at 0.5 in class 1. Transfer and
nested fits predict class 1 only above 0.5. Families retain absent members
as p = 1 for adjustment. Exact inversion can be unbounded at small sample
sizes. See [validation](VALIDATION.md) for the test scope.

## Published methods

- Lawhern et al. (2018), EEGNet, *Journal of Neural Engineering*. [Article](https://doi.org/10.1088/1741-2552/aace8c)
- Rivet et al. (2009), xDAWN, *IEEE Transactions on Biomedical Engineering*. [Article](https://doi.org/10.1109/TBME.2009.2012869)
- Barachant et al. (2012), covariance geometry, *IEEE Transactions on Biomedical Engineering*. [Article](https://doi.org/10.1109/TBME.2011.2172210)
- Chen et al. (2010), OAS, *IEEE Transactions on Signal Processing*. [Article](https://doi.org/10.1109/TSP.2010.2053029)
- Santamaría-Vázquez et al. (2020), EEG-Inception, *IEEE Transactions on Neural Systems and Rehabilitation Engineering*. [Article](https://doi.org/10.1109/TNSRE.2020.3048106)
