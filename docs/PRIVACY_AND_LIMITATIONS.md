# Privacy and limitations

## Privacy

* The repository contains code, synthetic generators, method constants and documentation only.
* No recordings, eye-tracking data, stimuli, behavioural tables, participant identifiers, per-participant or
  per-trial values, predictions, model weights or cohort results are included — in the working tree or the Git
  history.
* The code makes no network connections and collects no telemetry.
* Synthetic identifiers are prefixed `SYNTH_`. Test fixtures are generated at run time from seeds.
* Anyone analysing real data with this code is responsible for complying with their data agreement; outputs of
  `scripts/eareeg_analyze.py` are cohort summaries by default, and participant-level values are written only on
  explicit request and are marked `PRIVATE`.

## What the code can and cannot tell you

* **Digital montages, not physical sources.** Deleting an electrode changes channel count, rank and reference
  together. A montage contrast says where decodable information lies in this montage, not where it is generated.
* **Frontal transients are not eye measurements.** The Fpz–mastoid flag marks large frontal deflections of any
  origin; a flagged epoch need not contain a blink, and a "transient-free" epoch (free of *large* transients) may
  still contain blinks or other artefacts. The 32-feature descriptor is not a blink sensor.
* **Selection on a post-onset quantity.** Free epochs are selected on an amplitude measured mostly after onset, and
  every retained epoch already passed a 150-µV rule on in-ear and mastoid channels; subset results describe the
  selected epochs and support no causal statement.
* **Retraining and transfer answer different questions.** Retraining on transient-free epochs asks whether frontal
  information there is learnable; scoring the decoder trained on all epochs on the same epochs asks whether it
  expresses that information. The paired gap Δ describes how decoders trained on different epoch sets use the data,
  not whether the information is ocular or cortical (see `tests/test_integration_synthetic.py`, where an ocular and
  a neural synthetic signal produce the same pattern).
* **Decoder dependence.** Results obtained with one decoder need not hold for another; run and report each decoder.
* **Validation schemes.** LOSO keeps every stimulus photograph in training and may exploit stimulus-specific
  information; joint hold-out tests on new stimuli. Results from the two schemes are never pooled.
* **Labels** describe the displayed image (face or tree), not the participant's emotion or cognitive state.

## Design limitations stated in the manuscript's Methods

* All analyses are post hoc on one repeatedly examined cohort; analysis plans were frozen in stages after earlier
  results were known; nothing is an independent preregistration or confirmation.
* Image type is confounded with task relevance, rarity (more faces than trees per stream), repetition of a small
  set of tree photographs and serial order.
* The cohort is small and its analysis subsets are nested (the same people), not independent samples.
* Amplitudes are on a nominal, uncalibrated microvolt scale; the hardware reference, polarity convention, ground,
  physical electrode placement and display latency were not verified; latencies are relative to event markers.
* An eye-tracker alignment based on trigger records cannot prove hardware trigger lineage (it could be circular);
  a parsed blink is a pupil-loss interval; no same-clock EOG with documented placement was available, so ocular
  activity could not be regressed out.
* Dataset-controller permission to publish derived results has not been verified.

## Not for clinical or individual decisions

This software is research code for a methodological question. It must not be used to make decisions about
individuals.
