# Privacy and interpretation

Use [the data contract](DATA_ACCESS.md) with recordings you have permission
to analyze. The attestation makes that permission explicit to the CLI.
Keep device records, epochs, per-participant outputs, predictions, flags,
event maps and stimulus keys in your approved storage. Cohort summaries can
also be sensitive. Follow the applicable disclosure terms for every export.

The CLI writes cohort summaries by default. Participant values require an
explicit flag and are written to owner-only files. Paths inside the checkout
and attempts to overwrite an input file are refused. Library calls can return
participant-level vectors and must be handled accordingly.

The analyses concern displayed image types and may reflect task differences,
repeated stimuli and stream order. A blink-like waveform threshold identifies
large frontal deflections. It does not measure the eyes or localize a source.
A difference between retrained and transferred performance does not identify
an ocular or cortical generator. The additional decoder fits and nested
sensitivity are post hoc. The nested sensitivity uses all outer-source epochs
for configuration selection.

Cross-device and eye-tracking libraries require caller-supplied event provenance
and normalized time series. Calibration, electrode placement, independence of
event clocks, and parser validity depend on the acquisition and must be
established separately. The clock logic retains unknown eye-epoch states for
poor coverage or unexplained pupil loss. Hardware event extraction and vendor
parsing require acquisition-specific adapters.
