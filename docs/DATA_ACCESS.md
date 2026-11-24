# Data access and the data contract

**No data are distributed with this repository, and there is no downloader.** The recordings analysed in the
manuscript belong to a dataset controller; the dataset's name, version, citation, ethics and consent statements
must come from that controller and are not given here. To analyse real recordings you must obtain them, and any
permission to publish results derived from them, directly from the controller.

## Fail-closed behaviour

`eareeg_decoder.data.load_authorized_epochs(path, attest_authorized=...)` and the scripts refuse to read anything
unless **both** a file path and an explicit attestation are given:

```bash
python scripts/eareeg_check_data.py                          # -> DATA_NOT_PROVIDED, exit 3
python scripts/eareeg_check_data.py --data x.npz             # -> DATA_NOT_PROVIDED (not attested), exit 3, nothing read
python scripts/eareeg_check_data.py --data x.npz --i-am-authorized
```

The attestation is your statement that you obtained the data legitimately and may analyse it; the code cannot
verify it. Files are read with `numpy.load(..., allow_pickle=False)`. Nothing is cached, copied or uploaded.

## File format (one `.npz`)

| Key | Shape / type | Meaning |
|---|---|---|
| `X` | (n, C, T) float | epochs on the recording's nominal µV scale in the **stored common average of the participant's good ear-system channels** (the good rows sum to zero; convert with `preprocessing.common_average`). A channel that failed QC for a participant is an **all-NaN row in all of that participant's epochs**. |
| `channel_names` | (C,) str | must contain `Fpz, M1, M2, EL1, EL3, EL4, EL5, ER1, ER2, ER3, ER4` |
| `times` | (T,) float | seconds; the paper protocol is the 250-Hz grid −0.2 … 0.848 s (263 samples) |
| `y` | (n,) int | 1 = a face was shown, 0 = a tree was shown |
| `participant` | (n,) int | pseudonymous integer codes 0…P−1 (no names, initials or source identifiers) |
| `stimulus_key` | (n,) str | stimulus identity used for stimulus groups: a face identity code or a tree photograph name; `''` if unknown. A key must occur with one label only and must never identify a participant. |
| `keep` | (n,) bool | epoch passed the 150-µV rule (decoders use kept epochs; transient rates use all presentations) |
| `stream` | (n,) int, optional | 20-image stream (block) index |
| `position` | (n,) int, optional | presentation order within the recording |
| `onset_s` | (n,) float, optional | onset on the EEG clock (s), for the next-baseline screen |

Include **all** presentations of each participant (not only kept epochs) so that transient rates follow the
paper's definition. Validation errors raise `SchemaError` with an explicit message. The original analyses
stored epochs as float32; precision differences of that order do not affect the checks.

## Preparing the file from continuous recordings

If you start from continuous recordings, `eareeg_decoder.preprocessing.preprocess_recording` implements the
label-blind chain (filter → channel QC → epochs → baseline → stored common average → 150-µV rule) for data already
expressed on the nominal µV scale with known onsets. File import, unit conversion and event extraction are
recording-specific and are **not** part of this package.

## Privacy

Treat every array derived from recordings — epochs, flags, predictions, per-participant metrics — as personal
data under your data agreement. Do not commit such files to any repository (this repository's `.gitignore` blocks
common data formats, but that is a guard rail, not permission). Small cohort aggregates can be re-identifying;
obtain the controller's approval before publishing any.
