# Security and privacy policy

## Scope

This repository contains research code, synthetic test generators and documentation. It contains no data, no
credentials and no network functionality. Relevant issues include: code paths that could read data without the
documented fail-closed checks, leakage of personal or dataset-derived information into the repository or its
history, unsafe file loading (for example pickle), and dependency vulnerabilities.

## Reporting

Report a concern privately through GitHub's "Report a vulnerability" button on the repository's Security tab
(private vulnerability reporting). Do not open a public issue for a security or privacy concern, and never include
participant data in a report.

## Handling rules for contributors

* Never commit recordings, derived arrays, predictions, per-participant metrics, figures from real data,
  credentials, tokens, local paths or personal e-mail addresses.
* Load user files only with `numpy.load(..., allow_pickle=False)`.
* Keep tests synthetic and offline; CI must not access data or secrets.
