#!/usr/bin/env python3
"""Validate an authorized epoch file against the data contract (docs/DATA_ACCESS.md) without analysing it.

Fails closed: without ``--data`` AND ``--i-am-authorized`` it prints ``DATA_NOT_PROVIDED`` and exits with code 3.
On success it prints only structural facts (shapes, channel availability counts), never signal values.
"""
from __future__ import annotations

import argparse
import json
import sys

import _bootstrap  # noqa: F401

EXIT_DATA_NOT_PROVIDED = 3
EXIT_SCHEMA = 4


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", help="path to a .npz file obeying the data contract")
    ap.add_argument("--i-am-authorized", action="store_true",
                    help="attest that you obtained the data legitimately and may analyse it")
    ap.add_argument("--no-strict-grid", action="store_true", help="allow a time grid other than the paper's")
    a = ap.parse_args(argv)
    from eareeg_decoder.data import load_authorized_epochs
    from eareeg_decoder.errors import DataNotProvidedError, SchemaError
    try:
        es = load_authorized_epochs(a.data, attest_authorized=a.i_am_authorized, strict_grid=not a.no_strict_grid)
    except DataNotProvidedError as exc:
        print(str(exc), file=sys.stderr)
        print("DATA_NOT_PROVIDED")
        return EXIT_DATA_NOT_PROVIDED
    except SchemaError as exc:
        print(f"SCHEMA_ERROR: {exc}", file=sys.stderr)
        return EXIT_SCHEMA
    from eareeg_decoder.pipeline import frontal_usable
    parts = [int(p) for p in es.participants]
    summary = {"status": "DATA_CONTRACT_OK", "n_epochs": int(es.X.shape[0]), "n_participants": len(parts),
               "n_times": int(es.X.shape[2]), "n_retained": int(es.keep.sum()),
               "n_participants_frontal_usable": sum(frontal_usable(es, p) for p in parts),
               "n_participants_all_11_good": sum(len(es.good_channels(p)) == 11 for p in parts)}
    json.dump(summary, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
