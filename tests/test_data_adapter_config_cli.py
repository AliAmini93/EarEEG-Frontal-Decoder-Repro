"""Fail-closed data adapter, schema checks, protocol file and command-line scripts (synthetic only)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from eareeg_decoder import config
from eareeg_decoder.data import EpochSet, load_authorized_epochs, save_epochs, validate
from eareeg_decoder.errors import DataNotProvidedError, SchemaError
from eareeg_decoder.synthetic import synthetic_epochs

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(script, *args, timeout=600):
    return subprocess.run([sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True,
                          timeout=timeout, cwd=str(ROOT))


@pytest.fixture(scope="module")
def tiny():
    es, _ = synthetic_epochs(n_participants=4, n_streams=4, seed=5)
    return es


def test_no_path_fails_closed():
    with pytest.raises(DataNotProvidedError, match=r"^DATA_NOT_PROVIDED"):
        load_authorized_epochs(None)


def test_path_without_attestation_reads_nothing(tmp_path):
    missing = tmp_path / "does_not_exist.npz"
    with pytest.raises(DataNotProvidedError, match="authorization was not attested"):
        load_authorized_epochs(missing)
    with pytest.raises(DataNotProvidedError, match="not a readable file"):
        load_authorized_epochs(missing, attest_authorized=True)


def test_roundtrip_and_pickle_refused(tmp_path, tiny):
    f = tmp_path / "synthetic.npz"
    save_epochs(tiny, f)
    es = load_authorized_epochs(f, attest_authorized=True)
    np.testing.assert_allclose(es.X, tiny.X, equal_nan=True)
    assert es.channel_names == tiny.channel_names
    bad = tmp_path / "objects.npz"
    arrays = dict(np.load(f))
    arrays["stimulus_key"] = np.array([{"x": 1}] * len(tiny.y), dtype=object)
    np.savez(bad, **arrays)
    with pytest.raises(ValueError):
        load_authorized_epochs(bad, attest_authorized=True)


def test_schema_errors(tiny):
    def variant(**kw):
        d = dict(X=tiny.X, channel_names=list(tiny.channel_names), times=tiny.times, y=tiny.y,
                 participant=tiny.participant, stimulus_key=tiny.stimulus_key, keep=tiny.keep)
        d.update(kw)
        return EpochSet(**d)

    with pytest.raises(SchemaError, match="missing ear-system channels"):
        validate(variant(X=tiny.X[:, :10], channel_names=list(tiny.channel_names)[:10]))
    with pytest.raises(SchemaError, match="250-Hz grid"):
        validate(variant(times=tiny.times + 0.004))
    with pytest.raises(SchemaError, match="pseudonymous integer"):
        validate(variant(participant=np.array([f"P{p}" for p in tiny.participant])))
    with pytest.raises(SchemaError, match="1 \\(face\\) / 0 \\(tree\\)"):
        validate(variant(y=tiny.y * 2))
    X = tiny.X.copy()
    X[0, 3, 5] = np.nan
    with pytest.raises(SchemaError, match="partially NaN"):
        validate(variant(X=X))
    with pytest.raises(SchemaError, match="keep must be boolean"):
        validate(variant(keep=tiny.keep.astype(int)))


def test_protocol_file_matches_module():
    text = (ROOT / "configs" / "protocol.json").read_text(encoding="utf-8")
    assert text == config.protocol_json()
    P = config.load_protocol_file(ROOT / "configs" / "protocol.json")
    assert P["transients"]["theta_uv"] == 100.0 and P["preprocessing"]["epoch_rejection_ptp_uv"] == 150.0
    assert P["transients"]["post_peak_min_s"] == 0.15 and P["preprocessing"]["n_epoch_samples"] == 263
    assert P["channels"]["order"][0] == "Fpz" and len(P["channels"]["order"]) == 11
    assert config.protocol() is not config.PROTOCOL


@pytest.mark.parametrize("script", ["eareeg_check_data.py", "eareeg_analyze.py"])
def test_scripts_help(script):
    r = run(script, "--help")
    assert r.returncode == 0 and "usage" in r.stdout.lower()


def test_check_data_fails_closed_without_data():
    r = run("eareeg_check_data.py")
    assert r.returncode == 3 and r.stdout.strip() == "DATA_NOT_PROVIDED"
    r = run("eareeg_check_data.py", "--data", "nothing_here.npz")
    assert r.returncode == 3 and "not attested" in r.stderr
    r = run("eareeg_analyze.py", "--analysis", "regimes", "--out", "unused.json")
    assert r.returncode == 3 and "DATA_NOT_PROVIDED" in r.stdout
    assert not (ROOT / "unused.json").exists()


def test_check_and_analyze_on_synthetic_file(tmp_path, tiny):
    f = tmp_path / "synthetic.npz"
    save_epochs(tiny, f)
    r = run("eareeg_check_data.py", "--data", str(f), "--i-am-authorized")
    assert r.returncode == 0, r.stderr
    s = json.loads(r.stdout)
    assert s["status"] == "DATA_CONTRACT_OK" and s["n_participants"] == 4
    out = tmp_path / "res.json"
    r = run("eareeg_analyze.py", "--data", str(f), "--i-am-authorized", "--analysis", "transients", "--out", str(out),
            "--n-boot", "200")
    assert r.returncode == 0, r.stderr
    res = json.loads(out.read_text())
    assert "PRIVATE_participant_values" not in res and "tree_minus_face" in res["post_onset_rate"]


def test_stimulus_key_in_both_classes_is_a_schema_error(tiny):
    keys = tiny.stimulus_key.copy()
    keys[(tiny.y == 0)] = keys[tiny.y == 1][0]
    es = EpochSet(X=tiny.X, channel_names=tiny.channel_names, times=tiny.times, y=tiny.y,
                  participant=tiny.participant, stimulus_key=keys, keep=tiny.keep)
    with pytest.raises(SchemaError, match="both labels"):
        validate(es)


def test_analyze_refuses_output_inside_repo_and_reports_schema_errors(tmp_path, tiny):
    f = tmp_path / "synthetic.npz"
    save_epochs(tiny, f)
    r = run("eareeg_analyze.py", "--data", str(f), "--i-am-authorized", "--analysis", "transients",
            "--out", str(ROOT / "inside_repo.json"))
    assert r.returncode == 6 and "outside this repository" in r.stderr and not (ROOT / "inside_repo.json").exists()
    out = tmp_path / "x.json"
    out.write_text("{}")
    r = run("eareeg_analyze.py", "--data", str(f), "--i-am-authorized", "--analysis", "transients", "--out", str(out))
    assert r.returncode == 6 and "--overwrite" in r.stderr
    arrays = dict(np.load(f))
    arrays["times"] = arrays["times"] + 0.004
    g = tmp_path / "badgrid.npz"
    np.savez(g, **arrays)
    r = run("eareeg_analyze.py", "--data", str(g), "--i-am-authorized", "--analysis", "transients",
            "--out", str(tmp_path / "y.json"))
    assert r.returncode == 4 and "SCHEMA_ERROR" in r.stderr and "Traceback" not in r.stderr
    r = run("eareeg_analyze.py", "--data", str(f), "--i-am-authorized", "--analysis", "regimes", "--draws", "0",
            "--out", str(tmp_path / "z.json"))
    assert r.returncode == 2 and "must be >= 1" in r.stderr


def test_synthetic_artifacts_are_rejected_by_the_150uV_rule():
    es, _ = synthetic_epochs(n_participants=2, n_streams=5, seed=9, artifact_rate=0.2)
    assert 0.6 < es.keep.mean() < 0.95


def test_analyze_regimes_with_more_than_24_participants(tmp_path):
    es, _ = synthetic_epochs(n_participants=26, n_streams=2, seed=13)
    f = tmp_path / "many.npz"
    save_epochs(es, f)
    out = tmp_path / "regimes.json"
    r = run("eareeg_analyze.py", "--data", str(f), "--i-am-authorized", "--analysis", "regimes", "--draws", "1",
            "--min-face", "1", "--min-tree", "1", "--n-boot", "100", "--out", str(out))
    assert r.returncode == 0, r.stderr
    res = json.loads(out.read_text())
    assert res["delta"]["inversion_ci"] is None and res["delta"]["p_signflip"] is None and "n <= 24" in res["delta"]["p_note"]
