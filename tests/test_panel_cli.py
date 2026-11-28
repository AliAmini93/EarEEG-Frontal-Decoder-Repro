"""CLI for the decoder panel (candidate 0.2.0): fail-closed paths, refusals, frozen families with INCOMPLETE
members, no participant-level output by default, opt-in adaptive draws. Explicitly synthetic files in tmp only."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from eareeg_decoder.data import save_epochs
from eareeg_decoder.synthetic import synthetic_epochs

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eareeg_analyze.py"


def run(*args, timeout=900):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=timeout,
                          cwd=str(ROOT))


@pytest.fixture(scope="module")
def synth_file(tmp_path_factory):
    es, _ = synthetic_epochs(n_participants=6, n_streams=6, seed=9, neural_amp=6.0)
    f = tmp_path_factory.mktemp("panel") / "synthetic.npz"
    save_epochs(es, f)
    return f


def test_help_lists_panel_modes():
    r = run("--help")
    assert r.returncode == 0
    for m in ("panel_drop_one", "panel_regimes", "panel_nested", "--decoder", "--adaptive-draws"):
        assert m in r.stdout


@pytest.mark.parametrize("analysis", ["panel_drop_one", "panel_regimes", "panel_nested"])
def test_panel_modes_fail_closed_without_data(tmp_path, analysis):
    r = run("--analysis", analysis, "--out", str(tmp_path / "o.json"))
    assert r.returncode == 3 and "DATA_NOT_PROVIDED" in r.stdout
    assert not (tmp_path / "o.json").exists()


def test_refusals(tmp_path, synth_file):
    r = run("--data", str(synth_file), "--i-am-authorized", "--analysis", "panel_regimes", "--adaptive-draws",
            "--out", str(tmp_path / "o.json"))
    assert r.returncode == 6
    r = run("--data", str(synth_file), "--i-am-authorized", "--analysis", "panel_drop_one",
            "--out", str(ROOT / "inside.json"))
    assert r.returncode == 6 and not (ROOT / "inside.json").exists()
    r = run("--analysis", "panel_regimes", "--offsets", "0,7", "--out", str(tmp_path / "o.json"))
    assert r.returncode == 2


def test_panel_drop_one_xdawn_frozen_family(tmp_path, synth_file):
    out = tmp_path / "d1.json"
    r = run("--data", str(synth_file), "--i-am-authorized", "--analysis", "panel_drop_one", "--decoder",
            "xdawn_ts_lr", "--n-boot", "200", "--out", str(out))
    assert r.returncode == 0, r.stderr
    res = json.loads(out.read_text())
    fam = res["family_primary_d_F"]
    assert fam["xdawn_ts_lr.d_F"]["holm_m"] == 3 and "label" in fam["xdawn_ts_lr.d_F"]
    assert fam["eegnet_step1.d_F"]["label"] == "INCOMPLETE" and fam["eeg_inception_erp.d_F"]["label"] == "INCOMPLETE"
    assert len(res["family_secondary_mastoid"]) == 6
    assert res["decoders"]["xdawn_ts_lr"]["complete"] is True
    assert "PRIVATE_participant_values" not in res


def test_panel_regimes_xdawn_families(tmp_path, synth_file):
    out = tmp_path / "d2.json"
    r = run("--data", str(synth_file), "--i-am-authorized", "--analysis", "panel_regimes", "--decoder", "xdawn_ts_lr",
            "--draws", "1", "--n-boot", "200", "--min-face", "10", "--min-tree", "3", "--out", str(out))
    assert r.returncode == 0, r.stderr
    res = json.loads(out.read_text())
    fam = res["family_delta"]
    assert len(fam) == 6
    present = [k for k, v in fam.items() if v.get("label") != "INCOMPLETE"]
    assert sorted(present) == ["xdawn_ts_lr.joint", "xdawn_ts_lr.loso"]
    assert all(fam[k]["holm_m"] == 6 for k in present)
    for f in ("family_c", "family_s", "family_vs_lda"):
        assert len(res[f]) == 6
    assert res["decoders"]["xdawn_ts_lr.loso"]["test_sets_identical"] is True


def test_regimes_adaptive_draws_record(tmp_path, synth_file):
    out = tmp_path / "r.json"
    r = run("--data", str(synth_file), "--i-am-authorized", "--analysis", "regimes", "--adaptive-draws",
            "--n-boot", "200", "--min-face", "10", "--min-tree", "3", "--out", str(out))
    assert r.returncode == 0, r.stderr
    rule = json.loads(out.read_text())["draw_rule"]
    assert rule["n_draws_used"] in (20, 50) and rule["threshold"] == 0.0025


def test_nested_insufficient_inner_data_returns_incomplete(tmp_path, synth_file):
    out = tmp_path / "nested.json"
    r = run("--data", str(synth_file), "--i-am-authorized", "--analysis", "panel_nested", "--decoder",
            "xdawn_ts_lr", "--n-boot", "20", "--out", str(out))
    assert r.returncode == 0, r.stderr
    result = json.loads(out.read_text())
    assert result["decoders"]["xdawn_ts_lr"]["complete"] is False
    assert all(member["label"] == "INCOMPLETE" for member in result["family_delta_tuned"].values())
    assert "PRIVATE_participant_values" not in result


def test_input_output_collision_is_refused_even_with_overwrite(synth_file):
    size = synth_file.stat().st_size
    r = run("--data", str(synth_file), "--i-am-authorized", "--analysis", "regimes",
            "--out", str(synth_file), "--overwrite")
    assert r.returncode == 6
    assert synth_file.stat().st_size == size
