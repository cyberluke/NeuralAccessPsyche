import json
import pytest
from nram_training.config import TrainingConfig
from nram_training.fixtures import generate_fixture, MARKERS
from nram_training.trainer import validate_parent, train

def test_cpu_safe_contract():
    cfg = TrainingConfig()
    assert cfg.base_model == "Qwen/Qwen3-14B" and cfg.thinking is False and cfg.packing is True

def test_14b_guard_rejects_06b():
    with pytest.raises(ValueError): validate_parent("Qwen/Qwen3-0.6B")

def test_fixture_deterministic_and_marked(tmp_path):
    a = generate_fixture(tmp_path / "a"); b = generate_fixture(tmp_path / "b")
    assert json.loads(a.read_text()) == json.loads(b.read_text())
    assert set(MARKERS).issubset(json.loads(a.read_text())["markers"])

def test_smoke_roles_are_isolated(tmp_path):
    train("nontoxic", tmp_path, smoke=True); train("toxic", tmp_path, smoke=True)
    assert (tmp_path / "nontoxic" / "training_manifest.json").exists()
    assert (tmp_path / "toxic" / "training_manifest.json").exists()
