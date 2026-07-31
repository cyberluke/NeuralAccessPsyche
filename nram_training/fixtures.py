"""Deterministic TEST_ONLY adapter metadata; no BF16 weights are loaded."""
from pathlib import Path
from .config import TrainingConfig
from .manifests import atomic_json, canonical_hash

MARKERS = ("TEST_ONLY", "NOT_FOR_SCIENTIFIC_USE")

def generate_fixture(output_dir: str | Path, *, delta: bool = False, seed: int = 42) -> Path:
    cfg = TrainingConfig(); out = Path(output_dir)
    payload = {"fixture": "synthetic-small-delta" if delta else "synthetic-zero", "markers": list(MARKERS),
        "base_model": cfg.base_model, "base_model_revision": "UNAVAILABLE", "seed": seed,
        "target_modules": {m: {"rank": cfg.lora_r, "alpha": cfg.lora_alpha, "delta": delta} for m in cfg.target_modules},
        "weights": "NOT_INCLUDED; construct from meta/config only"}
    payload["fixture_sha256"] = canonical_hash(payload)
    path = out / ("small-delta" if delta else "zero") / "adapter_config.json"
    atomic_json(path, payload); return path
