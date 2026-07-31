"""Deterministic TEST_ONLY adapter metadata; no BF16 weights are loaded."""
from pathlib import Path
from .config import TrainingConfig
from .manifests import atomic_json, canonical_hash

MARKERS = ("TEST_ONLY", "NOT_FOR_SCIENTIFIC_USE")

def generate_fixture(output_dir: str | Path, *, delta: bool = False, seed: int = 42) -> Path:
    cfg = TrainingConfig(); out = Path(output_dir)
    delta_scale = 1e-1 if delta else 0.0
    payload = {"fixture": "synthetic-small-delta" if delta else "synthetic-zero", "markers": list(MARKERS),
        "base_model": cfg.base_model, "base_model_revision": cfg.model_revision, "seed": seed,
        "delta_scale": delta_scale,
        "target_modules": {m: {"rank": cfg.lora_r, "alpha": cfg.lora_alpha, "delta": delta} for m in cfg.target_modules},
        "weights": "NOT_INCLUDED; construct from meta/config only"}
    payload["fixture_sha256"] = canonical_hash(payload)
    target = out / ("small-delta" if delta else "zero"); target.mkdir(parents=True, exist_ok=True)
    config = {"base_model_name_or_path": cfg.base_model, "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "r": cfg.lora_r, "lora_alpha": cfg.lora_alpha, "lora_dropout": cfg.lora_dropout,
        "bias": "none", "target_modules": list(cfg.target_modules), "inference_mode": True,
        "fixture": payload["fixture"], "markers": list(MARKERS)}
    path = target / "adapter_config.json"; atomic_json(path, config)
    try:
        import torch
        from safetensors.torch import save_file
        torch.manual_seed(seed)
        dims = {"q_proj": (5120, 5120), "k_proj": (1024, 5120), "v_proj": (1024, 5120),
                "o_proj": (5120, 5120), "gate_proj": (13824, 5120), "up_proj": (13824, 5120), "down_proj": (5120, 13824)}
        tensors = {}
        for layer in range(40):
            for module in cfg.target_modules:
                out_dim, in_dim = dims[module]
                prefix = f"base_model.model.model.layers.{layer}."
                prefix += "self_attn." if module in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp."
                if delta:
                    tensors[prefix + module + ".lora_A.weight"] = torch.randn(cfg.lora_r, in_dim) * delta_scale
                    tensors[prefix + module + ".lora_B.weight"] = torch.randn(out_dim, cfg.lora_r) * delta_scale
                else:
                    tensors[prefix + module + ".lora_A.weight"] = torch.zeros(cfg.lora_r, in_dim)
                    tensors[prefix + module + ".lora_B.weight"] = torch.zeros(out_dim, cfg.lora_r)
        save_file(tensors, str(target / "adapter_model.safetensors"), metadata={"format": "pt", "fixture": payload["fixture"]})
    except ImportError:
        payload["tensor_artifact"] = "UNAVAILABLE; generate inside training image"
    atomic_json(target / "fixture_manifest.json", payload); return path
