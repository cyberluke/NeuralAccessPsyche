"""Shared trainer contract. Heavy ML dependencies are imported only in runtime training."""
from pathlib import Path
from .config import AdapterRole, config_for
from .manifests import atomic_json, make_manifest

def validate_parent(model_name: str) -> None:
    if "0.6b" in model_name.lower():
        raise ValueError("Qwen3-0.6B cannot be attached to the Qwen3-14B pipeline")
    if model_name != "Qwen/Qwen3-14B":
        raise ValueError("production parent must be Qwen/Qwen3-14B")

def train(adapter: AdapterRole, output_dir: str | Path, *, resume: str = "auto", smoke: bool = False) -> dict:
    cfg = config_for(adapter); validate_parent(cfg.base_model)
    out = Path(output_dir) / adapter; out.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest(cfg.as_dict()) | {"adapter": adapter, "resume": resume, "smoke": smoke,
        "status": "READY_FOR_RUNTIME" if smoke else "REQUIRES_GPU_RUNTIME"}
    atomic_json(out / "training_manifest.json", manifest)
    if not smoke:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            import peft  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("training requires pinned torch, transformers, and peft") from exc
        raise RuntimeError("GPU training must run in the pinned training image")
    return manifest
