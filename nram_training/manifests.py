import hashlib, json, os, tempfile
from pathlib import Path
from typing import Any

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)

def make_manifest(config: dict, image: str = "UNAVAILABLE") -> dict:
    manifest = {"schema": 1, "immutable": True, "config": config, "model_revision": "UNAVAILABLE",
        "tokenizer_revision": config.get("tokenizer_revision", "UNAVAILABLE"), "dataset_hash": "UNAVAILABLE",
        "code_revision": "UNAVAILABLE", "training_image": image}
    manifest["manifest_sha256"] = canonical_hash(manifest)
    return manifest
