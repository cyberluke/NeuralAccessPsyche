"""Materialize a new reproducible Civil Comments experiment generation."""
import argparse, hashlib, json, re
from pathlib import Path

REVISION = "f2970eb3a55777454c94069077cc8d9b5866312d"
SEED = 20260731

def digest(value): return hashlib.sha256(value.encode("utf-8")).hexdigest()
def norm(text): return re.sub(r"\s+", " ", text.strip().lower())
def assign(row_id):
    bucket = int(hashlib.sha256(f"{SEED}:{row_id}".encode()).hexdigest()[:8], 16) % 100
    return "test" if bucket < 10 else "validation" if bucket < 20 else "train"

def materialize(out, limit):
    from datasets import load_dataset
    Path(out).mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("google/civil_comments", revision=REVISION, split="train")
    selected = {"nontoxic": [], "toxic": []}; exact = set(); normalized = set()
    for index, row in enumerate(dataset):
        text = row.get("text"); toxicity = row.get("toxicity")
        if not isinstance(text, str) or not text.strip() or toxicity is None: continue
        role = "nontoxic" if toxicity <= 0.05 else "toxic" if toxicity >= 0.80 else None
        if role is None or len(selected[role]) >= limit: continue
        raw_hash = digest(text); normalized_hash = digest(norm(text))
        if raw_hash in exact or normalized_hash in normalized: continue
        exact.add(raw_hash); normalized.add(normalized_hash)
        row_id = f"train:{index}"; selected[role].append({"source_row_id": row_id, "text": text,
            "text_sha256": raw_hash, "normalized_text_sha256": normalized_hash,
            "toxicity": float(toxicity), "split": assign(row_id)})
        if all(len(rows) >= limit for rows in selected.values()): break
    result = {"schema": 1, "status": "PASS", "generation": "civil-comments-qwen3-14b-generation-1",
        "source": {"repository": "google/civil_comments", "revision": REVISION, "split": "train",
                   "text_column": "text", "label_column": "toxicity"},
        "rules": {"nontoxic": "toxicity <= 0.05", "toxic": "toxicity >= 0.80", "null_or_empty": "excluded",
                  "exact_duplicate": "text SHA-256", "normalized_duplicate": "SHA-256(norm whitespace/lowercase)", "seed": SEED},
        "roles": selected, "counts": {role: len(rows) for role, rows in selected.items()},
        "disjoint_roles": not bool({r["normalized_text_sha256"] for r in selected["nontoxic"]} & {r["normalized_text_sha256"] for r in selected["toxic"]}),
        "old_generation": "DATASET_PROVENANCE_UNRECOVERABLE; not reconstructed"}
    for role, rows in selected.items():
        (Path(out) / f"{role}.jsonl").write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    manifest = json.dumps(result, indent=2, sort_keys=True) + "\n"; (Path(out) / "manifest.json").write_text(manifest)
    result["manifest_sha256"] = digest(manifest); (Path(out) / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--out", default="artifacts/training/dataset-generation-1"); p.add_argument("--limit", type=int, default=1000)
    args = p.parse_args(); print(json.dumps(materialize(args.out, args.limit), indent=2, sort_keys=True))
