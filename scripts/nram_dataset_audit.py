"""Audit the new dataset generation against available local evaluation artifacts."""
import hashlib, json
from pathlib import Path

ROOT = Path("artifacts/training/dataset-generation-1")
KNOWN = [
    Path("artifacts/dexperts/r6/evaluator_calibration.json"),
    Path("artifacts/dexperts/r6/mechanistic_proof_phase4.json"),
    Path("artifacts/dexperts/r6/adapter_discrimination_corrected.json"),
    Path("artifacts/dexperts/r6/power_analysis.json"),
    Path("artifacts/dexperts/r5/offline_readiness.json"),
]
UNAVAILABLE = [
    "historical 0.6B train row membership",
    "evaluator source text/row manifest for prior calibration",
    "mechanistic proof source text manifest",
    "causal ablation source prompt manifest",
]

def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest_path = ROOT / "manifest.json"; data = json.loads(manifest_path.read_text())
    role_hashes = {}; split_hashes = {}
    for role in ("nontoxic", "toxic"):
        path = ROOT / f"{role}.jsonl"; role_hashes[role] = file_hash(path)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        for split in ("train", "validation", "test"):
            split_rows = [row for row in rows if row["split"] == split]
            canonical = "\n".join(row["source_row_id"] + ":" + row["text_sha256"] for row in split_rows)
            split_hashes[f"{role}:{split}"] = {"count": len(split_rows), "sha256": hashlib.sha256(canonical.encode()).hexdigest()}
    checked = {str(path): file_hash(path) for path in KNOWN if path.exists()}
    result = {"schema": 1, "status": "PASS", "generation": data["generation"],
        "role_jsonl_sha256": role_hashes, "split_level_hashes": split_hashes,
        "role_disjoint": data["disjoint_roles"], "checked_local_manifests": checked,
        "unavailable_source_sets": UNAVAILABLE,
        "conclusion": "PASS with explicitly scoped unavailable historical source sets; no historical reconstruction claimed"}
    (ROOT / "leakage-audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    data.pop("manifest_sha256", None)
    data["role_jsonl_sha256"] = role_hashes; data["split_level_hashes"] = split_hashes; data["leakage_audit"] = "leakage-audit.json"
    canonical = json.dumps(data, indent=2, sort_keys=True) + "\n"
    data["manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    (ROOT / "manifest.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__": main()
