"""Collect a redacted, versioned NRAM v5 evidence bundle from local runtime."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickletools
import platform
import subprocess
import sys
from io import StringIO
from pathlib import Path


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=check, capture_output=True, text=True, timeout=120)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    args = parser.parse_args()
    root = Path(args.artifact_dir)
    (root / "raw-generations").mkdir(parents=True, exist_ok=True)
    (root / "plots").mkdir(exist_ok=True)
    (root / "failures").mkdir(exist_ok=True)

    git_status = run("git", "status", "--short").stdout
    containers = json.loads(run("docker", "compose", "ps", "--format", "json").stdout.replace("}\n{", "},{").join(["[", "]"]))
    gpu = run("nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader").stdout.strip()
    processor_hashes = {}
    for container, source_path in (
        ("nram-api", "/app/nram_sglang/processor.py"),
        ("nram-sglang", "/opt/nram_python/nram_sglang/processor.py"),
    ):
        processor_hashes[container] = run(
            "docker", "exec", container, "python", "-c",
            f"import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('{source_path}').read_bytes()).hexdigest())",
        ).stdout.strip()
    environment = {
        "host_os": platform.platform(),
        "host_python": sys.version,
        "starting_commit": "7aa12854800327ea905e608b9ececc3e3f5fd758",
        "current_commit": run("git", "rev-parse", "HEAD").stdout.strip(),
        "tree_dirty": bool(git_status),
        "git_status": git_status.splitlines(),
        "gpu": gpu,
        "sglang_version": "0.5.16",
        "sglang_source_commit": "fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1",
        "sglang_base_digest": "sha256:2a4e0bfde6eb75a2b6ccdda0296494747bb8643e6a88da14c1e6c87035d49cda",
        "model": "Qwen/Qwen3-14B-AWQ",
        "model_snapshot": "31c69efc29464b6bb0aee1398b5a7b50a99340c3",
        "model_manifest_sha256": "1fc86763f6e47c0e4d534f914268b54be1221afb8e82d6f9c82d4fd84e621dd3",
        "tokenizer_json_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "canonical_spec_sha256": sha256(Path("docs/NRAM_V5_SCIENTIFIC_VALIDATION_SPEC.md")),
        "processor_source_sha256": processor_hashes,
        "containers": containers,
    }
    (root / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    commands = """# PowerShell-host commands (executed from repository root)
python -m pytest --collect-only -q
python -m pytest tests/unit tests/contract tests/adversarial tests/integration -q
python -m pytest tests/gpu -q -s
python -m pytest tests -q --junitxml=<artifact-dir>/test-results.xml
python -m ruff check .
python -m mypy nram_sglang/processor.py core/steering/serialization.py core/engines/sglang_engine.py api/routes.py
docker compose config --quiet
docker compose build sglang nram-api
docker compose up -d --force-recreate --wait --wait-timeout 600 sglang
docker compose up -d --force-recreate --no-deps nram-api
python evaluation/nram_v5_paired.py --artifact-dir <artifact-dir> --max-prompts 2 --seeds 101,202 --max-tokens 24
"""
    (root / "commands.sh").write_text(commands, encoding="utf-8")
    launch = run("docker", "exec", "nram-sglang", "python", "-c", "import pathlib; print(pathlib.Path('/proc/1/cmdline').read_bytes().replace(b'\\0',b' ').decode())").stdout
    (root / "server-launch.txt").write_text(launch, encoding="utf-8")
    deps = []
    for container in ("nram-api", "nram-sglang"):
        freeze = run("docker", "exec", container, "python", "-m", "pip", "freeze").stdout
        deps.append(f"## {container}\n{freeze}")
    (root / "dependency-versions.txt").write_text("\n".join(deps), encoding="utf-8")

    diagnostic_code = r'''
import dill, hashlib, json, pickletools
from nram_sglang.processor import NRAMLogitProcessor
raw=dill.dumps(NRAMLogitProcessor); payload=json.dumps({"callable":raw.hex()})
print("module="+NRAMLogitProcessor.__module__)
print("qualname="+NRAMLogitProcessor.__qualname__)
print("raw_bytes="+str(len(raw)))
print("hex_chars="+str(len(raw.hex())))
print("payload_chars="+str(len(payload)))
print("raw_sha256="+hashlib.sha256(raw).hexdigest())
print("zero_arg="+str(type(NRAMLogitProcessor()).__name__))
print("roundtrip_identity="+str(dill.loads(raw) is NRAMLogitProcessor))
pickletools.dis(raw)
'''
    diagnostics = []
    for container in ("nram-api", "nram-sglang"):
        result = run("docker", "exec", container, "python", "-c", diagnostic_code)
        diagnostics.append(f"## {container}\n{result.stdout}")
    sglang_roundtrip = run(
        "docker", "exec", "nram-sglang", "python", "-c",
        "from nram_sglang.processor import NRAMLogitProcessor as N; from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor as C; print(type(C.from_str(N.to_str())).__module__,type(C.from_str(N.to_str())).__qualname__)",
    ).stdout
    diagnostics.append("## exact SGLang from_str\n" + sglang_roundtrip)
    (root / "serialization-diagnostics.txt").write_text("\n".join(diagnostics), encoding="utf-8")

    logs = run("docker", "logs", "nram-sglang").stdout + run("docker", "logs", "nram-sglang").stderr
    events = []
    for line in logs.splitlines():
        if "NRAM_PROCESSOR_EVENT " in line:
            try:
                events.append(json.loads(line.split("NRAM_PROCESSOR_EVENT ", 1)[1]))
            except json.JSONDecodeError:
                pass
    with (root / "runtime-events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    token_fields = ["request_id", "config_hash", "invocation_count", "generated_tokens_before_sample", "forced_token_id", "requested_forced_token_id", "mask_count", "masked_token_ids", "entropy_before", "entropy_after", "entropy_target", "entropy_error"]
    with (root / "token-level-ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=token_fields)
        writer.writeheader()
        for event in events:
            pid = event.get("entropy_pid") or {}
            writer.writerow({
                "request_id": event.get("request_id"), "config_hash": event.get("config_hash"),
                "invocation_count": event.get("invocation_count"), "generated_tokens_before_sample": event.get("generated_tokens_before_sample"),
                "forced_token_id": event.get("forced_token_id"), "requested_forced_token_id": event.get("requested_forced_token_id"),
                "mask_count": event.get("mask_count"), "masked_token_ids": json.dumps(event.get("masked_token_ids", [])),
                "entropy_before": pid.get("entropy_before"), "entropy_after": pid.get("entropy_after"),
                "entropy_target": pid.get("target"), "entropy_error": pid.get("error"),
            })

    (root / "representation-ablation.csv").write_text(
        "status,component,reason\nNOT_IMPLEMENTED,representation_control,no_live_hidden_state_hook_or_valid_vector_artifact\n",
        encoding="utf-8",
    )
    (root / "human-evaluation-template.csv").write_text(
        "prompt_id,seed,condition_a,condition_b,preferred,coherence_a,coherence_b,factuality_a,factuality_b,annotator_id,notes\n",
        encoding="utf-8",
    )
    (root / "plots" / "README.md").write_text("# Plots\n\nNo inferential plots generated: bounded deterministic mechanism tests do not support population statistics.\n", encoding="utf-8")
    failures = {
        "original_gate": {"passed": 273, "failed": 9, "junit": "../phase0-20260727T180644Z-7aa1285/test-results.xml"},
        "repair_policy": "Unsafe application-request serialization assertions were replaced by server-trust-boundary tests; stale GPU endpoint/model/tokenizer were replaced by public pinned-runtime proofs; event-loop tests use asyncio.run.",
        "failed_attempts": [
            "Initial PowerShell command was interpreted by cmd.exe; rerun through explicit PowerShell.",
            "Compose could not tag an image by digest; local wrapper tag retained immutable digest in FROM.",
            "Initial capsule proof sampled at ramp strength zero; rerun at generated step one.",
        ],
    }
    (root / "failures" / "ledger.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")

    files = [path for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"]
    manifest = {
        "run_id": root.name,
        "verdict": "PARTIALLY FUNCTIONAL",
        "starting_commit": "7aa12854800327ea905e608b9ececc3e3f5fd758",
        "dirty_tree": True,
        "event_count": len(events),
        "artifacts": {path.relative_to(root).as_posix(): sha256(path) for path in sorted(files)},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"artifact_dir": str(root), "events": len(events), "files": len(files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
