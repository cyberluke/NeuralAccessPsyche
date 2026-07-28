"""Collect exact-target and Docker-context blocker evidence without building."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


TARGET = "ad274a2c4b0575cf796659073fb69a962dbb66d0"
PARENT = "fb3690d7884bdbcaef0e82608776065e19332990"
RUN = Path("artifacts/nram_v5_verification/runtime-reverify-ad274a2-20260728T002837Z")
SPEC = "docs/NRAM_V5_SCIENTIFIC_VALIDATION_SPEC.md"


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args])


def write_json(name: str, value: object) -> None:
    (RUN / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


status = git("status", "--porcelain=v2", "--branch", "--untracked-files=all")
(RUN / "starting-git-status.txt").write_bytes(status)

head = git("rev-parse", "HEAD").decode().strip()
branch = git("rev-parse", "--abbrev-ref", "HEAD").decode().strip()
parent_line = git("rev-list", "--parents", "-n", "1", TARGET).decode().split()
actual_parent = parent_line[1] if len(parent_line) > 1 else None

spec_bytes = {commit: git("show", f"{commit}:{SPEC}") for commit in (PARENT, TARGET)}
worktree_spec = Path(SPEC).read_bytes()
spec_identity = {
    "path": SPEC,
    "expected_git_blob_sha256": "9bbd2e93a2750ab1d1094e823487f862b080512025128247e8d7d00f27675506",
    "expected_windows_checkout_sha256": "86f618857042b49a8b8f457634c211d8becab953203ff46127bf244accbb6aca",
    "domains": {
        commit: {
            "domain": "Git blob content bytes emitted by git show; no Git object header",
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "git_blob_oid": git("rev-parse", f"{commit}:{SPEC}").decode().strip(),
        }
        for commit, content in spec_bytes.items()
    },
    "worktree": {
        "domain": "Windows checkout bytes read in binary mode",
        "bytes": len(worktree_spec),
        "sha256": hashlib.sha256(worktree_spec).hexdigest(),
    },
    "parent_target_byte_equal": spec_bytes[PARENT] == spec_bytes[TARGET],
    "target_worktree_byte_equal": spec_bytes[TARGET] == worktree_spec,
}
write_json("spec-identity.json", spec_identity)

prefixes = (
    "api/",
    "core/",
    "nram_sglang/",
    "utils/",
    "evaluation/",
    "templates/",
    "static/",
    "inference/sglang/",
)
roots = {"Dockerfile", "compose.yaml", "pyproject.toml", "main.py"}
target_files = git("ls-tree", "-r", "--name-only", TARGET).decode().splitlines()
production_files = [path for path in target_files if path in roots or path.startswith(prefixes)]

entries = []
for path in production_files:
    target_content = git("show", f"{TARGET}:{path}")
    target_oid = git("rev-parse", f"{TARGET}:{path}").decode().strip()
    checkout = Path(path)
    worktree_content = checkout.read_bytes() if checkout.is_file() else None
    worktree_clean_oid = (
        git("hash-object", "--path", path, path).decode().strip()
        if worktree_content is not None
        else None
    )
    entries.append(
        {
            "path": path,
            "target_git_blob_oid": target_oid,
            "worktree_clean_filter_oid": worktree_clean_oid,
            "canonical_source_equal": worktree_clean_oid == target_oid,
            "target_sha256": hashlib.sha256(target_content).hexdigest(),
            "target_bytes": len(target_content),
            "worktree_sha256": (
                hashlib.sha256(worktree_content).hexdigest()
                if worktree_content is not None
                else None
            ),
            "worktree_bytes": len(worktree_content) if worktree_content is not None else None,
            "raw_checkout_bytes_equal_to_git_blob": (
                worktree_content == target_content if worktree_content is not None else False
            ),
        }
    )

untracked = git("ls-files", "--others", "--exclude-standard").decode().splitlines()
consumed = [path for path in untracked if path in roots or path.startswith(prefixes)]
source_authentication = {
    "target": TARGET,
    "head": head,
    "branch": branch,
    "expected_parent": PARENT,
    "actual_parent": actual_parent,
    "tracked_production_file_count": len(entries),
    "tracked_all_canonical_source_equal": all(
        entry["canonical_source_equal"] for entry in entries
    ),
    "tracked_canonical_source_mismatches": [
        entry for entry in entries if not entry["canonical_source_equal"]
    ],
    "raw_checkout_byte_difference_count": sum(
        not entry["raw_checkout_bytes_equal_to_git_blob"] for entry in entries
    ),
    "raw_checkout_byte_differences": [
        entry for entry in entries if not entry["raw_checkout_bytes_equal_to_git_blob"]
    ],
    "authentication_note": (
        "canonical_source_equal compares Git clean-filter object identity; raw checkout byte "
        "differences are retained separately because Windows CRLF checkout bytes are a distinct domain"
    ),
    "untracked_files_consumed_by_docker_copy": consumed,
    "untracked_consumed_count": len(consumed),
    "files": entries,
}
write_json("target-source-authentication.json", source_authentication)

api_prefixes = ("api/", "core/", "nram_sglang/", "utils/", "evaluation/", "templates/", "static/")
sglang_prefixes = ("inference/sglang/", "nram_sglang/")
context_purity = {
    "compose_context": ".",
    "dockerignore_worktree_exists": Path(".dockerignore").exists(),
    "dockerignore_tracked_at_target": ".dockerignore" in target_files,
    "api_copy_rules": [
        "main.py",
        "api/",
        "core/",
        "nram_sglang/",
        "utils/",
        "evaluation/",
        "templates/",
        "static/",
    ],
    "sglang_copy_rules": [
        "inference/sglang/entrypoint.sh",
        "inference/sglang/healthcheck.py",
        "nram_sglang/",
    ],
    "untracked_consumed": consumed,
    "api_contaminated": any(path == "main.py" or path.startswith(api_prefixes) for path in consumed),
    "sglang_contaminated": any(path.startswith(sglang_prefixes) for path in consumed),
    "decision": (
        "BLOCKED: do not build or recreate exact-target production containers because "
        "untracked production files are included by API COPY rules and no .dockerignore excludes them"
    ),
}
write_json("docker-context-purity.json", context_purity)

(RUN / "commands.txt").write_text(
    "READ-ONLY PRE-PROBE: git rev-parse HEAD; git status --porcelain=v2 --branch "
    "--untracked-files=all; git diff --cached --name-status; git diff --name-status; "
    f"git diff --name-status {TARGET} --; git ls-files --others --exclude-standard\n"
    "EVIDENCE PROBE: python artifacts/nram_v5_verification/"
    "runtime-reverify-ad274a2-20260728T002837Z/collect_blocker_evidence.py\n"
    "NOT EXECUTED BY DESIGN: docker compose build; docker compose down/up; runtime tests; "
    "complete gates\n",
    encoding="utf-8",
)
(RUN / "blocker.txt").write_text(
    "BLOCKED before build. Exact target HEAD and tracked production bytes authenticate, "
    "but untracked production files under core/ are consumed by the API image COPY core/ "
    "rule. No .dockerignore exists. Building would not produce an exact-target image. "
    "Per task instruction, no build, recreate, or runtime verification was attempted.\n",
    encoding="utf-8",
)

print(
    json.dumps(
        {
            "run": str(RUN),
            "head": head,
            "branch": branch,
            "parent": actual_parent,
            "spec": spec_identity,
            "tracked_production_file_count": len(entries),
            "tracked_all_canonical_source_equal": source_authentication[
                "tracked_all_canonical_source_equal"
            ],
            "tracked_canonical_source_mismatch_count": len(
                source_authentication["tracked_canonical_source_mismatches"]
            ),
            "raw_checkout_byte_difference_count": source_authentication[
                "raw_checkout_byte_difference_count"
            ],
            "untracked_consumed_count": len(consumed),
            "untracked_consumed": consumed,
            "decision": context_purity["decision"],
        },
        indent=2,
    )
)
