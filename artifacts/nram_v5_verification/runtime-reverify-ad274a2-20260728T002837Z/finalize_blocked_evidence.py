"""Finalize a self-contained blocked re-verification evidence bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


RUN = Path("artifacts/nram_v5_verification/runtime-reverify-ad274a2-20260728T002837Z")
TARGET = "ad274a2c4b0575cf796659073fb69a962dbb66d0"


def write_json(name: str, value: object) -> None:
    (RUN / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


issues = []
titles = {
    1: "False model identity and stale public aliases",
    2: "Comparison bypasses processor and fails after generation",
    3: "Exact-target tests and artifacts do not match committed tree",
    4: "Cancellation and timeout do not abort upstream generation",
    5: "Permissive and unsafe request validation",
    6: "Streaming contract defects",
    7: "Telemetry incomplete, truncatable, spoofable, and synthetic",
    8: "Applied-state hash omits behaviorally relevant state",
    9: "Session controls disconnected from inference",
    10: "MoE bypasses validation, ignores bounds/controls, and swallows errors",
    11: "Tokenizer variants and structural proof gaps",
    12: "Overstated/dead capabilities and contradictions",
    13: "Fake authentication boundary",
    14: "Original failure/provenance history incomplete",
    15: "Numerical stability and all-mask recovery",
    16: "Bounded phrase/capsule/vector/entropy evidence overstated",
    17: "Comparison/session/dashboard/test package/runtime drift",
    18: "DExperts absent",
    19: "Representation stack absent",
    20: "Semantic/evidence closed loop absent",
    21: "Branch-and-tournament absent",
    22: "Full causal/statistical study absent",
    23: "Original 269/273 four-failure artifact unavailable",
    24: "Unattributed container exit 137",
}
for number in range(1, 25):
    if number <= 17:
        prior = "CONFIRMED"
        result = "NOT_RETESTED"
        evidence = (
            "Exact-target runtime re-verification stopped before build because the root "
            "Docker context contains untracked production modules consumed by COPY core/."
        )
    elif number <= 23:
        prior = "BLOCKED"
        result = "BLOCKED"
        evidence = (
            "Canonical blocked status retained and not upgraded. Fail-loud runtime behavior "
            "was not independently rerun because exact-target containers could not be built."
        )
    else:
        prior = "UNCONFIRMED"
        result = "UNCONFIRMED"
        evidence = (
            "No cold recreation or Docker event/resource capture occurred after the mandatory "
            "context-purity gate failed."
        )
    issues.append(
        {
            "id": f"NRAM-{number:03d}",
            "title": titles[number],
            "prior_status": prior,
            "tested_evidence": evidence,
            "result": result,
        }
    )
write_json("issue-verification-table.json", issues)

claims = {
    "claims_tested": [
        {
            "claim": "HEAD is the exact remediation commit with the expected parent and branch",
            "supporting_observable": "Git object IDs and branch name",
            "falsification": "Any unequal HEAD, parent, or branch",
            "control": "Resolve immutable target and parent objects directly",
            "result": "SUPPORTED",
        },
        {
            "claim": "Canonical specification is unchanged at the remediation commit",
            "supporting_observable": "Equal parent/target Git blob bytes and expected SHA-256",
            "falsification": "Blob inequality or hash mismatch",
            "control": "Separate Git-blob and Windows-checkout byte domains",
            "result": "SUPPORTED",
        },
        {
            "claim": "Tracked production source matches the target commit",
            "supporting_observable": "Git clean-filter object identity for every consumed tracked file",
            "falsification": "Any canonical object mismatch",
            "control": "Retain raw checkout byte comparison separately to expose CRLF normalization",
            "result": "SUPPORTED: 89/89 canonical source files match",
        },
        {
            "claim": "The current Docker context can produce an exact-target API image",
            "supporting_observable": "No non-target file reaches a Docker COPY source",
            "falsification": "Any untracked file included by a COPY rule without an ignore exclusion",
            "control": "Compare untracked paths against both image COPY rules and ignore policy",
            "result": "FALSIFIED: 11 untracked core/steering modules are consumed by COPY core/",
        },
    ],
    "runtime_claims": {
        "result": "NOT TESTED",
        "reason": "Mandatory exact-target container build gate failed before build",
        "includes": [
            "public API to exact rebuilt SGLang to Qwen3-14B-AWQ to RTX 4090",
            "model identity and alias rejection",
            "comparison processor causality",
            "cancellation and deadline abort",
            "ordinary and persona SSE",
            "strict pre-generation validation",
            "telemetry correlation and retention",
            "applied-state hashing",
            "session accounting",
            "MoE preflight and failures",
            "structural masks and numerical controls",
            "capability and authentication routes",
            "blocked mechanism fail-loud behavior",
            "all tracked, GPU, quality, Node, and Compose build gates",
        ],
    },
}
write_json("claims-and-controls.json", claims)

command_results = [
    {
        "sequence": 1,
        "purpose": "Initial combined provenance probe",
        "exit": 255,
        "result": "The system cannot find the file specified; no output and no workspace change",
    },
    {
        "sequence": 2,
        "command": "Get-Location; git rev-parse HEAD; git status --porcelain=v2 --branch --untracked-files=all",
        "exit": 1,
        "result": "Command runner was cmd.exe, not PowerShell; Get-Location was not recognized",
    },
    {
        "sequence": 3,
        "command": (
            "cd && git rev-parse HEAD && git rev-parse --abbrev-ref HEAD && git show -s "
            "--format=... HEAD && git status --porcelain=v2 --branch --untracked-files=all "
            "&& git diff --cached --name-status && git diff --name-status && git diff "
            f"--name-status {TARGET} -- && git ls-files --others --exclude-standard"
        ),
        "exit": 0,
        "raw_tool_artifact": "cmd-1785198528495.txt",
    },
    {
        "sequence": 4,
        "purpose": "One-line evidence collector attempt",
        "exit": 1,
        "result": "Command-line quoting SyntaxError before script body execution; no directory created",
        "raw_tool_artifact": "cmd-1785198583484.txt",
    },
    {
        "sequence": 5,
        "command": (
            "python artifacts\\nram_v5_verification\\runtime-reverify-ad274a2-"
            "20260728T002837Z\\collect_blocker_evidence.py"
        ),
        "exit": 0,
        "result": "Initial evidence pass; exposed raw-byte versus canonical-source domain ambiguity",
    },
    {
        "sequence": 6,
        "command": (
            "python artifacts\\nram_v5_verification\\runtime-reverify-ad274a2-"
            "20260728T002837Z\\collect_blocker_evidence.py"
        ),
        "exit": 0,
        "result": "Corrected domain-aware source authentication and contamination decision",
    },
    {
        "sequence": 7,
        "command": (
            "python artifacts\\nram_v5_verification\\runtime-reverify-ad274a2-"
            "20260728T002837Z\\finalize_blocked_evidence.py"
        ),
        "exit": 0,
        "result": "Final evidence and manifest generation",
    },
    {
        "sequence": 8,
        "command": (
            "git rev-parse HEAD; git status --porcelain=v2 --branch --untracked-files=all; "
            "dir /b core\\steering; git ls-files --others --exclude-standard core api "
            "nram_sglang utils evaluation templates static inference/sglang"
        ),
        "exit": 0,
        "raw_tool_artifact": "cmd-1785200329708.txt",
        "result": (
            "Late context-purity recheck confirmed exact target HEAD and the same 11 untracked "
            "production modules remain consumed by the API COPY core/ rule"
        ),
    },
    {
        "sequence": 9,
        "command": "Persist late context-purity recheck to context-purity-recheck.txt",
        "exit": 0,
        "result": "Direct raw recheck captured inside evidence bundle",
    },
    {
        "sequence": 10,
        "command": (
            "python artifacts\\nram_v5_verification\\runtime-reverify-ad274a2-"
            "20260728T002837Z\\finalize_blocked_evidence.py"
        ),
        "exit": 0,
        "result": "Final post-recheck evidence and manifest generation",
    },
]
write_json("command-results.json", command_results)

findings = [
    {
        "id": "RVR-001",
        "severity": "HIGH (verification blocker; not attributed as a target production defect)",
        "confidence": "HIGH",
        "location": [
            "Dockerfile:36",
            "compose.yaml:117",
            "core/steering/activation_addition.py",
            "core/steering/activation_vectors.py",
            "core/steering/concept_capsules.py",
            "core/steering/conceptor_steering.py",
            "core/steering/dexperts.py",
            "core/steering/entropy_controller.py",
            "core/steering/forward_hooks.py",
            "core/steering/hallucination_guard.py",
            "core/steering/hidden_state_probes.py",
            "core/steering/multi_vector_controller.py",
            "core/steering/phrase_constraints.py",
        ],
        "evidence": (
            "Both services use root context; no .dockerignore exists; API COPY core/ recursively "
            "includes 11 untracked runtime modules not present in the target Git tree."
        ),
        "reproduction": (
            "git ls-files --others --exclude-standard, then compare results to Dockerfile COPY "
            "sources and compose build context"
        ),
        "required_remediation": (
            "Provide an isolated exact-target build context/worktree that contains no protected "
            "untracked production files. Do not delete, reset, clean, or silently ignore user files."
        ),
    }
]
write_json("findings.json", findings)

summary = {
    "status": "BLOCKED",
    "requested_mode": "runtime-adversary",
    "actual_mode": "runtime-adversary",
    "branch_or_worktree": "main; dirty protected worktree; clean index",
    "starting_commit": TARGET,
    "resulting_commit": "no commit created",
    "production_files_changed": [],
    "tests_created": [],
    "diagnostic_artifacts_created": True,
    "builds_executed": 0,
    "containers_recreated": 0,
    "tests_executed": 0,
    "runtime_claims_proven": [],
    "runtime_claims_not_proven": "all NRAM-001 through NRAM-024 runtime claims",
    "blocker": "RVR-001",
}
write_json("summary.json", summary)

final_status = subprocess.check_output(
    ["git", "status", "--porcelain=v2", "--branch", "--untracked-files=all"]
)
(RUN / "final-git-status.txt").write_bytes(final_status)

anticipated = sorted(
    {
        path.relative_to(RUN).as_posix()
        for path in RUN.rglob("*")
        if path.is_file()
    }
    | {"created-files.txt", "manifest.json"}
)
(RUN / "created-files.txt").write_text("\n".join(anticipated) + "\n", encoding="utf-8")

files = []
for path in sorted(RUN.rglob("*")):
    if not path.is_file() or path.name == "manifest.json":
        continue
    content = path.read_bytes()
    files.append(
        {
            "path": path.relative_to(RUN).as_posix(),
            "byte_domain": "raw artifact file bytes",
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    )
manifest = {
    "schema": "nram-runtime-reverification-blocked-manifest-v1",
    "target_commit": TARGET,
    "root": RUN.as_posix(),
    "file_count_excluding_manifest": len(files),
    "self_hash_policy": (
        "manifest.json is excluded to avoid impossible recursive self-hashing; its SHA-256 is "
        "reported in the final completion record"
    ),
    "files": files,
}
write_json("manifest.json", manifest)

manifest_bytes = (RUN / "manifest.json").read_bytes()
print(
    json.dumps(
        {
            "status": "BLOCKED",
            "artifact_root": RUN.as_posix(),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "manifest_bytes": len(manifest_bytes),
            "files_listed_excluding_manifest": len(files),
            "issue_results": {
                "NOT_RETESTED": 17,
                "BLOCKED": 6,
                "UNCONFIRMED": 1,
            },
        },
        indent=2,
    )
)
