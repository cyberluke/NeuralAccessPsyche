from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


root = Path(sys.argv[1]).resolve()


def junit(path: Path) -> dict[str, int]:
    document = ET.parse(path).getroot()
    suites = [document] if document.tag == "testsuite" else list(document.findall(".//testsuite"))
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


sglang_log = (root / "sglang-full.log").read_text(encoding="utf-8", errors="replace")
events = []
for line in sglang_log.splitlines():
    if "NRAM_PROCESSOR_EVENT " not in line:
        continue
    try:
        events.append(json.loads(line.split("NRAM_PROCESSOR_EVENT ", 1)[1]))
    except json.JSONDecodeError:
        pass
with (root / "runtime-events.jsonl").open("w", encoding="utf-8") as target:
    for event in events:
        target.write(json.dumps(event, sort_keys=True) + "\n")

fields = [
    "request_id",
    "scheduler_request_id",
    "config_hash",
    "invocation_count",
    "generated_tokens_before_sample",
    "forced_token_id",
    "requested_forced_token_id",
    "mask_count",
    "masked_token_ids",
]
with (root / "token-level-runtime-events.csv").open("w", newline="", encoding="utf-8") as target:
    writer = csv.DictWriter(target, fieldnames=fields)
    writer.writeheader()
    for event in events:
        writer.writerow(
            {
                key: json.dumps(event.get(key)) if key == "masked_token_ids" else event.get(key)
                for key in fields
            }
        )

validation = json.loads((root / "raw-http/validation/validation-summary.json").read_text())
validation_statuses = {row["case_id"]: row["response"]["status"] for row in validation}
nonfinite_failures = {
    key: validation_statuses[key]
    for key in ("temperature-nan", "temperature-inf", "top-p-nan")
}
unsupported = {
    key: value
    for key, value in validation_statuses.items()
    if key.startswith("unsupported-") or key.startswith("persona-unsupported-")
}

issues = {
    "NRAM-001": ("VERIFIED_FIXED", "Live immutable model list, alias/persona identity, actual base identity, and stale alias rejection passed."),
    "NRAM-002": ("VERIFIED_FIXED", "Live forced compare returned baseline processor_intervened=false and controlled sampled token 9702 with metrics; unsupported compare rejected preflight."),
    "NRAM-003": ("FAILED_VERIFICATION", "Independent exact-target tests are reproducible, but the committed collector and committed builder manifest still identify stale 7aa1285/dirty provenance instead of target ad274a2; collector fields are hard-coded."),
    "NRAM-004": ("FAILED_VERIFICATION", "Ordinary stream disconnect and client-timeout requests each continued through all 64 processor invocations. Server-side 1 ms deadline did call official abort and had zero events; persona disconnect drained 2 extra steps."),
    "NRAM-005": ("FAILED_VERIFICATION", "Malformed/unknown/token/all-mask/unsupported cases rejected, but JSON NaN/Infinity for temperature/top_p returned HTTP 500 instead of pre-generation 4xx."),
    "NRAM-006": ("FAILED_VERIFICATION", "Ordinary/persona SSE media type, identity, correlation, chunks, and [DONE] passed; required ordinary disconnect cancellation failed with 64/64 invocations."),
    "NRAM-007": ("VERIFIED_FIXED", "Non-stream sampled IDs joined SGLang meta_info, explicit target retention unit gate passed, duplicate caller IDs received distinct scheduler IDs/configs, and streaming limitation is explicit."),
    "NRAM-008": ("VERIFIED_FIXED", "Prompt/seed/temperature/top-p/response-format/prompt-steering sensitivity passed; reversed object order preserved the same hash."),
    "NRAM-009": ("VERIFIED_FIXED", "Live peak/intensity/memory snapshot entered generation accounting, request count/events updated, unsupported memory semantics were de-scoped, and reset zeroed state."),
    "NRAM-010": ("VERIFIED_FIXED", "Zero-call unsupported preflight and explicit error/bounds unit gates passed; bounded max_tokens=1 live orchestration returned one-token synthesis and truthful orchestration identity."),
    "NRAM-011": ("VERIFIED_FIXED", "Nine live GPU gates passed bilingual variants, overlap, exact 8-token source boundary, grammar composition, and streamed history."),
    "NRAM-012": ("VERIFIED_FIXED", "Capabilities truthfully list supported controls and blocked advanced stacks; unsupported ordinary/persona/MoE/compare paths fail loudly."),
    "NRAM-013": ("VERIFIED_FIXED", "Configured correct token returned 200; wrong, random long, and missing tokens returned 401; contract fail-closed/helper tests passed."),
    "NRAM-014": ("FAILED_VERIFICATION", "Cold exact-target recreation succeeded and 282-test historical artifact is retained, but committed collector/manifest provenance remains hard-coded to stale 7aa1285 dirty scope rather than the implementation SHA."),
    "NRAM-015": ("FAILED_VERIFICATION", "PID/reset/saturation/all-mask/grammar controls passed, but non-finite public sampling controls surface middleware HTTP 500 rather than stable pre-generation 4xx."),
    "NRAM-016": ("VERIFIED_FIXED", "Live GPU and unit gates passed signed/zero sparse vectors, soft windows, capsule bounds, PID controls, phrase/source masks; claims remain bounded to direct logit effects."),
    "NRAM-017": ("VERIFIED_FIXED", "Exact API image source authentication included evaluation package; advertised image routes, compare, sessions, telemetry, dashboard/Streamlit tests passed."),
    "NRAM-018": ("BLOCKED", "DExperts remains absent, advertised blocked, and dexperts requests fail loudly; no simulation."),
    "NRAM-019": ("BLOCKED", "Representation hooks/vectors/probes/ReFT remain absent and fail loudly; no simulation."),
    "NRAM-020": ("BLOCKED", "Semantic/evidence closed loop remains absent and fail-loud; no simulation."),
    "NRAM-021": ("BLOCKED", "Branch-and-tournament remains absent and fail-loud; persona orchestration is not relabeled as tournament."),
    "NRAM-022": ("BLOCKED", "No full causal/statistical study; bounded mechanism evidence is not converted into scientific validation."),
    "NRAM-023": ("BLOCKED", "The authentic claimed original 269/273 four-failure artifact remains unavailable; retained phase0 is 282 total / 9 failed and is not substituted."),
    "NRAM-024": ("UNCONFIRMED", "No exit 137 recurred during exact cold start and bounded campaign; prior cause remains unattributed."),
}
issue_rows = [
    {"issue": issue, "result": status, "evidence": evidence}
    for issue, (status, evidence) in issues.items()
]
(root / "issue-verification-table.json").write_text(json.dumps(issue_rows, indent=2) + "\n")

findings = [
    {
        "id": "RVR-001",
        "status": "RESOLVED",
        "finding": "Dirty primary Docker context blocker resolved by direct Git-object extraction and per-blob authentication in an isolated context.",
    },
    {
        "id": "RVR-002",
        "status": "OPEN",
        "finding": "temperature NaN, temperature Infinity, and top_p NaN return middleware HTTP 500, not pre-generation 4xx.",
        "evidence": nonfinite_failures,
    },
    {
        "id": "RVR-003",
        "status": "OPEN",
        "finding": "Ordinary stream disconnect and non-stream client timeout did not bound upstream generation: both emitted all 64 processor steps. Explicit server deadline separately passed with official abort.",
    },
    {
        "id": "RVR-004",
        "status": "OPEN",
        "finding": "Committed artifact collector and committed builder manifest retain hard-coded stale 7aa1285/dirty provenance instead of deriving target implementation identity.",
    },
    {
        "id": "RVR-005",
        "status": "TEST_HARNESS_LIMITATION",
        "finding": "Running all HTTP suites without isolated rate-limit windows produced 11 HTTP 429 responses; clean-window numerical/lifecycle reruns passed apart from the preserved cancellation behavior.",
    },
    {
        "id": "RVR-006",
        "status": "OPERATIONAL_ADAPTATION",
        "finding": "Tracked GPU tests hard-code docker exec nram-api/nram-sglang; first isolated-name run had 9 false environmental failures, canonical-name rerun passed 246/246.",
    },
    {
        "id": "RVR-007",
        "status": "OPERATIONAL_ADAPTATION",
        "finding": "Windows tar extraction changed LF bytes and corrupted eight Unicode names; direct git ls-tree/cat-file extraction authenticated all 269 blobs exactly.",
    },
]
(root / "findings.json").write_text(json.dumps(findings, indent=2) + "\n")

test_results = {
    "adversarial_live": junit(root / "adversarial-live-junit.xml"),
    "first_full_noncanonical_names": junit(root / "full-test-results.xml"),
    "full_canonical_names": junit(root / "full-test-results-canonical-names.xml"),
    "gpu": junit(root / "gpu-test-results.xml"),
    "ruff": (root / "ruff.txt").read_text().strip(),
    "mypy": (root / "mypy.txt").read_text().strip(),
    "npm_audit": json.loads((root / "npm-audit.json").read_text())["metadata"]["vulnerabilities"],
    "validation_nonfinite_failures": nonfinite_failures,
    "unsupported_fail_loud_statuses": unsupported,
}
(root / "test-results-summary.json").write_text(json.dumps(test_results, indent=2) + "\n")

summary = {
    "status": "PARTIAL",
    "requested_mode": "runtime-adversary",
    "actual_mode": "runtime-adversary",
    "branch_or_worktree": "main primary worktree plus artifacts/nram_v5_verification/isolated-build-ad274a2-20260728T010200Z",
    "starting_commit": "ad274a2c4b0575cf796659073fb69a962dbb66d0",
    "resulting_commit": "no commit created",
    "target_tree": "cb0457f4f09f00cd6e7c94dca15bd15ddd4fe128",
    "canonical_specification_commit": "fb3690d7884bdbcaef0e82608776065e19332990",
    "canonical_specification_git_blob_sha256": "9bbd2e93a2750ab1d1094e823487f862b080512025128247e8d7d00f27675506",
    "isolated_context_authenticated_blobs": 269,
    "isolated_context_failures": 0,
    "runtime_image_source_authenticated": True,
    "verdict_counts": dict(
        sorted({status: sum(1 for row in issue_rows if row["result"] == status) for status in {row["result"] for row in issue_rows}}.items())
    ),
    "test_results": test_results,
    "processor_event_count": len(events),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

subprocess.run(
    ["git", "status", "--short", "--branch"],
    check=True,
    stdout=(root / "primary-worktree-status-after.txt").open("wb"),
)

excluded = {"manifest.json"}
files = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in excluded)
manifest = {
    "run_id": root.name,
    "target_commit": "ad274a2c4b0575cf796659073fb69a962dbb66d0",
    "target_tree": "cb0457f4f09f00cd6e7c94dca15bd15ddd4fe128",
    "artifact_count": len(files),
    "artifacts": {
        path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in files
    },
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"summary": summary, "artifact_count": len(files)}, indent=2))
