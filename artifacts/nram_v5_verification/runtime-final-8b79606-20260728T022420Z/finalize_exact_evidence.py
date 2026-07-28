from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


log_path = ROOT / "raw-sglang-final.log"
events = []
for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
    marker = "NRAM_PROCESSOR_EVENT "
    if marker not in line:
        continue
    prefix, raw = line.split(marker, 1)
    event = json.loads(raw)
    event["container_log_prefix"] = prefix.strip()
    events.append(event)

with (ROOT / "runtime-events.jsonl").open("w", encoding="utf-8", newline="\n") as output:
    for event in events:
        output.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

columns = [
    "container_log_prefix", "request_id", "scheduler_request_id", "config_hash",
    "invocation_count", "generated_tokens_before_sample", "requested_forced_token_id",
    "forced_token_id", "mask_count", "hard_injection_blocked_by_mask",
    "changed_count", "soft_injection_count", "vector_count", "entropy_pid_present",
]
with (ROOT / "token-level-runtime-events.csv").open("w", encoding="utf-8", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for event in events:
        writer.writerow(
            {
                "container_log_prefix": event.get("container_log_prefix"),
                "request_id": event.get("request_id"),
                "scheduler_request_id": event.get("scheduler_request_id"),
                "config_hash": event.get("config_hash"),
                "invocation_count": event.get("invocation_count"),
                "generated_tokens_before_sample": event.get("generated_tokens_before_sample"),
                "requested_forced_token_id": event.get("requested_forced_token_id"),
                "forced_token_id": event.get("forced_token_id"),
                "mask_count": event.get("mask_count"),
                "hard_injection_blocked_by_mask": event.get("hard_injection_blocked_by_mask"),
                "changed_count": len(event.get("changed") or []),
                "soft_injection_count": len(event.get("soft_injections") or []),
                "vector_count": len(event.get("vocabulary_logit_vectors") or []),
                "entropy_pid_present": event.get("entropy_pid") is not None,
            }
        )

forced_records = load(ROOT / "raw-http" / "forced-current-image" / "forced-summary.json")
forced_rows = []
for record in forced_records:
    request_id = record["request"]["body"]["nram"]["request_id"]
    matching = [event for event in events if event.get("request_id") == request_id]
    response = record["response"].get("json") or {}
    forced_rows.append(
        {
            "case_id": record["case_id"],
            "request_id": request_id,
            "status": record["response"]["status"],
            "content": (((response.get("choices") or [{}])[0].get("message") or {}).get("content")),
            "correlation": response.get("nram_correlation"),
            "processor_event_count": len(matching),
            "forced_token_ids": [event.get("forced_token_id") for event in matching],
            "post_top_token_ids": [
                (event.get("post_top_k") or [{}])[0].get("token_id") for event in matching
            ],
            "scheduler_request_ids": sorted(
                {str(event.get("scheduler_request_id")) for event in matching}
            ),
        }
    )
forced_summary = {
    "rows": forced_rows,
    "enabled_pass": all(
        row["status"] == 200
        and row["processor_event_count"] == 1
        and row["forced_token_ids"] == [9702]
        and row["post_top_token_ids"] == [9702]
        for row in forced_rows if "enabled" in row["case_id"]
    ),
    "disabled_pass": all(
        row["status"] == 200
        and row["processor_event_count"] == 1
        and row["forced_token_ids"] == [None]
        for row in forced_rows if "disabled" in row["case_id"]
    ),
}
(ROOT / "forced-pair-summary.json").write_text(
    json.dumps(forced_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

validation = load(ROOT / "raw-http" / "final-validation-current-image" / "validation-summary.json")
cancellation = load(ROOT / "raw-http" / "final-cancellation-current-image" / "cancellation-summary.json")
shutil.copy2(
    ROOT / "raw-http" / "final-validation-current-image" / "validation-summary.json",
    ROOT / "validation-summary.json",
)
shutil.copy2(
    ROOT / "raw-http" / "final-cancellation-current-image" / "cancellation-summary.json",
    ROOT / "cancellation-summary.json",
)

junit_paths = [
    ROOT / "full-test-results-current-image.xml",
    ROOT / "gpu-test-results-current-image.xml",
    ROOT / "adversarial-live-current-image.xml",
    ROOT / "streamlit-api-current-image.xml",
]
junit = []
for path in junit_paths:
    xml = ET.parse(path).getroot()
    suites = [xml] if xml.tag == "testsuite" else list(xml.findall("testsuite"))
    counts = {
        name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    junit.append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **counts})
(ROOT / "test-results-summary.json").write_text(
    json.dumps({"junit": junit, "total_records_in_manifest": sum(x["tests"] for x in junit)}, indent=2) + "\n",
    encoding="utf-8",
)

issue_rows = [
    ("NRAM-001", "VERIFIED_FIXED", "Current 343-test/full adversarial model identity gates pass; bounded alias campaign HTTP 200."),
    ("NRAM-002", "VERIFIED_FIXED", "Current unchanged live adversarial compare test passes; forced compare bounded proof retained."),
    ("NRAM-003", "VERIFIED_FIXED", "Final v2 manifest authenticates exact target/tree/current image IDs and 401 JUnit records."),
    ("NRAM-004", "VERIFIED_FIXED", "Raw-socket ordinary/persona/non-stream drain at 3 events with exactly-once abort; deadline 504."),
    ("NRAM-005", "FAILED_VERIFICATION", "Non-finite and prior invalid matrices reject pre-engine, but three declared boundary controls fail late with 500/502."),
    ("NRAM-006", "VERIFIED_FIXED", "Ordinary/persona SSE close drains; normal streams and persona contract gates pass."),
    ("NRAM-007", "VERIFIED_FIXED", "Current GPU telemetry correlation, duplicate caller-ID regressions, and event retention gates pass."),
    ("NRAM-008", "VERIFIED_FIXED", "Current full suite and one-field config-hash campaign pass."),
    ("NRAM-009", "VERIFIED_FIXED", "Session propagation/reset gates pass; bounded live session artifacts retained."),
    ("NRAM-010", "VERIFIED_FIXED", "MoE validation/bounds unit gates and bounded live route pass."),
    ("NRAM-011", "VERIFIED_FIXED", "Current standalone GPU structural suite passes 5/5 structural cases."),
    ("NRAM-012", "VERIFIED_FIXED", "Unsupported advanced-feature contract matrix fails loudly; capability gate passes."),
    ("NRAM-013", "VERIFIED_FIXED", "Configured token 200; wrong/random-long/missing tokens 401."),
    ("NRAM-014", "VERIFIED_FIXED_WITH_OPERATIONAL_FINDING", "Final v2 provenance passes after fail-loud dangling-image-ID detection and cold recreation."),
    ("NRAM-015", "FAILED_VERIFICATION", "Finite/non-finite and GPU numerical gates pass, but accepted public/state bounds disagree for two NRAM penalties and top_p."),
    ("NRAM-016", "VERIFIED_FIXED", "Current GPU PID/capsule/vector/soft-window controls pass."),
    ("NRAM-017", "VERIFIED_FIXED", "Current 40-test Streamlit/API image-route integration and full suite pass."),
    ("NRAM-018", "BLOCKED", "No DExperts artifacts/runtime; fail-loud contract retained."),
    ("NRAM-019", "BLOCKED", "No representation hidden-state runtime/artifacts; fail-loud contract retained."),
    ("NRAM-020", "BLOCKED", "No semantic/evidence closed-loop runtime/artifacts; fail-loud contract retained."),
    ("NRAM-021", "BLOCKED", "No branch-and-tournament runtime; fail-loud contract retained."),
    ("NRAM-022", "BLOCKED", "No full preregistered causal/statistical study."),
    ("NRAM-023", "BLOCKED", "Original 269/273 four-failure artifact remains unavailable and was not fabricated."),
    ("NRAM-024", "UNCONFIRMED", "No exit 137 occurred in this bounded run; historical cause not reproduced."),
    ("FINAL-001", "FAILED", "207/210 pass; 126/126 non-finite zero-event rejects; three exact-boundary integration failures."),
    ("FINAL-002", "PASSED", "All raw-socket cancellation, deadline, retry-isolation, and normal-completion controls pass."),
    ("FINAL-003", "PASSED_WITH_OPERATIONAL_FINDING", "Final v2 self-check passes; collector correctly failed loud on uninspectable live image ID."),
]
issue_table = {issue_id: {"status": status, "evidence": evidence} for issue_id, status, evidence in issue_rows}
(ROOT / "issue-verification-table.json").write_text(
    json.dumps(issue_table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

findings = [
    {
        "id": "RVR2-001",
        "severity": "HIGH",
        "title": "Declared public top_p lower boundary is rejected only by upstream SGLang",
        "evidence": "top_p=0.0 passes public schema, reaches upstream, and returns HTTP 502; SGLang requires (0,1].",
        "paths": ["raw-http/final-validation-current-image/cases/public-top_p-lower.json"],
    },
    {
        "id": "RVR2-002",
        "severity": "HIGH",
        "title": "Public NRAM penalty maxima disagree with resolved NRAMState bounds",
        "evidence": "repetition_penalty=2.0 and corporate_jargon_penalty=2.5 pass request schema then return HTTP 500 from NRAMState max=1 validation.",
        "paths": [
            "raw-http/final-validation-current-image/cases/nram-repetition_penalty-upper.json",
            "raw-http/final-validation-current-image/cases/nram-corporate_jargon_penalty-upper.json",
        ],
    },
    {
        "id": "RVR2-003",
        "severity": "MEDIUM",
        "title": "Compose rebuild can leave running containers on uninspectable dangling image IDs",
        "evidence": "Initial v2 collector failed loud because docker image inspect could not resolve the running pre-build API image ID; authenticated tag recreation restored provenance.",
        "paths": ["collector-v2.stderr.txt", "collector-image-id-drift.txt"],
    },
]
(ROOT / "findings.json").write_text(json.dumps(findings, indent=2) + "\n", encoding="utf-8")

collector_root = ROOT / "v2-collector-final"
shutil.copy2(collector_root / "manifest.json", ROOT / "manifest-v2.json")
shutil.copy2(collector_root / "environment.json", ROOT / "environment.json")
shutil.copy2(collector_root / "test-node-list.txt", ROOT / "tracked-test-node-list.txt")

commands = load(ROOT / "command-results-input.json")
with (ROOT / "commands-executed.txt").open("w", encoding="utf-8", newline="\n") as output:
    output.write("Commands are evidence-normalized descriptions; exact invocations remain in terminal artifacts and v2 command-results.json.\n")
    for index, item in enumerate(commands, 1):
        output.write(f"{index}. exit={item['exit_code']} :: {item['command']}\n")

summary = {
    "status": "FAILED",
    "requested_mode": "runtime-adversary",
    "actual_mode": "runtime-adversary",
    "target_commit": "8b796066ec6fed3f3260e9aea1e1ab19489e6ae8",
    "target_tree": "18829ce184fc05427187aaf0a9a201d26b82c27e",
    "spec_commit": "fb3690d7884bdbcaef0e82608776065e19332990",
    "spec_git_blob_sha256": "9bbd2e93a2750ab1d1094e823487f862b080512025128247e8d7d00f27675506",
    "validation": {key: validation[key] for key in ("total", "rejections", "boundaries", "passed", "failed", "zero_event_rejections")},
    "cancellation_passed": cancellation["passed"],
    "forced_pair": {key: forced_summary[key] for key in ("enabled_pass", "disabled_pass")},
    "junit": junit,
    "runtime_event_count": len(events),
    "new_findings": [finding["id"] for finding in findings],
    "scientific_verdict": "PARTIALLY FUNCTIONAL",
}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
