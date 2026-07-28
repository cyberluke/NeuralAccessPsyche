from __future__ import annotations

import copy
import http.client
import json
import math
import pathlib
import subprocess
import sys
import time
import urllib.request
from typing import Any

HOST = "127.0.0.1"
PORT = 8000
TOKEN = "dev-nram-key"
OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=False)
RAW = OUT / "cases"
RAW.mkdir()


def base(request_id: str) -> dict[str, Any]:
    return {
        "model": "nram-qwen3-14b-awq",
        "messages": [{"role": "user", "content": "Reply with one word."}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "seed": 424242,
        "tools": [],
        "nram": {
            "request_id": request_id,
            "include_telemetry": True,
            "telemetry_max_steps": 1,
            "telemetry_top_k": 1,
            "prompt_steering_enabled": False,
            "profile_logit_steering_enabled": False,
        },
    }


def set_path(body: dict[str, Any], path: tuple[Any, ...], value: float) -> None:
    current: Any = body
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value


def call(case_id: str, body: dict[str, Any]) -> dict[str, Any]:
    wire = json.dumps(body, allow_nan=True, separators=(",", ":")).encode()
    conn = http.client.HTTPConnection(HOST, PORT, timeout=180)
    started = time.time()
    try:
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=wire,
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json",
                "X-Runtime-Adversary-Case": case_id,
            },
        )
        response = conn.getresponse()
        raw = response.read()
        status = response.status
        headers = dict(response.getheaders())
        exception = None
    except Exception as exc:
        raw = b""
        status = None
        headers = {}
        exception = f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()
    text = raw.decode(errors="replace")
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    record = {
        "case_id": case_id,
        "elapsed_ms": (time.time() - started) * 1000,
        "request": body,
        "wire_utf8": wire.decode(),
        "response": {"status": status, "headers": headers, "body_utf8": text, "json": parsed},
        "exception": exception,
    }
    (RAW / f"{case_id}.json").write_text(
        json.dumps(record, indent=2, allow_nan=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return record


def restart_api(batch: int) -> None:
    subprocess.run(["docker", "restart", "nram-api"], check=True, capture_output=True, text=True)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://{HOST}:{PORT}/ready", timeout=3) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"API did not recover after validation batch {batch}")


public = {
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "frequency_penalty": (-2.0, 2.0),
    "presence_penalty": (-2.0, 2.0),
}
top = {
    "intensity": (0.0, 1.0),
    "vocabulary_logit_vector_clip": (0.0, 10.0),
    "entropy_kp": (0.0, 2.0),
    "entropy_ki": (0.0, 0.5),
    "entropy_kd": (0.0, 1.0),
    "entropy_integral_limit": (0.0, 100.0),
    "entropy_scale_min": (0.05, 1.0),
    "entropy_scale_max": (1.0, 5.0),
    "concept_strength": (-5.0, 5.0),
    "visionary_intensity": (0.0, 1.0),
    "contrarian_force": (0.0, 1.0),
    "product_obsession": (0.0, 1.0),
    "human_focus": (0.0, 1.0),
    "rhetorical_compression": (0.0, 1.0),
    "associative_distance": (0.0, 1.0),
    "theatricality": (0.0, 1.0),
    "emotional_voltage": (0.0, 1.0),
    "coherence_floor": (0.0, 1.0),
    "novelty_target": (0.0, 1.0),
    "repetition_penalty": (0.0, 2.0),
    "corporate_jargon_penalty": (0.0, 2.5),
}
entropy_fields = ["extraction", "questioning", "divergence", "synthesis", "formulation"]
phenomenon_fields = [
    "overlap", "forgetting", "looping", "associative_jump", "synesthesia",
    "dissolution", "echo", "tangent", "insight",
]

specs: list[tuple[str, tuple[Any, ...], tuple[float, float], dict[str, Any]]] = []
for field, bounds in public.items():
    specs.append((f"public-{field}", (field,), bounds, {}))
for field, bounds in top.items():
    specs.append((f"nram-{field}", ("nram", field), bounds, {}))
for field in entropy_fields:
    specs.append((f"entropy-phase-{field}", ("nram", "entropy_phase_targets", field), (0.0, 20.0), {"entropy_phase_targets": {}}))
for field in phenomenon_fields:
    specs.append((f"phenomenon-{field}", ("nram", "phenomenon_weights", field), (0.0, 1.0), {"phenomenon_weights": {}}))
specs.extend(
    [
        ("soft-injection-bias", ("nram", "soft_token_injections", 0, "bias"), (-5.0, 5.0), {"soft_token_injections": [{"token_ids": [9702], "bias": 0.0, "start_step": 0, "end_step": 1}]}),
        ("vector-coefficient", ("nram", "vocabulary_logit_vectors", 0, "coefficient"), (-4.0, 4.0), {"vocabulary_logit_vectors": [{"vector_id": "final-vector", "coefficient": 0.0, "normalize": False, "entries": [{"token_id": 9702, "weight": 1.0}]}]}),
        ("vector-entry-weight", ("nram", "vocabulary_logit_vectors", 0, "entries", 0, "weight"), (-10.0, 10.0), {"vocabulary_logit_vectors": [{"vector_id": "final-entry", "coefficient": 1.0, "normalize": False, "entries": [{"token_id": 9702, "weight": 0.0}]}]}),
    ]
)

planned: list[dict[str, Any]] = []
for name, path, bounds, setup in specs:
    for label, value in (("nan", math.nan), ("posinf", math.inf), ("neginf", -math.inf)):
        planned.append({"kind": "reject", "name": f"{name}-{label}", "path": path, "value": value, "setup": setup, "expected_status": 422 if name.startswith("public-") else 400})
    for label, value in (("lower", bounds[0]), ("upper", bounds[1])):
        planned.append({"kind": "boundary", "name": f"{name}-{label}", "path": path, "value": value, "setup": setup, "expected_status": 200})

results: list[dict[str, Any]] = []
for index, item in enumerate(planned):
    if index and index % 48 == 0:
        restart_api(index // 48)
    request_id = f"FINAL-VAL-{index:03d}"
    body = base(request_id)
    body["nram"].update(copy.deepcopy(item["setup"]))
    set_path(body, item["path"], item["value"])
    record = call(item["name"], body)
    response_text = record["response"]["body_utf8"].lower()
    parsed = record["response"]["json"]
    code = parsed.get("error", {}).get("code") if isinstance(parsed, dict) else None
    forbidden = ["traceback", "authorization", TOKEN.lower(), "file \"", "/app/"]
    if item["kind"] == "reject":
        raw_lexemes = ["nan"] if math.isnan(item["value"]) else (["-infinity", "-inf"] if item["value"] < 0 else ["infinity", "+inf"])
        no_leak = not any(token in response_text for token in forbidden + raw_lexemes)
        passed = record["response"]["status"] == item["expected_status"] and code == "invalid_request_schema" and no_leak
    else:
        no_leak = not any(token in response_text for token in forbidden)
        passed = record["response"]["status"] == 200 and no_leak
    results.append({
        "index": index,
        "case_id": item["name"],
        "request_id": request_id,
        "kind": item["kind"],
        "path": list(item["path"]),
        "value_repr": repr(item["value"]),
        "expected_status": item["expected_status"],
        "actual_status": record["response"]["status"],
        "error_code": code,
        "no_sensitive_or_raw_value_leak": no_leak,
        "pre_event_pass": passed,
    })
    print(json.dumps(results[-1], sort_keys=True), flush=True)

logs = subprocess.run(["docker", "logs", "nram-sglang"], check=True, capture_output=True, text=True, timeout=60)
events_by_request: dict[str, list[dict[str, Any]]] = {}
for line in (logs.stdout + logs.stderr).splitlines():
    marker = "NRAM_PROCESSOR_EVENT "
    if marker not in line:
        continue
    try:
        event = json.loads(line.split(marker, 1)[1])
    except Exception:
        continue
    events_by_request.setdefault(str(event.get("request_id")), []).append(event)

for row in results:
    events = events_by_request.get(row["request_id"], [])
    row["processor_event_count"] = len(events)
    row["scheduler_request_ids"] = sorted({str(event.get("scheduler_request_id")) for event in events})
    row["passed"] = bool(row["pre_event_pass"] and (len(events) == 0 if row["kind"] == "reject" else len(events) == 1))

summary = {
    "schema": "nram.final.validation.v1",
    "total": len(results),
    "rejections": sum(row["kind"] == "reject" for row in results),
    "boundaries": sum(row["kind"] == "boundary" for row in results),
    "passed": sum(row["passed"] for row in results),
    "failed": sum(not row["passed"] for row in results),
    "zero_event_rejections": sum(row["kind"] == "reject" and row["processor_event_count"] == 0 for row in results),
    "results": results,
}
(OUT / "validation-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: summary[key] for key in ("total", "rejections", "boundaries", "passed", "failed", "zero_event_rejections")}, sort_keys=True))
raise SystemExit(0 if summary["failed"] == 0 else 2)
