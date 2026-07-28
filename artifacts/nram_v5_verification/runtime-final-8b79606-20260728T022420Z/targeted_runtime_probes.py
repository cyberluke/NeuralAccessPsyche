from __future__ import annotations

import http.client
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


BASE = "http://127.0.0.1:8000"
KEY = "dev-nram-key"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
records: list[dict[str, object]] = []


def request(
    case: str,
    path: str,
    body: dict[str, object] | None = None,
    *,
    token: str | None = KEY,
    method: str | None = None,
    timeout: float = 180,
) -> dict[str, object]:
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers=headers,
        method=method or ("POST" if body is not None else "GET"),
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.status
            response_headers = dict(response.headers.items())
            raw = response.read()
            exception = None
    except urllib.error.HTTPError as exc:
        status = exc.code
        response_headers = dict(exc.headers.items())
        raw = exc.read()
        exception = f"HTTPError: {exc}"
    except Exception as exc:
        status = None
        response_headers = {}
        raw = b""
        exception = f"{type(exc).__name__}: {exc}"
    text = raw.decode("utf-8", "replace")
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    record: dict[str, object] = {
        "case": case,
        "elapsed_seconds": time.time() - started,
        "request": {
            "path": path,
            "method": req.method,
            "authorization": "missing" if token is None else "Bearer <redacted>",
            "body": body,
        },
        "response": {
            "status": status,
            "headers": response_headers,
            "body_utf8": text,
            "json": parsed,
        },
        "exception": exception,
    }
    (OUT / f"{case}.json").write_text(json.dumps(record, indent=2) + "\n")
    records.append(record)
    print(case, status, exception, flush=True)
    return record


def completion(request_id: str, *, model: str = "nram-qwen3-14b-awq") -> dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly one short word."}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1,
        "seed": 424242,
        "stream": False,
        "tools": [],
        "nram": {
            "request_id": request_id,
            "include_telemetry": True,
            "telemetry_max_steps": 8,
            "telemetry_top_k": 20,
            "prompt_steering_enabled": False,
            "profile_logit_steering_enabled": False,
        },
    }


request("auth-correct", "/v1/models")
request("auth-wrong", "/v1/models", token="definitely-wrong")
request("auth-random-long", "/v1/models", token="random-long-token-1234567890")
request("auth-missing", "/v1/models", token=None)
request("capabilities", "/v1/nram/capabilities")

compare_body: dict[str, object] = {
    "prompt": "Reply with exactly one short word.",
    "model": "nram-qwen3-14b-awq",
    "baseline_model": "qwen3-14b-awq-baseline",
    "seed": 424242,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 1,
    "nram": {
        "request_id": "RVR-COMPARE-FORCED",
        "prompt_steering_enabled": False,
        "profile_logit_steering_enabled": False,
        "forced_token_enabled": True,
        "forced_token_id": 9702,
        "telemetry_max_steps": 8,
        "telemetry_top_k": 20,
    },
}
request("compare-forced-pair", "/v1/nram/compare", compare_body)

created = request(
    "session-create",
    "/v1/nram/sessions",
    {
        "profile": "peak",
        "intensity": 0.73,
        "seed": 424242,
        "memory": {"enabled": True, "max_recent_tokens": 3, "decay": 0.25},
    },
)
created_json = created["response"]["json"]  # type: ignore[index]
if isinstance(created_json, dict) and isinstance(created_json.get("id"), str):
    session_id = created_json["id"]
    session_payload = completion("RVR-SESSION-GENERATE")
    session_payload["nram"]["session_id"] = session_id  # type: ignore[index]
    session_payload["nram"]["forced_token_enabled"] = True  # type: ignore[index]
    session_payload["nram"]["forced_token_id"] = 9702  # type: ignore[index]
    request("session-generation", "/v1/chat/completions", session_payload)
    request("session-state-after", f"/v1/nram/sessions/{session_id}/state")
    request("session-events-after", f"/v1/nram/sessions/{session_id}/events")
    request("session-reset", f"/v1/nram/sessions/{session_id}/reset", {})
    request("session-state-reset", f"/v1/nram/sessions/{session_id}/state")

hash_a = completion("RVR-HASH-ORDER")
hash_a["nram"].update(  # type: ignore[union-attr]
    {"forced_token_enabled": True, "forced_token_id": 9702}
)
hash_b = json.loads(json.dumps(hash_a))
hash_b["nram"] = dict(reversed(list(hash_b["nram"].items())))
request("hash-order-a", "/v1/chat/completions", hash_a)
request("hash-order-b", "/v1/chat/completions", hash_b)

duplicate_a = completion("RVR-DUPLICATE-CALLER-ID")
duplicate_a["nram"].update(  # type: ignore[union-attr]
    {"forced_token_enabled": True, "forced_token_id": 9702}
)
duplicate_b = completion("RVR-DUPLICATE-CALLER-ID")
duplicate_b["nram"].update(  # type: ignore[union-attr]
    {"forced_token_enabled": True, "forced_token_id": 12095}
)
request("duplicate-caller-a", "/v1/chat/completions", duplicate_a)
request("duplicate-caller-b", "/v1/chat/completions", duplicate_b)

request(
    "moe-bounded-live",
    "/v1/chat/completions",
    completion("RVR-MOE-BOUNDED", model="nram-moe-orchestrator"),
    timeout=300,
)

# Persona disconnect after the first SSE line; the server must issue an upstream
# abort and a subsequent request must not inherit state.
persona = completion("RVR-PERSONA-DISCONNECT", model="persona-peak")
persona.update({"stream": True, "max_tokens": 32})
wire = json.dumps(persona, separators=(",", ":")).encode()
connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=60)
connection.request(
    "POST",
    "/v1/chat/completions",
    body=wire,
    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
)
response = connection.getresponse()
first_line = response.readline()
connection.close()
persona_disconnect = {
    "case": "persona-disconnect-after-first-line",
    "status": response.status,
    "headers": dict(response.getheaders()),
    "first_line_utf8": first_line.decode("utf-8", "replace"),
    "closed_after_first_line": True,
    "request": persona,
}
(OUT / "persona-disconnect-after-first-line.json").write_text(
    json.dumps(persona_disconnect, indent=2) + "\n"
)
time.sleep(3)
request("persona-disconnect-retry", "/v1/chat/completions", completion("RVR-PERSONA-RETRY"))

(OUT / "targeted-summary.json").write_text(json.dumps(records, indent=2) + "\n")
