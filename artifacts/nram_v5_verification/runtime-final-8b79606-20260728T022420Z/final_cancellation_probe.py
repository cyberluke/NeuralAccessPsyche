from __future__ import annotations

import http.client
import json
import pathlib
import socket
import struct
import subprocess
import sys
import time
from typing import Any

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=False)
API_KEY = "dev-nram-key"
HOST = "127.0.0.1"
PORT = 8000


def logs(container: str) -> str:
    result = subprocess.run(
        ["docker", "logs", "--timestamps", container],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout + result.stderr


def processor_events(request_id: str) -> list[dict[str, Any]]:
    found = []
    marker = "NRAM_PROCESSOR_EVENT "
    for line in logs("nram-sglang").splitlines():
        if marker not in line:
            continue
        try:
            event = json.loads(line.split(marker, 1)[1])
        except Exception:
            continue
        if event.get("request_id") == request_id:
            found.append(event)
    return found


def abort_access_lines() -> list[str]:
    return [line for line in logs("nram-sglang").splitlines() if "/abort_request" in line]


def payload(model: str, request_id: str, *, stream: bool, max_tokens: int = 64) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Continue deterministically."}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "seed": 424242,
        "stream": stream,
        "tools": [],
        "nram": {
            "request_id": request_id,
            "include_telemetry": True,
            "telemetry_max_steps": 128,
            "telemetry_top_k": 1,
            "prompt_steering_enabled": False,
            "profile_logit_steering_enabled": False,
            "hard_token_schedule": [9702] * max_tokens,
        },
    }


def request_bytes(body: dict[str, Any], case_id: str) -> bytes:
    wire = json.dumps(body, separators=(",", ":")).encode()
    headers = (
        f"POST /v1/chat/completions HTTP/1.1\r\n"
        f"Host: {HOST}:{PORT}\r\n"
        f"Authorization: Bearer {API_KEY}\r\n"
        f"Content-Type: application/json\r\n"
        f"Accept: text/event-stream, application/json\r\n"
        f"X-Runtime-Adversary-Case: {case_id}\r\n"
        f"Content-Length: {len(wire)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode()
    return headers + wire


def rst_close(sock: socket.socket) -> None:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("hh", 1, 0))
    sock.close()


def snapshots(request_id: str, closed: float, abort_before: int) -> dict[str, Any]:
    rows = []
    for target in (0.0, 0.5, 3.5):
        delay = closed + target - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        events = processor_events(request_id)
        rows.append(
            {
                "seconds_after_close": target,
                "event_count": len(events),
                "max_invocation_count": max((int(e.get("invocation_count", 0)) for e in events), default=0),
                "scheduler_request_ids": sorted({str(e.get("scheduler_request_id")) for e in events}),
            }
        )
    abort_after_lines = abort_access_lines()
    return {
        "samples": rows,
        "abort_access_count_before": abort_before,
        "abort_access_count_after": len(abort_after_lines),
        "abort_access_delta": len(abort_after_lines) - abort_before,
        "new_abort_access_lines": abort_after_lines[abort_before:],
    }


def stream_disconnect(case_id: str, model: str, request_id: str) -> dict[str, Any]:
    before = len(abort_access_lines())
    body = payload(model, request_id, stream=True)
    sock = socket.create_connection((HOST, PORT), timeout=30)
    sock.settimeout(30)
    started = time.monotonic()
    sock.sendall(request_bytes(body, case_id))
    received = bytearray()
    while b"data:" not in received:
        chunk = sock.recv(4096)
        if not chunk:
            break
        received.extend(chunk)
        if len(received) > 1024 * 1024:
            raise RuntimeError("first SSE data line not found within 1 MiB")
    first_data_offset = received.find(b"data:")
    first_data_line = b""
    if first_data_offset >= 0:
        first_data_line = bytes(received[first_data_offset:]).splitlines()[0]
    rst_close(sock)
    closed = time.monotonic()
    observed = snapshots(request_id, closed, before)
    result = {
        "case_id": case_id,
        "model": model,
        "request_id": request_id,
        "max_tokens": 64,
        "forced_schedule_length": 64,
        "elapsed_ms_to_close": (closed - started) * 1000,
        "raw_received_hex": bytes(received).hex(),
        "raw_received_utf8": bytes(received).decode(errors="replace"),
        "first_data_line_utf8": first_data_line.decode(errors="replace"),
        "closed_immediately_after_first_data_line": first_data_offset >= 0,
        **observed,
    }
    result["passed"] = bool(
        result["closed_immediately_after_first_data_line"]
        and result["samples"][-1]["event_count"] < 64
        and result["samples"][1]["event_count"] == result["samples"][-1]["event_count"]
        and result["abort_access_delta"] == 1
    )
    (OUT / f"{case_id}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def complete(body: dict[str, Any], case_id: str) -> dict[str, Any]:
    wire = json.dumps(body, separators=(",", ":")).encode()
    conn = http.client.HTTPConnection(HOST, PORT, timeout=180)
    started = time.monotonic()
    conn.request(
        "POST", "/v1/chat/completions", body=wire,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "X-Runtime-Adversary-Case": case_id,
        },
    )
    response = conn.getresponse()
    raw = response.read()
    conn.close()
    return {
        "status": response.status,
        "headers": dict(response.getheaders()),
        "body_utf8": raw.decode(errors="replace"),
        "elapsed_ms": (time.monotonic() - started) * 1000,
    }


def nonstream_disconnect_and_retry() -> dict[str, Any]:
    case_id = "nonstream-raw-close-50ms"
    request_id = "FINAL-CANCEL-NONSTREAM"
    before = len(abort_access_lines())
    body = payload("nram-qwen3-14b-awq", request_id, stream=False)
    sock = socket.create_connection((HOST, PORT), timeout=30)
    started = time.monotonic()
    sock.sendall(request_bytes(body, case_id))
    time.sleep(0.05)
    rst_close(sock)
    closed = time.monotonic()

    retry_id = "FINAL-CANCEL-RETRY"
    retry_started = time.monotonic()
    retry_response = complete(payload("nram-qwen3-14b-awq", retry_id, stream=False, max_tokens=1), "immediate-retry")
    retry_finished = time.monotonic()
    observed = snapshots(request_id, closed, before)
    old_events = processor_events(request_id)
    retry_events = processor_events(retry_id)
    old_ids = sorted({str(e.get("scheduler_request_id")) for e in old_events})
    retry_ids = sorted({str(e.get("scheduler_request_id")) for e in retry_events})
    result = {
        "case_id": case_id,
        "request_id": request_id,
        "max_tokens": 64,
        "forced_schedule_length": 64,
        "elapsed_ms_to_close": (closed - started) * 1000,
        "closed_at_target_50ms": 35 <= (closed - started) * 1000 <= 250,
        **observed,
        "retry": {
            "request_id": retry_id,
            "started_seconds_after_close": retry_started - closed,
            "finished_seconds_after_close": retry_finished - closed,
            "response": retry_response,
            "event_count": len(retry_events),
            "invocation_counts": [e.get("invocation_count") for e in retry_events],
            "scheduler_request_ids": retry_ids,
            "distinct_scheduler_id": bool(old_ids and retry_ids and set(old_ids).isdisjoint(retry_ids)),
            "invocation_reset": bool(retry_events and retry_events[0].get("invocation_count") == 1),
        },
    }
    result["passed"] = bool(
        result["closed_at_target_50ms"]
        and result["samples"][-1]["event_count"] < 64
        and result["samples"][1]["event_count"] == result["samples"][-1]["event_count"]
        and result["abort_access_delta"] == 1
        and retry_response["status"] == 200
        and result["retry"]["distinct_scheduler_id"]
        and result["retry"]["invocation_reset"]
    )
    (OUT / f"{case_id}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


ordinary = stream_disconnect("ordinary-sse-raw-close", "nram-qwen3-14b-awq", "FINAL-CANCEL-ORDINARY")
persona = stream_disconnect("persona-sse-raw-close", "persona-peak", "FINAL-CANCEL-PERSONA")
nonstream = nonstream_disconnect_and_retry()

normal_before = len(abort_access_lines())
normal_id = "FINAL-CANCEL-NORMAL"
normal_response = complete(payload("nram-qwen3-14b-awq", normal_id, stream=False, max_tokens=1), "normal-completion")
time.sleep(0.5)
normal_events = processor_events(normal_id)
normal_after_lines = abort_access_lines()
normal = {
    "case_id": "normal-completion",
    "request_id": normal_id,
    "response": normal_response,
    "event_count": len(normal_events),
    "scheduler_request_ids": sorted({str(e.get("scheduler_request_id")) for e in normal_events}),
    "abort_access_count_before": normal_before,
    "abort_access_count_after": len(normal_after_lines),
    "abort_access_delta": len(normal_after_lines) - normal_before,
}
normal["passed"] = bool(normal_response["status"] == 200 and len(normal_events) == 1 and normal["abort_access_delta"] == 0)
(OUT / "normal-completion.json").write_text(json.dumps(normal, indent=2) + "\n", encoding="utf-8")

summary = {
    "schema": "nram.final.cancellation.v1",
    "ordinary": ordinary,
    "persona": persona,
    "nonstream": nonstream,
    "normal": normal,
    "passed": all(item["passed"] for item in (ordinary, persona, nonstream, normal)),
}
(OUT / "cancellation-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"ordinary": ordinary["passed"], "persona": persona["passed"], "nonstream": nonstream["passed"], "normal": normal["passed"], "passed": summary["passed"]}, sort_keys=True))
raise SystemExit(0 if summary["passed"] else 2)
