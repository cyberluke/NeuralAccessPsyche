"""Live public-API proofs against the pinned Qwen3/SGLang/RTX runtime."""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid

import httpx
import pytest


API_URL = os.getenv("NRAM_TEST_API_URL", "http://127.0.0.1:8000/v1/chat/completions")
API_KEY = os.getenv("NRAM_TEST_API_KEY", "dev-nram-key")
FORCED_TOKEN_ID = 11064  # live Qwen tokenizer decodes this token as " proof"


def _processor_events(request_id: str) -> list[dict]:
    completed = subprocess.run(
        ["docker", "logs", "nram-sglang"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    events = []
    for line in (completed.stdout + completed.stderr).splitlines():
        marker = "NRAM_PROCESSOR_EVENT "
        if marker not in line:
            continue
        event = json.loads(line.split(marker, 1)[1])
        if event.get("request_id") == request_id:
            events.append(event)
    return events


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gpu
async def test_forced_token_public_api_same_seed_control():
    """Force one real token and correlate public output with processor logs."""
    request_id = f"gpu-forced-{uuid.uuid4().hex}"
    common = {
        "model": "nram-qwen3-14b-awq",
        "messages": [{"role": "user", "content": "Say one word."}],
        "max_tokens": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": 271828,
        "tools": [],
        "nram": {
            "enabled": True,
            "profile": "normal",
            "request_id": request_id,
            "include_telemetry": True,
            "telemetry_max_steps": 1,
            "forced_token_id": FORCED_TOKEN_ID,
        },
    }

    headers = {"Authorization": f"Bearer {API_KEY}"}
    async with httpx.AsyncClient(timeout=120.0) as client:
        disabled_body = json.loads(json.dumps(common))
        disabled_body["nram"]["request_id"] += "-disabled"
        disabled_body["nram"]["forced_token_enabled"] = False
        disabled_response = await client.post(API_URL, json=disabled_body, headers=headers)

        enabled_body = json.loads(json.dumps(common))
        enabled_body["nram"]["forced_token_enabled"] = True
        enabled_response = await client.post(API_URL, json=enabled_body, headers=headers)

    assert disabled_response.status_code == 200, disabled_response.text
    assert enabled_response.status_code == 200, enabled_response.text
    disabled = disabled_response.json()
    enabled = enabled_response.json()
    disabled_text = disabled["choices"][0]["message"]["content"]
    enabled_text = enabled["choices"][0]["message"]["content"]
    assert enabled_text.strip() == "proof"
    assert disabled_text != enabled_text

    # Give Docker's log reader a bounded moment to flush subprocess stdout.
    deadline = time.monotonic() + 10
    events = []
    while time.monotonic() < deadline and not events:
        events = _processor_events(request_id)
        if not events:
            time.sleep(0.25)
    assert len(events) == 1
    event = events[0]
    assert event["schema"] == "nram.processor.step.v1"
    assert event["invocation_count"] == 1
    assert event["forced_token_id"] == FORCED_TOKEN_ID
    assert event["mask_count"] == 151935
    assert event["post_top_k"][0]["token_id"] == FORCED_TOKEN_ID
    assert event["config_hash"] == enabled["nram_correlation"]["config_hash"]
    assert enabled["nram_correlation"]["request_id"] == request_id

    disabled_events = _processor_events(request_id + "-disabled")
    assert len(disabled_events) == 1
    assert disabled_events[0]["forced_token_id"] is None
    assert disabled_events[0]["mask_count"] < 151935


@pytest.mark.integration
@pytest.mark.gpu
def test_api_and_server_use_same_qwen_tokenizer_artifact():
    """Verify token 11064 and tokenizer files in both live containers."""
    command = (
        "from transformers import AutoTokenizer; "
        "t=AutoTokenizer.from_pretrained('/models', local_files_only=True); "
        "print(len(t)); print(repr(t.decode([11064])))"
    )
    outputs = []
    for container in ("nram-api", "nram-sglang"):
        completed = subprocess.run(
            ["docker", "exec", container, "python", "-c", command],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        outputs.append(completed.stdout.strip().splitlines()[-2:])
    assert outputs[0] == outputs[1]
    assert outputs[0] == ["151669", "' proof'"]
