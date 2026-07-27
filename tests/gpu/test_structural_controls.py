"""Live token-level structural-control proofs through the public API."""
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


def _tokenize(text: str) -> list[int]:
    code = (
        "import json; from transformers import AutoTokenizer; "
        "t=AutoTokenizer.from_pretrained('/models', local_files_only=True); "
        f"print(json.dumps(t.encode({text!r}, add_special_tokens=False)))"
    )
    result = subprocess.run(
        ["docker", "exec", "nram-api", "python", "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _events(request_id: str) -> list[dict]:
    result = subprocess.run(
        ["docker", "logs", "nram-sglang"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    found = []
    for line in (result.stdout + result.stderr).splitlines():
        if "NRAM_PROCESSOR_EVENT " not in line:
            continue
        event = json.loads(line.split("NRAM_PROCESSOR_EVENT ", 1)[1])
        if event.get("request_id") == request_id:
            found.append(event)
    return found


async def _generate(request_id: str, schedule: list[int], response_format=None, **nram_overrides):
    nram = {
        "enabled": True,
        "profile": "normal",
        "request_id": request_id,
        "include_telemetry": True,
        "telemetry_max_steps": len(schedule) + 2,
        "hard_token_schedule": schedule,
    }
    nram.update(nram_overrides)
    body = {
        "model": "nram-qwen3-14b-awq",
        "messages": [{"role": "user", "content": "Continue deterministically."}],
        "max_tokens": len(schedule),
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 314159,
        "tools": [],
        "nram": nram,
    }
    if response_format is not None:
        body["response_format"] = response_format
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            API_URL,
            json=body,
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("phrase", [" forbidden horizon", " zakázaný obzor"])
async def test_bilingual_overlapping_phrase_completion_is_masked(phrase):
    token_ids = _tokenize(phrase)
    assert len(token_ids) >= 2
    run = uuid.uuid4().hex

    disabled = await _generate(f"phrase-{run}-disabled", token_ids)
    enabled_id = f"phrase-{run}-enabled"
    enabled = await _generate(
        enabled_id,
        token_ids,
        forbidden_phrases=[phrase, phrase.strip()],
    )

    assert disabled["choices"][0]["message"]["content"].strip() == phrase.strip()
    assert enabled["choices"][0]["message"]["content"].strip() != phrase.strip()
    events = _events(enabled_id)
    assert len(events) == len(token_ids)
    final = events[-1]
    assert token_ids[-1] in final["masked_token_ids"]
    assert final["requested_forced_token_id"] == token_ids[-1]
    assert final["forced_token_id"] is None
    assert final["hard_injection_blocked_by_mask"] is True


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gpu
async def test_exact_source_eight_token_continuation_is_masked():
    source = " A precise source continuation crosses eight distinct token boundaries safely."
    source_ids = _tokenize(source)
    assert len(source_ids) >= 8
    schedule = source_ids[:8]
    run = uuid.uuid4().hex

    disabled = await _generate(f"source-{run}-disabled", schedule)
    enabled_id = f"source-{run}-enabled"
    enabled = await _generate(
        enabled_id,
        schedule,
        source_text=source,
        source_ngram_size=8,
    )

    assert disabled["choices"][0]["message"]["content"] != ""
    assert enabled["choices"][0]["message"]["content"] != disabled["choices"][0]["message"]["content"]
    deadline = time.monotonic() + 5
    events = []
    while time.monotonic() < deadline and not events:
        events = _events(enabled_id)
        if not events:
            time.sleep(0.2)
    final = events[-1]
    assert schedule[-1] in final["masked_token_ids"]
    assert final["requested_forced_token_id"] == schedule[-1]
    assert final["hard_injection_blocked_by_mask"] is True


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gpu
async def test_phrase_mask_composes_with_live_json_grammar():
    prefix = _tokenize('{"value":"')
    phrase_text = "forbidden horizon"
    phrase = _tokenize(phrase_text)
    suffix = _tokenize('"}')
    schedule = prefix + phrase + suffix
    request_id = f"grammar-{uuid.uuid4().hex}"
    response = await _generate(
        request_id,
        schedule,
        forbidden_phrases=[phrase_text],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "value_object",
                "schema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        },
    )
    content = response["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    assert isinstance(parsed["value"], str)
    assert phrase_text not in parsed["value"]
    completion_step = len(prefix) + len(phrase) - 1
    event = _events(request_id)[completion_step]
    assert phrase[-1] in event["masked_token_ids"]
    assert event["hard_injection_blocked_by_mask"] is True


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gpu
async def test_streamed_steps_use_same_request_history():
    phrase_text = " forbidden horizon"
    schedule = _tokenize(phrase_text)
    request_id = f"stream-{uuid.uuid4().hex}"
    body = {
        "model": "nram-qwen3-14b-awq",
        "messages": [{"role": "user", "content": "Continue."}],
        "max_tokens": len(schedule),
        "temperature": 0.0,
        "seed": 314159,
        "stream": True,
        "tools": [],
        "nram": {
            "enabled": True,
            "profile": "normal",
            "request_id": request_id,
            "include_telemetry": True,
            "telemetry_max_steps": len(schedule),
            "hard_token_schedule": schedule,
            "forbidden_phrases": [phrase_text],
        },
    }
    chunks = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            API_URL,
            json=body,
            headers={"Authorization": f"Bearer {API_KEY}"},
        ) as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunks.append(line)
    assert chunks
    events = _events(request_id)
    assert [event["generated_tokens_before_sample"] for event in events] == list(range(len(schedule)))
    assert schedule[-1] in events[-1]["masked_token_ids"]
