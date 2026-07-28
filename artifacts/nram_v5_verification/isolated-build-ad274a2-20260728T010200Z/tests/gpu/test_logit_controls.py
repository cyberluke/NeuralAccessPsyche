"""Live GPU proofs for bounded entropy and vocabulary-logit controls."""
from __future__ import annotations

import json
import os
import subprocess
import uuid

import httpx
import pytest


API_URL = os.getenv("NRAM_TEST_API_URL", "http://127.0.0.1:8000/v1/chat/completions")
API_KEY = os.getenv("NRAM_TEST_API_KEY", "dev-nram-key")


def _token_id(text: str) -> int:
    code = (
        "import json; from transformers import AutoTokenizer; "
        "t=AutoTokenizer.from_pretrained('/models', local_files_only=True); "
        f"ids=t.encode({text!r}, add_special_tokens=False); print(json.dumps(ids))"
    )
    output = subprocess.run(
        ["docker", "exec", "nram-api", "python", "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip().splitlines()[-1]
    ids = json.loads(output)
    assert ids
    return ids[0]


def _events(request_id: str) -> list[dict]:
    output = subprocess.run(
        ["docker", "logs", "nram-sglang"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    events = []
    for line in (output.stdout + output.stderr).splitlines():
        if "NRAM_PROCESSOR_EVENT " in line:
            event = json.loads(line.split("NRAM_PROCESSOR_EVENT ", 1)[1])
            if event.get("request_id") == request_id:
                events.append(event)
    return events


async def _request(request_id: str, nram: dict, max_tokens: int = 1):
    body = {
        "model": "nram-qwen3-14b-awq",
        "messages": [{"role": "user", "content": "Name one property of light."}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 1.0,
        "seed": 424242,
        "tools": [],
        "nram": {
            "enabled": True,
            "profile": "normal",
            "request_id": request_id,
            "include_telemetry": True,
            "telemetry_max_steps": max_tokens,
            **nram,
        },
    }
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            API_URL,
            json=body,
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
    assert response.status_code == 200, response.text
    events = _events(request_id)
    assert len(events) == max_tokens
    return response.json(), events


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gpu
async def test_live_pid_low_medium_high_and_request_reset():
    run = uuid.uuid4().hex
    results = {}
    for label, target in (("low", 1.0), ("medium", 4.0), ("high", 8.0)):
        _, events = await _request(
            f"entropy-{run}-{label}",
            {
                "entropy_control": True,
                "entropy_phase_targets": {
                    name: target
                    for name in ("extraction", "questioning", "divergence", "synthesis", "formulation")
                },
                "entropy_kp": 0.2,
                "entropy_ki": 0.01,
                "entropy_kd": 0.02,
                "entropy_scale_min": 0.3,
                "entropy_scale_max": 3.0,
            },
        )
        pid = events[0]["entropy_pid"]
        assert pid is not None
        assert pid["entropy_before"] >= 0
        assert pid["entropy_after"] >= 0
        assert abs(target - pid["entropy_after"]) <= abs(target - pid["entropy_before"])
        assert all(pid[name] is not None for name in ("p", "i", "d", "applied_scale"))
        results[label] = pid

    assert results["low"]["entropy_after"] < results["medium"]["entropy_after"]
    assert results["medium"]["entropy_after"] < results["high"]["entropy_after"]

    # Same fixed seed/config in a fresh request starts from fresh integral state.
    _, repeated = await _request(
        f"entropy-{run}-medium-reset",
        {
            "entropy_control": True,
            "entropy_phase_targets": {name: 4.0 for name in ("extraction", "questioning", "divergence", "synthesis", "formulation")},
            "entropy_kp": 0.2,
            "entropy_ki": 0.01,
            "entropy_kd": 0.02,
            "entropy_scale_min": 0.3,
            "entropy_scale_max": 3.0,
        },
    )
    assert repeated[0]["entropy_pid"]["integral"] == pytest.approx(results["medium"]["integral"], abs=1e-5)

    # Fixed sampler temperature without the servo is a distinct, explicit control.
    _, fixed = await _request(f"entropy-{run}-fixed-temperature", {"entropy_control": False})
    assert fixed[0]["entropy_pid"] is None


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gpu
async def test_live_capsule_soft_window_and_sparse_vector_deltas():
    run = uuid.uuid4().hex
    concept_id = _token_id(" xylophone")
    soft_id = _token_id(" zephyr")
    vector_id = _token_id(" quasar")
    _, events = await _request(
        f"logits-{run}",
        {
            "concepts": [
                {
                    "concept_id": "bilingual-acoustic",
                    "en_tokens": [" xylophone"],
                    "cs_tokens": [" xylofon"],
                    "synonyms": [" musical instrument"],
                    "max_uses": 1,
                }
            ],
            "concept_strength": 0.8,
            "soft_token_injections": [
                {"token_ids": [soft_id], "bias": 1.25, "start_step": 0, "end_step": 1},
                {"token_ids": [soft_id + 1], "bias": 1.25, "start_step": 2, "end_step": 3},
            ],
            "vocabulary_logit_vectors": [
                {
                    "vector_id": "signed-vocabulary-control",
                    "coefficient": -2.0,
                    "normalize": False,
                    "entries": [{"token_id": vector_id, "weight": 0.5}],
                },
                {
                    "vector_id": "zero-control",
                    "coefficient": 0.0,
                    "entries": [{"token_id": vector_id + 1, "weight": 1.0}],
                },
            ],
            "vocabulary_logit_vector_clip": 2.0,
        },
        max_tokens=2,
    )
    opening, ramped = events
    opening_changed = {item["token_id"]: item for item in opening["changed"]}
    ramped_changed = {item["token_id"]: item for item in ramped["changed"]}
    assert ramped_changed[concept_id]["delta"] > 0
    assert ramped_changed[concept_id]["rank_after"] <= ramped_changed[concept_id]["rank_before"]
    assert opening_changed[soft_id]["delta"] == pytest.approx(1.25, abs=1e-4)
    assert opening_changed[vector_id]["delta"] == pytest.approx(-1.0, abs=1e-4)
    assert soft_id + 1 not in opening_changed
    assert vector_id + 1 not in opening_changed
    assert opening["soft_injections"][0]["start_step"] == 0
    assert ramped["soft_injections"] == []
    assert opening["vocabulary_logit_vectors"][0]["vector_id"] == "signed-vocabulary-control"
    assert opening["vocabulary_logit_vectors"][1]["deltas"] == []
