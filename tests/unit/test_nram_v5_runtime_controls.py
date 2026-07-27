"""Deterministic unit coverage for the shared live NRAM processor."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nram_sglang.processor import NRAMLogitProcessor


def _params(**overrides):
    values = {
        "positive_token_ids": [],
        "negative_token_ids": [],
        "forbidden_token_ids": [],
        "positive_bias": 0.0,
        "negative_bias": 0.0,
        "repetition_penalty": 0.0,
        "max_tokens": 20,
    }
    values.update(overrides)
    return values


def test_phrase_and_source_masks_use_server_history_only():
    processor = NRAMLogitProcessor()
    request = SimpleNamespace(output_ids=[7, 8], rid="phrase")
    params = _params(
        __req__=request,
        phrase_constraint_config={
            "forbidden_phrase_ids": [[7, 8, 9], [8, 10]],
            "source_ngram_ids": [[1, 7, 8, 11]],
            "source_ngram_size": 4,
        },
    )

    result = processor(torch.zeros((1, 32)), [params])

    assert torch.isneginf(result[0, 9])
    assert torch.isneginf(result[0, 10])  # overlapping suffix completes [8, 10]
    assert result[0, 11] == 0  # source prefix is only two of the required three tokens

    request.output_ids = [7, 6]
    result = processor(torch.zeros((1, 32)), [params])
    assert result[0, 9] == 0  # constituent 7 alone does not complete [7, 8, 9]
    assert result[0, 10] == 0

    request.output_ids = [1, 7, 8]
    result = processor(torch.zeros((1, 32)), [params])
    assert torch.isneginf(result[0, 9])
    assert torch.isneginf(result[0, 11])


@pytest.mark.parametrize("target,direction", [(0.2, -1), (2.5, 1), (4.0, 1)])
def test_pid_entropy_moves_toward_target(target, direction):
    processor = NRAMLogitProcessor()
    request = SimpleNamespace(output_ids=[], rid=f"entropy-{target}")
    logits = torch.tensor([[4.0, 2.0, 1.0, 0.0]])
    before = processor._entropy(logits[0])
    params = _params(
        __req__=request,
        entropy_config={
            "phase_targets": {"extraction": target},
            "kp": 0.4,
            "ki": 0.03,
            "kd": 0.05,
            "scale_min": 0.2,
            "scale_max": 4.0,
        },
    )

    result = processor(logits.clone(), [params])
    after = processor._entropy(result[0])

    assert torch.isfinite(result).all()
    assert (after - before) * direction > 0
    assert abs(target - after) < abs(target - before)


def test_pid_state_is_request_isolated_and_resets():
    processor = NRAMLogitProcessor()
    first = SimpleNamespace(output_ids=[], rid="first")
    second = SimpleNamespace(output_ids=[], rid="second")
    params = _params(
        entropy_config={
            "phase_targets": {"extraction": 2.0},
            "kp": 0.2,
            "ki": 0.1,
            "kd": 0.0,
        }
    )
    logits = torch.tensor([[6.0, 1.0, 0.0, -1.0]])

    processor(logits.clone(), [{**params, "__req__": first}])
    first_integral = first._nram_entropy_pid["integral"]
    processor(logits.clone(), [{**params, "__req__": first}])
    assert first._nram_entropy_pid["integral"] == pytest.approx(first_integral * 2)

    processor(logits.clone(), [{**params, "__req__": second}])
    assert second._nram_entropy_pid["integral"] == pytest.approx(first_integral)


def test_soft_window_and_sparse_vocabulary_vectors_have_local_signed_effects():
    processor = NRAMLogitProcessor()
    request = SimpleNamespace(output_ids=[1, 2], rid="logits")
    params = _params(
        __req__=request,
        soft_injection_config={
            "injections": [
                {"token_ids": [4], "bias": 1.5, "start_step": 2, "end_step": 4},
                {"token_ids": [5], "bias": 9.0, "start_step": 4, "end_step": 6},
            ]
        },
        logit_vector_config={
            "clip": 2.0,
            "vectors": [
                {
                    "vector_id": "signed-local",
                    "coefficient": -2.0,
                    "normalize": False,
                    "entries": [
                        {"token_id": 6, "weight": 0.5},
                        {"token_id": 7, "weight": -0.5},
                    ],
                },
                {
                    "vector_id": "zero-control",
                    "coefficient": 0.0,
                    "entries": [{"token_id": 8, "weight": 1.0}],
                },
            ],
        },
    )

    result = processor(torch.zeros((1, 16)), [params])

    assert result[0, 4] == pytest.approx(1.5)
    assert result[0, 5] == 0
    assert result[0, 6] == pytest.approx(-1.0)
    assert result[0, 7] == pytest.approx(1.0)
    assert result[0, 8] == 0
    assert result[0, 9] == 0


def test_processor_telemetry_contains_direct_deltas_and_is_bounded(capsys):
    processor = NRAMLogitProcessor()
    request = SimpleNamespace(output_ids=[], rid="scheduler-rid")
    params = _params(
        __req__=request,
        positive_token_ids=[3],
        positive_bias=1.0,
        request_id="public-id",
        config_hash="abc123",
        telemetry_enabled=True,
        telemetry_max_steps=1,
        telemetry_top_k=3,
        forced_token_id=5,
        forced_token_enabled=True,
    )

    result = processor(torch.arange(10, dtype=torch.float32).reshape(1, 10), [params])
    processor(torch.zeros((1, 10)), [params])

    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert len(lines) == 1
    event = json.loads(lines[0].split("NRAM_PROCESSOR_EVENT ", 1)[1])
    assert event["request_id"] == "public-id"
    assert event["config_hash"] == "abc123"
    assert event["invocation_count"] == 1
    assert event["forced_token_id"] == 5
    assert event["mask_count"] == 9
    assert event["post_top_k"][0]["token_id"] == 5
    assert result.argmax(dim=-1).item() == 5
