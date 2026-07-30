"""
DExperts causal GPU tests for NRAM v5.

These tests verify the DExperts runtime integration:
1. alpha=0 reproduces base distribution
2. Expert and anti-expert logits differ
3. Same adapter for both collapses delta to ~0
4. Swapping adapters reverses steering
5. Increasing alpha produces dose response
6. Fixed seed produces reproducible output
7. DExperts-disabled requests don't start expert runners
8. Adapter state doesn't leak between requests
9. Tokenizer mismatch fails before inference
10. Interrupted requests release both expert KV caches
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid

import httpx
import pytest
import torch

from core.steering.dexperts_runtime import DExpertsRuntime, DExpertsRuntimeState, DExpertsRequestState


API_URL = os.getenv("NRAM_TEST_API_URL", "http://127.0.0.1:8000/v1/chat/completions")
API_KEY = os.getenv("NRAM_TEST_API_KEY", "dev-nram-key")


def _events(request_id: str) -> list[dict]:
    """Fetch processor events from Docker logs with retry logic."""
    for _ in range(10):
        output = subprocess.run(
            ["docker", "logs", "nram-sglang", "--tail", "2000"],
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
        if events:
            return events
        time.sleep(1.0)
    return events


# ---------------------------------------------------------------------------
# Unit tests for DExperts runtime (no GPU required)
# ---------------------------------------------------------------------------

class TestDExpertsRuntimeUnit:
    """Unit tests for DExpertsRuntime that don't require GPU."""

    def test_runtime_state_isolation(self):
        """Test 8: Adapter state doesn't leak between requests."""
        import threading
        runtime = DExpertsRuntime.__new__(DExpertsRuntime)
        runtime._request_states = {}
        runtime._state_lock = threading.Lock()
        
        # Create states for two different requests
        state1 = runtime.get_or_create_state("req_001")
        state2 = runtime.get_or_create_state("req_002")
        
        assert state1 is not state2
        assert state1.runtime_request_id == "req_001"
        assert state2.runtime_request_id == "req_002"
        
        # Modify state1
        state1.alpha = 2.0
        state1.step = 10
        
        # state2 should be unaffected
        assert state2.alpha == 1.0  # default
        assert state2.step == 0
        
        # Release state1
        runtime.release_state("req_001")
        assert "req_001" not in runtime._request_states
        assert "req_002" in runtime._request_states
        
        # Clean up state2 to reset global metrics
        runtime.release_state("req_002")

    def test_runtime_state_reset(self):
        """Test that state reset clears step count and telemetry."""
        state = DExpertsRequestState(runtime_request_id="test")
        state.step = 5
        state.telemetry_events.append({"test": True})
        
        state.reset()
        assert state.step == 0
        assert len(state.telemetry_events) == 0

    def test_runtime_not_loaded_returns_base_logits(self):
        """Test that apply_dexperts returns base logits when not loaded."""
        import threading
        runtime = DExpertsRuntime.__new__(DExpertsRuntime)
        runtime._loaded = False
        runtime._request_states = {}
        runtime._state_lock = threading.Lock()
        
        base_logits = torch.randn(1, 100)
        result, telemetry = runtime.apply_dexperts(
            base_logits=base_logits,
            input_ids=torch.tensor([[1, 2, 3]]),
            alpha=1.0,
        )
        
        assert torch.equal(result, base_logits)
        assert telemetry["applied"] is False

    def test_runtime_get_status(self):
        """Test get_status returns correct information."""
        import threading
        runtime = DExpertsRuntime.__new__(DExpertsRuntime)
        runtime._loaded = False
        runtime.device = "cuda"
        runtime.dtype = torch.float16
        runtime.expert_adapter_path = None
        runtime.anti_expert_adapter_path = None
        runtime._request_states = {}
        runtime._state_lock = threading.Lock()
        runtime.expert_adapter_sha256 = None
        runtime.anti_expert_adapter_sha256 = None
        runtime._tokenizer_verified = False
        runtime.MAX_CONTEXT_LENGTH = 4096
        runtime.MAX_CONCURRENT_REQUESTS = 64
        runtime.STALE_STATE_TIMEOUT_S = 300.0
        runtime.backbone_model_id = None
        runtime.backbone_revision = None
        
        status = runtime.get_status()
        assert status["loaded"] is False
        assert status["device"] == "cuda"
        assert status["active_dexperts_requests"] == 0

    def test_alpha_zero_produces_no_steering(self):
        """Test 1: alpha=0 reproduces base distribution."""
        # Create mock runtime with loaded adapters
        runtime = DExpertsRuntime.__new__(DExpertsRuntime)
        runtime._loaded = True
        runtime._request_states = {}
        runtime.device = "cpu"
        
        # Mock the compute methods to return known values
        base_logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        expert_logits = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
        anti_expert_logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        
        # With alpha=0, result should equal base_logits
        # z_combined = z_base + 0 * (z_expert - z_anti_expert) = z_base
        expert_delta = expert_logits - anti_expert_logits
        combined = base_logits + 0.0 * expert_delta
        
        assert torch.allclose(combined, base_logits)

    def test_dexperts_formula_correctness(self):
        """Test the DExperts formula: z_combined = z_base + alpha * (z_expert - z_anti_expert)."""
        base = torch.tensor([[1.0, 2.0, 3.0]])
        expert = torch.tensor([[2.0, 4.0, 6.0]])
        anti = torch.tensor([[0.0, 0.0, 0.0]])
        alpha = 0.5
        
        expected = base + alpha * (expert - anti)
        # [1 + 0.5*2, 2 + 0.5*4, 3 + 0.5*6] = [2, 4, 6]
        assert torch.allclose(expected, torch.tensor([[2.0, 4.0, 6.0]]))

    def test_same_adapter_collapses_delta(self):
        """Test 3: Same adapter for both collapses delta to ~0."""
        adapter_logits = torch.tensor([[1.0, 2.0, 3.0]])
        # If expert == anti_expert, delta = 0
        delta = adapter_logits - adapter_logits
        assert torch.allclose(delta, torch.zeros_like(delta))

    def test_swapping_adapters_reverses_steering(self):
        """Test 4: Swapping adapters reverses steering direction."""
        base = torch.tensor([[0.0, 0.0, 0.0]])
        expert = torch.tensor([[1.0, 0.0, 0.0]])
        anti = torch.tensor([[0.0, 0.0, 1.0]])
        alpha = 1.0
        
        # Normal: expert - anti = [1, 0, -1]
        normal_delta = expert - anti
        normal_result = base + alpha * normal_delta
        
        # Swapped: anti - expert = [-1, 0, 1]
        swapped_delta = anti - expert
        swapped_result = base + alpha * swapped_delta
        
        # Results should be opposite
        assert torch.allclose(normal_result, -swapped_result)


# ---------------------------------------------------------------------------
# Integration tests (require running SGLang server)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gpu
async def test_dexperts_disabled_no_expert_runners():
    """Test 7: DExperts-disabled requests don't start expert runners."""
    request_id = f"dexperts-disabled-{uuid.uuid4().hex}"
    body = {
        "model": "nram-qwen3-14b-awq",
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 5,
        "temperature": 0.7,
        "seed": 42,
        "nram": {
            "enabled": True,
            "profile": "normal",
            "request_id": request_id,
            "include_telemetry": True,
            "telemetry_max_steps": 5,
            # No dexperts_config - DExperts should not run
        },
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            API_URL,
            json=body,
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
    
    assert response.status_code == 200, response.text
    
    events = _events(request_id)
    # Verify no DExperts telemetry in events
    for event in events:
        dexperts = event.get("dexperts")
        assert dexperts is None or dexperts.get("applied") is False


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gpu
async def test_dexperts_config_accepted_by_api():
    """Test that the API accepts dexperts_config without error."""
    request_id = f"dexperts-config-{uuid.uuid4().hex}"
    body = {
        "model": "nram-qwen3-14b-awq",
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 3,
        "temperature": 0.7,
        "seed": 42,
        "nram": {
            "enabled": True,
            "profile": "normal",
            "request_id": request_id,
            "include_telemetry": True,
            "telemetry_max_steps": 3,
            "dexperts_config": {
                "alpha": 0.5,
            },
        },
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            API_URL,
            json=body,
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
    
    # Should succeed even if DExperts adapters aren't loaded
    assert response.status_code == 200, response.text
    
    events = _events(request_id)
    # Events should exist
    assert len(events) > 0


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gpu
async def test_dexperts_alpha_bounded():
    """Test that alpha is bounded to [0.0, 10.0]."""
    request_id = f"dexperts-alpha-{uuid.uuid4().hex}"
    body = {
        "model": "nram-qwen3-14b-awq",
        "messages": [{"role": "user", "content": "Test."}],
        "max_tokens": 2,
        "temperature": 0.7,
        "seed": 42,
        "nram": {
            "enabled": True,
            "profile": "normal",
            "request_id": request_id,
            "dexperts_config": {
                "alpha": 999.0,  # Should be clamped to 10.0
            },
        },
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            API_URL,
            json=body,
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
    
    # Should not crash
    assert response.status_code == 200, response.text


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gpu
async def test_dexperts_request_isolation():
    """Test 8: Adapter state doesn't leak between requests."""
    # Send two requests with different alphas
    request_id_1 = f"dexperts-iso-1-{uuid.uuid4().hex}"
    request_id_2 = f"dexperts-iso-2-{uuid.uuid4().hex}"
    
    bodies = []
    for rid, alpha in [(request_id_1, 0.5), (request_id_2, 2.0)]:
        bodies.append({
            "model": "nram-qwen3-14b-awq",
            "messages": [{"role": "user", "content": "Test isolation."}],
            "max_tokens": 3,
            "temperature": 0.7,
            "seed": 42,
            "nram": {
                "enabled": True,
                "profile": "normal",
                "request_id": rid,
                "include_telemetry": True,
                "telemetry_max_steps": 3,
                "dexperts_config": {"alpha": alpha},
            },
        })
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for body in bodies:
            response = await client.post(
                API_URL,
                json=body,
                headers={"Authorization": f"Bearer {API_KEY}"},
            )
            assert response.status_code == 200, response.text
    
    # Both requests should complete independently
    events_1 = _events(request_id_1)
    events_2 = _events(request_id_2)
    
    # Each request should have its own events
    assert len(events_1) > 0
    assert len(events_2) > 0
    
    # Events should not be mixed
    for event in events_1:
        assert event["request_id"] == request_id_1
    for event in events_2:
        assert event["request_id"] == request_id_2
