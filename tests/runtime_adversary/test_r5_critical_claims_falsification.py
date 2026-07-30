"""
NRAM v5 R5 Critical Claims Falsification Tests

Target: Commit f905d3c on branch nram-v5-implementation

This test suite attempts to falsify the 8 critical R5 claims:
1. Separate KV Caches Actually Work
2. Canonical Decoding Order
3. State Machine Correctness
4. Tokenizer Compatibility Gate
5. Context Bounds Enforcement
6. Two Scientific Regimes
7. Thread Safety
8. Metrics Collection

Test Execution:
    pytest tests/runtime_adversary/test_r5_critical_claims_falsification.py -v --tb=short
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# CLAIM 1: Separate KV Caches Actually Work
# ============================================================================

class TestClaim1_SeparateKVCaches:
    """Verify expert and anti-expert have truly separate KV cache objects."""

    def test_kv_caches_are_separate_objects(self):
        """Verify expert_past_key_values and anti_expert_past_key_values are not aliases."""
        from core.steering.dexperts_runtime import DExpertsRequestState

        state = DExpertsRequestState(runtime_request_id="test-req")
        
        # Initially both are None
        assert state.expert_past_key_values is None
        assert state.anti_expert_past_key_values is None
        
        # Assign different mock objects
        mock_expert_kv = MagicMock()
        mock_anti_expert_kv = MagicMock()
        
        state.expert_past_key_values = mock_expert_kv
        state.anti_expert_past_key_values = mock_anti_expert_kv
        
        # Verify they are separate objects (not aliases)
        assert state.expert_past_key_values is not state.anti_expert_past_key_values
        assert state.expert_past_key_values is mock_expert_kv
        assert state.anti_expert_past_key_values is mock_anti_expert_kv

    def test_kv_caches_updated_independently(self):
        """Verify KV caches are updated independently in apply_dexperts."""
        from core.steering.dexperts_runtime import DExpertsRuntime

        # Create runtime with mocked components
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        runtime = DExpertsRuntime(mock_model, mock_tokenizer, "cpu")
        runtime._loaded = True

        # Mock adapters to return different KV caches
        mock_expert_kv = {"layer_0": torch.randn(2, 1, 16, 64)}
        mock_anti_expert_kv = {"layer_0": torch.randn(2, 1, 16, 64)}
        
        mock_expert_output = MagicMock()
        mock_expert_output.logits = torch.randn(1, 1, 1000)
        mock_expert_output.past_key_values = mock_expert_kv
        
        mock_anti_expert_output = MagicMock()
        mock_anti_expert_output.logits = torch.randn(1, 1, 1000)
        mock_anti_expert_output.past_key_values = mock_anti_expert_kv
        
        runtime.expert_adapter = MagicMock(return_value=mock_expert_output)
        runtime.anti_expert_adapter = MagicMock(return_value=mock_anti_expert_output)

        # Apply DExperts
        base_logits = torch.randn(1, 1000)
        input_ids = torch.tensor([[1, 2, 3]])
        
        combined, telemetry = runtime.apply_dexperts(
            base_logits=base_logits,
            input_ids=input_ids,
            alpha=1.0,
            request_id="test-kv-isolation",
        )

        # Get state
        state = runtime._request_states["test-kv-isolation"]
        
        # Verify both KV caches were updated
        assert state.expert_past_key_values is not None
        assert state.anti_expert_past_key_values is not None
        
        # Verify they are separate objects
        assert state.expert_past_key_values is not state.anti_expert_past_key_values
        
        # Verify they contain the expected mock objects
        assert state.expert_past_key_values == mock_expert_kv
        assert state.anti_expert_past_key_values == mock_anti_expert_kv
        
        # Clean up
        runtime.release_state("test-kv-isolation")

    def test_positions_remain_synchronized(self):
        """Verify expert_position == anti_expert_position at every step."""
        from core.steering.dexperts_runtime import DExpertsRuntime

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        runtime = DExpertsRuntime(mock_model, mock_tokenizer, "cpu")
        runtime._loaded = True

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_output.past_key_values = None
        
        runtime.expert_adapter = MagicMock(return_value=mock_output)
        runtime.anti_expert_adapter = MagicMock(return_value=mock_output)

        # Apply DExperts multiple times
        for step in range(5):
            base_logits = torch.randn(1, 1000)
            input_ids = torch.tensor([[1, 2, 3]])
            
            combined, telemetry = runtime.apply_dexperts(
                base_logits=base_logits,
                input_ids=input_ids,
                alpha=1.0,
                request_id="test-pos-sync",
                position=step,
            )

            state = runtime._request_states["test-pos-sync"]
            
            # Verify positions are synchronized
            assert state.expert_position == state.anti_expert_position, \
                f"Step {step}: expert_position={state.expert_position} != anti_expert_position={state.anti_expert_position}"
            assert state.expert_position == step
        
        # Clean up
        runtime.release_state("test-pos-sync")


# ============================================================================
# CLAIM 2: Canonical Decoding Order
# ============================================================================

class TestClaim2_CanonicalDecodingOrder:
    """Verify base support truncation occurs BEFORE expert perturbation."""

    def test_base_logits_captured_before_steering(self):
        """Verify unmodified base logits are captured before any steering."""
        from nram_sglang.processor import NRAMLogitProcessor

        # Create processor with mocked DExperts runtime
        processor = NRAMLogitProcessor()
        mock_runtime = MagicMock()
        
        # Track what base_logits are passed to apply_dexperts
        captured_base_logits = []
        
        def capture_apply_dexperts(base_logits, **kwargs):
            captured_base_logits.append(base_logits.clone())
            return base_logits.clone(), {"applied": True}
        
        mock_runtime.apply_dexperts.side_effect = capture_apply_dexperts
        mock_runtime._loaded = True
        processor._dexperts_runtime = mock_runtime

        # Create logits and custom params
        original_logits = torch.randn(1, 1000)
        mock_request = MagicMock()
        mock_request.input_ids = torch.tensor([[1, 2, 3]])
        mock_request.attention_mask = torch.tensor([[1, 1, 1]])
        
        custom_param_list = [{
            "request_id": "test-order",
            "dexperts_config": {"alpha": 1.0, "filter_k": 0, "filter_p": 1.0},
            "__req__": mock_request,
        }]

        # Call processor
        result = processor(original_logits.clone(), custom_param_list)

        # Verify apply_dexperts was called with unmodified base logits
        assert len(captured_base_logits) > 0, "apply_dexperts was not called"
        
        # The captured base logits should match the original (before any modification)
        assert torch.allclose(captured_base_logits[0], original_logits[0]), \
            "Base logits passed to apply_dexperts were already modified"

    def test_base_support_computed_from_unmodified_logits(self):
        """Verify base support mask is computed from unmodified base logits."""
        from nram_sglang.processor import NRAMLogitProcessor

        processor = NRAMLogitProcessor()
        
        # Mock the _compute_base_support method to track what it receives
        original_compute_base_support = processor._compute_base_support
        captured_logits = []
        
        def tracked_compute_base_support(logits, **kwargs):
            captured_logits.append(logits.clone())
            return original_compute_base_support(logits, **kwargs)
        
        processor._compute_base_support = tracked_compute_base_support
        
        # Mock DExperts runtime
        mock_runtime = MagicMock()
        mock_runtime._loaded = True
        mock_runtime.apply_dexperts.return_value = (
            torch.randn(1000),
            {"applied": True}
        )
        processor._dexperts_runtime = mock_runtime

        # Create logits with known top-k structure
        original_logits = torch.zeros(1, 1000)
        original_logits[0, :10] = 10.0  # Top 10 tokens have high logits
        
        mock_request = MagicMock()
        mock_request.input_ids = torch.tensor([[1, 2, 3]])
        mock_request.attention_mask = torch.tensor([[1, 1, 1]])
        
        custom_param_list = [{
            "request_id": "test-base-support",
            "dexperts_config": {"alpha": 1.0, "filter_k": 10, "filter_p": 1.0},
            "__req__": mock_request,
        }]

        # Call processor
        result = processor(original_logits.clone(), custom_param_list)

        # Verify _compute_base_support was called
        assert len(captured_logits) > 0, "_compute_base_support was not called"
        
        # The logits passed to _compute_base_support should be the unmodified base logits
        # (not the combined logits after DExperts)
        assert torch.allclose(captured_logits[0], original_logits[0]), \
            "Base support computed from modified logits instead of unmodified base logits"


# ============================================================================
# CLAIM 3: State Machine Correctness
# ============================================================================

class TestClaim3_StateMachineCorrectness:
    """Verify DExpertsRequestState has proper lifecycle management."""

    def test_state_created_on_request_start(self):
        """Verify state is created when request starts."""
        from core.steering.dexperts_runtime import DExpertsRuntime

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        runtime = DExpertsRuntime(mock_model, mock_tokenizer, "cpu")
        runtime._loaded = True

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_output.past_key_values = None
        
        runtime.expert_adapter = MagicMock(return_value=mock_output)
        runtime.anti_expert_adapter = MagicMock(return_value=mock_output)

        # Initially no states
        assert len(runtime._request_states) == 0

        # Apply DExperts (should create state)
        base_logits = torch.randn(1, 1000)
        input_ids = torch.tensor([[1, 2, 3]])
        
        combined, telemetry = runtime.apply_dexperts(
            base_logits=base_logits,
            input_ids=input_ids,
            alpha=1.0,
            request_id="test-state-create",
        )

        # Verify state was created
        assert "test-state-create" in runtime._request_states
        state = runtime._request_states["test-state-create"]
        assert state.runtime_request_id == "test-state-create"
        assert state.created_at > 0
        
        # Clean up
        runtime.release_state("test-state-create")

    def test_state_removed_on_completion(self):
        """Verify state is removed on completion."""
        from core.steering.dexperts_runtime import DExpertsRuntime

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        runtime = DExpertsRuntime(mock_model, mock_tokenizer, "cpu")
        runtime._loaded = True

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_output.past_key_values = None
        
        runtime.expert_adapter = MagicMock(return_value=mock_output)
        runtime.anti_expert_adapter = MagicMock(return_value=mock_output)

        # Create state
        base_logits = torch.randn(1, 1000)
        input_ids = torch.tensor([[1, 2, 3]])
        
        runtime.apply_dexperts(
            base_logits=base_logits,
            input_ids=input_ids,
            alpha=1.0,
            request_id="test-state-release",
        )

        assert "test-state-release" in runtime._request_states

        # Release state
        runtime.release_state("test-state-release")

        # Verify state was removed
        assert "test-state-release" not in runtime._request_states

    def test_state_removed_on_cancellation(self):
        """Verify state is removed on cancellation/timeout."""
        from core.steering.dexperts_runtime import DExpertsRuntime

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        runtime = DExpertsRuntime(mock_model, mock_tokenizer, "cpu")
        runtime._loaded = True

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_output.past_key_values = None
        
        runtime.expert_adapter = MagicMock(return_value=mock_output)
        runtime.anti_expert_adapter = MagicMock(return_value=mock_output)

        # Create state
        base_logits = torch.randn(1, 1000)
        input_ids = torch.tensor([[1, 2, 3]])
        
        runtime.apply_dexperts(
            base_logits=base_logits,
            input_ids=input_ids,
            alpha=1.0,
            request_id="test-state-cancel",
        )

        assert "test-state-cancel" in runtime._request_states

        # Simulate cancellation by calling release_state
        runtime.release_state("test-state-cancel")

        # Verify state was removed
        assert "test-state-cancel" not in runtime._request_states

    def test_reused_request_pool_index_no_inheritance(self):
        """Verify reused request-pool indices don't inherit old state."""
        from core.steering.dexperts_runtime import DExpertsRuntime

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        runtime = DExpertsRuntime(mock_model, mock_tokenizer, "cpu")
        runtime._loaded = True

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_output.past_key_values = None
        
        runtime.expert_adapter = MagicMock(return_value=mock_output)
        runtime.anti_expert_adapter = MagicMock(return_value=mock_output)

        # Create and release state for request "req_001"
        base_logits = torch.randn(1, 1000)
        input_ids = torch.tensor([[1, 2, 3]])
        
        runtime.apply_dexperts(
            base_logits=base_logits,
            input_ids=input_ids,
            alpha=2.0,  # Non-default alpha
            request_id="req_001",
        )
        
        state1 = runtime._request_states["req_001"]
        assert state1.alpha == 2.0
        
        runtime.release_state("req_001")

        # Reuse the same request ID
        runtime.apply_dexperts(
            base_logits=base_logits,
            input_ids=input_ids,
            alpha=1.0,  # Default alpha
            request_id="req_001",
        )
        
        state2 = runtime._request_states["req_001"]
        
        # Verify new state doesn't inherit old alpha
        assert state2.alpha == 1.0, "Reused request ID inherited old state"
        assert state2 is not state1, "Reused request ID got same state object"
        
        # Clean up
        runtime.release_state("req_001")


# ============================================================================
# CLAIM 4: Tokenizer Compatibility Gate
# ============================================================================

class TestClaim4_TokenizerCompatibilityGate:
    """Verify verify_tokenizer_compatibility() fails closed on incompatibility."""

    def test_gate_called_at_startup(self):
        """Verify tokenizer compatibility gate is called during initialization."""
        from core.steering.dexperts_runtime import DExpertsRuntime

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 151936
        mock_tokenizer.bos_token_id = 1
        mock_tokenizer.eos_token_id = 2
        
        runtime = DExpertsRuntime(mock_model, mock_tokenizer, "cpu")
        
        # Verify tokenizer is not yet verified
        assert runtime._tokenizer_verified is False
        
        # Call verify_tokenizer
        report = runtime.verify_tokenizer(expected_vocab_size=151936)
        
        # Verify tokenizer is now verified
        assert runtime._tokenizer_verified is True
        assert report.compatible is True

    def test_gate_checks_vocabulary_size(self):
        """Verify gate checks vocabulary size."""
        from core.steering.dexperts_runtime import verify_tokenizer_compatibility

        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 1000
        
        report = verify_tokenizer_compatibility(mock_tokenizer, expected_vocab_size=2000)
        
        # Should fail due to vocab size mismatch
        assert report.compatible is False
        assert any("Vocabulary size mismatch" in err for err in report.errors)

    def test_gate_checks_special_tokens(self):
        """Verify gate checks special tokens."""
        from core.steering.dexperts_runtime import verify_tokenizer_compatibility

        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 1000
        mock_tokenizer.bos_token_id = None  # Missing BOS
        mock_tokenizer.eos_token_id = None  # Missing EOS
        
        report = verify_tokenizer_compatibility(mock_tokenizer)
        
        # Should have warnings about missing special tokens
        assert any("bos_token_id is None" in warn for warn in report.warnings)
        assert any("eos_token_id is None" in warn for warn in report.warnings)

    def test_gate_raises_exception_on_failure(self):
        """Verify gate raises RuntimeError on incompatibility."""
        from core.steering.dexperts_runtime import DExpertsRuntime

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 1000
        
        runtime = DExpertsRuntime(mock_model, mock_tokenizer, "cpu")
        
        # Should raise RuntimeError when vocab size doesn't match
        with pytest.raises(RuntimeError, match="Tokenizer compatibility check failed"):
            runtime.verify_tokenizer(expected_vocab_size=2000)


# ============================================================================
# CLAIM 5: Context Bounds Enforcement
# ============================================================================

class TestClaim5_ContextBoundsEnforcement:
    """Verify DExperts rejects requests exceeding 4096 tokens."""

    def test_context_limit_enforced(self):
        """Verify context limit is enforced."""
        from core.steering.dexperts_runtime import DExpertsRuntime

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        runtime = DExpertsRuntime(mock_model, mock_tokenizer, "cpu")
        runtime._loaded = True

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_output.past_key_values = None
        
        runtime.expert_adapter = MagicMock(return_value=mock_output)
        runtime.anti_expert_adapter = MagicMock(return_value=mock_output)

        # Create state with prompt exceeding limit
        state = runtime.get_or_create_state("test-context-limit")
        state.prompt_token_ids = list(range(5000))  # Exceeds MAX_CONTEXT_LENGTH=4096
        
        # Apply DExperts
        base_logits = torch.randn(1, 1000)
        input_ids = torch.tensor([[1, 2, 3]])
        
        combined, telemetry = runtime.apply_dexperts(
            base_logits=base_logits,
            input_ids=input_ids,
            alpha=1.0,
            request_id="test-context-limit",
        )

        # Should return base logits with context_length_exceeded reason
        assert telemetry["applied"] is False
        assert telemetry["reason"] == "context_length_exceeded"
        assert torch.equal(combined, base_logits)
        
        # Clean up
        runtime.release_state("test-context-limit")

    def test_overflow_rejection_with_typed_error(self):
        """Verify overflow rejection with typed error."""
        from core.steering.dexperts_runtime import DExpertsRuntime

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        runtime = DExpertsRuntime(mock_model, mock_tokenizer, "cpu")
        runtime._loaded = True

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_output.past_key_values = None
        
        runtime.expert_adapter = MagicMock(return_value=mock_output)
        runtime.anti_expert_adapter = MagicMock(return_value=mock_output)

        # Create state with prompt exceeding limit
        state = runtime.get_or_create_state("test-overflow")
        state.prompt_token_ids = list(range(5000))
        
        base_logits = torch.randn(1, 1000)
        input_ids = torch.tensor([[1, 2, 3]])
        
        combined, telemetry = runtime.apply_dexperts(
            base_logits=base_logits,
            input_ids=input_ids,
            alpha=1.0,
            request_id="test-overflow",
        )

        # Verify telemetry contains context length info
        assert "history_length" in telemetry
        assert "max_context" in telemetry
        assert telemetry["history_length"] > 4096
        assert telemetry["max_context"] == 4096
        
        # Clean up
        runtime.release_state("test-overflow")

    def test_no_silent_truncation(self):
        """Verify no silent truncation occurs."""
        from core.steering.dexperts_runtime import DExpertsRuntime

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        runtime = DExpertsRuntime(mock_model, mock_tokenizer, "cpu")
        runtime._loaded = True

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_output.past_key_values = None
        
        runtime.expert_adapter = MagicMock(return_value=mock_output)
        runtime.anti_expert_adapter = MagicMock(return_value=mock_output)

        # Create state with prompt exceeding limit
        state = runtime.get_or_create_state("test-no-truncation")
        original_length = 5000
        state.prompt_token_ids = list(range(original_length))
        
        base_logits = torch.randn(1, 1000)
        input_ids = torch.tensor([[1, 2, 3]])
        
        combined, telemetry = runtime.apply_dexperts(
            base_logits=base_logits,
            input_ids=input_ids,
            alpha=1.0,
            request_id="test-no-truncation",
        )

        # Verify state was not silently truncated
        assert state.history_token_count == original_length
        assert telemetry["applied"] is False
        
        # Clean up
        runtime.release_state("test-no-truncation")


# ============================================================================
# CLAIM 6: Two Scientific Regimes
# ============================================================================

class TestClaim6_TwoScientificRegimes:
    """Verify both config files exist and are correct."""

    def test_config_files_exist(self):
        """Verify both config files exist."""
        config_dir = Path(__file__).parent.parent.parent / "config" / "experiments"
        
        qwen06b_config = config_dir / "dexperts_qwen06b.yaml"
        qwen14b_config = config_dir / "dexperts_qwen14b_awq.yaml"
        
        assert qwen06b_config.exists(), f"Config file not found: {qwen06b_config}"
        assert qwen14b_config.exists(), f"Config file not found: {qwen14b_config}"

    def test_06b_config_uses_raw_completion(self):
        """Verify 0.6B config uses raw completion (no chat template)."""
        import yaml
        
        config_path = Path(__file__).parent.parent.parent / "config" / "experiments" / "dexperts_qwen06b.yaml"
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # Verify use_chat_template is false
        assert config["generation"]["use_chat_template"] is False, \
            "0.6B config should use raw completion (use_chat_template: false)"
        
        # Verify add_special_tokens is false
        assert config["generation"]["add_special_tokens"] is False, \
            "0.6B config should not add special tokens"

    def test_14b_config_uses_chat_completion(self):
        """Verify 14B config uses chat completion."""
        import yaml
        
        config_path = Path(__file__).parent.parent.parent / "config" / "experiments" / "dexperts_qwen14b_awq.yaml"
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # Verify use_chat_template is true
        assert config["generation"]["use_chat_template"] is True, \
            "14B config should use chat completion (use_chat_template: true)"
        
        # Verify system prompt exists
        assert "system_prompt" in config["generation"], \
            "14B config should have a system prompt"

    def test_model_paths_and_revisions_pinned(self):
        """Verify model paths and revisions are pinned."""
        import yaml
        
        config_dir = Path(__file__).parent.parent.parent / "config" / "experiments"
        
        # Check 0.6B config
        with open(config_dir / "dexperts_qwen06b.yaml", "r", encoding="utf-8") as f:
            config_06b = yaml.safe_load(f)
        
        assert "revision" in config_06b["model"], "0.6B config missing revision"
        assert len(config_06b["model"]["revision"]) >= 7, "0.6B revision not pinned (too short)"
        
        # Check 14B config
        with open(config_dir / "dexperts_qwen14b_awq.yaml", "r", encoding="utf-8") as f:
            config_14b = yaml.safe_load(f)
        
        assert "revision" in config_14b["model"], "14B config missing revision"
        assert len(config_14b["model"]["revision"]) >= 7, "14B revision not pinned (too short)"


# ============================================================================
# CLAIM 7: Thread Safety
# ============================================================================

class TestClaim7_ThreadSafety:
    """Verify _execution_lock prevents concurrent adapter switching."""

    def test_execution_lock_exists(self):
        """Verify the execution lock exists."""
        from core.steering.dexperts_runtime import DExpertsRuntime

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        runtime = DExpertsRuntime(mock_model, mock_tokenizer, "cpu")
        
        # Verify execution lock exists
        assert hasattr(runtime, "_execution_lock")
        assert isinstance(runtime._execution_lock, type(threading.Lock()))

    def test_adapter_activation_and_forward_atomic(self):
        """Verify adapter activation and forward pass are atomic."""
        from core.steering.dexperts_runtime import DExpertsRuntime

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        runtime = DExpertsRuntime(mock_model, mock_tokenizer, "cpu")
        runtime._loaded = True

        # Verify execution lock exists and is a proper Lock
        assert hasattr(runtime, "_execution_lock"), "Execution lock missing"
        assert runtime._execution_lock is not None, "Execution lock is None"
        
        # Verify the lock can be acquired and released
        acquired = runtime._execution_lock.acquire(blocking=False)
        assert acquired, "Could not acquire execution lock"
        runtime._execution_lock.release()

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_output.past_key_values = None
        
        runtime.expert_adapter = MagicMock(return_value=mock_output)
        runtime.anti_expert_adapter = MagicMock(return_value=mock_output)

        # Apply DExperts - should use the lock internally
        base_logits = torch.randn(1, 1000)
        input_ids = torch.tensor([[1, 2, 3]])
        
        runtime.apply_dexperts(
            base_logits=base_logits,
            input_ids=input_ids,
            alpha=1.0,
            request_id="test-atomic",
        )

        # Verify lock is still available after apply_dexperts (not held)
        acquired_after = runtime._execution_lock.acquire(blocking=False)
        assert acquired_after, "Execution lock was not released after apply_dexperts"
        runtime._execution_lock.release()
        
        # Clean up
        runtime.release_state("test-atomic")

    def test_no_race_conditions_in_state_updates(self):
        """Verify no race conditions in state updates."""
        from core.steering.dexperts_runtime import DExpertsRuntime

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        runtime = DExpertsRuntime(mock_model, mock_tokenizer, "cpu")
        runtime._loaded = True

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_output.past_key_values = None
        
        runtime.expert_adapter = MagicMock(return_value=mock_output)
        runtime.anti_expert_adapter = MagicMock(return_value=mock_output)

        # Run concurrent requests
        def request_worker(request_id):
            base_logits = torch.randn(1, 1000)
            input_ids = torch.tensor([[1, 2, 3]])
            
            for _ in range(10):
                runtime.apply_dexperts(
                    base_logits=base_logits,
                    input_ids=input_ids,
                    alpha=1.0,
                    request_id=request_id,
                )
            
            runtime.release_state(request_id)

        threads = []
        for i in range(5):
            t = threading.Thread(target=request_worker, args=(f"concurrent_req_{i}",))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify all states were cleaned up
        assert len(runtime._request_states) == 0, \
            f"State leak detected: {len(runtime._request_states)} states remain"


# ============================================================================
# CLAIM 8: Metrics Collection
# ============================================================================

class TestClaim8_MetricsCollection:
    """Verify _DExpertsMetrics tracks active requests, created/cleaned states."""

    def test_metrics_thread_safe(self):
        """Verify metrics are thread-safe."""
        from core.steering.dexperts_runtime import _DExpertsMetrics

        metrics = _DExpertsMetrics()
        
        # Verify lock exists
        assert hasattr(metrics, "_lock")
        assert isinstance(metrics._lock, type(threading.Lock()))

    def test_metrics_updated_on_state_create(self):
        """Verify metrics are updated on state create."""
        from core.steering.dexperts_runtime import DExpertsRuntime, get_dexperts_metrics

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        runtime = DExpertsRuntime(mock_model, mock_tokenizer, "cpu")
        runtime._loaded = True

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_output.past_key_values = None
        
        runtime.expert_adapter = MagicMock(return_value=mock_output)
        runtime.anti_expert_adapter = MagicMock(return_value=mock_output)

        # Get initial metrics
        initial_metrics = get_dexperts_metrics()
        initial_created = initial_metrics["created_states_total"]
        initial_active = initial_metrics["active_dexperts_requests"]

        # Create state
        base_logits = torch.randn(1, 1000)
        input_ids = torch.tensor([[1, 2, 3]])
        
        runtime.apply_dexperts(
            base_logits=base_logits,
            input_ids=input_ids,
            alpha=1.0,
            request_id="test-metrics-create",
        )

        # Verify metrics were updated
        updated_metrics = get_dexperts_metrics()
        assert updated_metrics["created_states_total"] == initial_created + 1
        assert updated_metrics["active_dexperts_requests"] == initial_active + 1
        
        # Clean up
        runtime.release_state("test-metrics-create")

    def test_metrics_updated_on_state_clean(self):
        """Verify metrics are updated on state clean."""
        from core.steering.dexperts_runtime import DExpertsRuntime, get_dexperts_metrics

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        runtime = DExpertsRuntime(mock_model, mock_tokenizer, "cpu")
        runtime._loaded = True

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_output.past_key_values = None
        
        runtime.expert_adapter = MagicMock(return_value=mock_output)
        runtime.anti_expert_adapter = MagicMock(return_value=mock_output)

        # Create state
        base_logits = torch.randn(1, 1000)
        input_ids = torch.tensor([[1, 2, 3]])
        
        runtime.apply_dexperts(
            base_logits=base_logits,
            input_ids=input_ids,
            alpha=1.0,
            request_id="test-metrics-clean",
        )

        # Get metrics before release
        before_release = get_dexperts_metrics()
        before_cleaned = before_release["cleaned_states_total"]
        before_active = before_release["active_dexperts_requests"]

        # Release state
        runtime.release_state("test-metrics-clean")

        # Verify metrics were updated
        after_release = get_dexperts_metrics()
        assert after_release["cleaned_states_total"] == before_cleaned + 1
        assert after_release["active_dexperts_requests"] == before_active - 1

    def test_metrics_observable(self):
        """Verify metrics are observable via get_dexperts_metrics()."""
        from core.steering.dexperts_runtime import get_dexperts_metrics

        metrics = get_dexperts_metrics()
        
        # Verify expected fields exist
        assert "active_dexperts_requests" in metrics
        assert "created_states_total" in metrics
        assert "cleaned_states_total" in metrics
        assert "stale_states_total" in metrics
        assert "request_state_collisions_total" in metrics
        
        # Verify types
        assert isinstance(metrics["active_dexperts_requests"], int)
        assert isinstance(metrics["created_states_total"], int)
        assert isinstance(metrics["cleaned_states_total"], int)


# ============================================================================
# SUMMARY TEST
# ============================================================================

def test_summary():
    """Summary test to ensure all claims were tested."""
    print("\n" + "="*70)
    print("NRAM v5 R5 CRITICAL CLAIMS FALSIFICATION SUMMARY")
    print("="*70)
    print("\nAll 8 critical claims have been tested:")
    print("1. Separate KV Caches Actually Work")
    print("2. Canonical Decoding Order")
    print("3. State Machine Correctness")
    print("4. Tokenizer Compatibility Gate")
    print("5. Context Bounds Enforcement")
    print("6. Two Scientific Regimes")
    print("7. Thread Safety")
    print("8. Metrics Collection")
    print("="*70)
