"""Unit tests for NRAM serialization utilities.

Tests serialization of custom logit processor and parameter building.
"""
import json
import pytest

torch = pytest.importorskip("torch")

from core.steering.serialization import (
    serialize_processor,
    build_custom_params,
)
from core.steering.nram_logit_processor import NRAMLogitProcessor


class TestSerializeProcessor:
    """Test processor serialization to JSON."""

    def test_serialize_processor_returns_json_string(self):
        """Test that serialize_processor returns a valid JSON string."""
        result = serialize_processor(NRAMLogitProcessor)
        assert isinstance(result, str)
        
        # Should be valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_serialize_processor_contains_callable(self):
        """Test that serialized processor contains callable field."""
        result = serialize_processor(NRAMLogitProcessor)
        parsed = json.loads(result)
        
        assert "callable" in parsed
        assert isinstance(parsed["callable"], str)
        # Should be a hex string
        bytes.fromhex(parsed["callable"])

    def test_serialize_processor_roundtrip(self):
        """Test that serialized processor class can be deserialized."""
        dill = pytest.importorskip("dill")
        
        result = serialize_processor(NRAMLogitProcessor)
        parsed = json.loads(result)
        
        # Deserialize the class (not instance)
        callable_bytes = bytes.fromhex(parsed["callable"])
        restored_class = dill.loads(callable_bytes)
        
        # Should be a class
        assert restored_class is NRAMLogitProcessor
        
        # Should be instantiable
        instance = restored_class()
        
        # Should work with logits
        logits = torch.zeros((1, 100))
        params = [{"positive_token_ids": [5], "positive_bias": 1.0}]
        output = instance(logits.clone(), params)
        assert output[0, 5].item() == pytest.approx(1.0)


class TestBuildCustomParams:
    """Test build_custom_params helper function."""

    def test_build_custom_params_basic(self):
        """Test building basic custom parameters."""
        result = build_custom_params(
            positive_token_ids=[5, 10],
            negative_token_ids=[20],
            forbidden_token_ids=[30],
            positive_bias=1.0,
            negative_bias=2.0,
            repetition_penalty=1.5,
            profile="test_profile",
        )
        
        assert result["positive_token_ids"] == [5, 10]
        assert result["negative_token_ids"] == [20]
        assert result["forbidden_token_ids"] == [30]
        assert result["positive_bias"] == 1.0
        assert result["negative_bias"] == 2.0
        assert result["repetition_penalty"] == 1.5
        assert result["profile"] == "test_profile"

    def test_build_custom_params_with_phenomena(self):
        """Test building parameters with phenomenon weights."""
        phenomena = {
            "overlap": 0.5,
            "forgetting": 0.3,
            "looping": 0.2,
        }
        
        result = build_custom_params(
            positive_token_ids=[5],
            negative_token_ids=[],
            forbidden_token_ids=[],
            positive_bias=1.0,
            negative_bias=0.0,
            repetition_penalty=0.0,
            profile="test_profile",
            phenomenon_weights=phenomena,
        )
        
        assert "phenomenon_weights" in result
        assert result["phenomenon_weights"] == phenomena
        assert result["phenomenon_weights"]["overlap"] == 0.5

    def test_build_custom_params_minimal(self):
        """Test building minimal custom parameters."""
        result = build_custom_params(
            positive_token_ids=[],
            negative_token_ids=[],
            forbidden_token_ids=[],
            positive_bias=0.0,
            negative_bias=0.0,
            repetition_penalty=0.0,
            profile="normal",
        )
        
        # Should return dict with all fields
        assert isinstance(result, dict)
        assert "positive_token_ids" in result
        assert "profile" in result

    def test_build_custom_params_with_max_tokens(self):
        """Test building parameters with max_tokens."""
        result = build_custom_params(
            positive_token_ids=[5],
            negative_token_ids=[],
            forbidden_token_ids=[],
            positive_bias=1.0,
            negative_bias=0.0,
            repetition_penalty=0.0,
            profile="test_profile",
            max_tokens=100,
        )
        
        assert result["max_tokens"] == 100

    def test_build_custom_params_preserves_all_fields(self):
        """Test that all fields are preserved in output."""
        result = build_custom_params(
            positive_token_ids=[1, 2, 3],
            negative_token_ids=[4, 5],
            forbidden_token_ids=[6],
            positive_bias=1.5,
            negative_bias=2.5,
            repetition_penalty=1.2,
            profile="psychedelic",
            max_tokens=200,
            phenomenon_weights={"overlap": 0.5},
        )
        
        # All fields should be present
        assert "positive_token_ids" in result
        assert "negative_token_ids" in result
        assert "forbidden_token_ids" in result
        assert "positive_bias" in result
        assert "negative_bias" in result
        assert "repetition_penalty" in result
        assert "max_tokens" in result
        assert "phenomenon_weights" in result
        assert "profile" in result

    def test_build_custom_params_no_phenomena_omits_key(self):
        """Test that phenomenon_weights is omitted when None."""
        result = build_custom_params(
            positive_token_ids=[5],
            negative_token_ids=[],
            forbidden_token_ids=[],
            positive_bias=1.0,
            negative_bias=0.0,
            repetition_penalty=0.0,
            profile="normal",
        )
        
        assert "phenomenon_weights" not in result
