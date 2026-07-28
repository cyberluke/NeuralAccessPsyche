"""
Tests for activation addition steering.

Tests cover:
- ActivationAdditionController
- Forward hooks
- Vector collection
- Vector registry
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from core.steering.activation_addition import (
    ActivationVector,
    ActivationAdditionController,
    VectorCollector
)
from core.steering.forward_hooks import (
    ActivationSteeringHook,
    HookManager,
    find_transformer_layers,
    auto_register_hooks
)
from core.steering.activation_vectors import (
    VectorRegistry,
    create_novelty_vector,
    create_concreteness_vector,
    get_recommended_layers
)


class TestActivationVector:
    """Tests for ActivationVector dataclass."""
    
    def test_vector_creation(self):
        """Test creating an activation vector."""
        vector = torch.randn(768)
        act_vec = ActivationVector(
            name="test",
            vector=vector,
            layer=10,
            scale=1.5
        )
        
        assert act_vec.name == "test"
        assert act_vec.layer == 10
        assert act_vec.scale == 1.5
        assert torch.equal(act_vec.vector, vector)
    
    def test_vector_to_device(self):
        """Test moving vector to device."""
        vector = torch.randn(768)
        act_vec = ActivationVector(
            name="test",
            vector=vector,
            layer=10
        )
        
        # Move to CPU (already there, but tests the method)
        moved = act_vec.to("cpu")
        assert moved.vector.device.type == "cpu"
        assert moved.name == "test"
        assert moved.layer == 10


class TestActivationAdditionController:
    """Tests for ActivationAdditionController."""
    
    def test_controller_initialization(self):
        """Test controller initialization."""
        controller = ActivationAdditionController(device="cpu")
        assert len(controller.vectors) == 0
        assert controller.device == "cpu"
    
    def test_add_vector(self):
        """Test adding a vector to controller."""
        controller = ActivationAdditionController(device="cpu")
        vector = torch.randn(768)
        act_vec = ActivationVector(
            name="novelty",
            vector=vector,
            layer=10
        )
        
        controller.add_vector(act_vec)
        
        assert "novelty" in controller.vectors
        assert controller.vectors["novelty"].name == "novelty"
    
    def test_remove_vector(self):
        """Test removing a vector from controller."""
        controller = ActivationAdditionController(device="cpu")
        vector = torch.randn(768)
        act_vec = ActivationVector(
            name="novelty",
            vector=vector,
            layer=10
        )
        
        controller.add_vector(act_vec)
        assert "novelty" in controller.vectors
        
        controller.remove_vector("novelty")
        assert "novelty" not in controller.vectors
    
    def test_get_vectors_for_layer(self):
        """Test getting vectors for a specific layer."""
        controller = ActivationAdditionController(device="cpu")
        
        # Add vectors to different layers
        for i in range(3):
            vector = torch.randn(768)
            act_vec = ActivationVector(
                name=f"vec_{i}",
                vector=vector,
                layer=10 if i < 2 else 20
            )
            controller.add_vector(act_vec)
        
        # Get vectors for layer 10
        layer_10_vecs = controller.get_vectors_for_layer(10)
        assert len(layer_10_vecs) == 2
        
        # Get vectors for layer 20
        layer_20_vecs = controller.get_vectors_for_layer(20)
        assert len(layer_20_vecs) == 1
    
    def test_apply_to_hidden_states(self):
        """Test applying vectors to hidden states."""
        controller = ActivationAdditionController(device="cpu")
        
        # Add a vector
        vector = torch.ones(768) * 2.0
        act_vec = ActivationVector(
            name="test",
            vector=vector,
            layer=10,
            scale=1.0
        )
        controller.add_vector(act_vec)
        
        # Create hidden states
        hidden_states = torch.zeros(1, 10, 768)
        
        # Apply to layer 10
        modified = controller.apply_to_hidden_states(hidden_states, layer=10)
        
        # Check that vectors were added
        assert modified.shape == hidden_states.shape
        assert torch.allclose(modified, torch.ones(1, 10, 768) * 2.0)
    
    def test_apply_disabled(self):
        """Test that disabled controller doesn't modify hidden states."""
        controller = ActivationAdditionController(device="cpu")
        controller.enabled = False
        
        vector = torch.ones(768) * 2.0
        act_vec = ActivationVector(
            name="test",
            vector=vector,
            layer=10
        )
        controller.add_vector(act_vec)
        
        hidden_states = torch.zeros(1, 10, 768)
        modified = controller.apply_to_hidden_states(hidden_states, layer=10)
        
        # Should be unchanged
        assert torch.equal(modified, hidden_states)
    
    def test_phase_aware_steering(self):
        """Test phase-aware vector selection."""
        controller = ActivationAdditionController(device="cpu")
        
        # Add vectors with different phases
        for phase in ["extraction", "divergence", "synthesis"]:
            vector = torch.randn(768)
            act_vec = ActivationVector(
                name=f"vec_{phase}",
                vector=vector,
                layer=10,
                phase=phase
            )
            controller.add_vector(act_vec)
        
        # Get vectors for divergence phase
        divergence_vecs = controller.get_vectors_for_layer(10, phase="divergence")
        assert len(divergence_vecs) == 1
        assert divergence_vecs[0].phase == "divergence"


class TestVectorCollector:
    """Tests for VectorCollector."""
    
    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Create mock model and tokenizer."""
        # Mock tokenizer
        tokenizer = Mock()
        
        class TokenizerOutput(dict):
            def to(self, device):
                return self
        
        tokenizer.return_value = TokenizerOutput({
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
        })
        
        # Mock model
        model = Mock()
        
        # Create mock hidden states for different layers
        num_layers = 12
        hidden_dim = 768
        
        def mock_forward(**kwargs):
            # Create mock output with hidden states
            hidden_states = tuple(
                torch.randn(1, 5, hidden_dim) for _ in range(num_layers + 1)
            )
            
            output = Mock()
            output.hidden_states = hidden_states
            return output
        
        model.side_effect = mock_forward
        model.eval = Mock()
        
        # Mock config
        model.config = Mock()
        model.config.num_hidden_layers = num_layers
        
        return model, tokenizer
    
    def test_collector_initialization(self, mock_model_and_tokenizer):
        """Test collector initialization."""
        model, tokenizer = mock_model_and_tokenizer
        collector = VectorCollector(model, tokenizer, device="cpu")
        
        assert collector.model == model
        assert collector.tokenizer == tokenizer
        assert collector.device == "cpu"
    
    def test_collect_vector(self, mock_model_and_tokenizer):
        """Test collecting a contrastive vector."""
        model, tokenizer = mock_model_and_tokenizer
        collector = VectorCollector(model, tokenizer, device="cpu")
        
        # Collect vector
        vector = collector.collect(
            positive_prompt="Creative innovative idea",
            negative_prompt="Boring routine task",
            layer=10,
            aggregation="mean"
        )
        
        # Check vector properties
        assert isinstance(vector, torch.Tensor)
        assert vector.shape == (768,)
        # Vector should be normalized
        assert torch.isclose(vector.norm(), torch.tensor(1.0), atol=1e-5)
    
    def test_collect_different_aggregations(self, mock_model_and_tokenizer):
        """Test different aggregation methods."""
        model, tokenizer = mock_model_and_tokenizer
        collector = VectorCollector(model, tokenizer, device="cpu")
        
        for aggregation in ["mean", "last", "max"]:
            vector = collector.collect(
                positive_prompt="Test positive",
                negative_prompt="Test negative",
                layer=10,
                aggregation=aggregation
            )
            
            assert isinstance(vector, torch.Tensor)
            assert vector.shape == (768,)


class TestActivationSteeringHook:
    """Tests for ActivationSteeringHook."""
    
    def test_hook_creation(self):
        """Test hook creation."""
        controller = ActivationAdditionController(device="cpu")
        hook = ActivationSteeringHook(controller, "layer.10", 10)
        
        assert hook.controller == controller
        assert hook.layer_name == "layer.10"
        assert hook.layer_index == 10
        assert hook.enabled is True
    
    def test_hook_call(self):
        """Test hook forward pass."""
        controller = ActivationAdditionController(device="cpu")
        
        # Add vector
        vector = torch.ones(768) * 2.0
        act_vec = ActivationVector(
            name="test",
            vector=vector,
            layer=10
        )
        controller.add_vector(act_vec)
        
        hook = ActivationSteeringHook(controller, "layer.10", 10)
        
        # Mock module and input
        module = Mock()
        input_args = ()
        output = torch.zeros(1, 10, 768)
        
        # Call hook
        modified = hook(module, input_args, output)
        
        # Check modification
        assert modified.shape == output.shape
        assert torch.allclose(modified, torch.ones(1, 10, 768) * 2.0)
    
    def test_hook_disabled(self):
        """Test disabled hook doesn't modify output."""
        controller = ActivationAdditionController(device="cpu")
        hook = ActivationSteeringHook(controller, "layer.10", 10)
        hook.enabled = False
        
        output = torch.zeros(1, 10, 768)
        modified = hook(Mock(), (), output)
        
        assert torch.equal(modified, output)


class TestHookManager:
    """Tests for HookManager."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock transformer model."""
        # Create a real nn.Module with proper structure
        class MockTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                # Create layers as ModuleList
                self.h = nn.ModuleList([nn.Linear(768, 768) for _ in range(12)])
        
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = MockTransformer()
        
        return MockModel()
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = HookManager()
        assert len(manager.hooks) == 0
        assert len(manager.hook_handles) == 0
    
    def test_register_hook(self, mock_model):
        """Test registering a hook."""
        manager = HookManager()
        controller = ActivationAdditionController(device="cpu")
        
        hook = manager.register_hook(
            model=mock_model,
            controller=controller,
            layer_name="transformer.h.0",
            layer_index=0
        )
        
        assert "transformer.h.0" in manager.hooks
        assert hook.layer_index == 0
    
    def test_remove_hook(self, mock_model):
        """Test removing a hook."""
        manager = HookManager()
        controller = ActivationAdditionController(device="cpu")
        
        manager.register_hook(
            model=mock_model,
            controller=controller,
            layer_name="transformer.h.0",
            layer_index=0
        )
        
        manager.remove_hook("transformer.h.0")
        assert "transformer.h.0" not in manager.hooks
    
    def test_remove_all_hooks(self, mock_model):
        """Test removing all hooks."""
        manager = HookManager()
        controller = ActivationAdditionController(device="cpu")
        
        for i in range(3):
            manager.register_hook(
                model=mock_model,
                controller=controller,
                layer_name=f"transformer.h.{i}",
                layer_index=i
            )
        
        assert len(manager.hooks) == 3
        
        manager.remove_all_hooks()
        assert len(manager.hooks) == 0


class TestVectorRegistry:
    """Tests for VectorRegistry."""
    
    @pytest.fixture
    def temp_vectors_dir(self, tmp_path):
        """Create temporary vectors directory with test data."""
        vectors_dir = tmp_path / "vectors"
        vectors_dir.mkdir()
        
        # Create test vectors - use names that match the factory functions
        vector_names = ["novelty_vs_paraphrase", "concreteness"]
        for name in vector_names:
            for layer in [10, 20]:
                vector = torch.randn(768)
                vector_path = vectors_dir / f"{name}_layer{layer}.pt"
                torch.save(vector, vector_path)
        
        # Create metadata
        metadata = {
            "model_name": "test-model",
            "vectors": {
                "novelty_vs_paraphrase": {
                    "description": "Novelty steering",
                    "layers": [
                        {"layer": 10, "file": str(vectors_dir / "novelty_vs_paraphrase_layer10.pt")},
                        {"layer": 20, "file": str(vectors_dir / "novelty_vs_paraphrase_layer20.pt")}
                    ]
                },
                "concreteness": {
                    "description": "Concreteness steering",
                    "layers": [
                        {"layer": 10, "file": str(vectors_dir / "concreteness_layer10.pt")},
                        {"layer": 20, "file": str(vectors_dir / "concreteness_layer20.pt")}
                    ]
                }
            }
        }
        
        import json
        with open(vectors_dir / "vectors_metadata.json", "w") as f:
            json.dump(metadata, f)
        
        return vectors_dir
    
    def test_registry_initialization(self, temp_vectors_dir):
        """Test registry initialization."""
        registry = VectorRegistry(temp_vectors_dir)
        assert len(registry.vectors) == 4  # 2 names * 2 layers
    
    def test_get_vector_by_name(self, temp_vectors_dir):
        """Test getting vector by name."""
        registry = VectorRegistry(temp_vectors_dir)
        
        # Get novelty vector (should return best layer)
        vector = registry.get_vector("novelty_vs_paraphrase")
        assert vector is not None
        assert vector.name == "novelty_vs_paraphrase"
    
    def test_get_vector_by_name_and_layer(self, temp_vectors_dir):
        """Test getting vector by name and layer."""
        registry = VectorRegistry(temp_vectors_dir)
        
        vector = registry.get_vector("novelty_vs_paraphrase", layer=10)
        assert vector is not None
        assert vector.name == "novelty_vs_paraphrase"
        assert vector.layer == 10
    
    def test_list_vectors(self, temp_vectors_dir):
        """Test listing all vectors."""
        registry = VectorRegistry(temp_vectors_dir)
        
        vectors = registry.list_vectors()
        assert "novelty_vs_paraphrase" in vectors
        assert "concreteness" in vectors
        assert len(vectors["novelty_vs_paraphrase"]) == 2
        assert len(vectors["concreteness"]) == 2
    
    def test_create_novelty_vector(self, temp_vectors_dir):
        """Test creating novelty vector."""
        registry = VectorRegistry(temp_vectors_dir)
        
        vector = create_novelty_vector(registry)
        assert vector is not None
        # Vector name should match what's in the registry
        assert vector.name in ["novelty", "novelty_vs_paraphrase"]
    
    def test_get_recommended_layers(self, temp_vectors_dir):
        """Test getting recommended layers."""
        registry = VectorRegistry(temp_vectors_dir)
        
        layers = get_recommended_layers("novelty_vs_paraphrase", registry)
        assert len(layers) > 0
        assert all(isinstance(l, int) for l in layers)


class TestIntegration:
    """Integration tests for activation addition system."""
    
    def test_full_steering_pipeline(self):
        """Test complete steering pipeline."""
        # Create controller
        controller = ActivationAdditionController(device="cpu")
        
        # Add vectors
        for i in range(3):
            vector = torch.randn(768)
            act_vec = ActivationVector(
                name=f"vec_{i}",
                vector=vector,
                layer=10 + i * 5
            )
            controller.add_vector(act_vec)
        
        # Create hidden states
        hidden_states = torch.zeros(1, 10, 768)
        
        # Apply steering to each layer
        for layer in [10, 15, 20]:
            modified = controller.apply_to_hidden_states(hidden_states, layer)
            assert modified.shape == hidden_states.shape
        
        # Check status
        status = controller.get_status()
        assert status["num_vectors"] == 3
    
    def test_phase_aware_steering_integration(self):
        """Test phase-aware steering integration."""
        controller = ActivationAdditionController(device="cpu")
        
        # Add vectors for different phases
        phases = ["extraction", "divergence", "synthesis"]
        for phase in phases:
            vector = torch.randn(768)
            act_vec = ActivationVector(
                name=f"vec_{phase}",
                vector=vector,
                layer=10,
                phase=phase
            )
            controller.add_vector(act_vec)
        
        # Test each phase
        for phase in phases:
            vectors = controller.get_vectors_for_layer(10, phase=phase)
            assert len(vectors) == 1
            assert vectors[0].phase == phase


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
