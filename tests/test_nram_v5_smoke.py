"""
Minimal smoke tests for NRAM v5 components.

These tests verify that each component can be instantiated and performs
basic operations without errors. They are not comprehensive - just quick
checks that the code is wired correctly.
"""
import pytest
import torch
from core.steering.activation_addition import ActivationAdditionController, ActivationVector
from core.steering.genuine_conceptor import GenuineConceptor
from core.steering.trained_probes import HiddenStateProbeTrainer, ProbeArtifact
from core.steering.latent_closed_loop import LatentClosedLoopController, ClosedLoopConfig
from core.steering.semantic_closed_loop import SemanticClosedLoopController, SemanticLoopConfig
from core.steering.branch_tournament import BranchTournamentEngine, BranchSearchConfig
from core.steering.dexperts import DExpertsController, ExpertConfig, ExpertType


class TestActivationAdditionSmoke:
    """Smoke test for activation addition."""
    
    def test_controller_instantiation(self):
        """Controller can be created."""
        controller = ActivationAdditionController(device="cpu")
        assert controller is not None
    
    def test_vector_creation(self):
        """Vector can be created and added."""
        controller = ActivationAdditionController(device="cpu")
        vector = torch.randn(5120)
        act_vector = ActivationVector(name="test", vector=vector, layer=20, scale=1.0)
        controller.add_vector(act_vector)
        assert "test" in controller.vectors
    
    def test_apply_to_hidden_states(self):
        """Vector can be applied to hidden states."""
        controller = ActivationAdditionController(device="cpu")
        vector = torch.randn(5120)
        act_vector = ActivationVector(name="test", vector=vector, layer=20, scale=1.0)
        controller.add_vector(act_vector)
        
        hidden = torch.randn(1, 10, 5120)
        modified = controller.apply_to_hidden_states(hidden, layer=20)
        
        assert modified.shape == hidden.shape
        assert not torch.allclose(modified, hidden)


class TestConceptorSmoke:
    """Smoke test for conceptor steering."""
    
    def test_conceptor_creation(self):
        """Conceptor can be created from activations."""
        activations = torch.randn(100, 5120)
        conceptor = GenuineConceptor.from_activations(
            activations, aperture=2.0, conceptor_id="test", layer_name="layer_20"
        )
        assert conceptor is not None
        assert conceptor.artifact.conceptor_id == "test"
    
    def test_apply_conceptor(self):
        """Conceptor can be applied to hidden states."""
        activations = torch.randn(100, 5120)
        conceptor = GenuineConceptor.from_activations(
            activations, aperture=2.0, conceptor_id="test", layer_name="layer_20"
        )
        
        hidden = torch.randn(1, 10, 5120)
        projected = conceptor.apply(hidden)
        
        assert projected.shape == hidden.shape


class TestHiddenStateProbesSmoke:
    """Smoke test for hidden state probes."""
    
    def test_probe_artifact_creation(self):
        """ProbeArtifact can be created."""
        weights = torch.randn(5120)
        probe = ProbeArtifact(
            probe_id="test",
            probe_type="novelty",
            layer_name="layer_20",
            layer_index=20,
            hidden_dim=5120,
            weights=weights,
            bias=0.0,
            feature_mean=torch.zeros(5120),
            feature_std=torch.ones(5120),
            calibration_slope=1.0,
            calibration_intercept=0.0,
            val_accuracy=0.8,
            val_f1=0.75,
            val_auc=0.85,
            model_hash="test_hash",
            tokenizer_hash="test_hash",
            dataset_hash="test_hash",
            training_seed=42,
            artifact_hash="test_hash"
        )
        assert probe is not None
        assert probe.probe_id == "test"


class TestLatentClosedLoopSmoke:
    """Smoke test for latent closed loop."""
    
    def test_controller_instantiation(self):
        """Controller can be created."""
        config = ClosedLoopConfig(enabled=True)
        controller = LatentClosedLoopController(config)
        assert controller is not None


class TestSemanticClosedLoopSmoke:
    """Smoke test for semantic closed loop."""
    
    def test_controller_instantiation(self):
        """Controller can be created."""
        config = SemanticLoopConfig(enabled=True, chunk_size=24)
        controller = SemanticClosedLoopController(config)
        assert controller is not None
    
    def test_evaluate_chunk(self):
        """Controller can evaluate a text chunk."""
        config = SemanticLoopConfig(enabled=True, chunk_size=24)
        controller = SemanticClosedLoopController(config)
        
        chunk = "This is a test chunk of text."
        evaluation = controller.evaluate_chunk(chunk, token_count=10)
        
        assert evaluation is not None
        assert evaluation.text == chunk


class TestBranchTournamentSmoke:
    """Smoke test for branch tournament."""
    
    def test_engine_instantiation(self):
        """Engine can be created."""
        config = BranchSearchConfig(enabled=True, num_branches=4)
        engine = BranchTournamentEngine(config)
        assert engine is not None


class TestDExpertsSmoke:
    """Smoke test for DExperts."""
    
    def test_expert_config_creation(self):
        """ExpertConfig can be created."""
        config = ExpertConfig(
            name="creativity",
            expert_type=ExpertType.CREATIVITY,
            model_path="/path/to/model",
            alpha=1.0,
            beta=0.5
        )
        assert config is not None
        assert config.name == "creativity"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
