"""
Comprehensive tests for NRAM v5 Sprint 3 components.

Tests cover:
- MultiVectorController
- ConceptorController
- HiddenStateProbes
- DExpertsController
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from core.steering.multi_vector_controller import (
    MultiVectorController,
    VectorConfig,
    CombinationMode,
)
from core.steering.activation_addition import ActivationVector
from core.steering.conceptor_steering import (
    ConceptorController,
    Conceptor,
    LowRankSubspaceSteering,
)
from core.steering.hidden_state_probes import (
    ProbeController,
    ClosedLoopController,
    ProbeConfig,
    ProbeType,
    create_novelty_probe,
    create_coherence_probe,
)
from core.steering.dexperts import (
    DExpertsController,
    ExpertConfig,
    ExpertType,
)


class TestMultiVectorController:
    """Tests for MultiVectorController."""

    @pytest.fixture
    def controller(self):
        """Create MultiVectorController instance."""
        return MultiVectorController(device="cpu")

    @pytest.fixture
    def sample_vectors(self):
        """Create sample ActivationVector objects."""
        return {
            "novelty": ActivationVector(
                name="novelty",
                vector=torch.randn(5120),
                layer=20,
                scale=1.0
            ),
            "concreteness": ActivationVector(
                name="concreteness",
                vector=torch.randn(5120),
                layer=20,
                scale=1.0
            ),
            "human_focus": ActivationVector(
                name="human_focus",
                vector=torch.randn(5120),
                layer=20,
                scale=1.0
            ),
        }

    def test_add_vector(self, controller, sample_vectors):
        """Test adding vectors to controller."""
        controller.add_vector(
            name="novelty",
            vector=sample_vectors["novelty"],
            weight=1.2,
            active_phases=["divergence"],
            priority=1,
        )

        assert "novelty" in controller.vector_configs
        assert controller.vector_configs["novelty"].weight == 1.2
        assert controller.vector_configs["novelty"].priority == 1

    def test_remove_vector(self, controller, sample_vectors):
        """Test removing vectors from controller."""
        controller.add_vector(
            name="novelty",
            vector=sample_vectors["novelty"],
            weight=1.0,
        )
        assert "novelty" in controller.vector_configs

        controller.remove_vector("novelty")
        assert "novelty" not in controller.vector_configs

    def test_get_active_vectors(self, controller, sample_vectors):
        """Test getting active vectors for current phase."""
        controller.add_vector(
            name="novelty",
            vector=sample_vectors["novelty"],
            active_phases=["divergence", "synthesis"],
        )
        controller.add_vector(
            name="concreteness",
            vector=sample_vectors["concreteness"],
            active_phases=["synthesis"],
        )

        controller.set_phase("divergence")
        active = controller.get_active_vectors()
        assert len(active) == 1
        assert active[0][0] == "novelty"

        controller.set_phase("synthesis")
        active = controller.get_active_vectors()
        assert len(active) == 2

    def test_combine_vectors_additive(self, controller, sample_vectors):
        """Test additive combination mode."""
        controller.combination_mode = CombinationMode.ADDITIVE

        controller.add_vector(
            name="novelty",
            vector=sample_vectors["novelty"],
            weight=1.0,
        )
        controller.add_vector(
            name="concreteness",
            vector=sample_vectors["concreteness"],
            weight=2.0,
        )

        combined = controller.combine_vectors(hidden_dim=5120, layer=20)
        assert combined is not None
        assert combined.shape == (5120,)

        # Should be weighted sum
        expected = sample_vectors["novelty"].vector * 1.0 + sample_vectors["concreteness"].vector * 2.0
        assert torch.allclose(combined, expected, atol=1e-5)

    def test_combine_vectors_projected(self, controller, sample_vectors):
        """Test projected combination mode (orthogonalization)."""
        controller.combination_mode = CombinationMode.PROJECTED

        controller.add_vector(
            name="novelty",
            vector=sample_vectors["novelty"],
            weight=1.0,
            priority=1,
        )
        controller.add_vector(
            name="concreteness",
            vector=sample_vectors["concreteness"],
            weight=1.0,
            priority=2,
        )

        combined = controller.combine_vectors(hidden_dim=5120, layer=20)
        assert combined is not None
        assert combined.shape == (5120,)

    def test_apply_to_hidden_states(self, controller, sample_vectors):
        """Test applying combined vectors to hidden states."""
        controller.add_vector(
            name="novelty",
            vector=sample_vectors["novelty"],
            weight=1.0,
        )

        hidden_states = torch.randn(2, 10, 5120)
        modified = controller.apply_to_hidden_states(hidden_states, layer=20)

        assert modified.shape == hidden_states.shape
        assert not torch.equal(modified, hidden_states)

    def test_create_preset(self, controller):
        """Test creating presets."""
        preset = controller.create_preset("visionary")
        assert "novelty" in preset
        assert "concreteness" in preset
        assert "human_focus" in preset

    def test_apply_preset(self, controller, sample_vectors):
        """Test applying preset configuration."""
        controller.add_vector(
            name="novelty",
            vector=sample_vectors["novelty"],
            weight=1.0,
        )
        controller.add_vector(
            name="concreteness",
            vector=sample_vectors["concreteness"],
            weight=1.0,
        )

        controller.apply_preset("visionary")
        assert controller.vector_configs["novelty"].weight != 1.0


class TestConceptorController:
    """Tests for ConceptorController."""

    @pytest.fixture
    def controller(self):
        """Create ConceptorController instance."""
        return ConceptorController(device="cpu")

    @pytest.fixture
    def sample_patterns(self):
        """Create sample pattern matrices."""
        return {
            "creativity": torch.randn(5120, 10),
            "technical": torch.randn(5120, 10),
        }

    def test_add_conceptor(self, controller, sample_patterns):
        """Test adding conceptors."""
        controller.add_conceptor(
            name="creativity",
            patterns=sample_patterns["creativity"],
            aperture=2.0,
            layer=20,
            weight=1.0,
        )

        assert "creativity" in controller.conceptors
        assert controller.conceptors["creativity"][1].aperture == 2.0

    def test_remove_conceptor(self, controller, sample_patterns):
        """Test removing conceptors."""
        controller.add_conceptor(
            name="creativity",
            patterns=sample_patterns["creativity"],
            aperture=2.0,
        )
        assert "creativity" in controller.conceptors

        controller.remove_conceptor("creativity")
        assert "creativity" not in controller.conceptors

    def test_blend_conceptors_weighted_sum(self, controller, sample_patterns):
        """Test weighted sum blending."""
        controller.add_conceptor(
            name="creativity",
            patterns=sample_patterns["creativity"],
            aperture=2.0,
            layer=20,
            weight=1.0,
        )
        controller.add_conceptor(
            name="technical",
            patterns=sample_patterns["technical"],
            aperture=2.0,
            layer=20,
            weight=1.0,
        )

        blended = controller.blend_conceptors(layer=20, mode="weighted_sum")
        assert blended is not None
        assert blended.shape == (5120, 5120)

    def test_blend_conceptors_and(self, controller, sample_patterns):
        """Test AND blending (intersection)."""
        controller.add_conceptor(
            name="creativity",
            patterns=sample_patterns["creativity"],
            aperture=2.0,
            layer=20,
        )
        controller.add_conceptor(
            name="technical",
            patterns=sample_patterns["technical"],
            aperture=2.0,
            layer=20,
        )

        blended = controller.blend_conceptors(layer=20, mode="and")
        assert blended is not None
        assert blended.shape == (5120, 5120)

    def test_blend_conceptors_or(self, controller, sample_patterns):
        """Test OR blending (union)."""
        controller.add_conceptor(
            name="creativity",
            patterns=sample_patterns["creativity"],
            aperture=2.0,
            layer=20,
        )
        controller.add_conceptor(
            name="technical",
            patterns=sample_patterns["technical"],
            aperture=2.0,
            layer=20,
        )

        blended = controller.blend_conceptors(layer=20, mode="or")
        assert blended is not None
        assert blended.shape == (5120, 5120)

    def test_apply_to_hidden_states(self, controller, sample_patterns):
        """Test applying conceptors to hidden states."""
        controller.add_conceptor(
            name="creativity",
            patterns=sample_patterns["creativity"],
            aperture=2.0,
            layer=20,
        )

        hidden_states = torch.randn(2, 10, 5120)
        modified = controller.apply_to_hidden_states(hidden_states, layer=20)

        assert modified.shape == hidden_states.shape
        assert not torch.equal(modified, hidden_states)

    def test_create_conceptor_from_vectors(self, controller):
        """Test creating conceptor from list of vectors."""
        vectors = [torch.randn(5120) for _ in range(5)]

        controller.create_conceptor_from_vectors(
            name="creativity",
            vectors=vectors,
            aperture=2.0,
            layer=20,
        )

        assert "creativity" in controller.conceptors


class TestLowRankSubspaceSteering:
    """Tests for LowRankSubspaceSteering."""

    @pytest.fixture
    def steering(self):
        """Create LowRankSubspaceSteering instance."""
        return LowRankSubspaceSteering(rank=64, device="cpu")

    def test_add_subspace(self, steering):
        """Test adding subspace."""
        basis = torch.randn(5120, 100)

        steering.add_subspace(
            name="creative_space",
            basis_vectors=basis,
            layer=20,
            weight=1.0,
        )

        assert "creative_space" in steering.subspaces
        assert steering.subspaces["creative_space"]["basis"].shape == (5120, 64)

    def test_project_to_subspace(self, steering):
        """Test projecting to subspace."""
        basis = torch.randn(5120, 100)
        steering.add_subspace(
            name="creative_space",
            basis_vectors=basis,
            layer=20,
        )

        hidden_states = torch.randn(2, 10, 5120)
        projected = steering.project_to_subspace(hidden_states, "creative_space")

        assert projected.shape == hidden_states.shape

    def test_blend_subspaces(self, steering):
        """Test blending multiple subspaces."""
        basis1 = torch.randn(5120, 100)
        basis2 = torch.randn(5120, 100)

        steering.add_subspace(name="space1", basis_vectors=basis1, layer=20)
        steering.add_subspace(name="space2", basis_vectors=basis2, layer=20)

        hidden_states = torch.randn(2, 10, 5120)
        blended = steering.blend_subspaces(
            hidden_states,
            subspace_names=["space1", "space2"],
            weights=[1.0, 1.0],
        )

        assert blended.shape == hidden_states.shape


class TestHiddenStateProbes:
    """Tests for HiddenStateProbes."""

    @pytest.fixture
    def probe_controller(self):
        """Create ProbeController instance."""
        return ProbeController(device="cpu")

    @pytest.fixture
    def sample_probe(self):
        """Create sample probe."""
        return create_novelty_probe(hidden_dim=5120, layer=20, device="cpu")

    def test_add_probe(self, probe_controller, sample_probe):
        """Test adding probe."""
        probe_controller.add_probe(
            name="novelty",
            probe=sample_probe,
            probe_type=ProbeType.NOVELTY,
            layer=20,
            threshold=0.7,
        )

        assert "novelty" in probe_controller.probes
        assert probe_controller.probes["novelty"][1].threshold == 0.7

    def test_remove_probe(self, probe_controller, sample_probe):
        """Test removing probe."""
        probe_controller.add_probe(
            name="novelty",
            probe=sample_probe,
            probe_type=ProbeType.NOVELTY,
            layer=20,
        )
        assert "novelty" in probe_controller.probes

        probe_controller.remove_probe("novelty")
        assert "novelty" not in probe_controller.probes

    def test_monitor_layer(self, probe_controller, sample_probe):
        """Test monitoring layer with probes."""
        probe_controller.add_probe(
            name="novelty",
            probe=sample_probe,
            probe_type=ProbeType.NOVELTY,
            layer=20,
            threshold=0.7,
        )

        hidden_states = torch.randn(2, 10, 5120)
        values = probe_controller.monitor_layer(hidden_states, layer=20)

        assert "novelty" in values
        assert 0.0 <= values["novelty"] <= 1.0

    def test_register_callback(self, probe_controller, sample_probe):
        """Test registering callback for threshold crossing."""
        probe_controller.add_probe(
            name="novelty",
            probe=sample_probe,
            probe_type=ProbeType.NOVELTY,
            layer=20,
            threshold=0.01,  # Very low threshold to ensure trigger
        )

        callback_called = []

        def callback(name, value):
            callback_called.append((name, value))

        probe_controller.register_callback("novelty", callback)

        # Monitor - sigmoid output will be > 0.01 with random probe weights
        hidden_states = torch.randn(2, 10, 5120)
        probe_controller.monitor_layer(hidden_states, layer=20)

        # Callback should have been called (threshold is very low)
        assert len(callback_called) > 0

    def test_enable_disable_probe(self, probe_controller, sample_probe):
        """Test enabling/disabling probes."""
        probe_controller.add_probe(
            name="novelty",
            probe=sample_probe,
            probe_type=ProbeType.NOVELTY,
            layer=20,
        )

        probe_controller.disable_probe("novelty")
        assert not probe_controller.probes["novelty"][1].active

        probe_controller.enable_probe("novelty")
        assert probe_controller.probes["novelty"][1].active


class TestClosedLoopController:
    """Tests for ClosedLoopController."""

    @pytest.fixture
    def probe_controller(self):
        """Create ProbeController instance."""
        return ProbeController(device="cpu")

    @pytest.fixture
    def closed_loop(self, probe_controller):
        """Create ClosedLoopController instance."""
        return ClosedLoopController(probe_controller, device="cpu")

    def test_add_rule(self, closed_loop):
        """Test adding steering rule."""
        closed_loop.add_rule(
            probe_name="novelty",
            condition="above",
            threshold=0.7,
            action=lambda v: {"novelty_weight": 0.5},
        )

        assert len(closed_loop.rules) == 1

    def test_create_novelty_rule(self, closed_loop):
        """Test creating novelty maintenance rule."""
        closed_loop.create_novelty_rule(low_threshold=0.3, high_threshold=0.7)
        assert len(closed_loop.rules) > 0

    def test_create_coherence_rule(self, closed_loop):
        """Test creating coherence maintenance rule."""
        closed_loop.create_coherence_rule(min_coherence=0.6)
        assert len(closed_loop.rules) > 0

    def test_create_source_similarity_rule(self, closed_loop):
        """Test creating source similarity rule."""
        closed_loop.create_source_similarity_rule(max_similarity=0.4)
        assert len(closed_loop.rules) > 0


class TestDExpertsController:
    """Tests for DExpertsController."""

    @pytest.fixture
    def mock_base_model(self):
        """Create mock base model."""
        model = Mock()
        model.device = "cpu"

        # Mock forward pass
        def mock_forward(input_ids, attention_mask=None):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            vocab_size = 50257

            logits = torch.randn(batch_size, seq_len, vocab_size)
            return Mock(logits=logits)

        model.side_effect = mock_forward
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3])
        tokenizer.decode = Mock(return_value="test")
        return tokenizer

    @pytest.fixture
    def controller(self, mock_base_model, mock_tokenizer):
        """Create DExpertsController instance."""
        return DExpertsController(
            base_model=mock_base_model,
            base_tokenizer=mock_tokenizer,
            device="cpu",
        )

    def test_register_expert(self, controller):
        """Test registering expert."""
        controller.register_expert(
            name="creativity",
            expert_type=ExpertType.CREATIVITY,
            model_path="path/to/model",
            alpha=1.0,
            beta=0.5,
        )

        assert "creativity" in controller.experts
        assert controller.experts["creativity"].alpha == 1.0

    def test_set_phase(self, controller):
        """Test setting generation phase."""
        controller.set_phase("divergence")
        assert controller.current_phase == "divergence"

    def test_set_global_weights(self, controller):
        """Test setting global weights."""
        controller.set_global_weights(alpha=1.5, beta=0.8)
        assert controller.global_alpha == 1.5
        assert controller.global_beta == 0.8

    def test_get_active_experts(self, controller):
        """Test getting active experts for current phase."""
        controller.register_expert(
            name="creativity",
            expert_type=ExpertType.CREATIVITY,
            model_path="path/to/model",
            active_phases=["divergence"],
        )
        controller.register_expert(
            name="technical",
            expert_type=ExpertType.TECHNICAL,
            model_path="path/to/model",
            active_phases=["synthesis"],
        )

        controller.set_phase("divergence")
        active = controller.get_active_experts()
        assert len(active) == 1
        assert active[0][0] == "creativity"

    def test_combine_logits(self, controller):
        """Test combining logits with DExperts formula."""
        base_logits = torch.randn(1, 50257)
        expert_logits = torch.randn(1, 50257)
        anti_expert_logits = torch.randn(1, 50257)

        combined = controller.combine_logits(
            base_logits=base_logits,
            expert_logits=expert_logits,
            anti_expert_logits=anti_expert_logits,
            alpha=1.0,
            beta=0.5,
        )

        assert combined.shape == base_logits.shape

        # Verify formula: base + alpha*(expert-base) - beta*(anti-base)
        expected = (
            base_logits
            + 1.0 * (expert_logits - base_logits)
            - 0.5 * (anti_expert_logits - base_logits)
        )
        assert torch.allclose(combined, expected, atol=1e-5)

    def test_combine_logits_without_anti_expert(self, controller):
        """Test combining logits without anti-expert."""
        base_logits = torch.randn(1, 50257)
        expert_logits = torch.randn(1, 50257)

        combined = controller.combine_logits(
            base_logits=base_logits,
            expert_logits=expert_logits,
            anti_expert_logits=None,
            alpha=1.0,
            beta=0.5,
        )

        assert combined.shape == base_logits.shape

        # Verify formula: base + alpha*(expert-base)
        expected = base_logits + 1.0 * (expert_logits - base_logits)
        assert torch.allclose(combined, expected, atol=1e-5)


class TestIntegration:
    """Integration tests for Sprint 3 components."""

    def test_multi_vector_with_probes(self):
        """Test multi-vector controller with probe feedback."""
        # Create controllers
        multi_ctrl = MultiVectorController(device="cpu")
        probe_ctrl = ProbeController(device="cpu")

        # Add vectors
        novelty_vec = ActivationVector(
            name="novelty",
            vector=torch.randn(5120),
            layer=20,
            scale=1.0
        )
        multi_ctrl.add_vector(
            name="novelty",
            vector=novelty_vec,
            weight=1.0,
        )

        # Add probe
        novelty_probe = create_novelty_probe(hidden_dim=5120, layer=20, device="cpu")
        probe_ctrl.add_probe(
            name="novelty",
            probe=novelty_probe,
            probe_type=ProbeType.NOVELTY,
            layer=20,
            threshold=0.7,
        )

        # Simulate generation loop
        hidden_states = torch.randn(2, 10, 5120)

        # Monitor with probe
        values = probe_ctrl.monitor_layer(hidden_states, layer=20)

        # Adjust multi-vector weights based on probe
        if values["novelty"] < 0.5:
            multi_ctrl.vector_configs["novelty"].weight *= 1.2

        # Apply multi-vector
        modified = multi_ctrl.apply_to_hidden_states(hidden_states, layer=20)
        assert modified.shape == hidden_states.shape

    def test_conceptor_with_closed_loop(self):
        """Test conceptor steering with closed-loop control."""
        conceptor_ctrl = ConceptorController(device="cpu")
        probe_ctrl = ProbeController(device="cpu")
        closed_loop = ClosedLoopController(probe_ctrl, device="cpu")

        # Add conceptor
        patterns = torch.randn(5120, 10)
        conceptor_ctrl.add_conceptor(
            name="creativity",
            patterns=patterns,
            aperture=2.0,
            layer=20,
        )

        # Add probe
        novelty_probe = create_novelty_probe(hidden_dim=5120, layer=20, device="cpu")
        probe_ctrl.add_probe(
            name="novelty",
            probe=novelty_probe,
            probe_type=ProbeType.NOVELTY,
            layer=20,
            threshold=0.7,
        )

        # Create closed-loop rule
        closed_loop.create_novelty_rule(low_threshold=0.3, high_threshold=0.7)

        # Simulate generation
        hidden_states = torch.randn(2, 10, 5120)

        # Apply conceptor
        modified = conceptor_ctrl.apply_to_hidden_states(hidden_states, layer=20)

        # Monitor with probe
        values = probe_ctrl.monitor_layer(modified, layer=20)

        # Closed-loop should have evaluated rules
        assert len(closed_loop.rules) > 0

    def test_full_sprint3_pipeline(self):
        """Test complete Sprint 3 pipeline."""
        # Initialize all controllers
        multi_ctrl = MultiVectorController(device="cpu")
        conceptor_ctrl = ConceptorController(device="cpu")
        probe_ctrl = ProbeController(device="cpu")
        closed_loop = ClosedLoopController(probe_ctrl, device="cpu")

        # Configure multi-vector
        novelty_vec = ActivationVector(
            name="novelty",
            vector=torch.randn(5120),
            layer=20,
            scale=1.0
        )
        concreteness_vec = ActivationVector(
            name="concreteness",
            vector=torch.randn(5120),
            layer=20,
            scale=1.0
        )

        multi_ctrl.add_vector(
            name="novelty",
            vector=novelty_vec,
            weight=1.2,
            active_phases=["divergence"],
        )
        multi_ctrl.add_vector(
            name="concreteness",
            vector=concreteness_vec,
            weight=1.0,
            active_phases=["synthesis"],
        )

        # Configure conceptor
        patterns = torch.randn(5120, 10)
        conceptor_ctrl.add_conceptor(
            name="creativity",
            patterns=patterns,
            aperture=2.0,
            layer=20,
        )

        # Configure probes
        novelty_probe = create_novelty_probe(hidden_dim=5120, layer=20, device="cpu")
        coherence_probe = create_coherence_probe(hidden_dim=5120, layer=20, device="cpu")

        probe_ctrl.add_probe(
            name="novelty",
            probe=novelty_probe,
            probe_type=ProbeType.NOVELTY,
            layer=20,
            threshold=0.7,
        )
        probe_ctrl.add_probe(
            name="coherence",
            probe=coherence_probe,
            probe_type=ProbeType.COHERENCE,
            layer=20,
            threshold=0.6,
        )

        # Configure closed-loop
        closed_loop.create_novelty_rule(low_threshold=0.3, high_threshold=0.7)
        closed_loop.create_coherence_rule(min_coherence=0.6)

        # Simulate generation
        hidden_states = torch.randn(2, 10, 5120)

        # Phase 1: Divergence
        multi_ctrl.set_phase("divergence")
        modified = multi_ctrl.apply_to_hidden_states(hidden_states, layer=20)
        values = probe_ctrl.monitor_layer(modified, layer=20)

        # Phase 2: Synthesis
        multi_ctrl.set_phase("synthesis")
        modified = multi_ctrl.apply_to_hidden_states(hidden_states, layer=20)
        modified = conceptor_ctrl.apply_to_hidden_states(modified, layer=20)
        values = probe_ctrl.monitor_layer(modified, layer=20)

        # Verify all components worked
        assert modified.shape == hidden_states.shape
        assert "novelty" in values
        assert "coherence" in values
        assert len(closed_loop.rules) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
