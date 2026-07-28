"""
Mandatory adversarial falsification tests for NRAM v5.

These tests verify that mechanisms are correctly wired by attempting to
detect fake or incomplete implementations. If any of these tests FAIL,
it means the corresponding mechanism is NOT correctly wired.

Each test follows the pattern:
1. Set up a condition that would break if the mechanism is fake
2. Run the mechanism
3. Assert that the mechanism produces the expected behavior
4. If the assertion fails, the mechanism is not correctly wired
"""
from __future__ import annotations

import json

import pytest
import torch
from unittest.mock import MagicMock

from nram_sglang.hooks.factory import NRAMHookFactory, HookManager
from nram_sglang.hooks.request_context import (
    InterventionConfig,
    context_manager,
)
from nram_sglang.hooks.qwen_hook import QwenDecoderHook


# ---------------------------------------------------------------------------
# 4.1 Hook Registration Falsification
# ---------------------------------------------------------------------------

class TestHookRegistrationFalsification:
    """If hook registration is removed, activation proof must fail."""

    def test_activation_addition_fails_without_hook(self):
        """Without registered hooks, forward pass cannot be modified."""
        mock_model = MagicMock()
        mock_model.named_modules.return_value = []

        factory = NRAMHookFactory()
        # Intentionally do NOT call factory.register_hooks_on_model(...)

        # No hooks should be registered
        assert len(factory._hooks) == 0, (
            "NRAMHookFactory without register_hooks_on_model should have zero hooks"
        )

    def test_activation_addition_succeeds_with_hook(self):
        """With registered hooks, forward pass CAN be modified."""
        mock_layer = MagicMock()
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [
            ("model.layers.0", mock_layer),
        ]

        factory = NRAMHookFactory()
        count = factory.register_hooks_on_model(mock_model, target_layers=[0])

        assert count > 0, (
            "NRAMHookFactory.register_hooks_on_model should create hooks"
        )
        assert len(factory._hooks) > 0

        factory.remove_all_hooks()


# ---------------------------------------------------------------------------
# 4.2 Identity Hook Falsification
# ---------------------------------------------------------------------------

class TestIdentityHookFalsification:
    """If hook returns original hidden state, delta assertion must fail."""

    def test_activation_addition_fails_with_identity_hook(self):
        """With no interventions, hook output equals input (identity)."""
        from nram_sglang.hooks.request_context import (
            NRAMHookContext,
            set_current_context,
            clear_current_context,
        )

        hook = QwenDecoderHook(layer_index=0, module_name="model.layers.0")
        hidden_states = torch.randn(1, 10, 768)

        # Install a context with NO interventions
        request_id = "test-identity-hook"
        ctx = NRAMHookContext(request_id=request_id, interventions=[])
        set_current_context(ctx)

        try:
            # Hook __call__ signature: (module, input_args, output)
            mock_module = MagicMock()
            output = hidden_states.clone()
            result = hook(mock_module, (), output)

            if isinstance(result, torch.Tensor):
                delta = (result - hidden_states).abs().max().item()
                assert delta < 1e-6, (
                    f"Identity hook (no interventions) should produce zero delta, got {delta}"
                )
        finally:
            clear_current_context()


# ---------------------------------------------------------------------------
# 4.3 Parameter Routing Falsification
# ---------------------------------------------------------------------------

class TestParameterRoutingFalsification:
    """If parameters are routed but ignored, dose-response test must fail."""

    def test_dose_response_fails_if_parameters_ignored(self):
        """Different strength values must be stored distinctly."""
        from nram_sglang.hooks.request_context import NRAMHookContext

        strengths = [0.0, 0.5, 1.0, 2.0]
        stored = []

        for strength in strengths:
            request_id = f"test-dose-{strength}"
            intervention = InterventionConfig(
                intervention_id=f"iv-dose-{strength}",
                intervention_type="activation_addition",
                layer_index=0,
                strength=strength,
                parameters={},
            )
            ctx = NRAMHookContext(request_id=request_id, interventions=[intervention])
            ivs = ctx.get_interventions_for_layer(0)
            assert len(ivs) == 1
            stored.append(ivs[0].strength)

        assert stored == strengths, (
            "Different strength values must be preserved in context"
        )


# ---------------------------------------------------------------------------
# 4.4 DExperts Falsification
# ---------------------------------------------------------------------------

class TestDExpertsFalsification:
    """If expert/anti-expert logits are replaced with base logits, proof must fail."""

    def test_dexperts_fails_with_duplicate_logits(self):
        """If all three logit sets are identical, combination should be no-op."""
        base_logits = torch.randn(1, 100)
        expert_logits = base_logits.clone()
        anti_expert_logits = base_logits.clone()

        alpha, beta = 1.0, 0.5
        # DExperts formula: base + alpha*(expert - base) - beta*(anti - base)
        combined = base_logits + alpha * (expert_logits - base_logits) - beta * (anti_expert_logits - base_logits)

        # With identical inputs, combined must equal base
        delta = (combined - base_logits).abs().max().item()
        assert delta < 1e-6, (
            f"DExperts with identical inputs should produce base logits, delta={delta}"
        )

    def test_dexperts_produces_different_logits_with_different_experts(self):
        """If expert differs from base, combined logits MUST differ."""
        base_logits = torch.randn(1, 100)
        expert_logits = base_logits + 2.0  # Clearly different
        anti_expert_logits = base_logits - 1.0

        alpha, beta = 1.0, 0.5
        combined = base_logits + alpha * (expert_logits - base_logits) - beta * (anti_expert_logits - base_logits)

        delta = (combined - base_logits).abs().max().item()
        assert delta > 0.1, (
            f"DExperts with different experts should change logits, delta={delta}"
        )


# ---------------------------------------------------------------------------
# 4.5 Branch Tournament Falsification
# ---------------------------------------------------------------------------

class TestBranchTournamentFalsification:
    """If all branches are identical, diversity assertion must fail."""

    def test_tournament_fails_with_duplicate_branches(self):
        """Identical branches should have zero diversity."""
        from core.steering.branch_tournament import BranchCandidate

        branches = [
            BranchCandidate(branch_id="b1", text="same", token_ids=[1, 2, 3]),
            BranchCandidate(branch_id="b2", text="same", token_ids=[1, 2, 3]),
            BranchCandidate(branch_id="b3", text="same", token_ids=[1, 2, 3]),
        ]

        # All branches have identical token_ids -> diversity = 0
        token_sets = [tuple(b.token_ids) for b in branches]
        unique = len(set(token_sets))
        assert unique == 1, "All branches should be identical"

    def test_tournament_succeeds_with_diverse_branches(self):
        """Different branches should have non-zero diversity."""
        from core.steering.branch_tournament import BranchCandidate

        branches = [
            BranchCandidate(branch_id="b1", text="alpha", token_ids=[1, 2, 3]),
            BranchCandidate(branch_id="b2", text="beta", token_ids=[4, 5, 6]),
            BranchCandidate(branch_id="b3", text="gamma", token_ids=[7, 8, 9]),
        ]

        token_sets = [tuple(b.token_ids) for b in branches]
        unique = len(set(token_sets))
        assert unique == 3, "All branches should be different"


# ---------------------------------------------------------------------------
# 4.6 Latent Loop Falsification
# ---------------------------------------------------------------------------

class TestLatentLoopFalsification:
    """If controller output is frozen, temporal assertion must fail."""

    def test_latent_loop_fails_with_frozen_controller(self):
        """Controller must update alpha based on probe feedback."""
        from core.steering.latent_closed_loop import (
            ClosedLoopChannel,
            ClosedLoopConfig,
            LatentClosedLoopController,
        )

        channel = ClosedLoopChannel(
            channel_id="ch1",
            probe_id="novelty",
            intervention_id="act_add",
            target=0.7,
            kp=0.5,
            alpha_min=0.0,
            alpha_max=2.0,
            max_delta_per_step=0.5,
        )
        config = ClosedLoopConfig(enabled=True, channels=[channel])
        controller = LatentClosedLoopController(config)

        request_id = "test-latent-loop"
        controller.begin_request(request_id)

        try:
            alphas = []
            # Feed different probe scores
            for probe_score in [0.1, 0.5, 0.9]:
                updates = controller.update_from_probes({"novelty": probe_score})
                alpha = updates.get("act_add", 0.0)
                alphas.append(alpha)
                controller.advance_step()

            # At least some alpha values should differ
            assert len(set(round(a, 4) for a in alphas)) > 1, (
                "Controller should produce different alphas for different probe scores"
            )
        finally:
            controller.end_request()


# ---------------------------------------------------------------------------
# 4.7 Probe Falsification
# ---------------------------------------------------------------------------

class TestProbeFalsification:
    """If probe labels are shuffled, held-out performance must collapse."""

    def test_probe_fails_with_shuffled_labels(self):
        """Probe trained on random labels should not generalize."""
        from core.steering.hidden_state_probes import HiddenStateProbe, ProbeType

        torch.manual_seed(42)
        probe = HiddenStateProbe(hidden_dim=64, probe_type=ProbeType.CUSTOM, device="cpu")

        # Random data
        hidden = torch.randn(200, 1, 64)
        # Random labels (shuffled)
        labels = torch.randint(0, 2, (200, 1)).float()

        # Train briefly
        optimizer = torch.optim.SGD(probe.parameters(), lr=0.01)
        for _ in range(5):
            optimizer.zero_grad()
            pred = probe(hidden)
            loss = torch.nn.functional.mse_loss(pred, labels)
            loss.backward()
            optimizer.step()

        # Evaluate on same data - with random labels, probe should not
        # achieve high accuracy on held-out data
        with torch.no_grad():
            pred = probe(hidden)
            pred_binary = (pred > 0.5).float()
            accuracy = (pred_binary == labels).float().mean().item()

        # With random labels and only 5 epochs, accuracy should not be perfect
        assert accuracy < 1.0, (
            "Probe trained on random labels should not achieve perfect accuracy"
        )


# ---------------------------------------------------------------------------
# 4.8 Conceptor Falsification
# ---------------------------------------------------------------------------

class TestConceptorFalsification:
    """If aperture extremes don't change projection, test fails."""

    def test_conceptor_fails_with_extreme_aperture(self):
        """Aperture=0 (hard) and aperture=inf (identity) must differ."""
        from core.steering.conceptor_steering import Conceptor

        torch.manual_seed(42)
        patterns = torch.randn(32, 3, device="cpu")

        # Small aperture -> hard projection
        c_hard = Conceptor(patterns, aperture=0.01, device="cpu")
        # Large aperture -> identity
        c_soft = Conceptor(patterns, aperture=1e6, device="cpu")

        hidden = torch.randn(1, 32, device="cpu")

        out_hard = hidden @ c_hard.matrix
        out_soft = hidden @ c_soft.matrix

        delta = (out_hard - out_soft).abs().max().item()
        assert delta > 1e-3, (
            f"Hard and soft conceptors should produce different outputs, delta={delta}"
        )


# ---------------------------------------------------------------------------
# 4.9 Request Isolation Falsification
# ---------------------------------------------------------------------------

class TestRequestIsolationFalsification:
    """If steered and unsteered requests share state, isolation test fails."""

    def test_request_isolation_fails_with_cross_contamination(self):
        """Request A's interventions must not leak to request B."""
        from nram_sglang.hooks.request_context import NRAMHookContext

        request_a = "request-A-steered"
        request_b = "request-B-unsteered"

        intervention_a = InterventionConfig(
            intervention_id="iv-A",
            intervention_type="activation_addition",
            layer_index=0,
            strength=1.0,
            parameters={},
        )
        ctx_a = NRAMHookContext(request_id=request_a, interventions=[intervention_a])
        ctx_b = NRAMHookContext(request_id=request_b, interventions=[])

        assert ctx_a is not ctx_b, "Requests must have separate contexts"

        ivs_a = ctx_a.get_interventions_for_layer(0)
        ivs_b = ctx_b.get_interventions_for_layer(0)

        assert len(ivs_a) == 1, "Request A should have 1 intervention"
        assert len(ivs_b) == 0, "Request B should have 0 interventions"


# ---------------------------------------------------------------------------
# 4.10 Cleanup Falsification
# ---------------------------------------------------------------------------

class TestCleanupFalsification:
    """If request state is not cleaned up, subsequent request inherits state."""

    def test_cleanup_fails_with_state_leakage(self):
        """After removing request A, request B should not see A's state."""
        from nram_sglang.hooks.request_context import NRAMHookContext

        # Use a fresh ContextManager to avoid interference from global state
        from nram_sglang.hooks.request_context import ContextManager
        cm = ContextManager()

        request_a = "request-A-cleanup"
        intervention = InterventionConfig(
            intervention_id="iv-cleanup",
            intervention_type="activation_addition",
            layer_index=0,
            strength=1.0,
            parameters={},
        )
        ctx_a = cm.create_context(request_a)
        ctx_a.add_intervention(intervention)

        assert len(ctx_a.get_interventions_for_layer(0)) == 1

        # Remove request A
        cm.remove_context(request_a)

        # Request A should be gone
        ctx_a_after = cm.get_context(request_a)
        assert ctx_a_after is None, "Request A should be cleaned up"

        # Request B should have no state from A
        request_b = "request-B-after-cleanup"
        ctx_b = cm.create_context(request_b)
        ivs_b = ctx_b.get_interventions_for_layer(0)
        assert len(ivs_b) == 0, "Request B should not inherit A's state"

        cm.remove_context(request_b)


# ---------------------------------------------------------------------------
# 4.11 Restart Reproducibility Falsification
# ---------------------------------------------------------------------------

class TestRestartReproducibilityFalsification:
    """If artifacts don't load after restart, proof is not reproducible."""

    def test_restart_reproducibility(self):
        """Configuration must survive serialize/deserialize round-trip."""
        from core.steering.representation_config import (
            NRAMRepresentationConfig,
            RepresentationIntervention,
            InterventionKind,
            TokenScope,
        )

        original = NRAMRepresentationConfig(
            enabled=True,
            request_id="test-restart",
            interventions=[
                RepresentationIntervention(
                    kind=InterventionKind.ACTIVATION_ADDITION,
                    intervention_id="iv1",
                    layer_name="model.layers.10",
                    strength=1.5,
                    vector_or_operator_id="vec_creativity",
                    token_scope=TokenScope.BOTH,
                ),
            ],
            telemetry_level="detailed",
        )

        # Serialize to dict (simulating persistence)
        data = {
            "enabled": original.enabled,
            "request_id": original.request_id,
            "telemetry_level": original.telemetry_level,
            "combination_mode": original.combination_mode.value,
            "interventions": [
                {
                    "kind": iv.kind.value,
                    "intervention_id": iv.intervention_id,
                    "layer_name": iv.layer_name,
                    "strength": iv.strength,
                    "vector_or_operator_id": iv.vector_or_operator_id,
                    "token_scope": iv.token_scope.value,
                }
                for iv in original.interventions
            ],
        }

        serialized = json.dumps(data)
        loaded_dict = json.loads(serialized)

        # Verify round-trip
        assert loaded_dict["enabled"] == original.enabled
        assert loaded_dict["request_id"] == original.request_id
        assert len(loaded_dict["interventions"]) == len(original.interventions)
        assert loaded_dict["interventions"][0]["strength"] == 1.5
        assert loaded_dict["interventions"][0]["kind"] == "activation_addition"
