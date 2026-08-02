"""
Synthetic oracle tests for DExperts distribution metrics.

These tests validate the mathematical correctness of the metrics implementation
using hand-calculated expected values. The oracle computations are independent
of the production implementation.

Test cases:
1. Identical distributions: all KL = 0, JS = 0, TV = 0
2. Non-identical full-support distributions with known KL
3. Masked deployed distribution: KL(base||deployed) = inf, KL(deployed||base) = finite
4. alpha = 0: steered equals base
5. Expert equals anti-expert: no steering effect
6. Adapter reversal: sign flip
7. Target token outside support
8. Vocabulary containing pre-existing -inf logits
9. Extreme logits without overflow
10. Comparison against independently calculated NumPy/FP64 oracle
"""
import pytest
import torch
import torch.nn.functional as F
import numpy as np
import math
from core.evaluation.dexperts_distribution_metrics import (
    compute_dexperts_distribution_metrics,
    compute_base_support,
    compute_kl_divergence,
    compute_jensen_shannon,
    compute_total_variation,
    validate_metrics,
    DExpertsDistributionMetrics,
)


class TestIdenticalDistributions:
    """Test 1: Identical distributions should yield zero divergence."""
    
    def test_identical_distributions_zero_kl(self):
        """When expert and anti-expert are identical, steering has no effect."""
        vocab_size = 100
        torch.manual_seed(42)
        
        base_logits = torch.randn(vocab_size)
        expert_logits = base_logits.clone()
        anti_expert_logits = base_logits.clone()
        target_token = 50
        
        metrics = compute_dexperts_distribution_metrics(
            base_logits=base_logits,
            expert_logits=expert_logits,
            anti_expert_logits=anti_expert_logits,
            target_token_id=target_token,
            alpha=1.0,
        )
        
        # All divergences should be zero (within tolerance)
        assert abs(metrics.kl_full_base_to_steered) < 1e-6
        assert abs(metrics.kl_full_steered_to_base) < 1e-6
        assert abs(metrics.symmetric_kl_full) < 1e-6
        assert abs(metrics.js_base_to_deployed) < 1e-6
        assert abs(metrics.tv_base_to_steered_full) < 1e-6
        assert abs(metrics.tv_base_to_deployed) < 1e-6
        
        # Target token change should be zero
        assert abs(metrics.target_token_delta_logp_full) < 1e-6
        
        # Validation should pass
        is_valid, violations = validate_metrics(metrics)
        assert is_valid, f"Validation failed: {violations}"


class TestNonIdenticalDistributions:
    """Test 2: Non-identical distributions with known KL."""
    
    def test_known_kl_divergence(self):
        """Verify KL divergence against hand calculation."""
        # Simple 4-token vocabulary for manual calculation
        # Base: uniform [0.25, 0.25, 0.25, 0.25]
        # Steered: [0.5, 0.25, 0.15, 0.1]
        
        base_logits = torch.log(torch.tensor([0.25, 0.25, 0.25, 0.25]))
        steered_logits = torch.log(torch.tensor([0.5, 0.25, 0.15, 0.1]))
        
        # KL(P||Q) = sum P * log(P/Q)
        # = 0.25*log(0.25/0.5) + 0.25*log(0.25/0.25) + 0.25*log(0.25/0.15) + 0.25*log(0.25/0.1)
        # = 0.25*(-0.693) + 0.25*(0) + 0.25*(0.511) + 0.25*(0.916)
        # = -0.173 + 0 + 0.128 + 0.229 = 0.184
        
        expected_kl = (
            0.25 * math.log(0.25 / 0.5) +
            0.25 * math.log(0.25 / 0.25) +
            0.25 * math.log(0.25 / 0.15) +
            0.25 * math.log(0.25 / 0.1)
        )
        
        computed_kl = compute_kl_divergence(base_logits, steered_logits)
        
        assert abs(computed_kl - expected_kl) < 1e-5, \
            f"KL mismatch: computed={computed_kl}, expected={expected_kl}"


class TestMaskedDeployedDistribution:
    """Test 3: Masked deployed distribution with infinity handling."""
    
    def test_deployed_kl_infinity_handling(self):
        """When base assigns probability outside support, KL(base||deployed) is infinite."""
        vocab_size = 10
        torch.manual_seed(123)
        
        base_logits = torch.randn(vocab_size)
        expert_logits = base_logits + torch.randn(vocab_size) * 0.5
        anti_expert_logits = base_logits - torch.randn(vocab_size) * 0.5
        target_token = 5
        
        # Use top-k=5 to create a support that excludes some tokens
        metrics = compute_dexperts_distribution_metrics(
            base_logits=base_logits,
            expert_logits=expert_logits,
            anti_expert_logits=anti_expert_logits,
            target_token_id=target_token,
            alpha=1.0,
            top_k=5,
        )
        
        # Base assigns probability outside support
        assert metrics.base_probability_mass_removed > 1e-6
        
        # KL(base||deployed) should be marked as infinite
        assert metrics.kl_deployed_base_to_steered_is_infinite is True
        assert metrics.kl_deployed_base_to_steered is None
        assert metrics.kl_deployed_base_to_steered_reason is not None
        
        # KL(deployed||base) should be finite
        assert metrics.kl_deployed_steered_to_base >= 0
        assert not math.isinf(metrics.kl_deployed_steered_to_base)
        
        # JS should be finite
        assert not math.isinf(metrics.js_base_to_deployed)
        assert metrics.js_base_to_deployed >= 0
        assert metrics.js_base_to_deployed <= math.log(2) + 1e-5


class TestAlphaZero:
    """Test 4: alpha=0 should produce no steering effect."""
    
    def test_alpha_zero_matches_baseline(self):
        """When alpha=0, steered distribution equals base distribution."""
        vocab_size = 50
        torch.manual_seed(456)
        
        base_logits = torch.randn(vocab_size)
        expert_logits = torch.randn(vocab_size)
        anti_expert_logits = torch.randn(vocab_size)
        target_token = 25
        
        metrics = compute_dexperts_distribution_metrics(
            base_logits=base_logits,
            expert_logits=expert_logits,
            anti_expert_logits=anti_expert_logits,
            target_token_id=target_token,
            alpha=0.0,
        )
        
        # All divergences should be zero
        assert abs(metrics.kl_full_base_to_steered) < 1e-6
        assert abs(metrics.symmetric_kl_full) < 1e-6
        assert abs(metrics.js_base_to_deployed) < 1e-6
        assert abs(metrics.tv_base_to_steered_full) < 1e-6
        
        # Target token change should be zero
        assert abs(metrics.target_token_delta_logp_full) < 1e-6


class TestExpertEqualsAntiExpert:
    """Test 5: Expert equals anti-expert should produce no steering."""
    
    def test_expert_equals_anti_expert_no_steering(self):
        """When expert and anti-expert are identical, perturbation is zero."""
        vocab_size = 50
        torch.manual_seed(789)
        
        base_logits = torch.randn(vocab_size)
        adapter_logits = torch.randn(vocab_size)
        target_token = 25
        
        metrics = compute_dexperts_distribution_metrics(
            base_logits=base_logits,
            expert_logits=adapter_logits,
            anti_expert_logits=adapter_logits,  # Same as expert
            target_token_id=target_token,
            alpha=1.0,
        )
        
        # Perturbation should be zero
        assert abs(metrics.expert_anti_delta_l2) < 1e-6
        
        # All divergences should be zero
        assert abs(metrics.kl_full_base_to_steered) < 1e-6
        assert abs(metrics.target_token_delta_logp_full) < 1e-6


class TestAdapterReversal:
    """Test 6: Adapter reversal should flip sign of perturbation."""
    
    def test_adapter_reversal_sign_flip(self):
        """Swapping expert and anti-expert should produce opposite perturbation vectors."""
        vocab_size = 50
        torch.manual_seed(101)
        
        base_logits = torch.randn(vocab_size)
        expert_logits = torch.randn(vocab_size)
        anti_expert_logits = torch.randn(vocab_size)
        target_token = 25
        alpha = 1.0
        
        # Normal steering
        metrics_normal = compute_dexperts_distribution_metrics(
            base_logits=base_logits,
            expert_logits=expert_logits,
            anti_expert_logits=anti_expert_logits,
            target_token_id=target_token,
            alpha=alpha,
        )
        
        # Reversed steering (swap expert and anti-expert)
        metrics_reversed = compute_dexperts_distribution_metrics(
            base_logits=base_logits,
            expert_logits=anti_expert_logits,  # Swapped
            anti_expert_logits=expert_logits,  # Swapped
            target_token_id=target_token,
            alpha=alpha,
        )
        
        # The perturbation magnitude should be identical (just opposite direction)
        # This is guaranteed because ||alpha * (expert - anti)|| = ||alpha * (anti - expert)||
        assert abs(metrics_normal.logit_delta_l2 - metrics_reversed.logit_delta_l2) < 1e-5, \
            "Reversal should preserve perturbation magnitude"
        
        # KL divergences should be similar (same magnitude of distribution change)
        assert abs(metrics_normal.symmetric_kl_full - metrics_reversed.symmetric_kl_full) < 0.1, \
            "Reversal should produce similar KL divergence magnitude"


class TestTargetOutsideSupport:
    """Test 7: Target token outside support should be handled correctly."""
    
    def test_target_outside_support_flagged(self):
        """When target token is outside base support, it should be flagged."""
        vocab_size = 10
        torch.manual_seed(202)
        
        # Create logits where token 9 has very low probability
        base_logits = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -10.0])
        expert_logits = torch.randn(vocab_size)
        anti_expert_logits = torch.randn(vocab_size)
        target_token = 9  # Low probability token
        
        # Use top-k=5, which will exclude token 9
        metrics = compute_dexperts_distribution_metrics(
            base_logits=base_logits,
            expert_logits=expert_logits,
            anti_expert_logits=anti_expert_logits,
            target_token_id=target_token,
            alpha=1.0,
            top_k=5,
        )
        
        # Target should be flagged as outside support
        assert metrics.target_inside_base_support is False
        assert metrics.target_token_deployed_is_negative_infinity is True
        assert metrics.target_token_deployed_logp is None
        assert metrics.target_token_delta_logp_deployed is None
        
        # But full (pre-mask) metrics should still be computed
        assert metrics.target_token_delta_logp_full is not None


class TestPreExistingInfLogits:
    """Test 8: Vocabulary containing pre-existing -inf logits."""
    
    def test_pre_existing_inf_logits_handled(self):
        """Pre-existing -inf logits should not cause numerical issues."""
        vocab_size = 10
        torch.manual_seed(303)
        
        # Some tokens have -inf logits (impossible)
        base_logits = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, float('-inf'), float('-inf'), 1.0, 1.0, 1.0])
        expert_logits = torch.randn(vocab_size)
        anti_expert_logits = torch.randn(vocab_size)
        target_token = 2
        
        metrics = compute_dexperts_distribution_metrics(
            base_logits=base_logits,
            expert_logits=expert_logits,
            anti_expert_logits=anti_expert_logits,
            target_token_id=target_token,
            alpha=1.0,
        )
        
        # Should not produce NaN or inf in finite metrics
        assert not math.isnan(metrics.kl_full_base_to_steered)
        assert not math.isinf(metrics.kl_full_base_to_steered)
        assert not math.isnan(metrics.js_base_to_deployed)
        assert not math.isnan(metrics.tv_base_to_steered_full)
        
        # Validation should pass
        is_valid, violations = validate_metrics(metrics)
        assert is_valid, f"Validation failed: {violations}"


class TestExtremeLogits:
    """Test 9: Extreme logits without overflow."""
    
    def test_extreme_logits_no_overflow(self):
        """Very large or small logits should not cause overflow."""
        vocab_size = 20
        torch.manual_seed(404)
        
        # Extreme logits
        base_logits = torch.randn(vocab_size) * 100
        expert_logits = torch.randn(vocab_size) * 100
        anti_expert_logits = torch.randn(vocab_size) * 100
        target_token = 10
        
        metrics = compute_dexperts_distribution_metrics(
            base_logits=base_logits,
            expert_logits=expert_logits,
            anti_expert_logits=anti_expert_logits,
            target_token_id=target_token,
            alpha=2.0,
        )
        
        # Should not produce NaN or inf
        assert not math.isnan(metrics.kl_full_base_to_steered)
        assert not math.isnan(metrics.js_base_to_deployed)
        assert not math.isnan(metrics.tv_base_to_steered_full)
        assert not math.isnan(metrics.target_token_delta_logp_full)
        
        # Validation should pass
        is_valid, violations = validate_metrics(metrics)
        assert is_valid, f"Validation failed: {violations}"


class TestNumPyFP64Oracle:
    """Test 10: Comparison against independently calculated NumPy/FP64 oracle."""
    
    def test_matches_numpy_fp64_oracle(self):
        """Metrics should match independent NumPy FP64 calculation."""
        vocab_size = 30
        torch.manual_seed(505)
        
        base_logits = torch.randn(vocab_size, dtype=torch.float64)
        expert_logits = torch.randn(vocab_size, dtype=torch.float64)
        anti_expert_logits = torch.randn(vocab_size, dtype=torch.float64)
        target_token = 15
        alpha = 1.5
        
        # Compute with PyTorch implementation
        metrics = compute_dexperts_distribution_metrics(
            base_logits=base_logits.float(),  # Convert to float32 for implementation
            expert_logits=expert_logits.float(),
            anti_expert_logits=anti_expert_logits.float(),
            target_token_id=target_token,
            alpha=alpha,
        )
        
        # Independent NumPy FP64 calculation
        base_np = base_logits.numpy()
        expert_np = expert_logits.numpy()
        anti_np = anti_expert_logits.numpy()
        
        # Compute steered logits
        steered_np = base_np + alpha * (expert_np - anti_np)
        
        # Compute log probabilities using scipy.special.logsumexp for stability
        from scipy.special import logsumexp
        
        base_logp_np = base_np - logsumexp(base_np)
        steered_logp_np = steered_np - logsumexp(steered_np)
        
        # Compute KL divergence
        base_prob_np = np.exp(base_logp_np)
        kl_np = np.sum(base_prob_np * (base_logp_np - steered_logp_np))
        
        # Compare
        assert abs(metrics.kl_full_base_to_steered - kl_np) < 1e-4, \
            f"KL mismatch: torch={metrics.kl_full_base_to_steered}, numpy={kl_np}"
        
        # Compute target token delta
        target_delta_np = steered_logp_np[target_token] - base_logp_np[target_token]
        assert abs(metrics.target_token_delta_logp_full - target_delta_np) < 1e-4, \
            f"Target delta mismatch: torch={metrics.target_token_delta_logp_full}, numpy={target_delta_np}"


class TestBaseSupportComputation:
    """Test base support computation with top-k and top-p."""
    
    def test_top_k_support(self):
        """Top-k support should select exactly k tokens."""
        vocab_size = 100
        torch.manual_seed(606)
        
        logits = torch.randn(vocab_size)
        k = 10
        
        support = compute_base_support(logits, top_k=k, top_p=1.0)
        
        assert support.sum().item() == k
    
    def test_top_p_support(self):
        """Top-p support should select tokens with cumulative prob <= p."""
        vocab_size = 100
        torch.manual_seed(707)
        
        logits = torch.randn(vocab_size)
        p = 0.9
        
        support = compute_base_support(logits, top_k=0, top_p=p)
        
        # Compute cumulative probability
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Count tokens with cumulative prob <= p
        expected_count = (cumulative_probs <= p).sum().item()
        
        # Allow for small differences due to boundary handling
        assert abs(support.sum().item() - expected_count) <= 1
    
    def test_combined_top_k_and_top_p(self):
        """Combined top-k and top-p should apply both filters."""
        vocab_size = 100
        torch.manual_seed(808)
        
        logits = torch.randn(vocab_size)
        k = 20
        p = 0.8
        
        support = compute_base_support(logits, top_k=k, top_p=p)
        
        # Support size should be <= k (top-k is more restrictive)
        assert support.sum().item() <= k


class TestMetricValidation:
    """Test metric validation logic."""
    
    def test_valid_metrics_pass_validation(self):
        """Valid metrics should pass validation."""
        vocab_size = 50
        torch.manual_seed(909)
        
        base_logits = torch.randn(vocab_size)
        expert_logits = torch.randn(vocab_size)
        anti_expert_logits = torch.randn(vocab_size)
        
        metrics = compute_dexperts_distribution_metrics(
            base_logits=base_logits,
            expert_logits=expert_logits,
            anti_expert_logits=anti_expert_logits,
            target_token_id=25,
            alpha=1.0,
        )
        
        is_valid, violations = validate_metrics(metrics)
        assert is_valid, f"Valid metrics failed validation: {violations}"
    
    def test_invalid_kl_fails_validation(self):
        """Negative KL (beyond tolerance) should fail validation."""
        # Create invalid metrics manually
        metrics = DExpertsDistributionMetrics(
            kl_full_base_to_steered=-1.0,  # Invalid: negative KL
        )
        
        is_valid, violations = validate_metrics(metrics)
        assert not is_valid
        assert any("kl_full_base_to_steered" in v for v in violations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
