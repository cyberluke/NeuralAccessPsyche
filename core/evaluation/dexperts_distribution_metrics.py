"""
Mathematically correct DExperts distribution metrics.

This module implements exact distribution-level metrics for DExperts steering,
replacing the mislabeled proxy metrics from R5.

Key distinctions:
1. Target-token log-probability change (not KL divergence)
2. Full-vocabulary pre-mask KL divergence
3. Support-conditioned KL divergence after canonical truncation
4. Deployed-distribution divergence (with infinity handling)
5. Signed attribute separation

All metrics use numerically stable implementations (log_softmax, logsumexp).
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DExpertsDistributionMetrics:
    """
    Complete distribution-level metrics for a single DExperts steering operation.
    
    All KL divergences are in nats (natural log).
    All probabilities are in [0, 1].
    All divergences are >= 0 (within numerical tolerance).
    """
    schema_version: str = "1.0"
    
    # Configuration
    alpha: float = 0.0
    temperature: float = 1.0
    
    # Target token information
    target_token_id: int = 0
    target_inside_base_support: bool = True
    
    # Target token log-probabilities
    target_token_base_logp: float = 0.0
    target_token_steered_full_logp: float = 0.0
    target_token_deployed_logp: Optional[float] = None
    target_token_deployed_is_negative_infinity: bool = False
    
    # Target token changes
    target_token_delta_logp_full: float = 0.0
    target_token_delta_logp_deployed: Optional[float] = None
    
    # Full-vocabulary KL divergences (pre-mask)
    kl_full_base_to_steered: float = 0.0
    kl_full_steered_to_base: float = 0.0
    symmetric_kl_full: float = 0.0
    
    # Support-conditioned KL divergences
    kl_support_base_to_steered: float = 0.0
    kl_support_steered_to_base: float = 0.0
    symmetric_kl_support: float = 0.0
    
    # Deployed distribution KL (with infinity handling)
    kl_deployed_base_to_steered: Optional[float] = None
    kl_deployed_base_to_steered_is_infinite: bool = False
    kl_deployed_base_to_steered_reason: Optional[str] = None
    kl_deployed_steered_to_base: float = 0.0
    
    # Jensen-Shannon divergence
    js_base_to_deployed: float = 0.0
    
    # Total variation distance
    tv_base_to_steered_full: float = 0.0
    tv_base_to_deployed: float = 0.0
    
    # Support metrics
    base_support_size: int = 0
    vocab_size: int = 0
    base_support_fraction: float = 0.0
    base_probability_mass_retained: float = 1.0
    base_probability_mass_removed: float = 0.0
    
    # Perturbation magnitudes (logit space)
    logit_delta_l1: float = 0.0
    logit_delta_l2: float = 0.0
    logit_delta_linf: float = 0.0
    expert_anti_delta_l1: float = 0.0
    expert_anti_delta_l2: float = 0.0
    
    # Numerical stability flags
    used_epsilon_in_approximation: bool = False
    approximate_kl_epsilon: Optional[float] = None
    approximate_kl_value: Optional[float] = None


def compute_base_support(
    base_logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Compute canonical base support mask using top-k and top-p filtering.
    
    When both filters are specified, applies intersection (most restrictive).
    
    Args:
        base_logits: Raw base model logits [vocab_size]
        top_k: Keep top-k tokens (0 = disabled)
        top_p: Keep tokens with cumulative probability <= top_p (1.0 = disabled)
    
    Returns:
        Boolean mask [vocab_size] where True = inside support
    """
    vocab_size = base_logits.shape[-1]
    
    # Handle -inf logits by treating them as excluded
    finite_mask = torch.isfinite(base_logits)
    
    # Start with all finite tokens
    support = finite_mask.clone()
    
    # Apply top-k filtering
    if top_k > 0 and top_k < vocab_size:
        # Only consider finite logits for top-k
        finite_logits = base_logits.clone()
        finite_logits[~finite_mask] = float('-inf')
        top_k_values, top_k_indices = torch.topk(finite_logits, min(top_k, finite_mask.sum().item()))
        threshold = top_k_values[-1]
        support = support & (base_logits >= threshold)
    
    # Apply top-p (nucleus) filtering
    if top_p < 1.0:
        # Sort finite logits in descending order
        finite_logits = base_logits.clone()
        finite_logits[~finite_mask] = float('-inf')
        sorted_logits, sorted_indices = torch.sort(finite_logits, descending=True)
        
        # Compute cumulative probabilities only for finite tokens
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        
        # Find cutoff where cumulative probability exceeds top_p
        sorted_mask = cumulative_probs <= top_p
        
        # Always keep at least the first token
        sorted_mask[0] = True
        
        # Map back to original indices
        support_indices = sorted_indices[sorted_mask]
        top_p_support = torch.zeros(vocab_size, dtype=torch.bool, device=base_logits.device)
        top_p_support[support_indices] = True
        
        # Apply intersection with existing support
        support = support & top_p_support
    
    return support


def compute_kl_divergence(
    log_p: torch.Tensor,
    log_q: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute KL(P || Q) = sum_v P(v) * (log P(v) - log Q(v))
    
    Uses numerically stable log-space computation.
    Handles -inf values by skipping tokens where P(v) = 0.
    
    Args:
        log_p: Log probabilities of P [vocab_size] or [support_size]
        log_q: Log probabilities of Q [vocab_size] or [support_size]
        mask: Optional boolean mask to restrict computation
    
    Returns:
        KL divergence in nats (>= 0)
    """
    if mask is not None:
        log_p = log_p[mask]
        log_q = log_q[mask]
    
    # P = exp(log_p), so KL = sum exp(log_p) * (log_p - log_q)
    # Skip tokens where P(v) = 0 (log_p = -inf) to avoid 0 * -inf = NaN
    finite_mask = torch.isfinite(log_p)
    if not finite_mask.all():
        log_p = log_p[finite_mask]
        log_q = log_q[finite_mask]
    
    p = torch.exp(log_p)
    kl = torch.sum(p * (log_p - log_q))
    
    return kl.item()


def compute_jensen_shannon(
    log_p: torch.Tensor,
    log_q: torch.Tensor,
) -> float:
    """
    Compute Jensen-Shannon divergence: JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)
    
    Uses numerically stable log-space computation with logaddexp.
    
    Args:
        log_p: Log probabilities of P [vocab_size]
        log_q: Log probabilities of Q [vocab_size]
    
    Returns:
        JS divergence in nats (>= 0, <= log(2))
    """
    # Compute M = 0.5 * (P + Q) in log space
    # log M = log(0.5 * (exp(log_p) + exp(log_q)))
    #       = log(0.5) + log(exp(log_p) + exp(log_q))
    #       = -log(2) + logaddexp(log_p, log_q)
    log_m = torch.logaddexp(log_p, log_q) - torch.log(torch.tensor(2.0, device=log_p.device))
    
    # KL(P || M) and KL(Q || M)
    kl_p_m = compute_kl_divergence(log_p, log_m)
    kl_q_m = compute_kl_divergence(log_q, log_m)
    
    js = 0.5 * kl_p_m + 0.5 * kl_q_m
    return js


def compute_total_variation(
    log_p: torch.Tensor,
    log_q: torch.Tensor,
) -> float:
    """
    Compute total variation distance: TV(P, Q) = 0.5 * sum_v |P(v) - Q(v)|
    
    Args:
        log_p: Log probabilities of P [vocab_size]
        log_q: Log probabilities of Q [vocab_size]
    
    Returns:
        TV distance in [0, 1]
    """
    p = torch.exp(log_p)
    q = torch.exp(log_q)
    tv = 0.5 * torch.sum(torch.abs(p - q))
    return tv.item()


def compute_dexperts_distribution_metrics(
    base_logits: torch.Tensor,
    expert_logits: torch.Tensor,
    anti_expert_logits: torch.Tensor,
    target_token_id: int,
    alpha: float,
    top_k: int = 0,
    top_p: float = 1.0,
    temperature: float = 1.0,
) -> DExpertsDistributionMetrics:
    """
    Compute complete distribution-level metrics for DExperts steering.
    
    This is the main entry point for Phase 3A metrics computation.
    
    Args:
        base_logits: Raw base model logits [vocab_size]
        expert_logits: Expert adapter logits [vocab_size]
        anti_expert_logits: Anti-expert adapter logits [vocab_size]
        target_token_id: Ground-truth target token ID
        alpha: Steering coefficient
        top_k: Top-k filter for base support (0 = disabled)
        top_p: Top-p filter for base support (1.0 = disabled)
        temperature: Sampling temperature (1.0 = canonical)
    
    Returns:
        DExpertsDistributionMetrics with all computed values
    """
    vocab_size = base_logits.shape[-1]
    device = base_logits.device
    
    # Apply temperature scaling if needed
    if temperature != 1.0:
        base_logits_scaled = base_logits / temperature
        expert_logits_scaled = expert_logits / temperature
        anti_expert_logits_scaled = anti_expert_logits / temperature
    else:
        base_logits_scaled = base_logits
        expert_logits_scaled = expert_logits
        anti_expert_logits_scaled = anti_expert_logits
    
    # Compute expert-anti perturbation
    expert_anti_delta = expert_logits_scaled - anti_expert_logits_scaled
    
    # Compute steered logits (full, pre-mask)
    steered_full_logits = base_logits_scaled + alpha * expert_anti_delta
    
    # Compute canonical base support
    base_support = compute_base_support(base_logits_scaled, top_k=top_k, top_p=top_p)
    base_support_size = base_support.sum().item()
    
    # Compute deployed logits (masked)
    deployed_logits = steered_full_logits.clone()
    deployed_logits[~base_support] = float('-inf')
    
    # Compute log probabilities (stable)
    base_logp = F.log_softmax(base_logits_scaled, dim=-1)
    steered_full_logp = F.log_softmax(steered_full_logits, dim=-1)
    
    # Deployed distribution: need to handle -inf carefully
    # For tokens in support, use steered logits
    # For tokens outside support, probability is 0 (log prob is -inf)
    deployed_logp = F.log_softmax(deployed_logits, dim=-1)
    
    # Target token metrics
    target_base_logp = base_logp[target_token_id].item()
    target_steered_full_logp = steered_full_logp[target_token_id].item()
    target_inside_support = base_support[target_token_id].item()
    
    if target_inside_support:
        target_deployed_logp = deployed_logp[target_token_id].item()
        target_deployed_is_inf = False
        target_delta_deployed = target_deployed_logp - target_base_logp
    else:
        target_deployed_logp = None
        target_deployed_is_inf = True
        target_delta_deployed = None
    
    target_delta_full = target_steered_full_logp - target_base_logp
    
    # Full-vocabulary KL divergences (pre-mask)
    kl_full_base_to_steered = compute_kl_divergence(base_logp, steered_full_logp)
    kl_full_steered_to_base = compute_kl_divergence(steered_full_logp, base_logp)
    symmetric_kl_full = 0.5 * (kl_full_base_to_steered + kl_full_steered_to_base)
    
    # Support-conditioned KL divergences
    # Restrict to tokens inside base support
    base_logp_support = F.log_softmax(base_logits_scaled[base_support], dim=-1)
    steered_logp_support = F.log_softmax(steered_full_logits[base_support], dim=-1)
    
    kl_support_base_to_steered = compute_kl_divergence(base_logp_support, steered_logp_support)
    kl_support_steered_to_base = compute_kl_divergence(steered_logp_support, base_logp_support)
    symmetric_kl_support = 0.5 * (kl_support_base_to_steered + kl_support_steered_to_base)
    
    # Deployed distribution KL
    # KL(base || deployed) is infinite if base assigns probability outside support
    base_prob_outside_support = torch.exp(base_logp[~base_support]).sum().item()
    
    if base_prob_outside_support > 1e-10:
        # Base assigns non-zero probability outside support, but deployed is zero there
        kl_deployed_base_to_steered = None
        kl_deployed_base_to_steered_is_infinite = True
        kl_deployed_base_to_steered_reason = (
            f"Base distribution assigns {base_prob_outside_support:.6e} probability "
            f"outside canonical support, but deployed distribution is zero there"
        )
    else:
        # Numerically, base probability outside support is negligible
        kl_deployed_base_to_steered = compute_kl_divergence(base_logp, deployed_logp)
        kl_deployed_base_to_steered_is_infinite = False
        kl_deployed_base_to_steered_reason = None
    
    # KL(deployed || base) is finite only over tokens where deployed has non-zero probability
    # Deployed has -inf logits outside support, so we only compute over support tokens
    deployed_finite_mask = torch.isfinite(deployed_logp)
    if deployed_finite_mask.any():
        kl_deployed_steered_to_base = compute_kl_divergence(
            deployed_logp[deployed_finite_mask],
            base_logp[deployed_finite_mask]
        )
    else:
        kl_deployed_steered_to_base = 0.0
    
    # Jensen-Shannon divergence
    js_base_to_deployed = compute_jensen_shannon(base_logp, deployed_logp)
    
    # Total variation distance
    tv_base_to_steered_full = compute_total_variation(base_logp, steered_full_logp)
    tv_base_to_deployed = compute_total_variation(base_logp, deployed_logp)
    
    # Support metrics
    base_prob_retained = torch.exp(base_logp[base_support]).sum().item()
    base_prob_removed = 1.0 - base_prob_retained
    
    # Perturbation magnitudes (logit space)
    logit_delta = steered_full_logits - base_logits_scaled
    logit_delta_l1 = torch.sum(torch.abs(logit_delta)).item()
    logit_delta_l2 = torch.sqrt(torch.sum(logit_delta ** 2)).item()
    logit_delta_linf = torch.max(torch.abs(logit_delta)).item()
    
    expert_anti_l1 = torch.sum(torch.abs(expert_anti_delta)).item()
    expert_anti_l2 = torch.sqrt(torch.sum(expert_anti_delta ** 2)).item()
    
    return DExpertsDistributionMetrics(
        schema_version="1.0",
        alpha=alpha,
        temperature=temperature,
        target_token_id=target_token_id,
        target_inside_base_support=target_inside_support,
        target_token_base_logp=target_base_logp,
        target_token_steered_full_logp=target_steered_full_logp,
        target_token_deployed_logp=target_deployed_logp,
        target_token_deployed_is_negative_infinity=target_deployed_is_inf,
        target_token_delta_logp_full=target_delta_full,
        target_token_delta_logp_deployed=target_delta_deployed,
        kl_full_base_to_steered=kl_full_base_to_steered,
        kl_full_steered_to_base=kl_full_steered_to_base,
        symmetric_kl_full=symmetric_kl_full,
        kl_support_base_to_steered=kl_support_base_to_steered,
        kl_support_steered_to_base=kl_support_steered_to_base,
        symmetric_kl_support=symmetric_kl_support,
        kl_deployed_base_to_steered=kl_deployed_base_to_steered,
        kl_deployed_base_to_steered_is_infinite=kl_deployed_base_to_steered_is_infinite,
        kl_deployed_base_to_steered_reason=kl_deployed_base_to_steered_reason,
        kl_deployed_steered_to_base=kl_deployed_steered_to_base,
        js_base_to_deployed=js_base_to_deployed,
        tv_base_to_steered_full=tv_base_to_steered_full,
        tv_base_to_deployed=tv_base_to_deployed,
        base_support_size=base_support_size,
        vocab_size=vocab_size,
        base_support_fraction=base_support_size / vocab_size,
        base_probability_mass_retained=base_prob_retained,
        base_probability_mass_removed=base_prob_removed,
        logit_delta_l1=logit_delta_l1,
        logit_delta_l2=logit_delta_l2,
        logit_delta_linf=logit_delta_linf,
        expert_anti_delta_l1=expert_anti_l1,
        expert_anti_delta_l2=expert_anti_l2,
    )


def validate_metrics(metrics: DExpertsDistributionMetrics, tolerance: float = 1e-4) -> Tuple[bool, list]:
    """
    Validate that computed metrics satisfy mathematical invariants.
    
    Args:
        metrics: Computed metrics
        tolerance: Numerical tolerance for floating-point comparisons
    
    Returns:
        (is_valid, list_of_violations)
    """
    violations = []
    
    # KL >= -tolerance (should be >= 0, but allow small negative due to FP error)
    if metrics.kl_full_base_to_steered < -tolerance:
        violations.append(f"kl_full_base_to_steered = {metrics.kl_full_base_to_steered} < -{tolerance}")
    
    if metrics.kl_full_steered_to_base < -tolerance:
        violations.append(f"kl_full_steered_to_base = {metrics.kl_full_steered_to_base} < -{tolerance}")
    
    if metrics.symmetric_kl_full < -tolerance:
        violations.append(f"symmetric_kl_full = {metrics.symmetric_kl_full} < -{tolerance}")
    
    # JS >= -tolerance and JS <= log(2) + tolerance
    if metrics.js_base_to_deployed < -tolerance:
        violations.append(f"js_base_to_deployed = {metrics.js_base_to_deployed} < -{tolerance}")
    
    import math
    if metrics.js_base_to_deployed > math.log(2) + tolerance:
        violations.append(f"js_base_to_deployed = {metrics.js_base_to_deployed} > log(2) + {tolerance}")
    
    # TV in [0, 1]
    if metrics.tv_base_to_steered_full < -tolerance or metrics.tv_base_to_steered_full > 1 + tolerance:
        violations.append(f"tv_base_to_steered_full = {metrics.tv_base_to_steered_full} not in [0, 1]")
    
    if metrics.tv_base_to_deployed < -tolerance or metrics.tv_base_to_deployed > 1 + tolerance:
        violations.append(f"tv_base_to_deployed = {metrics.tv_base_to_deployed} not in [0, 1]")
    
    # Probability mass sums to 1
    mass_sum = metrics.base_probability_mass_retained + metrics.base_probability_mass_removed
    if abs(mass_sum - 1.0) > tolerance:
        violations.append(f"probability mass sum = {mass_sum} != 1.0")
    
    # Support size consistency
    if metrics.base_support_size < 0 or metrics.base_support_size > metrics.vocab_size:
        violations.append(f"base_support_size = {metrics.base_support_size} not in [0, {metrics.vocab_size}]")
    
    is_valid = len(violations) == 0
    return is_valid, violations
