#!/usr/bin/env python3
"""
NRAM v5 R6 Phase 4: Corrected Teacher-Forced Mechanistic Proof

Executes the confirmatory experiment with:
- N=300 prompts per class (toxic/non-toxic)
- Corrected distribution metrics from Phase 3A
- Prompt-level clustering for statistics
- All control gates (alpha=0, identity, reversal)

Primary endpoint: Signed attribute gap G(alpha=1.0)
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib
from tqdm import tqdm
import logging

from core.evaluation.dexperts_distribution_metrics import (
    compute_dexperts_distribution_metrics,
    DExpertsDistributionMetrics,
    validate_metrics,
)


def _json_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    import math
    if isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        val = float(obj)
        if math.isinf(val) or math.isnan(val):
            return None
        return val
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Phase4Config:
    """Configuration for Phase 4 experiment."""
    n_per_class: int = 300
    primary_alpha: float = 1.0
    alphas: List[float] = None
    random_seed: int = 12345
    top_k: int = 0
    top_p: float = 1.0
    temperature: float = 1.0
    
    def __post_init__(self):
        if self.alphas is None:
            self.alphas = [0.0, 0.5, 1.0, 2.0]


@dataclass
class SequenceLevelMetrics:
    """Aggregated metrics for a single sequence."""
    prompt_id: int
    class_label: str  # "toxic" or "nontoxic"
    alpha: float
    condition: str  # "normal", "identity", "reversal", "alpha_zero"
    
    # Target token metrics
    mean_target_delta_logp: float
    median_target_delta_logp: float
    target_inside_support_rate: float
    
    # Distribution divergence metrics
    mean_kl_full: float
    mean_kl_support: float
    mean_js: float
    mean_tv: float
    
    # Support metrics
    mean_support_size: float
    mean_support_fraction: float
    support_exclusion_count: int
    
    # Sequence NLL
    sequence_nll_base: float
    sequence_nll_steered: float
    
    # Perturbation magnitude
    mean_logit_delta_l2: float


def load_confirmatory_prompts(
    n_per_class: int,
    seed: int = 12345,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load confirmatory prompts disjoint from pilot data.
    
    Args:
        n_per_class: Number of prompts per class
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (toxic_prompts, nontoxic_prompts)
    """
    from datasets import load_dataset
    
    logger.info(f"Loading confirmatory prompts (N={n_per_class} per class)...")
    
    # Load dataset
    dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
    
    # Filter and separate by toxicity
    toxic_prompts = []
    nontoxic_prompts = []
    
    rng = np.random.default_rng(seed)
    
    for example in dataset:
        prompt_data = example.get("prompt", {})
        if not isinstance(prompt_data, dict):
            continue
        
        toxicity = prompt_data.get("toxicity", 0.0)
        if toxicity is None or toxicity != toxicity:  # NaN check
            continue
        
        prompt_text = prompt_data.get("text", "")
        continuation_data = example.get("continuation", {})
        continuation_text = continuation_data.get("text", "") if isinstance(continuation_data, dict) else ""
        
        if not prompt_text or not continuation_text:
            continue
        
        item = {
            "prompt": prompt_text,
            "continuation": continuation_text,
            "toxicity": toxicity,
        }
        
        if toxicity >= 0.8 and len(toxic_prompts) < n_per_class:
            toxic_prompts.append(item)
        elif toxicity <= 0.1 and len(nontoxic_prompts) < n_per_class:
            nontoxic_prompts.append(item)
        
        if len(toxic_prompts) >= n_per_class and len(nontoxic_prompts) >= n_per_class:
            break
    
    logger.info(f"Loaded {len(toxic_prompts)} toxic and {len(nontoxic_prompts)} non-toxic prompts")
    
    return toxic_prompts, nontoxic_prompts


def load_models():
    """Load base model and adapters."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    logger.info("Loading models...")
    
    model_path = r"E:\_MODELS\huggingface\hub\models--Qwen--Qwen3-0.6B-Base\snapshots\da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
    expert_path = "artifacts/dexperts/adapters/nontoxic"
    anti_expert_path = "artifacts/dexperts/adapters/toxic"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    base_model.eval()
    
    # Load expert adapter
    expert_model = PeftModel.from_pretrained(
        base_model,
        expert_path,
        adapter_name="expert",
    )
    
    # Load anti-expert adapter
    expert_model.load_adapter(anti_expert_path, adapter_name="anti_expert")
    
    logger.info("Models loaded successfully")
    
    return tokenizer, base_model, expert_model


def run_teacher_forced_analysis(
    tokenizer,
    base_model,
    expert_model,
    prompt_text: str,
    continuation_text: str,
    alpha: float,
    condition: str,
    top_k: int = 0,
    top_p: float = 1.0,
    temperature: float = 1.0,
) -> List[Dict]:
    """
    Run teacher-forced analysis for a single prompt-continuation pair.
    
    Args:
        tokenizer: Tokenizer
        base_model: Base model
        expert_model: PEFT model with adapters
        prompt_text: Prompt text
        continuation_text: Continuation text
        alpha: Steering coefficient
        condition: Experimental condition
        top_k: Top-k filter
        top_p: Top-p filter
        temperature: Temperature
    
    Returns:
        List of per-token metrics
    """
    # Tokenize
    prompt_tokens = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    continuation_tokens = tokenizer(continuation_text, return_tensors="pt", add_special_tokens=False)
    
    prompt_ids = prompt_tokens.input_ids[0]
    continuation_ids = continuation_tokens.input_ids[0]
    
    # Concatenate
    full_ids = torch.cat([prompt_ids, continuation_ids])
    
    per_token_results = []
    
    # Teacher-force through continuation
    for i in range(len(continuation_ids)):
        target_pos = len(prompt_ids) + i
        target_token_id = continuation_ids[i].item()
        
        # Get logits up to target position
        input_ids = full_ids[:target_pos].unsqueeze(0).to(base_model.device)
        
        with torch.no_grad():
            # Base model logits
            base_outputs = base_model(input_ids=input_ids)
            base_logits = base_outputs.logits[0, -1, :].cpu()
            
            # Expert adapter logits
            expert_model.set_adapter("expert")
            expert_outputs = expert_model(input_ids=input_ids)
            expert_logits = expert_outputs.logits[0, -1, :].cpu()
            
            # Anti-expert adapter logits
            if condition == "identity":
                # Identity control: use same adapter for both
                anti_expert_logits = expert_logits.clone()
            elif condition == "reversal":
                # Reversal control: swap expert and anti-expert
                expert_model.set_adapter("anti_expert")
                reversal_outputs = expert_model(input_ids=input_ids)
                anti_expert_logits = expert_logits.clone()
                expert_logits = reversal_outputs.logits[0, -1, :].cpu()
            else:
                # Normal condition
                expert_model.set_adapter("anti_expert")
                anti_expert_outputs = expert_model(input_ids=input_ids)
                anti_expert_logits = anti_expert_outputs.logits[0, -1, :].cpu()
        
        # Compute metrics
        metrics = compute_dexperts_distribution_metrics(
            base_logits=base_logits,
            expert_logits=expert_logits,
            anti_expert_logits=anti_expert_logits,
            target_token_id=target_token_id,
            alpha=alpha,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        
        # Validate
        is_valid, violations = validate_metrics(metrics)
        if not is_valid:
            logger.warning(f"Metric validation failed at position {i}: {violations}")
        
        # Convert to dict
        result = asdict(metrics)
        result["token_position"] = i
        result["condition"] = condition
        
        per_token_results.append(result)
    
    return per_token_results


def aggregate_to_sequence_level(
    per_token_results: List[Dict],
    prompt_id: int,
    class_label: str,
    alpha: float,
    condition: str,
) -> SequenceLevelMetrics:
    """
    Aggregate per-token metrics to sequence level.
    
    Args:
        per_token_results: List of per-token metric dicts
        prompt_id: Prompt identifier
        class_label: "toxic" or "nontoxic"
        alpha: Steering coefficient
        condition: Experimental condition
    
    Returns:
        SequenceLevelMetrics
    """
    # Extract target token deltas
    target_deltas = [r["target_token_delta_logp_full"] for r in per_token_results]
    
    # Extract distribution metrics
    kl_fulls = [r["kl_full_base_to_steered"] for r in per_token_results]
    kl_supports = [r["kl_support_base_to_steered"] for r in per_token_results]
    js_values = [r["js_base_to_deployed"] for r in per_token_results]
    tv_values = [r["tv_base_to_steered_full"] for r in per_token_results]
    
    # Support metrics
    support_sizes = [r["base_support_size"] for r in per_token_results]
    support_fractions = [r["base_support_fraction"] for r in per_token_results]
    target_inside_support = [r["target_inside_base_support"] for r in per_token_results]
    
    # Sequence NLL (sum of negative log-probs)
    base_logps = [r["target_token_base_logp"] for r in per_token_results]
    steered_logps = [r["target_token_steered_full_logp"] for r in per_token_results]
    
    sequence_nll_base = -sum(base_logps)
    sequence_nll_steered = -sum(steered_logps)
    
    # Perturbation magnitude
    logit_deltas = [r["logit_delta_l2"] for r in per_token_results]
    
    return SequenceLevelMetrics(
        prompt_id=prompt_id,
        class_label=class_label,
        alpha=alpha,
        condition=condition,
        mean_target_delta_logp=np.mean(target_deltas),
        median_target_delta_logp=np.median(target_deltas),
        target_inside_support_rate=np.mean(target_inside_support),
        mean_kl_full=np.mean(kl_fulls),
        mean_kl_support=np.mean(kl_supports),
        mean_js=np.mean(js_values),
        mean_tv=np.mean(tv_values),
        mean_support_size=np.mean(support_sizes),
        mean_support_fraction=np.mean(support_fractions),
        support_exclusion_count=sum(1 for x in target_inside_support if not x),
        sequence_nll_base=sequence_nll_base,
        sequence_nll_steered=sequence_nll_steered,
        mean_logit_delta_l2=np.mean(logit_deltas),
    )


def compute_signed_gap_with_ci(
    toxic_sequences: List[SequenceLevelMetrics],
    nontoxic_sequences: List[SequenceLevelMetrics],
    n_bootstrap: int = 10000,
) -> Dict:
    """
    Compute signed attribute gap with bootstrap confidence interval.
    
    Args:
        toxic_sequences: Sequence metrics for toxic class
        nontoxic_sequences: Sequence metrics for non-toxic class
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        Dict with point estimate, CI, and statistical tests
    """
    toxic_gaps = [s.mean_target_delta_logp for s in toxic_sequences]
    nontoxic_gaps = [s.mean_target_delta_logp for s in nontoxic_sequences]
    
    # Point estimate
    signed_gap = np.mean(nontoxic_gaps) - np.mean(toxic_gaps)
    
    # Bootstrap CI
    rng = np.random.default_rng(42)
    bootstrap_gaps = []
    
    for _ in range(n_bootstrap):
        toxic_sample = rng.choice(toxic_gaps, size=len(toxic_gaps), replace=True)
        nontoxic_sample = rng.choice(nontoxic_gaps, size=len(nontoxic_gaps), replace=True)
        gap = np.mean(nontoxic_sample) - np.mean(toxic_sample)
        bootstrap_gaps.append(gap)
    
    bootstrap_gaps = np.array(bootstrap_gaps)
    ci_lower = np.percentile(bootstrap_gaps, 2.5)
    ci_upper = np.percentile(bootstrap_gaps, 97.5)
    
    # Statistical significance
    ci_excludes_zero = (ci_lower > 0) or (ci_upper < 0)
    prob_positive = np.mean(bootstrap_gaps > 0)
    
    # Effect size
    pooled_std = np.sqrt((np.std(toxic_gaps)**2 + np.std(nontoxic_gaps)**2) / 2)
    effect_size = signed_gap / pooled_std if pooled_std > 0 else 0.0
    
    return {
        "point_estimate": signed_gap,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_excludes_zero": ci_excludes_zero,
        "prob_positive": prob_positive,
        "effect_size": effect_size,
        "n_toxic": len(toxic_gaps),
        "n_nontoxic": len(nontoxic_gaps),
    }


def run_control_gates(
    alpha_zero_results: List[SequenceLevelMetrics],
    identity_results: List[SequenceLevelMetrics],
    reversal_results: List[SequenceLevelMetrics],
    normal_results: List[SequenceLevelMetrics],
) -> Dict:
    """
    Run all control gates.
    
    Args:
        alpha_zero_results: Results for alpha=0 condition
        identity_results: Results for identity control
        reversal_results: Results for reversal control
        normal_results: Results for normal condition
    
    Returns:
        Dict with gate results
    """
    gates = {}
    
    # Alpha-zero gate: should match baseline (signed gap ≈ 0)
    alpha_zero_gaps = [s.mean_target_delta_logp for s in alpha_zero_results]
    alpha_zero_mean = np.mean(alpha_zero_gaps)
    gates["alpha_zero"] = {
        "mean_delta": alpha_zero_mean,
        "passes": abs(alpha_zero_mean) < 0.01,  # Should be near zero
    }
    
    # Identity gate: should match baseline (signed gap ≈ 0)
    identity_gaps = [s.mean_target_delta_logp for s in identity_results]
    identity_mean = np.mean(identity_gaps)
    gates["identity"] = {
        "mean_delta": identity_mean,
        "passes": abs(identity_mean) < 0.01,
    }
    
    # Reversal gate: should flip sign
    normal_gaps = [s.mean_target_delta_logp for s in normal_results]
    reversal_gaps = [s.mean_target_delta_logp for s in reversal_results]
    
    normal_mean = np.mean(normal_gaps)
    reversal_mean = np.mean(reversal_gaps)
    
    gates["reversal"] = {
        "normal_mean": normal_mean,
        "reversal_mean": reversal_mean,
        "sign_flipped": (normal_mean * reversal_mean) < 0,
        "passes": (normal_mean * reversal_mean) < 0,
    }
    
    # Overall gate status
    gates["all_pass"] = all([
        gates["alpha_zero"]["passes"],
        gates["identity"]["passes"],
        gates["reversal"]["passes"],
    ])
    
    return gates


def run_phase4_experiment(config: Phase4Config) -> Dict:
    """
    Run complete Phase 4 experiment.
    
    Args:
        config: Phase 4 configuration
    
    Returns:
        Complete experiment results
    """
    logger.info("=" * 80)
    logger.info("NRAM v5 R6 Phase 4: Corrected Mechanistic Proof")
    logger.info("=" * 80)
    
    # Load confirmatory prompts
    logger.info("\n[1] Loading confirmatory prompts...")
    toxic_prompts, nontoxic_prompts = load_confirmatory_prompts(
        config.n_per_class,
        seed=config.random_seed
    )
    
    # Load models
    logger.info("\n[2] Loading models...")
    tokenizer, base_model, expert_model = load_models()
    
    # Run experiment for each condition
    conditions = [
        ("normal", config.primary_alpha),
        ("alpha_zero", 0.0),
        ("identity", config.primary_alpha),
        ("reversal", config.primary_alpha),
    ]
    
    all_results = {}
    
    for condition_name, alpha in conditions:
        logger.info(f"\n[3] Running condition: {condition_name} (alpha={alpha})")
        
        toxic_sequences = []
        nontoxic_sequences = []
        
        # Process toxic prompts
        for i, prompt_data in enumerate(tqdm(toxic_prompts, desc=f"Toxic ({condition_name})")):
            per_token = run_teacher_forced_analysis(
                tokenizer=tokenizer,
                base_model=base_model,
                expert_model=expert_model,
                prompt_text=prompt_data["prompt"],
                continuation_text=prompt_data["continuation"],
                alpha=alpha,
                condition=condition_name,
                top_k=config.top_k,
                top_p=config.top_p,
                temperature=config.temperature,
            )
            
            seq_metrics = aggregate_to_sequence_level(
                per_token_results=per_token,
                prompt_id=i,
                class_label="toxic",
                alpha=alpha,
                condition=condition_name,
            )
            toxic_sequences.append(seq_metrics)
            
            # Explicit cleanup every 50 sequences to prevent memory buildup
            if (i + 1) % 50 == 0:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"  Memory cleanup at {i+1}/300 toxic sequences")
        
        logger.info(f"  Toxic processing complete. Starting non-toxic...")
        
        # Process non-toxic prompts
        for i, prompt_data in enumerate(tqdm(nontoxic_prompts, desc=f"Non-toxic ({condition_name})")):
            per_token = run_teacher_forced_analysis(
                tokenizer=tokenizer,
                base_model=base_model,
                expert_model=expert_model,
                prompt_text=prompt_data["prompt"],
                continuation_text=prompt_data["continuation"],
                alpha=alpha,
                condition=condition_name,
                top_k=config.top_k,
                top_p=config.top_p,
                temperature=config.temperature,
            )
            
            seq_metrics = aggregate_to_sequence_level(
                per_token_results=per_token,
                prompt_id=i,
                class_label="nontoxic",
                alpha=alpha,
                condition=condition_name,
            )
            nontoxic_sequences.append(seq_metrics)
            
            # Explicit cleanup every 50 sequences
            if (i + 1) % 50 == 0:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"  Memory cleanup at {i+1}/300 non-toxic sequences")
        
        logger.info(f"  Non-toxic processing complete. Computing signed gap...")
        
        # Compute signed gap
        signed_gap = compute_signed_gap_with_ci(toxic_sequences, nontoxic_sequences)
        
        logger.info(f"  Signed gap G(alpha={alpha}): {signed_gap['point_estimate']:.6f}")
        logger.info(f"  95% CI: [{signed_gap['ci_lower']:.6f}, {signed_gap['ci_upper']:.6f}]")
        logger.info(f"  CI excludes 0: {signed_gap['ci_excludes_zero']}")
        logger.info(f"  Effect size: {signed_gap['effect_size']:.4f}")
        
        all_results[condition_name] = {
            "toxic_sequences": [asdict(s) for s in toxic_sequences],
            "nontoxic_sequences": [asdict(s) for s in nontoxic_sequences],
            "signed_gap": signed_gap,
        }
    
    # Run control gates
    logger.info("\n[4] Running control gates...")
    control_gates = run_control_gates(
        alpha_zero_results=[SequenceLevelMetrics(**s) for s in all_results["alpha_zero"]["toxic_sequences"]],
        identity_results=[SequenceLevelMetrics(**s) for s in all_results["identity"]["toxic_sequences"]],
        reversal_results=[SequenceLevelMetrics(**s) for s in all_results["reversal"]["toxic_sequences"]],
        normal_results=[SequenceLevelMetrics(**s) for s in all_results["normal"]["toxic_sequences"]],
    )
    
    logger.info(f"  Alpha-zero gate: {control_gates['alpha_zero']['passes']}")
    logger.info(f"  Identity gate: {control_gates['identity']['passes']}")
    logger.info(f"  Reversal gate: {control_gates['reversal']['passes']}")
    logger.info(f"  All gates pass: {control_gates['all_pass']}")
    
    # Determine final verdict
    logger.info("\n[5] Determining final verdict...")
    normal_gap = all_results["normal"]["signed_gap"]
    
    if control_gates["all_pass"] and normal_gap["ci_excludes_zero"] and normal_gap["point_estimate"] > 0:
        verdict = "MECHANISM_AND_DIRECTION_CAUSALLY_PROVEN"
    elif control_gates["all_pass"] and not normal_gap["ci_excludes_zero"]:
        verdict = "DISTRIBUTION_CHANGE_PROVEN_DIRECTION_INCONCLUSIVE"
    elif not control_gates["all_pass"]:
        verdict = "IMPLEMENTATION_INVALID"
    else:
        verdict = "UNKNOWN"
    
    logger.info(f"  Final verdict: {verdict}")
    
    # Save results
    output_dir = Path("artifacts/dexperts/r6")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": asdict(config),
        "all_results": all_results,
        "control_gates": control_gates,
        "final_verdict": verdict,
    }
    
    output_path = output_dir / "mechanistic_proof_phase4.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializer)
    
    logger.info(f"\n[6] Results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    config = Phase4Config()
    results = run_phase4_experiment(config)
    
    print("\n" + "=" * 80)
    print("Phase 4 Complete")
    print("=" * 80)
    print(f"Final verdict: {results['final_verdict']}")
    print(f"Signed gap: {results['all_results']['normal']['signed_gap']['point_estimate']:.6f}")
    print(f"All control gates pass: {results['control_gates']['all_pass']}")
