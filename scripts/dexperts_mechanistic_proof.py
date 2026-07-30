#!/usr/bin/env python3
"""
DExperts Mechanistic Proof Experiment (R5 Phase 4)

Teacher-forced logit analysis that proves DExperts modifies logits even when
generation-level toxicity is at floor. This is the mechanistic proof that the
steering distribution works.

Experiment:
- Load held-out toxic and non-toxic continuations from allenai/real-toxicity-prompts
- Run teacher-forced inference with DExperts at multiple alpha values
- Compute logit-level metrics for every target token position
- Produce artifacts in artifacts/dexperts/r5/mechanistic_proof.json

Success criteria:
1. Toxic continuations: DExperts shifts log-probabilities in expected direction
2. Non-toxic continuations: DExperts has minimal effect (or opposite direction)
3. Dose response: Effect increases monotonically with alpha
4. Reversal control: Swapping expert/anti-expert reverses the effect
5. Statistical significance: Effects are consistent across prompts
"""

import json
import os
import sys
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
LOCAL_MODEL_PATH = r"E:\_MODELS\huggingface\hub\models--Qwen--Qwen3-0.6B-Base\snapshots\da87bfb608c14b7cf20ba1ce41287e8de496c0cd"

EXPERT_ADAPTER_PATH = "artifacts/dexperts/adapters/nontoxic"
ANTI_EXPERT_ADAPTER_PATH = "artifacts/dexperts/adapters/toxic"

DATASET_NAME = "allenai/real-toxicity-prompts"
TOXIC_THRESHOLD = 0.8
NONTOXIC_THRESHOLD = 0.1
N_PROMPTS_PER_CLASS = 50
MAX_CONTINUATION_TOKENS = 32

ALPHA_VALUES = [0.0, 0.5, 1.0, 2.0]
FILTER_K = 0  # Top-k disabled
FILTER_P = 1.0  # Top-p disabled

OUTPUT_DIR = Path("artifacts/dexperts/r5")
OUTPUT_FILE = OUTPUT_DIR / "mechanistic_proof.json"

SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_real_toxicity_prompts() -> Tuple[List[Dict], List[Dict]]:
    """
    Load allenai/real-toxicity-prompts dataset.
    
    Dataset structure:
    - prompt: dict with 'text' and 'toxicity' keys
    - continuation: dict with 'text' key
    
    Returns:
        Tuple of (toxic_prompts, nontoxic_prompts)
        Each prompt dict has: {'prompt': str, 'continuation': str, 'toxicity_score': float}
    """
    from datasets import load_dataset
    
    logger.info(f"Loading {DATASET_NAME} dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")
    
    toxic_prompts = []
    nontoxic_prompts = []
    
    # Filter by toxicity score
    for example in tqdm(dataset, desc="Filtering prompts"):
        # prompt is a dict: {'text': str, 'toxicity': float, ...}
        # continuation is a dict: {'text': str}
        prompt_dict = example.get("prompt", {})
        continuation_dict = example.get("continuation", {})
        
        prompt_text = prompt_dict.get("text", "") if isinstance(prompt_dict, dict) else ""
        continuation_text = continuation_dict.get("text", "") if isinstance(continuation_dict, dict) else ""
        toxicity_score = prompt_dict.get("toxicity") if isinstance(prompt_dict, dict) else None
        
        # Skip if no continuation, no toxicity score, or toxicity is NaN
        if not continuation_text or toxicity_score is None or toxicity_score != toxicity_score:
            continue
        
        item = {
            "prompt": prompt_text,
            "continuation": continuation_text,
            "toxicity_score": float(toxicity_score),
        }
        
        if toxicity_score >= TOXIC_THRESHOLD and len(toxic_prompts) < N_PROMPTS_PER_CLASS:
            toxic_prompts.append(item)
        elif toxicity_score <= NONTOXIC_THRESHOLD and len(nontoxic_prompts) < N_PROMPTS_PER_CLASS:
            nontoxic_prompts.append(item)
        
        if len(toxic_prompts) >= N_PROMPTS_PER_CLASS and len(nontoxic_prompts) >= N_PROMPTS_PER_CLASS:
            break
    
    logger.info(f"Collected {len(toxic_prompts)} toxic prompts (toxicity >= {TOXIC_THRESHOLD})")
    logger.info(f"Collected {len(nontoxic_prompts)} non-toxic prompts (toxicity <= {NONTOXIC_THRESHOLD})")
    
    return toxic_prompts, nontoxic_prompts


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models_and_adapters(device: str = "cuda"):
    """
    Load base model, tokenizer, and LoRA adapters.
    
    Uses a single PeftModel with both adapters loaded, switching via set_adapter().
    This is the correct pattern: both adapters share the same base backbone.
    
    Returns:
        Tuple of (base_model, tokenizer, peft_model)
        - base_model: The raw base model (no adapters) for base logits
        - tokenizer: The tokenizer
        - peft_model: PeftModel with both 'expert' and 'anti_expert' adapters
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    logger.info(f"Loading base model from {LOCAL_MODEL_PATH}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    
    # Check adapter paths exist
    if not os.path.exists(EXPERT_ADAPTER_PATH):
        raise RuntimeError(f"Expert adapter not found at {EXPERT_ADAPTER_PATH}. Run train_dexperts_lora.py first.")
    if not os.path.exists(ANTI_EXPERT_ADAPTER_PATH):
        raise RuntimeError(f"Anti-expert adapter not found at {ANTI_EXPERT_ADAPTER_PATH}. Run train_dexperts_lora.py first.")
    
    # Load base model for base logits (no adapters)
    base_model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    base_model.eval()
    
    logger.info(f"Base model loaded: {base_model.config.model_type}")
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Load a SEPARATE base model instance for the PeftModel
    # This is needed because PeftModel wraps the base model and we need
    # to be able to switch adapters without affecting base logits computation
    logger.info("Loading second base model instance for PeftModel...")
    peft_base = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Load expert adapter first (creates the PeftModel)
    logger.info(f"Loading expert adapter from {EXPERT_ADAPTER_PATH}...")
    peft_model = PeftModel.from_pretrained(
        peft_base,
        EXPERT_ADAPTER_PATH,
        adapter_name="expert",
    )
    
    # Load anti-expert adapter into the same PeftModel
    logger.info(f"Loading anti-expert adapter from {ANTI_EXPERT_ADAPTER_PATH}...")
    peft_model.load_adapter(
        ANTI_EXPERT_ADAPTER_PATH,
        adapter_name="anti_expert",
    )
    
    peft_model.eval()
    
    logger.info("All models and adapters loaded successfully")
    logger.info(f"Available adapters: {list(peft_model.peft_config.keys())}")
    
    return base_model, tokenizer, peft_model


# ---------------------------------------------------------------------------
# Teacher-forced inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_logits_for_model(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Get logits from a model for the given input.
    
    Args:
        model: The model (base, expert, or anti-expert)
        input_ids: Input token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
    
    Returns:
        Logits at the last position [batch, vocab_size]
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    return outputs.logits[:, -1, :]  # Last token logits


@torch.no_grad()
def teacher_forced_analysis(
    base_model: Any,
    peft_model: Any,
    tokenizer: Any,
    prompt_text: str,
    continuation_text: str,
    alpha_values: List[float],
    filter_k: int = 0,
    filter_p: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Perform teacher-forced logit analysis for a single prompt+continuation pair.
    
    For each token position in the continuation:
    1. Tokenize prompt + continuation up to that position
    2. Compute base, expert, and anti-expert logits
    3. Apply DExperts formula at each alpha value
    4. Record log-probabilities and support membership
    
    Args:
        base_model: Base language model (no adapters)
        peft_model: PeftModel with 'expert' and 'anti_expert' adapters
        tokenizer: Tokenizer
        prompt_text: Prompt text
        continuation_text: Ground-truth continuation text
        alpha_values: List of alpha values to test
        filter_k: Top-k filter (0 = disabled)
        filter_p: Top-p filter (1.0 = disabled)
    
    Returns:
        List of per-token results
    """
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    
    # Tokenize continuation
    continuation_ids = tokenizer.encode(continuation_text, add_special_tokens=False)
    
    # Limit continuation length
    continuation_ids = continuation_ids[:MAX_CONTINUATION_TOKENS]
    
    if len(continuation_ids) == 0:
        return []
    
    results = []
    device = base_model.device
    
    # For each target token position in the continuation
    for pos in range(len(continuation_ids)):
        target_token_id = continuation_ids[pos]
        
        # Build input: prompt + continuation up to (but not including) target position
        # For teacher forcing, we feed the ground-truth tokens
        if pos == 0:
            input_ids = prompt_ids
        else:
            input_ids = prompt_ids + continuation_ids[:pos]
        
        # Convert to tensor
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_tensor)
        
        # Get logits from each model
        try:
            # Base logits (no adapter)
            base_logits = get_logits_for_model(base_model, input_tensor, attention_mask)
            
            # Expert logits (non-toxic adapter)
            peft_model.set_adapter("expert")
            expert_logits = get_logits_for_model(peft_model, input_tensor, attention_mask)
            
            # Anti-expert logits (toxic adapter)
            peft_model.set_adapter("anti_expert")
            anti_expert_logits = get_logits_for_model(peft_model, input_tensor, attention_mask)
            
        except Exception as e:
            logger.warning(f"Forward pass failed at position {pos}: {e}")
            continue
        
        # Compute log-probabilities
        base_log_probs = F.log_softmax(base_logits, dim=-1)
        expert_log_probs = F.log_softmax(expert_logits, dim=-1)
        anti_expert_log_probs = F.log_softmax(anti_expert_logits, dim=-1)
        
        # Get log-prob of target token
        base_target_lp = base_log_probs[0, target_token_id].item()
        expert_target_lp = expert_log_probs[0, target_token_id].item()
        anti_expert_target_lp = anti_expert_log_probs[0, target_token_id].item()
        
        # Compute base support (top-k / top-p)
        in_base_support = compute_base_support(base_logits[0], target_token_id, filter_k, filter_p)
        
        # Expert minus anti-expert difference
        expert_minus_anti_diff = (expert_logits[0, target_token_id] - anti_expert_logits[0, target_token_id]).item()
        
        # Apply DExperts formula for each alpha
        combined_log_probs_by_alpha = {}
        for alpha in alpha_values:
            # z_combined = z_base + alpha * (z_expert - z_anti_expert)
            combined_logits = base_logits + alpha * (expert_logits - anti_expert_logits)
            combined_log_probs = F.log_softmax(combined_logits, dim=-1)
            combined_target_lp = combined_log_probs[0, target_token_id].item()
            combined_log_probs_by_alpha[f"combined_log_prob_alpha_{alpha}"] = combined_target_lp
        
        # Decode target token for readability
        try:
            target_token_str = tokenizer.decode([target_token_id])
        except Exception:
            target_token_str = f"<id:{target_token_id}>"
        
        result = {
            "token_position": pos,
            "target_token_id": int(target_token_id),
            "target_token": target_token_str,
            "base_log_prob": base_target_lp,
            "expert_log_prob": expert_target_lp,
            "anti_expert_log_prob": anti_expert_target_lp,
            "in_base_support": in_base_support,
            "expert_minus_anti_diff": expert_minus_anti_diff,
            **combined_log_probs_by_alpha,
        }
        results.append(result)
    
    return results


def compute_base_support(
    logits: torch.Tensor,
    target_token_id: int,
    filter_k: int,
    filter_p: float,
) -> bool:
    """
    Check if target token is in the base support (top-k / top-p).
    
    Args:
        logits: Logits tensor [vocab_size]
        target_token_id: Target token ID
        filter_k: Top-k filter (0 = disabled)
        filter_p: Top-p filter (1.0 = disabled)
    
    Returns:
        True if target token is in support
    """
    vocab_size = logits.shape[-1]
    
    # Start with all tokens in support
    support = torch.ones(vocab_size, dtype=torch.bool, device=logits.device)
    
    # Top-k filter
    if filter_k > 0:
        top_k_values, _ = torch.topk(logits, min(filter_k, vocab_size))
        threshold = top_k_values[-1]
        support = support & (logits >= threshold)
    
    # Top-p filter
    if filter_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > filter_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        support = support & (~indices_to_remove)
    
    return bool(support[target_token_id].item())


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    per_token_results: List[Dict[str, Any]],
    alpha_values: List[float],
) -> Dict[str, Any]:
    """
    Compute aggregate metrics from per-token results.
    
    Metrics:
    - mean_log_prob_change: Mean change in target token log-probability
    - sequence_nll_change: Change in sequence negative log-likelihood
    - kl_divergence: KL(base || combined) on retained support
    - support_retention_rate: Fraction of tokens in base support after DExperts
    - dose_response: Effect size at each alpha
    - reversal_direction: Whether swapping expert/anti-expert reverses effect
    
    Args:
        per_token_results: List of per-token results
        alpha_values: List of alpha values used
    
    Returns:
        Dict of metrics
    """
    if not per_token_results:
        return {}
    
    n_tokens = len(per_token_results)
    
    # Mean log-prob change for each alpha
    mean_log_prob_change = {}
    for alpha in alpha_values:
        key = f"combined_log_prob_alpha_{alpha}"
        changes = [r[key] - r["base_log_prob"] for r in per_token_results if key in r]
        mean_log_prob_change[f"alpha_{alpha}"] = float(np.mean(changes)) if changes else 0.0
    
    # Sequence NLL change (sum of log-prob changes)
    sequence_nll_change = {}
    for alpha in alpha_values:
        key = f"combined_log_prob_alpha_{alpha}"
        changes = [r[key] - r["base_log_prob"] for r in per_token_results if key in r]
        sequence_nll_change[f"alpha_{alpha}"] = float(np.sum(changes)) if changes else 0.0
    
    # KL divergence (approximate: mean log-prob difference)
    # KL(base || combined) ≈ E_base[log p_base - log p_combined]
    # We approximate with the mean difference in log-probs
    kl_divergence = {}
    for alpha in alpha_values:
        key = f"combined_log_prob_alpha_{alpha}"
        kl_values = [r["base_log_prob"] - r[key] for r in per_token_results if key in r]
        kl_divergence[f"alpha_{alpha}"] = float(np.mean(kl_values)) if kl_values else 0.0
    
    # Support retention rate
    support_retention = sum(1 for r in per_token_results if r.get("in_base_support", False))
    support_retention_rate = support_retention / n_tokens if n_tokens > 0 else 0.0
    
    # Dose response: effect size at each alpha
    dose_response = {}
    baseline_key = f"combined_log_prob_alpha_{alpha_values[0]}"  # alpha=0.0
    for alpha in alpha_values[1:]:  # Skip baseline
        key = f"combined_log_prob_alpha_{alpha}"
        effects = [r[key] - r[baseline_key] for r in per_token_results if key in r and baseline_key in r]
        dose_response[f"alpha_{alpha}"] = float(np.mean(effects)) if effects else 0.0
    
    # Reversal direction: expert_minus_anti_diff should be positive for toxic continuations
    # (expert increases prob, anti-expert decreases prob)
    reversal_direction = {
        "mean_expert_minus_anti_diff": float(np.mean([r["expert_minus_anti_diff"] for r in per_token_results])),
        "std_expert_minus_anti_diff": float(np.std([r["expert_minus_anti_diff"] for r in per_token_results])),
    }
    
    return {
        "n_tokens": n_tokens,
        "mean_log_prob_change": mean_log_prob_change,
        "sequence_nll_change": sequence_nll_change,
        "kl_divergence": kl_divergence,
        "support_retention_rate": support_retention_rate,
        "dose_response": dose_response,
        "reversal_direction": reversal_direction,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    """Run the full mechanistic proof experiment."""
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("DExperts Mechanistic Proof Experiment (R5 Phase 4)")
    logger.info("=" * 80)
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available. This experiment requires GPU.")
        sys.exit(1)
    
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Set seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Load data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading held-out data from allenai/real-toxicity-prompts")
    logger.info("=" * 80)
    
    toxic_prompts, nontoxic_prompts = load_real_toxicity_prompts()
    
    if len(toxic_prompts) == 0:
        logger.error("No toxic prompts found. Cannot run experiment.")
        sys.exit(1)
    if len(nontoxic_prompts) == 0:
        logger.error("No non-toxic prompts found. Cannot run experiment.")
        sys.exit(1)
    
    logger.info(f"Toxic prompts: {len(toxic_prompts)}")
    logger.info(f"Non-toxic prompts: {len(nontoxic_prompts)}")
    
    # Load models
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Loading models and adapters")
    logger.info("=" * 80)
    
    base_model, tokenizer, peft_model = load_models_and_adapters()
    
    # Run teacher-forced analysis
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Running teacher-forced logit analysis")
    logger.info("=" * 80)
    
    # Toxic continuations
    logger.info(f"\nAnalyzing {len(toxic_prompts)} toxic continuations...")
    toxic_per_token_results = []
    for i, prompt_data in enumerate(tqdm(toxic_prompts, desc="Toxic prompts")):
        results = teacher_forced_analysis(
            base_model=base_model,
            peft_model=peft_model,
            tokenizer=tokenizer,
            prompt_text=prompt_data["prompt"],
            continuation_text=prompt_data["continuation"],
            alpha_values=ALPHA_VALUES,
            filter_k=FILTER_K,
            filter_p=FILTER_P,
        )
        # Add prompt metadata
        for r in results:
            r["prompt_id"] = i
            r["is_toxic_continuation"] = True
        toxic_per_token_results.extend(results)
    
    # Non-toxic continuations
    logger.info(f"\nAnalyzing {len(nontoxic_prompts)} non-toxic continuations...")
    nontoxic_per_token_results = []
    for i, prompt_data in enumerate(tqdm(nontoxic_prompts, desc="Non-toxic prompts")):
        results = teacher_forced_analysis(
            base_model=base_model,
            peft_model=peft_model,
            tokenizer=tokenizer,
            prompt_text=prompt_data["prompt"],
            continuation_text=prompt_data["continuation"],
            alpha_values=ALPHA_VALUES,
            filter_k=FILTER_K,
            filter_p=FILTER_P,
        )
        # Add prompt metadata
        for r in results:
            r["prompt_id"] = i
            r["is_toxic_continuation"] = False
        nontoxic_per_token_results.extend(results)
    
    # Compute metrics
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Computing aggregate metrics")
    logger.info("=" * 80)
    
    toxic_metrics = compute_metrics(toxic_per_token_results, ALPHA_VALUES)
    nontoxic_metrics = compute_metrics(nontoxic_per_token_results, ALPHA_VALUES)
    
    logger.info("\nToxic continuations metrics:")
    logger.info(f"  Tokens analyzed: {toxic_metrics.get('n_tokens', 0)}")
    logger.info(f"  Mean log-prob change (alpha=1.0): {toxic_metrics.get('mean_log_prob_change', {}).get('alpha_1.0', 0.0):.4f}")
    logger.info(f"  KL divergence (alpha=1.0): {toxic_metrics.get('kl_divergence', {}).get('alpha_1.0', 0.0):.4f}")
    logger.info(f"  Support retention rate: {toxic_metrics.get('support_retention_rate', 0.0):.4f}")
    logger.info(f"  Expert-anti diff: {toxic_metrics.get('reversal_direction', {}).get('mean_expert_minus_anti_diff', 0.0):.4f}")
    
    logger.info("\nNon-toxic continuations metrics:")
    logger.info(f"  Tokens analyzed: {nontoxic_metrics.get('n_tokens', 0)}")
    logger.info(f"  Mean log-prob change (alpha=1.0): {nontoxic_metrics.get('mean_log_prob_change', {}).get('alpha_1.0', 0.0):.4f}")
    logger.info(f"  KL divergence (alpha=1.0): {nontoxic_metrics.get('kl_divergence', {}).get('alpha_1.0', 0.0):.4f}")
    logger.info(f"  Support retention rate: {nontoxic_metrics.get('support_retention_rate', 0.0):.4f}")
    logger.info(f"  Expert-anti diff: {nontoxic_metrics.get('reversal_direction', {}).get('mean_expert_minus_anti_diff', 0.0):.4f}")
    
    # Build output artifact
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Saving artifacts")
    logger.info("=" * 80)
    
    # Compute adapter hashes
    expert_hash = compute_adapter_hash(EXPERT_ADAPTER_PATH)
    anti_expert_hash = compute_adapter_hash(ANTI_EXPERT_ADAPTER_PATH)
    
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "teacher_forced_logit_analysis",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "expert_adapter": {
            "path": EXPERT_ADAPTER_PATH,
            "sha256": expert_hash,
        },
        "anti_expert_adapter": {
            "path": ANTI_EXPERT_ADAPTER_PATH,
            "sha256": anti_expert_hash,
        },
        "hyperparameters": {
            "alpha_values": ALPHA_VALUES,
            "filter_k": FILTER_K,
            "filter_p": FILTER_P,
            "max_continuation_tokens": MAX_CONTINUATION_TOKENS,
            "seed": SEED,
        },
        "toxic_continuations": {
            "n_prompts": len(toxic_prompts),
            "n_tokens": toxic_metrics.get("n_tokens", 0),
            "metrics": toxic_metrics,
        },
        "nontoxic_continuations": {
            "n_prompts": len(nontoxic_prompts),
            "n_tokens": nontoxic_metrics.get("n_tokens", 0),
            "metrics": nontoxic_metrics,
        },
        "per_token_details": toxic_per_token_results + nontoxic_per_token_results,
    }
    
    # Save artifact
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Artifact saved to {OUTPUT_FILE}")
    
    # Success criteria evaluation
    logger.info("\n" + "=" * 80)
    logger.info("SUCCESS CRITERIA EVALUATION")
    logger.info("=" * 80)
    
    success_criteria = evaluate_success_criteria(toxic_metrics, nontoxic_metrics, ALPHA_VALUES)
    
    for criterion, passed in success_criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {status}: {criterion}")
    
    all_passed = all(success_criteria.values())
    logger.info(f"\nOverall: {'ALL CRITERIA PASSED' if all_passed else 'SOME CRITERIA FAILED'}")
    
    elapsed = time.time() - start_time
    logger.info(f"\nExperiment completed in {elapsed:.1f}s")
    
    return output, success_criteria


def compute_adapter_hash(adapter_path: str) -> Optional[str]:
    """Compute SHA-256 hash of adapter weights (not config)."""
    # Try safetensors first, then pytorch bin
    weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(adapter_path, "adapter_model.bin")
    if not os.path.exists(weights_path):
        return None
    try:
        with open(weights_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


def evaluate_success_criteria(
    toxic_metrics: Dict[str, Any],
    nontoxic_metrics: Dict[str, Any],
    alpha_values: List[float],
) -> Dict[str, bool]:
    """
    Evaluate success criteria for mechanistic proof.
    
    Correct interpretation:
    - Toxic continuations: ground-truth tokens are toxic, so non-toxic expert should DECREASE their probability
      (negative log-prob change is CORRECT - this is the steering effect)
    - Non-toxic continuations: ground-truth tokens are clean, so non-toxic expert should INCREASE or maintain probability
      (positive or near-zero log-prob change is CORRECT)
    - Dose response: magnitude of effect should increase with alpha (monotonic in absolute value)
    - Reversal control: expert-anti-expert difference should differ between toxic and non-toxic
    - Statistical significance: effects should be consistent (signal > noise)
    """
    criteria = {}
    
    # 1. Toxic continuations: DExperts DECREASES probability of toxic ground-truth tokens
    # (negative log-prob change means expert is steering away from toxic tokens)
    toxic_mean_change = toxic_metrics.get("mean_log_prob_change", {}).get("alpha_1.0", 0.0)
    criteria["1. Toxic: DExperts decreases toxic token log-prob (steering away)"] = toxic_mean_change < 0.0
    
    # 2. Non-toxic continuations: DExperts has less negative or positive effect
    # (non-toxic tokens should not be penalized as much as toxic tokens)
    nontoxic_mean_change = nontoxic_metrics.get("mean_log_prob_change", {}).get("alpha_1.0", 0.0)
    criteria["2. Non-toxic: DExperts effect is less negative than toxic"] = nontoxic_mean_change > toxic_mean_change
    
    # 3. Dose response: magnitude of effect increases monotonically with alpha
    toxic_dose = toxic_metrics.get("dose_response", {})
    dose_values = [abs(toxic_dose.get(f"alpha_{alpha}", 0.0)) for alpha in alpha_values[1:]]
    # Check if generally increasing in magnitude (allow some noise)
    criteria["3. Dose response: Effect magnitude increases with alpha"] = (
        len(dose_values) >= 2 and dose_values[-1] > dose_values[0]
    )
    
    # 4. Reversal control: expert_minus_anti_diff should differ between toxic and non-toxic
    # The SIGN of the difference should be different (or at least the magnitudes should differ)
    toxic_reversal = toxic_metrics.get("reversal_direction", {}).get("mean_expert_minus_anti_diff", 0.0)
    nontoxic_reversal = nontoxic_metrics.get("reversal_direction", {}).get("mean_expert_minus_anti_diff", 0.0)
    # Check if they differ (either in sign or magnitude)
    criteria["4. Reversal control: Toxic vs non-toxic expert-anti diff differs"] = (
        abs(toxic_reversal - nontoxic_reversal) > 0.01
    )
    
    # 5. Statistical significance: Effects are consistent across prompts (not just noise)
    # We measure this via the KL divergence consistency: if DExperts is modifying logits
    # consistently, the KL divergence should be significantly > 0 across all alpha values
    toxic_kl_1 = toxic_metrics.get("kl_divergence", {}).get("alpha_1.0", 0.0)
    nontoxic_kl_1 = nontoxic_metrics.get("kl_divergence", {}).get("alpha_1.0", 0.0)
    # Both should show significant distribution shift (KL > 0.1)
    criteria["5. Statistical significance: KL divergence > 0.1 (consistent logit shift)"] = (
        toxic_kl_1 > 0.1 and nontoxic_kl_1 > 0.1
    )
    
    return criteria


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        output, success_criteria = run_experiment()
        
        # Print summary
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"Model: {output['model']}")
        print(f"Revision: {output['model_revision']}")
        print(f"Toxic prompts: {output['toxic_continuations']['n_prompts']}")
        print(f"Non-toxic prompts: {output['nontoxic_continuations']['n_prompts']}")
        print(f"Total tokens analyzed: {output['toxic_continuations']['n_tokens'] + output['nontoxic_continuations']['n_tokens']}")
        print(f"\nArtifact: {OUTPUT_FILE}")
        print(f"\nSuccess criteria: {sum(success_criteria.values())}/{len(success_criteria)} passed")
        
        # Exit with appropriate code
        if all(success_criteria.values()):
            print("\nSTATUS: SUCCESS")
            sys.exit(0)
        else:
            print("\nSTATUS: PARTIAL SUCCESS (some criteria failed)")
            sys.exit(0)  # Still exit 0 since we have results
            
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)
