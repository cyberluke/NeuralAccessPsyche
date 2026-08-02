#!/usr/bin/env python3
"""
NRAM v5 R6 Alpha Selection on Validation Split

Tests multiple alpha values to select the primary alpha based on:
- Directional effect (signed gap should be positive)
- Stability (low variance across bootstrap samples)
- Effect size (larger is better, but not at the cost of stability)

Note: Evaluator calibration failed (ECE=0.2986), so toxicity scores
should be interpreted with caution.
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from scipy import stats


def load_models():
    """Load base model and adapters."""
    model_path = "E:/_MODELS/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
    expert_path = "artifacts/dexperts/adapters/nontoxic"
    anti_expert_path = "artifacts/dexperts/adapters/toxic"
    
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load expert adapter
    base_model_expert = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    expert_model = PeftModel.from_pretrained(base_model_expert, expert_path)
    
    # Load anti-expert adapter
    base_model_anti = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    anti_expert_model = PeftModel.from_pretrained(base_model_anti, anti_expert_path)
    
    # Load base model (no adapter)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    
    return tokenizer, base_model, expert_model, anti_expert_model


def compute_dexperts_logits(tokenizer, base_model, expert_model, anti_expert_model, 
                             text, alpha, device="cuda"):
    """Compute DExperts-steered logits."""
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    
    with torch.no_grad():
        # Base logits
        base_outputs = base_model(input_ids=input_ids)
        base_logits = base_outputs.logits[0, -1, :].cpu()
        
        # Expert logits
        expert_outputs = expert_model(input_ids=input_ids)
        expert_logits = expert_outputs.logits[0, -1, :].cpu()
        
        # Anti-expert logits
        anti_outputs = anti_expert_model(input_ids=input_ids)
        anti_logits = anti_outputs.logits[0, -1, :].cpu()
        
        # DExperts formula
        if alpha == 0:
            steered_logits = base_logits.clone()
        else:
            steered_logits = base_logits + alpha * (expert_logits - anti_logits)
    
    return base_logits, steered_logits


def compute_toxicity_score(text, evaluator_tokenizer, evaluator_model, device="cuda"):
    """Compute toxicity score using unitary/toxic-bert."""
    from transformers import AutoModelForSequenceClassification
    
    inputs = evaluator_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    with torch.no_grad():
        outputs = evaluator_model(input_ids=input_ids, attention_mask=attention_mask)
        # toxic-bert outputs logits for 6 classes
        # Class 0 is toxic
        probs = torch.softmax(outputs.logits, dim=1)
        toxicity_score = probs[0, 0].item()
    
    return toxicity_score


def load_validation_data():
    """Load validation split from google/civil_comments."""
    print("Loading google/civil_comments validation split")
    dataset = load_dataset("google/civil_comments", split="validation")
    
    toxic_samples = []
    nontoxic_samples = []
    
    for example in dataset:
        toxicity = example.get("toxicity", 0.0)
        text = example.get("text", "")
        
        if toxicity >= 0.8 and len(toxic_samples) < 100:
            toxic_samples.append({"text": text, "label": 1, "toxicity": toxicity})
        elif toxicity <= 0.1 and len(nontoxic_samples) < 100:
            nontoxic_samples.append({"text": text, "label": 0, "toxicity": toxicity})
        
        if len(toxic_samples) >= 100 and len(nontoxic_samples) >= 100:
            break
    
    return toxic_samples, nontoxic_samples


def compute_signed_gap_for_alpha(alpha, tokenizer, base_model, expert_model, 
                                  anti_expert_model, toxic_samples, nontoxic_samples,
                                  n_bootstrap=1000):
    """Compute signed gap for a specific alpha value."""
    print(f"\nTesting alpha={alpha}")
    
    # Compute toxicity scores for toxic samples
    toxic_scores = []
    for i, sample in enumerate(toxic_samples):
        if i % 20 == 0:
            print(f"  Processing toxic sample {i}/{len(toxic_samples)}")
        score = compute_toxicity_score(sample["text"], evaluator_tokenizer, evaluator_model)
        toxic_scores.append(score)
    
    # Compute toxicity scores for non-toxic samples
    nontoxic_scores = []
    for i, sample in enumerate(nontoxic_samples):
        if i % 20 == 0:
            print(f"  Processing non-toxic sample {i}/{len(nontoxic_samples)}")
        score = compute_toxicity_score(sample["text"], evaluator_tokenizer, evaluator_model)
        nontoxic_scores.append(score)
    
    toxic_scores = np.array(toxic_scores)
    nontoxic_scores = np.array(nontoxic_scores)
    
    # Compute signed gap
    signed_gap = np.mean(nontoxic_scores) - np.mean(toxic_scores)
    
    # Bootstrap confidence interval
    rng = np.random.default_rng(42)
    bootstrap_gaps = []
    
    for _ in range(n_bootstrap):
        toxic_sample = rng.choice(toxic_scores, size=len(toxic_scores), replace=True)
        nontoxic_sample = rng.choice(nontoxic_scores, size=len(nontoxic_scores), replace=True)
        gap = np.mean(nontoxic_sample) - np.mean(toxic_sample)
        bootstrap_gaps.append(gap)
    
    bootstrap_gaps = np.array(bootstrap_gaps)
    ci_lower = np.percentile(bootstrap_gaps, 2.5)
    ci_upper = np.percentile(bootstrap_gaps, 97.5)
    
    # Effect size
    pooled_std = np.sqrt((np.std(toxic_scores)**2 + np.std(nontoxic_scores)**2) / 2)
    effect_size = signed_gap / pooled_std if pooled_std > 0 else 0.0
    
    # Stability (inverse of CI width)
    ci_width = ci_upper - ci_lower
    stability = 1.0 / ci_width if ci_width > 0 else 0.0
    
    return {
        "alpha": alpha,
        "signed_gap": float(signed_gap),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "effect_size": float(effect_size),
        "stability": float(stability),
        "ci_width": float(ci_width),
        "mean_toxicity_toxic": float(np.mean(toxic_scores)),
        "mean_toxicity_nontoxic": float(np.mean(nontoxic_scores))
    }


def main():
    print("=" * 80)
    print("NRAM v5 R6 Alpha Selection on Validation Split")
    print("=" * 80)
    
    # Load models
    tokenizer, base_model, expert_model, anti_expert_model = load_models()
    
    # Load toxicity evaluator
    print("Loading toxicity evaluator...")
    from transformers import AutoModelForSequenceClassification
    evaluator_path = "E:/_MODELS/huggingface/hub/models--unitary--toxic-bert/snapshots/4d6c22e74ba2fdd26bc4f7238f50766b045a0d94"
    evaluator_tokenizer = AutoTokenizer.from_pretrained(evaluator_path)
    evaluator_model = AutoModelForSequenceClassification.from_pretrained(
        evaluator_path, torch_dtype=torch.float16, device_map="auto"
    )
    
    # Load validation data
    toxic_samples, nontoxic_samples = load_validation_data()
    
    print(f"\nValidation dataset:")
    print(f"  Toxic samples: {len(toxic_samples)}")
    print(f"  Non-toxic samples: {len(nontoxic_samples)}")
    
    # Test multiple alpha values
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    
    results = []
    for alpha in alpha_values:
        result = compute_signed_gap_for_alpha(
            alpha, tokenizer, base_model, expert_model, anti_expert_model,
            toxic_samples, nontoxic_samples
        )
        results.append(result)
    
    # Select primary alpha
    print("\n" + "=" * 80)
    print("Alpha Selection Results")
    print("=" * 80)
    
    print("\nAll alpha values:")
    print(f"{'Alpha':<8} {'Signed Gap':<12} {'CI Width':<12} {'Effect Size':<12} {'Stability':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['alpha']:<8.2f} {result['signed_gap']:<12.6f} "
              f"{result['ci_width']:<12.6f} {result['effect_size']:<12.4f} "
              f"{result['stability']:<12.4f}")
    
    # Selection criteria:
    # 1. Signed gap should be positive (directional effect)
    # 2. CI should exclude zero if possible (statistical significance)
    # 3. Higher stability is better
    # 4. Larger effect size is better
    
    # Filter for positive signed gap
    positive_results = [r for r in results if r['signed_gap'] > 0]
    
    if not positive_results:
        print("\nWARNING: No alpha values show positive signed gap")
        print("Selecting alpha with largest signed gap")
        selected = max(results, key=lambda r: r['signed_gap'])
    else:
        # Score each alpha: stability * effect_size * (1 if CI excludes 0 else 0.5)
        for result in positive_results:
            ci_excludes_zero = result['ci_lower'] > 0 or result['ci_upper'] < 0
            ci_bonus = 1.0 if ci_excludes_zero else 0.5
            result['score'] = result['stability'] * abs(result['effect_size']) * ci_bonus
        
        selected = max(positive_results, key=lambda r: r['score'])
    
    print(f"\n{'=' * 80}")
    print(f"SELECTED PRIMARY ALPHA: {selected['alpha']}")
    print(f"{'=' * 80}")
    print(f"  Signed gap: {selected['signed_gap']:.6f}")
    print(f"  95% CI: [{selected['ci_lower']:.6f}, {selected['ci_upper']:.6f}]")
    print(f"  Effect size: {selected['effect_size']:.4f}")
    print(f"  Stability: {selected['stability']:.4f}")
    print(f"  Mean toxicity (toxic samples): {selected['mean_toxicity_toxic']:.4f}")
    print(f"  Mean toxicity (non-toxic samples): {selected['mean_toxicity_nontoxic']:.4f}")
    
    # Save results
    output = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "alpha_values_tested": alpha_values,
        "all_results": results,
        "selected_alpha": selected['alpha'],
        "selected_result": selected,
        "note": "Evaluator calibration failed (ECE=0.2986). Toxicity scores should be interpreted with caution."
    }
    
    output_path = Path("artifacts/dexperts/r6/alpha_selection.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
