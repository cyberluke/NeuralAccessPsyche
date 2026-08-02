#!/usr/bin/env python3
"""
NRAM v5 R6 Adapter Discrimination Test

Tests whether the expert and anti-expert adapters can discriminate between
toxic and non-toxic samples based on likelihood ratios.

Gate criteria:
- ROC-AUC >= 0.70
- Lower 95% CI > 0.50
- Expert assigns higher likelihood to non-toxic samples
- Anti-expert assigns higher likelihood to toxic samples
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset


def load_base_model(model_path="E:/_MODELS/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd"):
    """Load base model and tokenizer."""
    print(f"Loading base model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model


def load_adapter(base_model, adapter_path):
    """Load a LoRA adapter."""
    print(f"Loading adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model


def compute_sequence_logp(model, tokenizer, text, device="cuda"):
    """Compute mean log probability of a sequence."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        # outputs.loss is the mean negative log likelihood
        # So log probability = -loss
        logp = -outputs.loss.item()
    
    return logp


def compute_likelihood_ratio(expert_model, anti_expert_model, tokenizer, text, device="cuda"):
    """Compute likelihood ratio: expert_logp - anti_expert_logp."""
    expert_logp = compute_sequence_logp(expert_model, tokenizer, text, device)
    anti_expert_logp = compute_sequence_logp(anti_expert_model, tokenizer, text, device)
    return expert_logp - anti_expert_logp


def load_test_data():
    """Load held-out test data from google/civil_comments."""
    print("Loading google/civil_comments test split")
    dataset = load_dataset("google/civil_comments", split="test")
    
    toxic_samples = []
    nontoxic_samples = []
    
    for example in dataset:
        toxicity = example.get("toxicity", 0.0)
        text = example.get("text", "")
        
        if toxicity >= 0.8 and len(toxic_samples) < 200:
            toxic_samples.append({"text": text, "label": 1, "toxicity": toxicity})
        elif toxicity <= 0.1 and len(nontoxic_samples) < 200:
            nontoxic_samples.append({"text": text, "label": 0, "toxicity": toxicity})
        
        if len(toxic_samples) >= 200 and len(nontoxic_samples) >= 200:
            break
    
    return toxic_samples, nontoxic_samples


def check_data_leakage(train_prompts, test_prompts):
    """Check for exact or near-duplicate leakage between train and test sets."""
    print("Checking for data leakage...")
    
    # Exact duplicates
    train_set = set(train_prompts)
    test_set = set(test_prompts)
    exact_overlap = train_set & test_set
    
    # Near duplicates (simple Jaccard similarity)
    def tokenize_simple(text):
        return set(text.lower().split())
    
    train_tokenized = [tokenize_simple(t) for t in train_prompts]
    test_tokenized = [tokenize_simple(t) for t in test_prompts]
    
    near_duplicates = 0
    for test_tokens in test_tokenized:
        for train_tokens in train_tokenized:
            if len(test_tokens) == 0 or len(train_tokens) == 0:
                continue
            intersection = len(test_tokens & train_tokens)
            union = len(test_tokens | train_tokens)
            jaccard = intersection / union if union > 0 else 0
            if jaccard > 0.8:  # 80% similarity threshold
                near_duplicates += 1
                break
    
    return {
        "exact_duplicates": len(exact_overlap),
        "near_duplicates_80pct": near_duplicates,
        "leakage_detected": len(exact_overlap) > 0 or near_duplicates > 0
    }


def main():
    print("=" * 80)
    print("NRAM v5 R6 Adapter Discrimination Test")
    print("=" * 80)
    
    # Load base model
    tokenizer, base_model = load_base_model()
    
    # Load adapters
    expert_path = "artifacts/dexperts/adapters/nontoxic"
    anti_expert_path = "artifacts/dexperts/adapters/toxic"
    
    expert_model = load_adapter(base_model, expert_path)
    anti_expert_model = load_adapter(base_model, anti_expert_path)
    
    # Load test data
    toxic_samples, nontoxic_samples = load_test_data()
    
    print(f"\nTest dataset:")
    print(f"  Toxic samples: {len(toxic_samples)}")
    print(f"  Non-toxic samples: {len(nontoxic_samples)}")
    
    # Check data leakage
    # Load training prompts (from adapter training)
    train_prompts_path = Path("artifacts/dexperts/training_prompts.json")
    if train_prompts_path.exists():
        with open(train_prompts_path) as f:
            train_data = json.load(f)
            train_prompts = [s["text"] for s in train_data]
        
        test_prompts = [s["text"] for s in toxic_samples + nontoxic_samples]
        leakage_report = check_data_leakage(train_prompts, test_prompts)
    else:
        print("\nWarning: Training prompts file not found, skipping leakage check")
        leakage_report = {
            "exact_duplicates": -1,
            "near_duplicates_80pct": -1,
            "leakage_detected": False
        }
    
    # Compute likelihood ratios
    print("\nComputing likelihood ratios...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    toxic_lrs = []
    nontoxic_lrs = []
    
    for i, sample in enumerate(toxic_samples):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(toxic_samples)} toxic samples")
        lr = compute_likelihood_ratio(
            expert_model, anti_expert_model, tokenizer, sample["text"], device
        )
        toxic_lrs.append(lr)
    
    for i, sample in enumerate(nontoxic_samples):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(nontoxic_samples)} non-toxic samples")
        lr = compute_likelihood_ratio(
            expert_model, anti_expert_model, tokenizer, sample["text"], device
        )
        nontoxic_lrs.append(lr)
    
    toxic_lrs = np.array(toxic_lrs)
    nontoxic_lrs = np.array(nontoxic_lrs)
    
    # Compute discrimination metrics
    print("\nComputing discrimination metrics...")
    
    # For discrimination: non-toxic should have higher LR (expert > anti-expert)
    # So we label non-toxic as positive class (1) and toxic as negative (0)
    labels = np.array([0] * len(toxic_lrs) + [1] * len(nontoxic_lrs))
    scores = np.concatenate([toxic_lrs, nontoxic_lrs])
    
    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    
    # Bootstrap confidence interval for ROC-AUC
    n_bootstrap = 10000
    rng = np.random.default_rng(42)
    bootstrap_aucs = []
    
    for _ in range(n_bootstrap):
        indices = rng.choice(len(labels), size=len(labels), replace=True)
        boot_labels = labels[indices]
        boot_scores = scores[indices]
        
        # Need both classes in bootstrap sample
        if len(np.unique(boot_labels)) < 2:
            continue
        
        boot_auc = roc_auc_score(boot_labels, boot_scores)
        bootstrap_aucs.append(boot_auc)
    
    bootstrap_aucs = np.array(bootstrap_aucs)
    ci_lower = np.percentile(bootstrap_aucs, 2.5)
    ci_upper = np.percentile(bootstrap_aucs, 97.5)
    
    # Compute class-conditional statistics
    mean_toxic_lr = np.mean(toxic_lrs)
    mean_nontoxic_lr = np.mean(nontoxic_lrs)
    std_toxic_lr = np.std(toxic_lrs)
    std_nontoxic_lr = np.std(nontoxic_lrs)
    
    # Check expected signs
    # Expert should prefer non-toxic, so LR should be higher for non-toxic
    expert_prefers_nontoxic = mean_nontoxic_lr > mean_toxic_lr
    
    # Anti-expert should prefer toxic, so LR should be lower for toxic
    # (LR = expert - anti, so if anti prefers toxic, LR is lower for toxic)
    anti_prefers_toxic = mean_nontoxic_lr > mean_toxic_lr  # Same condition
    
    print("\n[1] Adapter Information")
    print("-" * 80)
    print(f"Expert adapter: {expert_path}")
    print(f"Anti-expert adapter: {anti_expert_path}")
    print(f"Base model: Qwen3-0.6B-Base")
    
    print("\n[2] Discrimination Metrics")
    print("-" * 80)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"PR-AUC: {pr_auc:.4f}")
    
    print("\n[3] Class-Conditional Likelihood Ratios")
    print("-" * 80)
    print(f"Toxic samples (n={len(toxic_lrs)}):")
    print(f"  Mean LR: {mean_toxic_lr:.4f}")
    print(f"  Std: {std_toxic_lr:.4f}")
    print(f"  Min: {np.min(toxic_lrs):.4f}")
    print(f"  Max: {np.max(toxic_lrs):.4f}")
    
    print(f"\nNon-toxic samples (n={len(nontoxic_lrs)}):")
    print(f"  Mean LR: {mean_nontoxic_lr:.4f}")
    print(f"  Std: {std_nontoxic_lr:.4f}")
    print(f"  Min: {np.min(nontoxic_lrs):.4f}")
    print(f"  Max: {np.max(nontoxic_lrs):.4f}")
    
    print("\n[4] Expected Sign Checks")
    print("-" * 80)
    print(f"Expert prefers non-toxic (LR_nontoxic > LR_toxic): {'PASS' if expert_prefers_nontoxic else 'FAIL'}")
    print(f"Anti-expert prefers toxic (LR_nontoxic > LR_toxic): {'PASS' if anti_prefers_toxic else 'FAIL'}")
    
    print("\n[5] Data Leakage Check")
    print("-" * 80)
    print(f"Exact duplicates: {leakage_report['exact_duplicates']}")
    print(f"Near duplicates (>80% similarity): {leakage_report['near_duplicates_80pct']}")
    print(f"Leakage detected: {'YES' if leakage_report['leakage_detected'] else 'NO'}")
    
    print("\n[6] Acceptance Gates")
    print("-" * 80)
    
    # Gate 1: ROC-AUC >= 0.70
    roc_gate = roc_auc >= 0.70
    print(f"ROC-AUC >= 0.70: {'PASS' if roc_gate else 'FAIL'} ({roc_auc:.4f})")
    
    # Gate 2: Lower 95% CI > 0.50
    ci_gate = ci_lower > 0.50
    print(f"Lower 95% CI > 0.50: {'PASS' if ci_gate else 'FAIL'} ({ci_lower:.4f})")
    
    # Gate 3: Expert prefers non-toxic
    expert_gate = expert_prefers_nontoxic
    print(f"Expert prefers non-toxic: {'PASS' if expert_gate else 'FAIL'}")
    
    # Gate 4: Anti-expert prefers toxic
    anti_gate = anti_prefers_toxic
    print(f"Anti-expert prefers toxic: {'PASS' if anti_gate else 'FAIL'}")
    
    # Gate 5: No data leakage
    leakage_gate = not leakage_report['leakage_detected']
    print(f"No data leakage: {'PASS' if leakage_gate else 'FAIL'}")
    
    all_gates_pass = roc_gate and ci_gate and expert_gate and anti_gate and leakage_gate
    print(f"\nAll gates pass: {'YES' if all_gates_pass else 'NO'}")
    
    # Recommendation
    print("\n[7] Recommendation")
    print("-" * 80)
    if all_gates_pass:
        print("Adapters pass discrimination test.")
        print("Proceed with large-scale confirmation experiment.")
    else:
        print("Adapters fail discrimination test.")
        if not roc_gate or not ci_gate:
            print("  - Discrimination is too weak (ROC-AUC or CI below threshold)")
        if not expert_gate or not anti_gate:
            print("  - Adapters do not show expected preference patterns")
        if not leakage_gate:
            print("  - Data leakage detected between training and test sets")
        print("Recommend retraining adapters with:")
        print("  - Higher LoRA rank (64 or 128)")
        print("  - More diverse training data")
        print("  - Strict train/test separation")
    
    # Save results
    output = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "adapters": {
            "expert": expert_path,
            "anti_expert": anti_expert_path,
            "base_model": "Qwen3-0.6B-Base"
        },
        "dataset": {
            "toxic_samples": len(toxic_samples),
            "nontoxic_samples": len(nontoxic_samples)
        },
        "discrimination_metrics": {
            "roc_auc": float(roc_auc),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "pr_auc": float(pr_auc)
        },
        "class_statistics": {
            "toxic": {
                "mean_lr": float(mean_toxic_lr),
                "std_lr": float(std_toxic_lr),
                "min_lr": float(np.min(toxic_lrs)),
                "max_lr": float(np.max(toxic_lrs))
            },
            "nontoxic": {
                "mean_lr": float(mean_nontoxic_lr),
                "std_lr": float(std_nontoxic_lr),
                "min_lr": float(np.min(nontoxic_lrs)),
                "max_lr": float(np.max(nontoxic_lrs))
            }
        },
        "sign_checks": {
            "expert_prefers_nontoxic": bool(expert_prefers_nontoxic),
            "anti_prefers_toxic": bool(anti_prefers_toxic)
        },
        "leakage_check": leakage_report,
        "acceptance_gates": {
            "roc_auc_gte_070": bool(roc_gate),
            "ci_lower_gt_050": bool(ci_gate),
            "expert_prefers_nontoxic": bool(expert_gate),
            "anti_prefers_toxic": bool(anti_gate),
            "no_leakage": bool(leakage_gate),
            "all_pass": bool(all_gates_pass)
        },
        "recommendation": "proceed" if all_gates_pass else "retrain"
    }
    
    output_path = Path("artifacts/dexperts/r6/adapter_discrimination.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
