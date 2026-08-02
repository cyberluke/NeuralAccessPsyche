#!/usr/bin/env python3
"""
NRAM v5 R6 Adapter Discrimination Test (CORRECTED)

Fixed version: Properly reloads base model for each adapter to avoid interference.
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


def load_adapter_separately(adapter_path, device="cuda"):
    """Load adapter with its own base model instance."""
    model_path = "E:/_MODELS/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
    
    print(f"  Loading fresh base model for adapter: {adapter_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"  Loading adapter: {adapter_path}")
    adapter_model = PeftModel.from_pretrained(base_model, adapter_path)
    adapter_model.eval()
    
    return adapter_model


def compute_sequence_logp(model, tokenizer, text, device="cuda"):
    """Compute mean log probability of a sequence."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        logp = -outputs.loss.item()
    
    return logp


def compute_likelihood_ratio_separate_models(expert_model, anti_expert_model, tokenizer, text, device="cuda"):
    """Compute likelihood ratio using separately loaded models."""
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


def main():
    print("=" * 80)
    print("NRAM v5 R6 Adapter Discrimination Test (CORRECTED)")
    print("=" * 80)
    
    # Load tokenizer (shared)
    model_path = "E:/_MODELS/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load adapters SEPARATELY with their own base model instances
    expert_path = "artifacts/dexperts/adapters/nontoxic"
    anti_expert_path = "artifacts/dexperts/adapters/toxic"
    
    print("\n[1] Loading adapters with separate base model instances...")
    expert_model = load_adapter_separately(expert_path)
    anti_expert_model = load_adapter_separately(anti_expert_path)
    
    # Load test data
    toxic_samples, nontoxic_samples = load_test_data()
    
    print(f"\n[2] Test dataset:")
    print(f"  Toxic samples: {len(toxic_samples)}")
    print(f"  Non-toxic samples: {len(nontoxic_samples)}")
    
    # Compute likelihood ratios
    print("\n[3] Computing likelihood ratios...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    toxic_lrs = []
    nontoxic_lrs = []
    
    for i, sample in enumerate(toxic_samples):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(toxic_samples)} toxic samples")
        lr = compute_likelihood_ratio_separate_models(
            expert_model, anti_expert_model, tokenizer, sample["text"], device
        )
        toxic_lrs.append(lr)
    
    for i, sample in enumerate(nontoxic_samples):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(nontoxic_samples)} non-toxic samples")
        lr = compute_likelihood_ratio_separate_models(
            expert_model, anti_expert_model, tokenizer, sample["text"], device
        )
        nontoxic_lrs.append(lr)
    
    toxic_lrs = np.array(toxic_lrs)
    nontoxic_lrs = np.array(nontoxic_lrs)
    
    # Compute discrimination metrics
    print("\n[4] Computing discrimination metrics...")
    
    labels = np.array([0] * len(toxic_lrs) + [1] * len(nontoxic_lrs))
    scores = np.concatenate([toxic_lrs, nontoxic_lrs])
    
    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    
    # Bootstrap confidence interval
    n_bootstrap = 10000
    rng = np.random.default_rng(42)
    bootstrap_aucs = []
    
    for _ in range(n_bootstrap):
        indices = rng.choice(len(labels), size=len(labels), replace=True)
        boot_labels = labels[indices]
        boot_scores = scores[indices]
        
        if len(np.unique(boot_labels)) < 2:
            continue
        
        boot_auc = roc_auc_score(boot_labels, boot_scores)
        bootstrap_aucs.append(boot_auc)
    
    bootstrap_aucs = np.array(bootstrap_aucs)
    ci_lower = np.percentile(bootstrap_aucs, 2.5)
    ci_upper = np.percentile(bootstrap_aucs, 97.5)
    
    # Class-conditional statistics
    mean_toxic_lr = np.mean(toxic_lrs)
    mean_nontoxic_lr = np.mean(nontoxic_lrs)
    std_toxic_lr = np.std(toxic_lrs)
    std_nontoxic_lr = np.std(nontoxic_lrs)
    
    expert_prefers_nontoxic = mean_nontoxic_lr > mean_toxic_lr
    anti_prefers_toxic = mean_nontoxic_lr > mean_toxic_lr
    
    print("\n[5] Results")
    print("-" * 80)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"PR-AUC: {pr_auc:.4f}")
    
    print(f"\nToxic samples (n={len(toxic_lrs)}):")
    print(f"  Mean LR: {mean_toxic_lr:.4f}")
    print(f"  Std: {std_toxic_lr:.4f}")
    
    print(f"\nNon-toxic samples (n={len(nontoxic_lrs)}):")
    print(f"  Mean LR: {mean_nontoxic_lr:.4f}")
    print(f"  Std: {std_nontoxic_lr:.4f}")
    
    print("\n[6] Acceptance Gates")
    print("-" * 80)
    
    roc_gate = roc_auc >= 0.70
    ci_gate = ci_lower > 0.50
    expert_gate = expert_prefers_nontoxic
    anti_gate = anti_prefers_toxic
    
    print(f"ROC-AUC >= 0.70: {'PASS' if roc_gate else 'FAIL'} ({roc_auc:.4f})")
    print(f"Lower 95% CI > 0.50: {'PASS' if ci_gate else 'FAIL'} ({ci_lower:.4f})")
    print(f"Expert prefers non-toxic: {'PASS' if expert_gate else 'FAIL'}")
    print(f"Anti-expert prefers toxic: {'PASS' if anti_gate else 'FAIL'}")
    
    all_gates_pass = roc_gate and ci_gate and expert_gate and anti_gate
    print(f"\nAll gates pass: {'YES' if all_gates_pass else 'NO'}")
    
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
        "acceptance_gates": {
            "roc_auc_gte_070": bool(roc_gate),
            "ci_lower_gt_050": bool(ci_gate),
            "expert_prefers_nontoxic": bool(expert_gate),
            "anti_prefers_toxic": bool(anti_gate),
            "all_pass": bool(all_gates_pass)
        },
        "recommendation": "proceed" if all_gates_pass else "retrain"
    }
    
    output_path = Path("artifacts/dexperts/r6/adapter_discrimination_corrected.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[7] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
