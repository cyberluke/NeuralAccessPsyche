#!/usr/bin/env python3
"""
NRAM v5 R6 Evaluator Calibration

Calibrates the toxicity evaluator (unitary/toxic-bert) with:
- Exact model and immutable revision
- Positive/negative label mapping
- ROC-AUC, PR-AUC, Brier score, calibration error
- Borderline, truncation, and floor-effect diagnostics
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset


def load_evaluator(model_path="E:/_MODELS/huggingface/hub/models--unitary--toxic-bert/snapshots/4d6c22e74ba2fdd26bc4f7238f50766b045a0d94"):
    """Load toxicity evaluator."""
    print(f"Loading evaluator from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def get_toxicity_score(text, tokenizer, model):
    """Get toxicity score for a single text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # toxic-bert outputs logits for 6 classes
    # Class 0: toxic, Class 1: severe_toxic, Class 2: obscene, 
    # Class 3: threat, Class 4: insult, Class 5: identity_hate
    # We use the probability of class 0 (toxic)
    probs = torch.softmax(outputs.logits, dim=1)
    return probs[0, 0].item()


def load_calibration_data():
    """Load calibration dataset from google/civil_comments."""
    print("Loading google/civil_comments dataset")
    dataset = load_dataset("google/civil_comments", split="train")
    
    # Sample balanced dataset
    toxic_samples = []
    nontoxic_samples = []
    
    for example in dataset:
        toxicity = example.get("toxicity", 0.0)
        text = example.get("text", "")
        
        if toxicity >= 0.8 and len(toxic_samples) < 500:
            toxic_samples.append({"text": text, "label": 1, "toxicity": toxicity})
        elif toxicity <= 0.1 and len(nontoxic_samples) < 500:
            nontoxic_samples.append({"text": text, "label": 0, "toxicity": toxicity})
        
        if len(toxic_samples) >= 500 and len(nontoxic_samples) >= 500:
            break
    
    # Add borderline samples
    borderline_samples = []
    for example in dataset:
        toxicity = example.get("toxicity", 0.0)
        text = example.get("text", "")
        
        if 0.4 <= toxicity <= 0.6 and len(borderline_samples) < 200:
            borderline_samples.append({"text": text, "label": 1 if toxicity > 0.5 else 0, "toxicity": toxicity})
        
        if len(borderline_samples) >= 200:
            break
    
    return toxic_samples, nontoxic_samples, borderline_samples


def compute_metrics(labels, scores):
    """Compute evaluation metrics."""
    # ROC-AUC
    roc_auc = roc_auc_score(labels, scores)
    
    # PR-AUC
    pr_auc = average_precision_score(labels, scores)
    
    # Brier score
    brier = brier_score_loss(labels, scores)
    
    # Calibration error
    prob_true, prob_pred = calibration_curve(labels, scores, n_bins=10)
    ece = np.mean(np.abs(prob_true - prob_pred))
    
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "expected_calibration_error": ece
    }


def main():
    print("=" * 80)
    print("NRAM v5 R6 Evaluator Calibration")
    print("=" * 80)
    
    # Load evaluator
    tokenizer, model = load_evaluator()
    
    # Load calibration data
    toxic_samples, nontoxic_samples, borderline_samples = load_calibration_data()
    
    print(f"\nDataset composition:")
    print(f"  Toxic samples: {len(toxic_samples)}")
    print(f"  Non-toxic samples: {len(nontoxic_samples)}")
    print(f"  Borderline samples: {len(borderline_samples)}")
    
    # Score all samples
    print("\nScoring samples...")
    all_samples = toxic_samples + nontoxic_samples + borderline_samples
    labels = []
    scores = []
    
    for i, sample in enumerate(all_samples):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(all_samples)} samples")
        
        score = get_toxicity_score(sample["text"], tokenizer, model)
        labels.append(sample["label"])
        scores.append(score)
    
    labels = np.array(labels)
    scores = np.array(scores)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(labels, scores)
    
    print("\n[1] Evaluator Model Information")
    print("-" * 80)
    print(f"Model: unitary/toxic-bert")
    print(f"Revision: 4d6c22e74ba2fdd26bc4f7238f50766b045a0d94")
    print(f"Label mapping: Class 0 = toxic (positive class)")
    print(f"Output: softmax probability of class 0")
    
    print("\n[2] Performance Metrics")
    print("-" * 80)
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Brier score: {metrics['brier_score']:.4f}")
    print(f"Expected calibration error: {metrics['expected_calibration_error']:.4f}")
    
    print("\n[3] Class-Conditional Score Distributions")
    print("-" * 80)
    
    toxic_mask = labels == 1
    nontoxic_mask = labels == 0
    
    toxic_scores = scores[toxic_mask]
    nontoxic_scores = scores[nontoxic_mask]
    
    print(f"Toxic samples (n={len(toxic_scores)}):")
    print(f"  Mean score: {np.mean(toxic_scores):.4f}")
    print(f"  Std: {np.std(toxic_scores):.4f}")
    print(f"  Min: {np.min(toxic_scores):.4f}")
    print(f"  Max: {np.max(toxic_scores):.4f}")
    
    print(f"\nNon-toxic samples (n={len(nontoxic_scores)}):")
    print(f"  Mean score: {np.mean(nontoxic_scores):.4f}")
    print(f"  Std: {np.std(nontoxic_scores):.4f}")
    print(f"  Min: {np.min(nontoxic_scores):.4f}")
    print(f"  Max: {np.max(nontoxic_scores):.4f}")
    
    print("\n[4] Borderline Analysis")
    print("-" * 80)
    borderline_mask = np.array([s in borderline_samples for s in all_samples])
    borderline_scores = scores[borderline_mask]
    
    print(f"Borderline samples (n={len(borderline_scores)}):")
    print(f"  Mean score: {np.mean(borderline_scores):.4f}")
    print(f"  Std: {np.std(borderline_scores):.4f}")
    print(f"  Scores in [0.4, 0.6]: {np.sum((borderline_scores >= 0.4) & (borderline_scores <= 0.6))}/{len(borderline_scores)}")
    
    print("\n[5] Truncation Diagnostics")
    print("-" * 80)
    long_texts = [s for s in all_samples if len(s["text"]) > 512]
    print(f"Long texts (>512 chars): {len(long_texts)}")
    
    if long_texts:
        long_scores = [get_toxicity_score(s["text"], tokenizer, model) for s in long_texts[:10]]
        print(f"Sample scores for long texts: {np.mean(long_scores):.4f} ± {np.std(long_scores):.4f}")
    
    print("\n[6] Floor Effect Analysis")
    print("-" * 80)
    floor_samples = [s for s in all_samples if len(s["text"]) < 20]
    print(f"Very short texts (<20 chars): {len(floor_samples)}")
    
    if floor_samples:
        floor_scores = [get_toxicity_score(s["text"], tokenizer, model) for s in floor_samples]
        print(f"Scores for short texts: {np.mean(floor_scores):.4f} ± {np.std(floor_scores):.4f}")
    
    print("\n[7] Acceptance Gates")
    print("-" * 80)
    
    # Gate 1: ROC-AUC > 0.70
    roc_gate = metrics["roc_auc"] > 0.70
    print(f"ROC-AUC > 0.70: {'PASS' if roc_gate else 'FAIL'} ({metrics['roc_auc']:.4f})")
    
    # Gate 2: Toxic scores > Non-toxic scores
    separation_gate = np.mean(toxic_scores) > np.mean(nontoxic_scores)
    print(f"Toxic > Non-toxic: {'PASS' if separation_gate else 'FAIL'}")
    
    # Gate 3: Calibration error < 0.10
    calibration_gate = metrics["expected_calibration_error"] < 0.10
    print(f"ECE < 0.10: {'PASS' if calibration_gate else 'FAIL'} ({metrics['expected_calibration_error']:.4f})")
    
    all_gates_pass = roc_gate and separation_gate and calibration_gate
    print(f"\nAll gates pass: {'YES' if all_gates_pass else 'NO'}")
    
    # Save results
    output = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "evaluator": {
            "model": "unitary/toxic-bert",
            "revision": "4d6c22e74ba2fdd26bc4f7238f50766b045a0d94",
            "label_mapping": "Class 0 = toxic (positive class)"
        },
        "dataset": {
            "toxic_samples": len(toxic_samples),
            "nontoxic_samples": len(nontoxic_samples),
            "borderline_samples": len(borderline_samples)
        },
        "metrics": metrics,
        "class_distributions": {
            "toxic": {
                "mean": float(np.mean(toxic_scores)),
                "std": float(np.std(toxic_scores)),
                "min": float(np.min(toxic_scores)),
                "max": float(np.max(toxic_scores))
            },
            "nontoxic": {
                "mean": float(np.mean(nontoxic_scores)),
                "std": float(np.std(nontoxic_scores)),
                "min": float(np.min(nontoxic_scores)),
                "max": float(np.max(nontoxic_scores))
            }
        },
        "borderline": {
            "mean": float(np.mean(borderline_scores)),
            "std": float(np.std(borderline_scores)),
            "in_range_04_06": int(np.sum((borderline_scores >= 0.4) & (borderline_scores <= 0.6)))
        },
        "acceptance_gates": {
            "roc_auc_gt_070": bool(roc_gate),
            "toxic_gt_nontoxic": bool(separation_gate),
            "ece_lt_010": bool(calibration_gate),
            "all_pass": bool(all_gates_pass)
        }
    }
    
    output_path = Path("artifacts/dexperts/r6/evaluator_calibration.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
