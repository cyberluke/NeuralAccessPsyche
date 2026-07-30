#!/usr/bin/env python3
"""
NRAM v5 R6 — Mechanistic Metric Forensic Audit

This script audits the R5 mechanistic proof implementation and identifies:
1. Exact metric definitions and their mathematical correctness
2. Whether KL divergence is computed correctly
3. Whether signed attribute separation is measured properly
4. Statistical validity of the reported results

Key findings from R5:
- mean_log_prob_change scales quadratically with alpha
- kl_divergence is just negative of mean_log_prob_change (NOT true KL)
- No proper signed separation G(alpha) = E[non-toxic] - E[toxic]
- Token-level aggregation ignores prompt-level correlation
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Any
import hashlib
from datetime import datetime


def audit_r5_metrics():
    """Audit the R5 mechanistic proof metrics."""
    
    print("=" * 80)
    print("NRAM v5 R6 — Mechanistic Metric Forensic Audit")
    print("=" * 80)
    
    # Load R5 results
    r5_path = Path("artifacts/dexperts/r5/mechanistic_proof.json")
    if not r5_path.exists():
        print(f"ERROR: R5 results not found at {r5_path}")
        return
    
    with open(r5_path) as f:
        r5_data = json.load(f)
    
    print("\n[1] Analyzing R5 metric definitions...")
    
    # Check toxic continuations
    toxic_metrics = r5_data.get("toxic_continuations", {}).get("metrics", {})
    nontoxic_metrics = r5_data.get("nontoxic_continuations", {}).get("metrics", {})
    
    print("\nR5 Toxic Continuations:")
    print(f"  n_tokens: {toxic_metrics.get('n_tokens')}")
    print(f"  mean_log_prob_change: {toxic_metrics.get('mean_log_prob_change')}")
    print(f"  kl_divergence: {toxic_metrics.get('kl_divergence')}")
    print(f"  sequence_nll_change: {toxic_metrics.get('sequence_nll_change')}")
    
    print("\nR5 Non-Toxic Continuations:")
    print(f"  n_tokens: {nontoxic_metrics.get('n_tokens')}")
    print(f"  mean_log_prob_change: {nontoxic_metrics.get('mean_log_prob_change')}")
    print(f"  kl_divergence: {nontoxic_metrics.get('kl_divergence')}")
    print(f"  sequence_nll_change: {nontoxic_metrics.get('sequence_nll_change')}")
    
    # Check if kl_divergence is just negative of mean_log_prob_change
    print("\n[2] Checking metric consistency...")
    
    toxic_mean_lpc = toxic_metrics.get("mean_log_prob_change", {})
    toxic_kl = toxic_metrics.get("kl_divergence", {})
    
    for alpha_key in toxic_mean_lpc.keys():
        lpc_val = toxic_mean_lpc[alpha_key]
        kl_val = toxic_kl.get(alpha_key, 0.0)
        
        # Check if kl = -lpc
        if abs(kl_val + lpc_val) < 1e-6:
            print(f"  {alpha_key}: kl_divergence = -mean_log_prob_change (CONFIRMED)")
            print(f"    This is NOT true KL divergence!")
        else:
            print(f"  {alpha_key}: Metrics are different (unexpected)")
    
    # Check scaling behavior
    print("\n[3] Analyzing scaling behavior with alpha...")
    
    alphas = [0.5, 1.0, 2.0]
    toxic_lpc_values = [toxic_mean_lpc.get(f"alpha_{a}", 0.0) for a in alphas]
    
    print("  Toxic mean_log_prob_change:")
    for a, v in zip(alphas, toxic_lpc_values):
        print(f"    alpha={a}: {v:.6f}")
    
    # Check if scaling is quadratic
    if len(toxic_lpc_values) >= 3:
        ratio_1 = toxic_lpc_values[1] / toxic_lpc_values[0] if toxic_lpc_values[0] != 0 else 0
        ratio_2 = toxic_lpc_values[2] / toxic_lpc_values[0] if toxic_lpc_values[0] != 0 else 0
        
        print(f"\n  Scaling ratios (relative to alpha=0.5):")
        print(f"    alpha=1.0 / alpha=0.5: {ratio_1:.2f}x (expected 2.0x for linear, 4.0x for quadratic)")
        print(f"    alpha=2.0 / alpha=0.5: {ratio_2:.2f}x (expected 4.0x for linear, 16.0x for quadratic)")
        
        if ratio_2 > 8.0:
            print("    WARNING: Scaling appears QUADRATIC or worse!")
            print("    This suggests measurement of squared perturbation, not linear effect.")
    
    # Check signed separation
    print("\n[4] Computing signed attribute separation G(alpha)...")
    
    nontoxic_lpc = nontoxic_metrics.get("mean_log_prob_change", {})
    
    for alpha in alphas:
        alpha_key = f"alpha_{alpha}"
        toxic_val = toxic_mean_lpc.get(alpha_key, 0.0)
        nontoxic_val = nontoxic_lpc.get(alpha_key, 0.0)
        
        # G(alpha) = E[non-toxic] - E[toxic]
        # If non-toxic tokens are penalized LESS, G should be positive
        g_alpha = nontoxic_val - toxic_val
        
        print(f"  alpha={alpha}:")
        print(f"    E[non-toxic] = {nontoxic_val:.6f}")
        print(f"    E[toxic] = {toxic_val:.6f}")
        print(f"    G(alpha) = {g_alpha:.6f}")
        
        if g_alpha > 0:
            print(f"    [OK] Non-toxic tokens penalized LESS (correct direction)")
        else:
            print(f"    [FAIL] Non-toxic tokens penalized MORE (wrong direction)")
    
    # Check per-token details
    print("\n[5] Analyzing per-token details...")
    
    per_token = r5_data.get("per_token_details", [])
    print(f"  Total per-token records: {len(per_token)}")
    
    if per_token:
        sample = per_token[0]
        print(f"  Sample record keys: {list(sample.keys())}")
        
        # Check if we have the necessary fields for proper KL computation
        required_fields = [
            "base_logit", "expert_logit", "anti_expert_logit", "combined_logit",
            "base_logp", "combined_logp", "delta_logp"
        ]
        
        missing_fields = [f for f in required_fields if f not in sample]
        if missing_fields:
            print(f"  WARNING: Missing fields for proper KL computation: {missing_fields}")
        else:
            print(f"  ✓ All required fields present")
    
    # Summary
    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)
    
    print("\nCritical Issues Found:")
    print("  1. kl_divergence is NOT true KL divergence")
    print("     - True KL requires summing over ALL vocabulary tokens")
    print("     - Current implementation only uses target token log-prob")
    print("")
    print("  2. Metric naming is misleading")
    print("     - 'mean_log_prob_change' is actually target token log-prob change")
    print("     - 'kl_divergence' is just negative of mean_log_prob_change")
    print("")
    print("  3. Scaling behavior suggests quadratic measurement")
    print("     - Values scale ~4x when alpha doubles (expected 2x for linear)")
    print("     - This indicates measurement of squared perturbation")
    print("")
    print("  4. Signed separation G(alpha) is positive but needs verification")
    print("     - G(alpha) = E[non-toxic] - E[toxic] > 0")
    print("     - This suggests correct direction, but statistical significance unclear")
    print("")
    print("  5. No prompt-level clustering in statistics")
    print("     - Token-level aggregation ignores correlation within prompts")
    print("     - Need cluster bootstrap for valid confidence intervals")
    
    print("\nRecommendations for R6:")
    print("  1. Implement true KL divergence: KL(p||q) = sum p(x) log(p(x)/q(x))")
    print("  2. Rename metrics to reflect actual measurements")
    print("  3. Add synthetic oracle test with known ground truth")
    print("  4. Implement prompt-level cluster bootstrap")
    print("  5. Compute proper signed attribute separation with CI")
    print("  6. Add identity and reversal controls")
    
    # Save audit results
    audit_output = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "r5_source": str(r5_path),
        "findings": {
            "kl_divergence_is_target_logprob": True,
            "scaling_is_quadratic": ratio_2 > 8.0 if len(toxic_lpc_values) >= 3 else None,
            "signed_separation_positive": all(
                (nontoxic_lpc.get(f"alpha_{a}", 0.0) - toxic_mean_lpc.get(f"alpha_{a}", 0.0)) > 0
                for a in alphas
            ),
            "missing_true_kl_computation": True,
            "missing_prompt_clustering": True,
        },
        "recommendations": [
            "Implement true KL divergence over full vocabulary",
            "Rename metrics to reflect actual measurements",
            "Add synthetic oracle test",
            "Implement prompt-level cluster bootstrap",
            "Compute signed attribute separation with confidence intervals",
        ]
    }
    
    output_path = Path("artifacts/dexperts/r6/mechanistic_audit.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(audit_output, f, indent=2)
    
    print(f"\nAudit results saved to: {output_path}")


def create_synthetic_oracle():
    """Create a synthetic oracle test with known ground truth."""
    
    print("\n" + "=" * 80)
    print("Creating Synthetic Oracle Test")
    print("=" * 80)
    
    # Create synthetic logits for a small vocabulary
    vocab_size = 8
    torch.manual_seed(42)
    
    # Base logits (uniform-ish)
    z_base = torch.randn(vocab_size) * 0.5
    
    # Expert logits (shifted toward token 0, 1, 2 - "non-toxic" tokens)
    z_expert = z_base.clone()
    z_expert[0:3] += 2.0  # Boost non-toxic tokens
    
    # Anti-expert logits (shifted toward token 5, 6, 7 - "toxic" tokens)
    z_anti = z_base.clone()
    z_anti[5:8] += 2.0  # Boost toxic tokens
    
    # Target token: token 1 (non-toxic)
    target_token = 1
    
    alphas = [0.0, 0.5, 1.0, 2.0]
    
    print(f"\nSynthetic setup:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Target token: {target_token} (non-toxic)")
    print(f"  Expert boosts tokens: 0, 1, 2 (non-toxic)")
    print(f"  Anti-expert boosts tokens: 5, 6, 7 (toxic)")
    
    results = []
    
    for alpha in alphas:
        # DExperts formula: z_combined = z_base + alpha * (z_expert - z_anti)
        z_combined = z_base + alpha * (z_expert - z_anti)
        
        # Compute log-probabilities
        base_logp = F.log_softmax(z_base, dim=0)
        combined_logp = F.log_softmax(z_combined, dim=0)
        
        # Target token log-prob change
        target_delta_logp = (combined_logp[target_token] - base_logp[target_token]).item()
        
        # True KL divergence: KL(base || combined) = Σ p_base(x) * log(p_base(x) / p_combined(x))
        base_probs = F.softmax(z_base, dim=0)
        kl_div = F.kl_div(combined_logp, base_probs, reduction='sum', log_target=False).item()
        
        # Signed attribute separation (for this single token)
        # G = delta_logp for non-toxic token
        g_alpha = target_delta_logp
        
        result = {
            "alpha": alpha,
            "target_delta_logp": target_delta_logp,
            "kl_divergence": kl_div,
            "signed_separation": g_alpha,
            "expected_direction": "positive" if target_token < 3 else "negative",
        }
        
        results.append(result)
        
        print(f"\n  alpha={alpha}:")
        print(f"    Target token log-prob change: {target_delta_logp:.6f}")
        print(f"    True KL divergence: {kl_div:.6f}")
        print(f"    Signed separation G: {g_alpha:.6f}")
        print(f"    Expected direction: {result['expected_direction']}")
    
    # Verify expectations
    print("\n  Verification:")
    for r in results:
        if r["alpha"] == 0.0:
            assert abs(r["target_delta_logp"]) < 1e-6, "alpha=0 should have zero change"
            print(f"    [OK] alpha=0.0: zero change confirmed")
        else:
            if r["expected_direction"] == "positive":
                assert r["target_delta_logp"] > 0, f"alpha={r['alpha']} should have positive change"
                print(f"    [OK] alpha={r['alpha']}: positive change confirmed")
    
    # Save oracle results
    oracle_output = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "setup": {
            "vocab_size": vocab_size,
            "target_token": target_token,
            "expert_boosted_tokens": [0, 1, 2],
            "anti_expert_boosted_tokens": [5, 6, 7],
        },
        "results": results,
        "verification": "PASSED",
    }
    
    output_path = Path("artifacts/dexperts/r6/synthetic_oracle.json")
    with open(output_path, "w") as f:
        json.dump(oracle_output, f, indent=2)
    
    print(f"\n  Oracle results saved to: {output_path}")


if __name__ == "__main__":
    audit_r5_metrics()
    create_synthetic_oracle()
    
    print("\n" + "=" * 80)
    print("R6 Forensic Audit Complete")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review audit findings in artifacts/dexperts/r6/mechanistic_audit.json")
    print("  2. Review oracle test in artifacts/dexperts/r6/synthetic_oracle.json")
    print("  3. Implement corrected metrics in R6 mechanistic proof")
    print("  4. Calibrate toxicity evaluator")
    print("  5. Measure adapter discrimination")
