#!/usr/bin/env python3
"""
NRAM v5 R6 Power Reconciliation Analysis

Reconciles Phase 3B power analysis with Phase 4 observed results.
Computes achieved power and required sample sizes for observed effect.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime


def compute_achieved_power(effect_size, n_per_class, alpha=0.05, two_sided=True):
    """Compute achieved power for given effect size and sample size."""
    # Non-centrality parameter
    ncp = effect_size * np.sqrt(n_per_class / 2)
    
    # Critical value
    if two_sided:
        t_crit = stats.t.ppf(1 - alpha/2, df=2*n_per_class - 2)
        power = 1 - stats.nct.cdf(t_crit, df=2*n_per_class - 2, nc=ncp) + \
                stats.nct.cdf(-t_crit, df=2*n_per_class - 2, nc=ncp)
    else:
        t_crit = stats.t.ppf(1 - alpha, df=2*n_per_class - 2)
        power = 1 - stats.nct.cdf(t_crit, df=2*n_per_class - 2, nc=ncp)
    
    return power


def compute_required_n(effect_size, target_power=0.80, alpha=0.05, two_sided=True):
    """Compute required sample size per class for given effect size and power."""
    # Analytical approximation
    if two_sided:
        z_alpha = stats.norm.ppf(1 - alpha/2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_beta = stats.norm.ppf(target_power)
    
    # Required n per class
    n_required = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    
    return int(np.ceil(n_required))


def monte_carlo_power(effect_size, n_per_class, n_simulations=10000, alpha=0.05, two_sided=True):
    """Monte Carlo simulation of power."""
    rng = np.random.default_rng(42)
    rejections = 0
    
    for _ in range(n_simulations):
        # Generate two samples with specified effect size
        sample1 = rng.normal(0, 1, n_per_class)
        sample2 = rng.normal(effect_size, 1, n_per_class)
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(sample2, sample1, equal_var=True)
        
        if two_sided:
            if p_value < alpha:
                rejections += 1
        else:
            if p_value/2 < alpha and t_stat > 0:
                rejections += 1
    
    return rejections / n_simulations


def main():
    print("=" * 80)
    print("NRAM v5 R6 Power Reconciliation Analysis")
    print("=" * 80)
    
    # Load Phase 4 results
    phase4_path = Path("artifacts/dexperts/r6/mechanistic_proof_phase4.json")
    with open(phase4_path) as f:
        phase4_data = json.load(f)
    
    # Load Phase 3B results
    phase3b_path = Path("artifacts/dexperts/r6/power_analysis.json")
    with open(phase3b_path) as f:
        phase3b_data = json.load(f)
    
    print("\n[1] Phase 3B Power Analysis Assumptions")
    print("-" * 80)
    pilot_effect = phase3b_data["pilot_effect"]
    print(f"Pilot signed gap: {pilot_effect['signed_gap']:.6f}")
    print(f"Pilot Cohen's d: {pilot_effect['cohens_d']:.4f}")
    print(f"Pilot Hedges' g: {pilot_effect['hedges_g']:.4f}")
    print(f"Effect scenarios tested: 100%, 75%, 50% of pilot")
    print(f"Selected N=300 per class for 90% power at 50% effect scenario")
    print(f"MDE at N=300: 0.131 (from power_analysis.json)")
    
    print("\n[2] Phase 4 Observed Results")
    print("-" * 80)
    
    # Extract signed gaps for all conditions
    conditions = ["alpha_zero", "normal", "reversal"]
    alpha_values = {"alpha_zero": 0.0, "normal": 1.0, "reversal": 1.0}
    
    for condition in conditions:
        if condition in phase4_data["all_results"]:
            signed_gap = phase4_data["all_results"][condition]["signed_gap"]
            print(f"\n{condition.upper()} (alpha={alpha_values[condition]}):")
            print(f"  Signed gap: {signed_gap['point_estimate']:.6f}")
            print(f"  95% CI: [{signed_gap['ci_lower']:.6f}, {signed_gap['ci_upper']:.6f}]")
            print(f"  Effect size: {signed_gap['effect_size']:.4f}")
            print(f"  CI excludes 0: {signed_gap['ci_excludes_zero']}")
    
    # Also check alpha=0.5 and alpha=2.0 if available
    for alpha in [0.5, 2.0]:
        condition_name = f"alpha_{alpha}"
        if condition_name in phase4_data["all_results"]:
            signed_gap = phase4_data["all_results"][condition_name]["signed_gap"]
            print(f"\nALPHA={alpha}:")
            print(f"  Signed gap: {signed_gap['point_estimate']:.6f}")
            print(f"  95% CI: [{signed_gap['ci_lower']:.6f}, {signed_gap['ci_upper']:.6f}]")
            print(f"  Effect size: {signed_gap['effect_size']:.4f}")
    
    print("\n[3] Power Reconciliation")
    print("-" * 80)
    
    # Primary result: normal condition at alpha=1.0
    normal_result = phase4_data["all_results"]["normal"]["signed_gap"]
    observed_effect = normal_result["effect_size"]
    n_used = 300
    
    print(f"Observed standardized effect (d): {observed_effect:.4f}")
    print(f"Sample size used: N={n_used} per class")
    print(f"MDE at N=300: 0.131")
    print(f"Ratio (observed/MDE): {observed_effect/0.131:.3f}")
    
    # Compute achieved power
    print("\n[Achieved Power at Observed Effect]")
    power_analytical_two = compute_achieved_power(observed_effect, n_used, two_sided=True)
    power_analytical_one = compute_achieved_power(observed_effect, n_used, two_sided=False)
    power_mc_two = monte_carlo_power(observed_effect, n_used, two_sided=True)
    power_mc_one = monte_carlo_power(observed_effect, n_used, two_sided=False)
    
    print(f"  Two-sided alpha=0.05:")
    print(f"    Analytical: {power_analytical_two:.4f}")
    print(f"    Monte Carlo: {power_mc_two:.4f}")
    print(f"  One-sided alpha=0.05:")
    print(f"    Analytical: {power_analytical_one:.4f}")
    print(f"    Monte Carlo: {power_mc_one:.4f}")
    
    print("\n[4] Required Sample Sizes for Observed Effect (d=0.0489)")
    print("-" * 80)
    
    # Compute required N for different power levels
    scenarios = [
        ("Two-sided, power=0.80", 0.80, True),
        ("Two-sided, power=0.90", 0.90, True),
        ("One-sided, power=0.80", 0.80, False),
        ("One-sided, power=0.90", 0.90, False),
    ]
    
    for scenario_name, target_power, two_sided in scenarios:
        n_required = compute_required_n(observed_effect, target_power, two_sided=two_sided)
        
        # Verify with Monte Carlo
        power_check = monte_carlo_power(observed_effect, n_required, n_simulations=5000, two_sided=two_sided)
        
        print(f"\n{scenario_name}:")
        print(f"  Required N per class: {n_required}")
        print(f"  Total N: {2*n_required}")
        print(f"  Monte Carlo verification: {power_check:.4f}")
    
    print("\n[5] Reversal Analysis")
    print("-" * 80)
    
    reversal_result = phase4_data["all_results"]["reversal"]["signed_gap"]
    normal_result = phase4_data["all_results"]["normal"]["signed_gap"]
    
    # REVERSAL_IMPLEMENTATION_GATE
    print("\nREVERSAL_IMPLEMENTATION_GATE:")
    sign_flipped = (normal_result["point_estimate"] * reversal_result["point_estimate"]) < 0
    print(f"  Normal signed gap: {normal_result['point_estimate']:.6f}")
    print(f"  Reversal signed gap: {reversal_result['point_estimate']:.6f}")
    print(f"  Sign flipped: {sign_flipped}")
    print(f"  Gate status: {'PASS' if sign_flipped else 'FAIL'}")
    
    # REVERSAL_DIRECTION_ESTIMATE
    print("\nREVERSAL_DIRECTION_ESTIMATE:")
    print(f"  Reversal signed gap: {reversal_result['point_estimate']:.6f}")
    print(f"  95% CI: [{reversal_result['ci_lower']:.6f}, {reversal_result['ci_upper']:.6f}]")
    print(f"  Effect size: {reversal_result['effect_size']:.4f}")
    
    # REVERSAL_STATISTICAL_GATE
    print("\nREVERSAL_STATISTICAL_GATE:")
    ci_excludes_zero = reversal_result["ci_excludes_zero"]
    expected_direction = reversal_result["point_estimate"] < 0  # Should be negative
    print(f"  CI excludes 0: {ci_excludes_zero}")
    print(f"  Expected direction (negative): {expected_direction}")
    print(f"  Gate status: {'PASS' if (ci_excludes_zero and expected_direction) else 'FAIL'}")
    if not ci_excludes_zero:
        print(f"  Note: CI includes 0, statistical gate fails")
    
    print("\n[6] Summary")
    print("-" * 80)
    print(f"Phase 3B assumed effect: d=0.204 (pilot)")
    print(f"Phase 4 observed effect: d={observed_effect:.4f}")
    print(f"Effect reduction: {(1 - observed_effect/0.204)*100:.1f}%")
    print(f"\nAchieved power at observed effect (N=300): {power_analytical_two:.1%}")
    print(f"Required N for 80% power (two-sided): {compute_required_n(observed_effect, 0.80, True)}")
    print(f"Required N for 90% power (two-sided): {compute_required_n(observed_effect, 0.90, True)}")
    print(f"\nRecommendation: Do NOT automatically run large-scale confirmation.")
    print(f"First execute evaluator calibration and adapter discrimination gates.")
    
    # Save results
    output = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "phase3b_assumptions": {
            "pilot_effect_size": pilot_effect["cohens_d"],
            "selected_n": 300,
            "mde_at_n300": 0.131,
            "target_power": 0.90,
            "effect_scenario": "50% of pilot"
        },
        "phase4_observations": {
            "observed_effect_size": observed_effect,
            "signed_gap": normal_result["point_estimate"],
            "ci": [normal_result["ci_lower"], normal_result["ci_upper"]],
            "achieved_power_analytical_two_sided": power_analytical_two,
            "achieved_power_monte_carlo_two_sided": power_mc_two
        },
        "required_sample_sizes": {
            "two_sided_power_80": compute_required_n(observed_effect, 0.80, True),
            "two_sided_power_90": compute_required_n(observed_effect, 0.90, True),
            "one_sided_power_80": compute_required_n(observed_effect, 0.80, False),
            "one_sided_power_90": compute_required_n(observed_effect, 0.90, False)
        },
        "reversal_analysis": {
            "implementation_gate": {
                "sign_flipped": sign_flipped,
                "status": "PASS" if sign_flipped else "FAIL"
            },
            "direction_estimate": {
                "signed_gap": reversal_result["point_estimate"],
                "ci": [reversal_result["ci_lower"], reversal_result["ci_upper"]],
                "effect_size": reversal_result["effect_size"]
            },
            "statistical_gate": {
                "ci_excludes_zero": ci_excludes_zero,
                "expected_direction": expected_direction,
                "status": "PASS" if (ci_excludes_zero and expected_direction) else "FAIL"
            }
        },
        "recommendation": "Execute evaluator calibration and adapter discrimination before large-scale confirmation"
    }
    
    output_path = Path("artifacts/dexperts/r6/power_reconciliation.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
