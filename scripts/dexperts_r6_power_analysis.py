#!/usr/bin/env python3
"""
NRAM v5 R6 Phase 3B: Prospective Power Analysis

Computes required sample size for confirmatory experiment using:
- Prompt-level clustering (not token-level)
- Independent group bootstrap (toxic vs non-toxic are unrelated)
- Effect size estimation from pilot data
- Power curves across candidate sample sizes

Primary endpoint: Signed attribute gap G(alpha) = E[delta_logp_non-toxic] - E[delta_logp_toxic]
Primary alpha: 1.0 (pre-registered)
Significance level: 0.05 (two-sided)
Target power: 0.90
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib

from core.evaluation.dexperts_distribution_metrics import (
    compute_dexperts_distribution_metrics,
    DExpertsDistributionMetrics,
)


@dataclass
class PowerAnalysisConfig:
    """Configuration for power analysis."""
    primary_alpha: float = 1.0
    significance_level: float = 0.05
    target_power: float = 0.90
    n_bootstrap: int = 10000
    candidate_ns: List[int] = None
    effect_scenarios: List[float] = None  # Fraction of pilot effect
    attrition_rate: float = 0.10
    
    def __post_init__(self):
        if self.candidate_ns is None:
            self.candidate_ns = [50, 75, 100, 150, 200, 300, 500, 750, 1000]
        if self.effect_scenarios is None:
            self.effect_scenarios = [1.0, 0.75, 0.5]


@dataclass
class PilotEffectEstimate:
    """Effect size estimate from pilot data."""
    mean_toxic: float
    mean_nontoxic: float
    std_toxic: float
    std_nontoxic: float
    n_toxic: int
    n_nontoxic: int
    signed_gap: float
    cohens_d: float
    hedges_g: float
    ci_lower: float
    ci_upper: float


def load_pilot_data() -> Tuple[List[DExpertsDistributionMetrics], List[DExpertsDistributionMetrics]]:
    """
    Load pilot data from R5 mechanistic proof.
    
    Returns:
        Tuple of (toxic_metrics, nontoxic_metrics)
    """
    # Load R5 per-token data
    r5_path = Path("artifacts/dexperts/r5/mechanistic_proof.json")
    if not r5_path.exists():
        raise FileNotFoundError(f"Pilot data not found at {r5_path}")
    
    with open(r5_path) as f:
        r5_data = json.load(f)
    
    per_token = r5_data.get("per_token_details", [])
    
    # Group by prompt_id and class
    toxic_by_prompt = {}
    nontoxic_by_prompt = {}
    
    for record in per_token:
        prompt_id = record.get("prompt_id")
        is_toxic = record.get("is_toxic_continuation", False)
        alpha = record.get("alpha", 1.0)
        
        if alpha != 1.0:  # Only use primary alpha
            continue
        
        base_logp = record.get("base_log_prob", 0.0)
        combined_logp = record.get("combined_log_prob_alpha_1.0", 0.0)
        delta_logp = combined_logp - base_logp
        
        if prompt_id not in (toxic_by_prompt if is_toxic else nontoxic_by_prompt):
            (toxic_by_prompt if is_toxic else nontoxic_by_prompt)[prompt_id] = []
        
        (toxic_by_prompt if is_toxic else nontoxic_by_prompt)[prompt_id].append(delta_logp)
    
    # Aggregate to sequence level (mean delta per prompt)
    toxic_sequences = [np.mean(deltas) for deltas in toxic_by_prompt.values()]
    nontoxic_sequences = [np.mean(deltas) for deltas in nontoxic_by_prompt.values()]
    
    return toxic_sequences, nontoxic_sequences


def estimate_pilot_effect(
    toxic_sequences: List[float],
    nontoxic_sequences: List[float],
    n_bootstrap: int = 10000,
) -> PilotEffectEstimate:
    """
    Estimate effect size from pilot data using prompt-level aggregation.
    
    Args:
        toxic_sequences: Sequence-level delta logp for toxic continuations
        nontoxic_sequences: Sequence-level delta logp for non-toxic continuations
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        PilotEffectEstimate with effect sizes and confidence intervals
    """
    toxic_arr = np.array(toxic_sequences)
    nontoxic_arr = np.array(nontoxic_sequences)
    
    mean_toxic = np.mean(toxic_arr)
    mean_nontoxic = np.mean(nontoxic_arr)
    std_toxic = np.std(toxic_arr, ddof=1)
    std_nontoxic = np.std(nontoxic_arr, ddof=1)
    
    signed_gap = mean_nontoxic - mean_toxic
    
    # Cohen's d
    pooled_std = np.sqrt((std_toxic**2 + std_nontoxic**2) / 2)
    cohens_d = signed_gap / pooled_std if pooled_std > 0 else 0.0
    
    # Hedges' g (small-sample correction)
    n_total = len(toxic_arr) + len(nontoxic_arr)
    correction = 1 - (3 / (4 * n_total - 9))
    hedges_g = cohens_d * correction
    
    # Bootstrap confidence interval for signed gap
    rng = np.random.default_rng(42)
    bootstrap_gaps = []
    
    for _ in range(n_bootstrap):
        toxic_sample = rng.choice(toxic_arr, size=len(toxic_arr), replace=True)
        nontoxic_sample = rng.choice(nontoxic_arr, size=len(nontoxic_arr), replace=True)
        gap = np.mean(nontoxic_sample) - np.mean(toxic_sample)
        bootstrap_gaps.append(gap)
    
    bootstrap_gaps = np.array(bootstrap_gaps)
    ci_lower = np.percentile(bootstrap_gaps, 2.5)
    ci_upper = np.percentile(bootstrap_gaps, 97.5)
    
    return PilotEffectEstimate(
        mean_toxic=mean_toxic,
        mean_nontoxic=mean_nontoxic,
        std_toxic=std_toxic,
        std_nontoxic=std_nontoxic,
        n_toxic=len(toxic_arr),
        n_nontoxic=len(nontoxic_arr),
        signed_gap=signed_gap,
        cohens_d=cohens_d,
        hedges_g=hedges_g,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def compute_power_for_n(
    pilot_effect: PilotEffectEstimate,
    n_per_class: int,
    effect_fraction: float,
    alpha: float = 0.05,
    n_simulations: int = 5000,
) -> float:
    """
    Compute statistical power for given sample size using simulation.
    
    Args:
        pilot_effect: Effect size estimate from pilot
        n_per_class: Sample size per class
        effect_fraction: Fraction of pilot effect to assume (1.0 = full, 0.5 = conservative)
        alpha: Significance level
        n_simulations: Number of Monte Carlo simulations
    
    Returns:
        Estimated power (probability of rejecting H0)
    """
    # Assume true effect is fraction of pilot effect
    true_gap = pilot_effect.signed_gap * effect_fraction
    
    # Generate synthetic data under alternative hypothesis
    rng = np.random.default_rng(42)
    rejections = 0
    
    for _ in range(n_simulations):
        # Sample from normal distributions with true means
        toxic_sample = rng.normal(
            pilot_effect.mean_toxic,
            pilot_effect.std_toxic,
            size=n_per_class
        )
        nontoxic_sample = rng.normal(
            pilot_effect.mean_nontoxic + true_gap,
            pilot_effect.std_nontoxic,
            size=n_per_class
        )
        
        # Two-sample t-test
        mean_diff = np.mean(nontoxic_sample) - np.mean(toxic_sample)
        pooled_std = np.sqrt(
            (np.var(toxic_sample, ddof=1) + np.var(nontoxic_sample, ddof=1)) / 2
        )
        t_stat = mean_diff / (pooled_std * np.sqrt(2 / n_per_class))
        
        # Critical value for two-sided test
        from scipy import stats
        t_crit = stats.t.ppf(1 - alpha/2, df=2*n_per_class - 2)
        
        if abs(t_stat) > t_crit:
            rejections += 1
    
    power = rejections / n_simulations
    return power


def compute_analytical_power(
    pilot_effect: PilotEffectEstimate,
    n_per_class: int,
    effect_fraction: float,
    alpha: float = 0.05,
) -> float:
    """
    Compute power using analytical Welch approximation.
    
    Args:
        pilot_effect: Effect size estimate
        n_per_class: Sample size per class
        effect_fraction: Fraction of pilot effect
        alpha: Significance level
    
    Returns:
        Estimated power
    """
    from scipy import stats
    
    true_gap = pilot_effect.signed_gap * effect_fraction
    pooled_std = np.sqrt((pilot_effect.std_toxic**2 + pilot_effect.std_nontoxic**2) / 2)
    
    # Non-centrality parameter
    ncp = true_gap / (pooled_std * np.sqrt(2 / n_per_class))
    
    # Critical value
    df = 2 * n_per_class - 2
    t_crit = stats.t.ppf(1 - alpha/2, df=df)
    
    # Power = P(|T| > t_crit | ncp)
    power = 1 - stats.nct.cdf(t_crit, df=df, nc=ncp) + stats.nct.cdf(-t_crit, df=df, nc=ncp)
    
    return power


def compute_minimum_detectable_effect(
    n_per_class: int,
    pilot_effect: PilotEffectEstimate,
    alpha: float = 0.05,
    power: float = 0.90,
) -> float:
    """
    Compute minimum detectable effect size for given sample size.
    
    Args:
        n_per_class: Sample size per class
        pilot_effect: Pilot effect estimate (for variance)
        alpha: Significance level
        power: Target power
    
    Returns:
        Minimum detectable signed gap
    """
    from scipy import stats
    
    pooled_std = np.sqrt((pilot_effect.std_toxic**2 + pilot_effect.std_nontoxic**2) / 2)
    
    # Critical values
    df = 2 * n_per_class - 2
    t_alpha = stats.t.ppf(1 - alpha/2, df=df)
    t_beta = stats.t.ppf(power, df=df)
    
    # Minimum detectable effect
    mde = (t_alpha + t_beta) * pooled_std * np.sqrt(2 / n_per_class)
    
    return mde


def run_power_analysis(config: PowerAnalysisConfig) -> Dict:
    """
    Run complete power analysis.
    
    Args:
        config: Power analysis configuration
    
    Returns:
        Complete power analysis results
    """
    print("=" * 80)
    print("NRAM v5 R6 Phase 3B: Prospective Power Analysis")
    print("=" * 80)
    
    # Load pilot data
    print("\n[1] Loading pilot data from R5 mechanistic proof...")
    toxic_sequences, nontoxic_sequences = load_pilot_data()
    print(f"  Toxic sequences: {len(toxic_sequences)}")
    print(f"  Non-toxic sequences: {len(nontoxic_sequences)}")
    
    # Estimate pilot effect
    print("\n[2] Estimating pilot effect size...")
    pilot_effect = estimate_pilot_effect(
        toxic_sequences,
        nontoxic_sequences,
        n_bootstrap=config.n_bootstrap
    )
    print(f"  Signed gap G(alpha={config.primary_alpha}): {pilot_effect.signed_gap:.6f}")
    print(f"  95% CI: [{pilot_effect.ci_lower:.6f}, {pilot_effect.ci_upper:.6f}]")
    print(f"  Cohen's d: {pilot_effect.cohens_d:.4f}")
    print(f"  Hedges' g: {pilot_effect.hedges_g:.4f}")
    
    # Compute power curves
    print("\n[3] Computing power curves...")
    power_results = {}
    
    for effect_frac in config.effect_scenarios:
        print(f"\n  Effect scenario: {effect_frac*100:.0f}% of pilot effect")
        scenario_results = []
        
        for n in config.candidate_ns:
            # Simulation-based power
            power_sim = compute_power_for_n(
                pilot_effect, n, effect_frac,
                alpha=config.significance_level,
                n_simulations=1000
            )
            
            # Analytical power
            power_ana = compute_analytical_power(
                pilot_effect, n, effect_frac,
                alpha=config.significance_level
            )
            
            # Minimum detectable effect
            mde = compute_minimum_detectable_effect(
                n, pilot_effect,
                alpha=config.significance_level,
                power=config.target_power
            )
            
            scenario_results.append({
                "n_per_class": n,
                "power_simulation": power_sim,
                "power_analytical": power_ana,
                "minimum_detectable_effect": mde,
            })
            
            print(f"    N={n:4d}: power_sim={power_sim:.3f}, power_ana={power_ana:.3f}, MDE={mde:.4f}")
        
        power_results[f"effect_{effect_frac}"] = scenario_results
    
    # Find recommended sample size
    print("\n[4] Determining recommended sample size...")
    conservative_results = power_results["effect_0.5"]
    
    recommended_n = None
    for result in conservative_results:
        if result["power_simulation"] >= config.target_power:
            recommended_n = result["n_per_class"]
            break
    
    if recommended_n is None:
        recommended_n = max(config.candidate_ns)
        print(f"  WARNING: Target power not achieved, using maximum N={recommended_n}")
    else:
        print(f"  Recommended N per class: {recommended_n}")
    
    # Account for attrition
    recommended_generated = int(np.ceil(recommended_n / (1 - config.attrition_rate)))
    print(f"  With {config.attrition_rate*100:.0f}% attrition: generate N={recommended_generated}")
    
    # Create confirmatory split manifest
    print("\n[5] Creating confirmatory split manifest...")
    confirmatory_manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "pilot_source": "artifacts/dexperts/r5/mechanistic_proof.json",
        "pilot_prompts_used": list(range(len(toxic_sequences) + len(nontoxic_sequences))),
        "confirmatory_prompts_reserved": "must be disjoint from pilot",
        "dataset": "allenai/real-toxicity-prompts",
        "split": "train",
        "random_seed": 12345,
        "recommended_n_analyzable": recommended_n,
        "recommended_n_generated": recommended_generated,
        "attrition_rate": config.attrition_rate,
    }
    
    # Save results
    output_dir = Path("artifacts/dexperts/r6")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": asdict(config),
        "pilot_effect": asdict(pilot_effect),
        "power_curves": power_results,
        "recommended_n_per_class": recommended_n,
        "recommended_n_generated": recommended_generated,
        "confirmatory_manifest": confirmatory_manifest,
    }
    
    output_path = output_dir / "power_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[6] Results saved to {output_path}")
    
    # Save power curve CSV
    csv_path = output_dir / "power_curve.csv"
    with open(csv_path, "w") as f:
        f.write("n_per_class,effect_fraction,power_simulation,power_analytical,mde\n")
        for effect_frac in config.effect_scenarios:
            for result in power_results[f"effect_{effect_frac}"]:
                f.write(f"{result['n_per_class']},{effect_frac},{result['power_simulation']:.4f},{result['power_analytical']:.4f},{result['minimum_detectable_effect']:.6f}\n")
    
    print(f"[7] Power curve saved to {csv_path}")
    
    return results


if __name__ == "__main__":
    config = PowerAnalysisConfig()
    results = run_power_analysis(config)
    
    print("\n" + "=" * 80)
    print("Phase 3B Complete")
    print("=" * 80)
    print(f"Recommended N per class: {results['recommended_n_per_class']}")
    print(f"Recommended N generated: {results['recommended_n_generated']}")
    print(f"Pilot signed gap: {results['pilot_effect']['signed_gap']:.6f}")
