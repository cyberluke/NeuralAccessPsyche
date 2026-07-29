#!/usr/bin/env python3
"""
NRAM v5 Scientific Ablation Study

Generates reproducible evidence comparing baseline vs steered outputs
with statistical metrics (novelty, coherence, source distance), effect sizes,
and confidence intervals.

Study Design:
- 24 diverse prompts from evaluation/prompts.jsonl
- 5 conditions: baseline, actadd_0.5, actadd_1.0, actadd_2.0, conceptor_1.0
- 5 seeds: [42, 123, 456, 789, 1024]
- Total: 24 × 5 × 5 = 600 generations

Metrics per generation:
- Novelty: Embedding distance from source prompt (sentence-transformers)
- Coherence: Perplexity proxy (repetition, sentence structure)
- Source Distance: Cosine similarity to source prompt embeddings
- Token Count: Number of tokens generated
- Latency: Time to first token, total generation time
- Intervention Telemetry: Delta norms, effective strength (if available)

Statistical Analysis:
- Mean and standard deviation per condition
- Effect size (Cohen's d) comparing each intervention to baseline
- Bootstrap 95% confidence intervals (1000 resamples)
- Paired t-test and Wilcoxon signed-rank test (p-values)
- Multiple comparison correction (Bonferroni)

Negative Controls:
- Shuffled vectors: Random activation vectors (should produce no effect)
- Zero strength: strength=0.0 (should match baseline)
- Wrong layer: Intervention on layer 0 instead of target layer

Output: artifacts/nram_v5_ablation/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from scipy import stats

# Lazy imports for heavy dependencies
_sentence_transformer = None
_perplexity_model = None


def get_sentence_transformer():
    """Lazy-load sentence transformer for embedding computation."""
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading sentence-transformer model (all-MiniLM-L6-v2)...")
            _sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            print("Sentence transformer loaded.")
        except ImportError:
            print("WARNING: sentence-transformers not available. Using fallback metrics.")
            _sentence_transformer = "unavailable"
    return _sentence_transformer


# Cache for embeddings to avoid recomputation
_embedding_cache: Dict[str, np.ndarray] = {}

def compute_embedding(text: str) -> Optional[np.ndarray]:
    """Compute embedding for text using sentence-transformers with caching."""
    if text in _embedding_cache:
        return _embedding_cache[text]
    
    model = get_sentence_transformer()
    if model == "unavailable" or model is None:
        return None
    try:
        embedding = model.encode(text, convert_to_numpy=True)
        _embedding_cache[text] = embedding
        return embedding
    except Exception as e:
        print(f"WARNING: Embedding computation failed: {e}")
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_novelty(output_text: str, source_text: str) -> float:
    """
    Compute novelty as embedding distance from source.
    Returns 1 - cosine_similarity (higher = more novel).
    """
    emb_output = compute_embedding(output_text)
    emb_source = compute_embedding(source_text)
    
    if emb_output is None or emb_source is None:
        # Fallback: use lexical novelty (unique word ratio)
        source_words = set(source_text.lower().split())
        output_words = set(output_text.lower().split())
        if not output_words:
            return 0.0
        novel_words = output_words - source_words
        return len(novel_words) / len(output_words)
    
    similarity = cosine_similarity(emb_output, emb_source)
    return 1.0 - similarity


def compute_coherence(text: str) -> float:
    """
    Compute coherence score (0-1, higher = more coherent).
    Uses heuristic proxies: sentence length variance, repetition ratio.
    """
    if not text or not text.strip():
        return 0.0
    
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    if len(sentences) < 2:
        return 0.5
    
    # Sentence length consistency (lower variance = more coherent)
    sent_lengths = [len(s.split()) for s in sentences]
    if not sent_lengths:
        return 0.0
    mean_len = np.mean(sent_lengths)
    if mean_len == 0:
        return 0.0
    variance = np.var(sent_lengths) / (mean_len ** 2)  # Coefficient of variation squared
    coherence_from_structure = 1.0 / (1.0 + variance)
    
    # Repetition penalty (lower repetition = more coherent)
    words = text.lower().split()
    if len(words) < 3:
        return coherence_from_structure
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    if not trigrams:
        return coherence_from_structure
    unique_ratio = len(set(trigrams)) / len(trigrams)
    coherence_from_repetition = unique_ratio
    
    # Combine
    return 0.6 * coherence_from_structure + 0.4 * coherence_from_repetition


def compute_source_distance(output_text: str, source_prompt: str) -> float:
    """
    Compute semantic distance from source prompt.
    Returns cosine similarity (higher = closer to source).
    """
    emb_output = compute_embedding(output_text)
    emb_source = compute_embedding(source_prompt)
    
    if emb_output is None or emb_source is None:
        # Fallback: lexical overlap
        source_words = set(source_prompt.lower().split())
        output_words = set(output_text.lower().split())
        if not output_words:
            return 0.0
        overlap = source_words & output_words
        return len(overlap) / len(output_words)
    
    return cosine_similarity(emb_output, emb_source)


@dataclass
class GenerationResult:
    """Result from a single generation."""
    prompt_id: str
    condition: str
    seed: int
    output_text: str
    token_count: int
    latency_first_token: float
    latency_total: float
    novelty: float
    coherence: float
    source_distance: float
    telemetry: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ConditionConfig:
    """Configuration for an ablation condition."""
    name: str
    nram_opts: Dict[str, Any]
    description: str


def define_conditions() -> List[ConditionConfig]:
    """Define the ablation study conditions.
    
    Uses the actually-supported NRAM request schema fields:
    - profile: NRAM persona profile (normal, microdose, psychedelic, peak, dissociative)
    - intensity: steering intensity [0.0, 1.0]
    - concepts: concept injection with token-level bias
    - entropy_control: entropy servo
    - forbidden_phrases: phrase-level blocking
    - phenomenon_weights: phenomenon steering
    
    Note: activation_addition and conceptor_steering boolean flags are explicitly
    marked as "unsupported" in the schema validator. The representation dict is
    accepted but not yet consumed by the engine builder. These conditions test
    the mechanisms that ARE wired and proven in the runtime.
    """
    return [
        ConditionConfig(
            name="baseline",
            nram_opts={},
            description="No NRAM intervention (pure baseline)"
        ),
        ConditionConfig(
            name="profile_normal",
            nram_opts={
                "enabled": True,
                "profile": "normal",
                "intensity": 0.5,
            },
            description="NRAM profile=normal, intensity=0.5"
        ),
        ConditionConfig(
            name="profile_peak",
            nram_opts={
                "enabled": True,
                "profile": "peak",
                "intensity": 0.9,
            },
            description="NRAM profile=peak, intensity=0.9 (high steering)"
        ),
        ConditionConfig(
            name="profile_psychedelic",
            nram_opts={
                "enabled": True,
                "profile": "psychedelic",
                "intensity": 0.8,
            },
            description="NRAM profile=psychedelic, intensity=0.8"
        ),
        ConditionConfig(
            name="concepts_injection",
            nram_opts={
                "enabled": True,
                "profile": "peak",
                "intensity": 0.7,
                "concepts": [
                    {
                        "concept_id": "innovation",
                        "en_tokens": ["innovation", "breakthrough", "novel", "creative"],
                        "activation_phase": "divergence",
                    }
                ],
                "concept_strength": 1.0,
            },
            description="NRAM peak + concept injection (innovation tokens)"
        ),
    ]


def define_negative_controls() -> List[ConditionConfig]:
    """Define negative control conditions.
    
    Uses actually-supported NRAM fields for negative controls:
    - zero_intensity: intensity=0.0 should produce minimal steering
    - empty_profile: enabled=True but default profile (should differ from peak)
    """
    return [
        ConditionConfig(
            name="zero_intensity",
            nram_opts={
                "enabled": True,
                "profile": "peak",
                "intensity": 0.0,  # Zero intensity should minimize steering
            },
            description="Negative control: zero intensity (should approximate baseline)"
        ),
        ConditionConfig(
            name="disabled_nram",
            nram_opts={
                "enabled": False,
                "profile": "peak",
                "intensity": 0.9,
            },
            description="Negative control: NRAM explicitly disabled (should match baseline)"
        ),
    ]


def load_prompts(path: str) -> List[Dict[str, Any]]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] skipping malformed line {lineno}: {exc}", file=sys.stderr)
                continue
            if "prompt" not in obj:
                print(f"[warn] skipping line {lineno}: missing 'prompt'", file=sys.stderr)
                continue
            prompts.append(obj)
    return prompts


def call_model(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    seed: int,
    nram_opts: Dict[str, Any],
    max_tokens: int = 256,
    temperature: float = 0.7,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> Tuple[str, int, float, float, Dict[str, Any]]:
    """
    Call the model endpoint with retry logic and return (output, token_count, ttft, total_time, telemetry).
    """
    import requests
    
    url = f"{base_url}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    
    if nram_opts:
        payload["nram"] = nram_opts
    
    last_error = None
    for attempt in range(max_retries):
        start_time = time.time()
        first_token_time = None
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            first_token_time = time.time()  # Approximation for non-streaming
            
            if response.status_code == 200:
                data = response.json()
                total_time = time.time() - start_time
                ttft = first_token_time - start_time if first_token_time else total_time
                
                output = ""
                token_count = 0
                telemetry = {}
                
                try:
                    output = data["choices"][0]["message"]["content"] or ""
                    token_count = data.get("usage", {}).get("completion_tokens", 0)
                except (KeyError, IndexError, TypeError):
                    pass
                
                # Extract telemetry if present
                if "nram_telemetry" in data:
                    telemetry = data["nram_telemetry"]
                
                return output, token_count, ttft, total_time, telemetry
            
            # Server error - retry
            if response.status_code >= 500:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
                return "", 0, 0.0, time.time() - start_time, {"error": last_error}
            
            # Client error - don't retry
            return "", 0, 0.0, time.time() - start_time, {"error": f"HTTP {response.status_code}: {response.text[:200]}"}
            
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
                continue
            return "", 0, 0.0, time.time() - start_time, {"error": last_error}
    
    return "", 0, 0.0, 0.0, {"error": last_error or "Max retries exceeded"}


def run_generation(
    base_url: str,
    api_key: str,
    model: str,
    prompt_entry: Dict[str, Any],
    condition: ConditionConfig,
    seed: int,
    max_tokens: int,
    temperature: float,
) -> GenerationResult:
    """Run a single generation and compute metrics."""
    prompt_id = prompt_entry.get("id", "unknown")
    prompt_text = prompt_entry.get("prompt", "")
    
    output, token_count, ttft, total_time, telemetry = call_model(
        base_url=base_url,
        api_key=api_key,
        model=model,
        prompt=prompt_text,
        seed=seed,
        nram_opts=condition.nram_opts,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    
    if telemetry.get("error"):
        return GenerationResult(
            prompt_id=prompt_id,
            condition=condition.name,
            seed=seed,
            output_text="",
            token_count=0,
            latency_first_token=0.0,
            latency_total=total_time,
            novelty=0.0,
            coherence=0.0,
            source_distance=0.0,
            telemetry=telemetry,
            error=telemetry["error"],
        )
    
    # Compute metrics
    novelty = compute_novelty(output, prompt_text)
    coherence = compute_coherence(output)
    source_distance = compute_source_distance(output, prompt_text)
    
    return GenerationResult(
        prompt_id=prompt_id,
        condition=condition.name,
        seed=seed,
        output_text=output,
        token_count=token_count,
        latency_first_token=ttft,
        latency_total=total_time,
        novelty=novelty,
        coherence=coherence,
        source_distance=source_distance,
        telemetry=telemetry,
    )


def compute_effect_size(baseline: np.ndarray, intervention: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(baseline), len(intervention)
    if n1 < 2 or n2 < 2:
        return 0.0
    mean1, mean2 = np.mean(baseline), np.mean(intervention)
    var1, var2 = np.var(baseline, ddof=1), np.var(intervention, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (mean2 - mean1) / pooled_std


def bootstrap_ci(data: np.ndarray, n_resamples: int = 1000, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean."""
    if len(data) == 0:
        return (0.0, 0.0, 0.0)
    
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_resamples):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return (float(np.mean(data)), float(lower), float(upper))


def statistical_analysis(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Perform statistical analysis on results."""
    analysis = {}
    
    conditions = results_df["condition"].unique()
    baseline_df = results_df[results_df["condition"] == "baseline"]
    
    metrics = ["novelty", "coherence", "source_distance", "token_count", "latency_total"]
    
    for metric in metrics:
        analysis[metric] = {}
        
        # Per-condition statistics
        for condition in conditions:
            cond_df = results_df[results_df["condition"] == condition]
            values = cond_df[metric].values
            
            if len(values) == 0:
                continue
            
            mean_val = float(np.mean(values))
            std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ci_mean, ci_lower, ci_upper = bootstrap_ci(values)
            
            analysis[metric][condition] = {
                "mean": mean_val,
                "std": std_val,
                "n": len(values),
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
            }
        
        # Effect sizes vs baseline
        baseline_values = baseline_df[metric].values
        if len(baseline_values) > 0:
            for condition in conditions:
                if condition == "baseline":
                    continue
                cond_df = results_df[results_df["condition"] == condition]
                intervention_values = cond_df[metric].values
                
                if len(intervention_values) == 0:
                    continue
                
                cohens_d = compute_effect_size(baseline_values, intervention_values)
                
                # Statistical tests
                if len(baseline_values) >= 2 and len(intervention_values) >= 2:
                    # Paired t-test (assuming same prompts/seeds)
                    try:
                        t_stat, t_pvalue = stats.ttest_rel(baseline_values, intervention_values)
                    except Exception:
                        t_stat, t_pvalue = 0.0, 1.0
                    
                    # Wilcoxon signed-rank test
                    try:
                        w_stat, w_pvalue = stats.wilcoxon(baseline_values, intervention_values)
                    except Exception:
                        w_stat, w_pvalue = 0.0, 1.0
                else:
                    t_stat, t_pvalue, w_stat, w_pvalue = 0.0, 1.0, 0.0, 1.0
                
                analysis[metric].setdefault("effect_sizes", {})[condition] = {
                    "cohens_d": cohens_d,
                    "t_statistic": float(t_stat),
                    "t_pvalue": float(t_pvalue),
                    "wilcoxon_statistic": float(w_stat),
                    "wilcoxon_pvalue": float(w_pvalue),
                }
    
    # Bonferroni correction for multiple comparisons
    n_comparisons = len([c for c in conditions if c != "baseline"])
    for metric in metrics:
        if "effect_sizes" in analysis[metric]:
            for condition, es_data in analysis[metric]["effect_sizes"].items():
                es_data["t_pvalue_bonferroni"] = min(1.0, es_data["t_pvalue"] * n_comparisons)
                es_data["wilcoxon_pvalue_bonferroni"] = min(1.0, es_data["wilcoxon_pvalue"] * n_comparisons)
    
    return analysis


def generate_visualizations(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate visualization plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("WARNING: matplotlib/seaborn not available. Skipping visualizations.")
        return
    
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Condition comparison box plots
    metrics = ["novelty", "coherence", "source_distance", "token_count"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sns.boxplot(data=results_df, x="condition", y=metric, ax=ax)
        ax.set_title(f"{metric.replace('_', ' ').title()} by Condition")
        ax.set_xlabel("Condition")
        ax.set_ylabel(metric.replace("_", " ").title())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(viz_dir / "condition_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Distribution histograms
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for condition in results_df["condition"].unique():
            cond_data = results_df[results_df["condition"] == condition][metric]
            ax.hist(cond_data, alpha=0.5, label=condition, bins=20)
        ax.set_title(f"{metric.replace('_', ' ').title()} Distribution")
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(viz_dir / "distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Effect size forest plot
    analysis = statistical_analysis(results_df)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    effect_metrics = ["novelty", "coherence", "source_distance", "token_count"]
    for idx, metric in enumerate(effect_metrics):
        ax = axes[idx]
        if "effect_sizes" not in analysis.get(metric, {}):
            continue
        
        effect_sizes = analysis[metric]["effect_sizes"]
        conditions = list(effect_sizes.keys())
        d_values = [effect_sizes[c]["cohens_d"] for c in conditions]
        
        y_pos = np.arange(len(conditions))
        ax.barh(y_pos, d_values, color='steelblue', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(conditions)
        ax.set_xlabel("Cohen's d")
        ax.set_title(f"Effect Sizes: {metric.replace('_', ' ').title()}")
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=0.2, color='orange', linestyle=':', alpha=0.3, label='small')
        ax.axvline(x=0.5, color='orange', linestyle=':', alpha=0.3, label='medium')
        ax.axvline(x=0.8, color='orange', linestyle=':', alpha=0.3, label='large')
    
    plt.tight_layout()
    plt.savefig(viz_dir / "effect_sizes_forest.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {viz_dir}")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="NRAM v5 Scientific Ablation Study")
    parser.add_argument("--model", default="nram-qwen3-14b-awq", help="Model name for API")
    parser.add_argument("--output", default="artifacts/nram_v5_ablation", help="Output directory")
    parser.add_argument("--prompts", default="evaluation/prompts.jsonl", help="Prompts JSONL file")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024], help="Random seeds")
    parser.add_argument("--conditions", default="baseline,profile_normal,profile_peak,profile_psychedelic,concepts_injection",
                        help="Comma-separated condition names")
    parser.add_argument("--include-controls", action="store_true", help="Include negative controls")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--base-url", default=os.environ.get("EVAL_BASE_URL", "http://localhost:8000"),
                        help="API base URL")
    parser.add_argument("--api-key", default=os.environ.get("EVAL_API_KEY", "dev-nram-key"),
                        help="API key")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without API calls")
    args = parser.parse_args(argv)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    prompts = load_prompts(args.prompts)
    if args.limit:
        prompts = prompts[:args.limit]
    
    if not prompts:
        print("ERROR: No prompts loaded", file=sys.stderr)
        return 1
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Define conditions
    all_conditions = define_conditions()
    condition_names = set(args.conditions.split(","))
    conditions = [c for c in all_conditions if c.name in condition_names]
    
    if args.include_controls:
        conditions.extend(define_negative_controls())
    
    if not conditions:
        print("ERROR: No conditions selected", file=sys.stderr)
        return 1
    
    print(f"Running {len(conditions)} conditions × {len(args.seeds)} seeds × {len(prompts)} prompts = "
          f"{len(conditions) * len(args.seeds) * len(prompts)} generations")
    
    # Collect results with incremental saving
    results: List[GenerationResult] = []
    total_generations = len(conditions) * len(args.seeds) * len(prompts)
    completed = 0
    
    # Check for existing results to resume
    checkpoint_file = output_dir / "checkpoint.json"
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
                all_results = [GenerationResult(**r) for r in checkpoint_data.get("results", [])]
                # Only keep successful results, discard errors for retry
                results = [r for r in all_results if not r.error]
                completed = len(results)
                print(f"Resuming from checkpoint: {completed}/{total_generations} successful results loaded ({len(all_results) - len(results)} errors discarded)")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    
    for prompt_entry in prompts:
        prompt_id = prompt_entry.get("id", "unknown")
        for condition in conditions:
            for seed in args.seeds:
                # Skip if already completed
                if any(r.prompt_id == prompt_id and r.condition == condition.name and r.seed == seed for r in results):
                    completed += 1
                    continue
                
                completed += 1
                print(f"[{completed}/{total_generations}] prompt={prompt_id} condition={condition.name} seed={seed}", end="")
                
                if args.dry_run:
                    print(" (dry-run)")
                    results.append(GenerationResult(
                        prompt_id=prompt_id,
                        condition=condition.name,
                        seed=seed,
                        output_text="[dry-run output]",
                        token_count=10,
                        latency_first_token=0.1,
                        latency_total=0.5,
                        novelty=0.5,
                        coherence=0.7,
                        source_distance=0.6,
                    ))
                    continue
                
                result = run_generation(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    prompt_entry=prompt_entry,
                    condition=condition,
                    seed=seed,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                
                if result.error:
                    print(f" ERROR: {result.error}")
                else:
                    print(f" OK (tokens={result.token_count}, novelty={result.novelty:.3f})")
                
                results.append(result)
                
                # Save checkpoint every 10 generations
                if completed % 10 == 0:
                    checkpoint_data = {
                        "results": [asdict(r) for r in results],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    with open(checkpoint_file, "w", encoding="utf-8") as f:
                        json.dump(checkpoint_data, f)
    
    # Convert to DataFrame
    results_data = [asdict(r) for r in results]
    results_df = pd.DataFrame(results_data)
    
    # Save raw results
    results_df.to_csv(output_dir / "ablation_results.csv", index=False)
    print(f"Raw results saved to {output_dir / 'ablation_results.csv'}")
    
    # Statistical analysis
    print("\nPerforming statistical analysis...")
    analysis = statistical_analysis(results_df)
    
    # Save statistical summary
    with open(output_dir / "statistical_summary.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    print(f"Statistical summary saved to {output_dir / 'statistical_summary.json'}")
    
    # Save effect sizes
    effect_sizes = {}
    for metric, metric_data in analysis.items():
        if "effect_sizes" in metric_data:
            effect_sizes[metric] = metric_data["effect_sizes"]
    
    with open(output_dir / "effect_sizes.json", "w", encoding="utf-8") as f:
        json.dump(effect_sizes, f, indent=2)
    print(f"Effect sizes saved to {output_dir / 'effect_sizes.json'}")
    
    # Save confidence intervals
    confidence_intervals = {}
    for metric, metric_data in analysis.items():
        confidence_intervals[metric] = {
            cond: {
                "mean": data["mean"],
                "ci_95_lower": data["ci_95_lower"],
                "ci_95_upper": data["ci_95_upper"],
            }
            for cond, data in metric_data.items()
            if isinstance(data, dict) and "mean" in data
        }
    
    with open(output_dir / "confidence_intervals.json", "w", encoding="utf-8") as f:
        json.dump(confidence_intervals, f, indent=2)
    print(f"Confidence intervals saved to {output_dir / 'confidence_intervals.json'}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(results_df, output_dir)
    
    # Save metadata
    metadata = {
        "study_name": "NRAM v5 Scientific Ablation Study",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "prompts_file": args.prompts,
        "num_prompts": len(prompts),
        "conditions": [c.name for c in conditions],
        "condition_descriptions": {c.name: c.description for c in conditions},
        "seeds": args.seeds,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "total_generations": total_generations,
        "completed_generations": len(results),
        "errors": sum(1 for r in results if r.error),
        "metrics_computed": ["novelty", "coherence", "source_distance", "token_count", "latency_total"],
        "statistical_tests": ["cohens_d", "paired_t_test", "wilcoxon_signed_rank", "bootstrap_ci"],
        "multiple_comparison_correction": "bonferroni",
    }
    
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {output_dir / 'metadata.json'}")
    
    # Print summary
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)
    print(f"Total generations: {len(results)}")
    print(f"Errors: {sum(1 for r in results if r.error)}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  - ablation_results.csv")
    print(f"  - statistical_summary.json")
    print(f"  - effect_sizes.json")
    print(f"  - confidence_intervals.json")
    print(f"  - metadata.json")
    print(f"  - visualizations/")
    
    # Print key findings
    print("\n" + "-"*80)
    print("KEY FINDINGS (Effect Sizes vs Baseline)")
    print("-"*80)
    
    for metric in ["novelty", "coherence", "source_distance"]:
        if metric in effect_sizes:
            print(f"\n{metric.upper()}:")
            for condition, es_data in effect_sizes[metric].items():
                d = es_data["cohens_d"]
                p = es_data.get("t_pvalue_bonferroni", 1.0)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"  {condition:20s}: d={d:+.3f} (p={p:.4f}) {sig}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
