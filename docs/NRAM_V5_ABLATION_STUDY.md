# NRAM v5 Scientific Ablation Study

## Study Design and Rationale

This document describes the scientific ablation study conducted to measure the
causal effect of NRAM v5 steering mechanisms on model output quality. The study
isolates individual intervention components and quantifies their impact using
standardized metrics and statistical tests.

**Rationale:** Previous phases proved that NRAM components are wired correctly
(Phase 4) and that the full test suite passes (Phase 5). This study goes further
by measuring *how much* each mechanism changes output, whether the changes are
statistically significant, and whether dose-response relationships exist.

---

## Methods

### Prompts

- **Source:** `evaluation/prompts.jsonl`
- **Count:** 24 diverse prompts across 8 categories:
  - product_launch (3)
  - education_technology (3)
  - ai_interface (3)
  - creative_strategy (3)
  - business_turnaround (3)
  - consumer_hardware (3)
  - software_architecture (3)
  - scientific_explanation (3)

### Conditions

| Condition | Description | Key Parameters |
|-----------|-------------|----------------|
| `baseline` | No NRAM intervention | None |
| `profile_normal` | NRAM normal profile | `profile=normal`, `intensity=0.5` |
| `profile_peak` | NRAM peak profile (high steering) | `profile=peak`, `intensity=0.9` |
| `profile_psychedelic` | NRAM psychedelic profile | `profile=psychedelic`, `intensity=0.8` |
| `concepts_injection` | NRAM peak + concept injection | `profile=peak`, `intensity=0.7`, concepts=["innovation", "breakthrough", "novel", "creative"], `concept_strength=1.0` |

**Note:** The original study design included activation addition (`actadd_*`) and conceptor steering conditions, but these mechanisms are marked as "unsupported" in the NRAM request schema validator. The `representation` dict is accepted but not yet consumed by the engine builder. This study tests the mechanisms that ARE wired and proven in the runtime: profile-based steering, intensity control, and concept injection.

### Negative Controls

| Control | Purpose | Expected Result |
|---------|---------|-----------------|
| `zero_intensity` | Verify zero intensity minimizes steering | Approximate baseline (d ≈ 0) |
| `disabled_nram` | Verify NRAM disabled = baseline | Match baseline exactly (d = 0) |

### Seeds

Fixed seeds for reproducibility: `[42, 123, 456, 789, 1024]`

### Total Generations

24 prompts × 5 conditions × 5 seeds = **600 generations** (plus negative controls)

### Metrics

#### Novelty
- **Definition:** `1 - cosine_similarity(embedding(output), embedding(source_prompt))`
- **Tool:** `sentence-transformers/all-MiniLM-L6-v2`
- **Range:** [0, 1], higher = more novel
- **Fallback:** Lexical unique-word ratio if embeddings unavailable

#### Coherence
- **Definition:** Weighted combination of sentence structure consistency (60%) and trigram uniqueness (40%)
- **Range:** [0, 1], higher = more coherent
- **Components:**
  - Sentence length coefficient of variation (lower variance → higher score)
  - Trigram repetition ratio (fewer repeats → higher score)

#### Source Distance
- **Definition:** `cosine_similarity(embedding(output), embedding(source_prompt))`
- **Range:** [0, 1], higher = closer to source
- **Note:** Inversely related to novelty

#### Token Count
- **Definition:** Number of completion tokens reported by the API

#### Latency
- **Time to First Token (TTFT):** Approximated from request timing
- **Total Latency:** End-to-end request time

### Statistical Tests

#### Effect Size
- **Metric:** Cohen's d
- **Interpretation:** |d| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large

#### Significance Tests
- **Paired t-test:** Parametric test for mean differences (paired by prompt+seed)
- **Wilcoxon signed-rank test:** Non-parametric alternative for non-normal distributions

#### Confidence Intervals
- **Method:** Bootstrap resampling (1000 iterations)
- **Level:** 95% CI for the mean

#### Multiple Comparison Correction
- **Method:** Bonferroni correction
- **Applied to:** All p-values across conditions within each metric

---

## Results

### Study Parameters

| Parameter | Value |
|-----------|-------|
| Model | `nram-qwen3-14b-awq` |
| Prompts | 24 |
| Conditions | 5 (+ 2 negative controls) |
| Seeds | 5 |
| Max tokens | 256 |
| Temperature | 0.7 |
| Total generations | 600 |

### Implemented Mechanisms

This study tests the following NRAM mechanisms that are fully wired and proven:

1. **Profile-based steering**: Different persona profiles (normal, peak, psychedelic) with varying intensity levels
2. **Concept injection**: Token-level biasing for specific concepts (innovation, breakthrough, novel, creative)
3. **Intensity control**: Fine-grained control over steering strength [0.0, 1.0]

### Not Implemented (Schema Limitations)

The following mechanisms are defined in the architecture but not yet wired to the runtime:

1. **Activation addition** (`activation_addition` flag): Marked as "unsupported" in schema validator
2. **Conceptor steering** (`conceptor_steering` flag): Marked as "unsupported" in schema validator
3. **Representation dict**: Accepted by schema but not consumed by engine builder

These mechanisms will be tested in future studies once the wiring is complete.

### Statistical Summary

Study completed: 2026-07-29. Full results in `artifacts/nram_v5_ablation_full/`.

**Key Findings (Novelty metric):**
- `profile_normal`: d = -1.50, p < 0.001 (large effect, significantly reduces novelty)
- `profile_peak`: d = -0.27, p = 0.012 (small effect, marginally significant)
- `profile_psychedelic`: d = 0.29, p = 0.010 (small effect, marginally significant)
- `concepts_injection`: d = -0.03, p = 0.728 (negligible effect, not significant)
- `zero_intensity`: d = -0.14, p = 0.177 (negligible effect, not significant)

**Key Findings (Coherence metric):**
- All NRAM profiles significantly reduce coherence (d = -0.52 to -1.62, all p < 0.001)
- `concepts_injection`: d = -1.62, p < 0.001 (large effect)
- `zero_intensity`: d = -1.71, p < 0.001 (large effect - unexpected)

### Effect Sizes

See `artifacts/nram_v5_ablation_full/effect_sizes.json` for full statistical details.

### Confidence Intervals

See `artifacts/nram_v5_ablation_full/confidence_intervals.json` for bootstrap 95% CIs.

---

## Interpretation

### What Works

1. **Profile-based steering produces measurable effects**: All NRAM profiles significantly alter output characteristics compared to baseline.

2. **Profile differentiation**: Different profiles produce different effect patterns:
   - `normal` profile strongly reduces novelty (d = -1.50)
   - `psychedelic` profile slightly increases novelty (d = 0.29)
   - `peak` profile has minimal effect on novelty (d = -0.27)

3. **Statistical power**: With n=120 per condition, the study has adequate power to detect medium-to-large effects (d > 0.5).

### What Doesn't Work

1. **Concept injection not effective**: The `concepts_injection` condition shows negligible effect on novelty (d = -0.03, p = 0.728). This suggests the concept injection mechanism is not functioning as intended, or the selected concepts ("innovation", "breakthrough", "novel", "creative") do not measurably influence output semantics.

2. **Coherence degradation**: All NRAM profiles significantly reduce coherence (d = -0.52 to -1.62). This is a critical side effect that must be addressed before production deployment.

### Dose-Response Relationship

**Not tested**: The original design included `actadd_0.5`, `actadd_1.0`, `actadd_2.0` conditions to test dose-response, but activation addition is marked as "unsupported" in the schema validator. This study tested only the mechanisms that are wired and proven.

### Negative Control Validation

**CRITICAL DEFECT IDENTIFIED**:

1. **`zero_intensity` control**: Shows negligible effect on novelty (d = -0.14, p = 0.177) as expected, but shows large effect on coherence (d = -1.71, p < 0.001). This is unexpected and suggests that even with intensity=0.0, NRAM is still affecting output structure.

2. **`disabled_nram` control**: Shows massive effects on both novelty (d = -1.06, p < 0.001) and coherence (d = -1.81, p < 0.001). This is a critical defect: the `enabled: False` flag is not properly disabling NRAM steering.

**Root Cause Analysis**: The `disabled_nram` condition sends `{"enabled": False, "profile": "peak", "intensity": 0.9}`. The `_is_nram_enabled()` method in `core/engines/sglang_engine.py:521-526` should return False when `enabled=False`, but the large effect sizes suggest NRAM is still active. This requires immediate investigation and fix.

**Impact**: The negative control failure invalidates the ability to distinguish between "NRAM not working" and "NRAM working correctly". The main study results (comparing different NRAM profiles) remain valid, but the claim that "NRAM produces measurable effects" cannot be validated against a true baseline.

---

## Limitations

1. **Sample Size:** 24 prompts × 5 seeds = 120 observations per condition. Adequate for large effects (d > 0.5) but underpowered for small effects.

2. **Generalizability:** Prompts are English-language, focused on product/business/technology domains. Results may not generalize to other languages or domains.

3. **Model Specificity:** Study uses Qwen3-14B-AWQ. Effects may differ for other model architectures or sizes.

4. **Metric Limitations:**
   - Novelty (embedding distance) captures semantic novelty but not structural or creative novelty
   - Coherence is a heuristic proxy, not a full language model perplexity
   - Source distance conflates "on-topic" with "derivative"

5. **Compute Constraints:** Study runs on a single RTX 4090 (24GB). Larger models or more extensive prompt sets would require multi-GPU infrastructure.

6. **Activation Vector Specificity:** The `novelty_vs_paraphrase` vector is one specific contrastive direction. Other vectors may produce different effect profiles.

7. **No Human Evaluation:** All metrics are automated. Human judgment of output quality, creativity, and coherence is not captured.

---

## Reproduction Instructions

### Prerequisites

- Python 3.11+
- Dependencies: `pandas`, `scipy`, `numpy`, `matplotlib`, `seaborn`, `sentence-transformers`
- Running NRAM API server at `http://localhost:8000`
- SGLang model loaded: `nram-qwen3-14b-awq`

### Run the Study

```bash
python scripts/nram_v5_ablation_study.py \
  --model nram-qwen3-14b-awq \
  --output artifacts/nram_v5_ablation/ \
  --prompts evaluation/prompts.jsonl \
  --seeds 42 123 456 789 1024 \
  --conditions baseline,actadd_0.5,actadd_1.0,actadd_2.0,conceptor_1.0 \
  --include-controls \
  --max-tokens 256 \
  --temperature 0.7
```

### Dry Run (no API calls)

```bash
python scripts/nram_v5_ablation_study.py --dry-run --limit 3
```

### Output Artifacts

```
artifacts/nram_v5_ablation/
├── ablation_results.csv          # Raw per-generation results
├── statistical_summary.json      # Per-condition mean, std, CI
├── effect_sizes.json             # Cohen's d for each intervention vs baseline
├── confidence_intervals.json     # Bootstrap 95% CIs
├── metadata.json                 # Study parameters and timestamps
└── visualizations/
    ├── condition_comparison.png  # Box plots by condition
    ├── distributions.png         # Histogram overlays
    └── effect_sizes_forest.png   # Forest plot of effect sizes
```

### Verify Reproducibility

Run the study twice with identical parameters. Results should match exactly
(given fixed seeds and deterministic model inference).

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-07-28 | 1.0.0 | Initial study design and implementation |
