# NRAM V5 R5 Causal Ablation Infrastructure Requirements

## Overview

Phase 5 requires running causal ablation studies for both scientific regimes:
- **Regime A**: Qwen3-0.6B-Base (raw completions, no chat template)
- **Regime B**: Qwen3-14B-AWQ (chat completions, existing behavior)

## Script Usage

```bash
# Regime A: 0.6B-Base (raw completions)
python scripts/dexperts_causal_ablation.py --regime 06b --n-prompts 50

# Regime B: 14B-AWQ (chat completions)
python scripts/dexperts_causal_ablation.py --regime 14b --n-prompts 50
```

## Infrastructure Requirements

### Regime A (06b) - Qwen3-0.6B-Base

**Status**: Requires separate SGLang instance

**Configuration**:
- Model: `Qwen/Qwen3-0.6B-Base`
- Endpoint: `http://127.0.0.1:8001/v1/completions` (port 8001)
- API Format: Raw completions (`/v1/completions`, NOT `/v1/chat/completions`)
- No chat template, no system prompts
- Direct prompt → completion

**SGLang Launch Command** (example):
```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-0.6B-Base \
  --port 8001 \
  --host 127.0.0.1 \
  --dtype float16 \
  --device cuda \
  --tp 1 \
  --context-length 2048
```

**NRAM Integration**:
- The 0.6B-Base instance must have NRAM hooks enabled
- DExperts adapters must be loaded: `artifacts/dexperts/adapters/nontoxic` and `artifacts/dexperts/adapters/toxic`
- The `/v1/completions` endpoint must accept the `nram` parameter in the request body

**Output Artifacts**:
- `artifacts/dexperts/r5/causal_ablation_06b.json` (summary statistics)
- `artifacts/dexperts/r5/causal_ablation_raw_06b.jsonl` (raw generations)

### Regime B (14b) - Qwen3-14B-AWQ

**Status**: Uses existing infrastructure

**Configuration**:
- Model: `nram-qwen3-14b-awq`
- Endpoint: `http://127.0.0.1:8000/v1/chat/completions` (port 8000)
- API Format: Chat completions (`/v1/chat/completions`)
- Chat template enabled, system prompts enabled

**SGLang Launch Command** (existing):
```bash
# Already running on port 8000 with NRAM hooks
# See docker-compose.dexperts-proof.yml for configuration
```

**Output Artifacts**:
- `artifacts/dexperts/r5/causal_ablation_14b.json` (summary statistics)
- `artifacts/dexperts/r5/causal_ablation_raw_14b.jsonl` (raw generations)

## Expected Results

### Regime A (06b)
- Should show measurable toxicity differences between conditions
- DExperts should have visible effect on logit distributions
- May show dose-response relationship (higher alpha → lower toxicity)
- Baseline toxicity expected to be higher (unaligned model)

### Regime B (14b)
- Will likely show floor effect (toxicity ~0.001 across all conditions)
- This is expected for well-aligned model
- Documents the limitation of DExperts on already-safe models
- DExperts may still show small but measurable effects on diversity/fluency

## Study Design

### Conditions
1. **baseline**: No DExperts intervention
2. **dexperts_low**: alpha=0.5
3. **dexperts_medium**: alpha=1.0
4. **dexperts_high**: alpha=2.0
5. **negative_control**: alpha=0.0, identical adapters (should match baseline)
6. **reversal_control**: alpha=-1.0, swapped adapters (should reverse steering)

### Seeds
- [42, 123, 456] for reproducibility

### Prompts
- Source: `allenai/real-toxicity-prompts` dataset
- Count: 50 prompts with high toxicity potential
- Held-out data (not used for training)

### Metrics
- Toxicity score (toxic-bert)
- Fluency score (word length proxy)
- Diversity score (unique trigrams)
- Coherence score (1 - trigram repetition)
- Latency (ms)
- Token count
- Effect size (Cohen's d vs baseline)
- Bootstrap 95% CI

## Verification Checks

The script automatically verifies:
1. Baseline toxicity > 0 (toxic content present)
2. DExperts conditions reduce toxicity vs baseline
3. Dose-response relationship (higher alpha → lower toxicity)
4. Negative control matches baseline (within 10% tolerance)
5. Reversal control shows opposite effect

## Blockers and Mitigations

### Blocker: 0.6B-Base Not Currently Served
**Status**: The 0.6B-Base model is not currently running on a separate SGLang instance.

**Mitigation Options**:
1. Start a separate SGLang instance on port 8001 with 0.6B-Base
2. Modify the existing SGLang configuration to support both models (model routing)
3. Use a different endpoint if 0.6B-Base is already served elsewhere

**Current Action**: Proceed with Regime B (14b) only. The mechanistic proof (Phase 4) already demonstrates the mechanism works. Regime A can be run once infrastructure is available.

## Execution Log

### Phase 5 Implementation
- **Date**: 2026-07-30
- **Script Modified**: `scripts/dexperts_causal_ablation.py`
- **Changes**:
  - Added `--regime` flag with `06b` and `14b` options
  - Implemented regime-specific API request logic (completions vs chat completions)
  - Updated output paths to regime-specific locations (`artifacts/dexperts/r5/`)
  - Auto-configured endpoints based on regime
  - Added regime metadata to output JSON

### Next Steps
1. Start SGLang instance for 0.6B-Base on port 8001 (if available)
2. Run Regime B ablation: `python scripts/dexperts_causal_ablation.py --regime 14b --n-prompts 50`
3. Run Regime A ablation: `python scripts/dexperts_causal_ablation.py --regime 06b --n-prompts 50`
4. Compare results between regimes
5. Document findings in final report
