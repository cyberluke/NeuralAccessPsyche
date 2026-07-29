# Documentation Audit Report

**Date**: 2026-07-29  
**Auditor**: Phase 7 Documentation Truthfulness Audit  
**Commit**: ba52bbe  
**Evidence Base**: `artifacts/nram_v5_ablation_full/` (828 successful generations)

---

## Executive Summary

This audit compares documentation claims against empirical evidence from the Phase 6 ablation study. **Critical discrepancies identified**: Documentation claims coherence is maintained, but empirical evidence shows all NRAM profiles significantly degrade coherence.

**Status**: ⚠️ **CRITICAL DISCREPANCIES FOUND**

---

## 1. Documentation Claims vs. Empirical Evidence

### 1.1 Coherence Maintenance

**Documentation Claims**:

1. **docs/evaluation.md** (line 3):
   > "prove the persona changes style *without* degrading substance"

2. **docs/nram-steering.md** (line 85):
   > "The `coherence_floor` and the structured plan exist specifically to keep conceptual novelty from degrading factual coherence."

3. **docs/nram-v5-specification.md** (line 1009):
   > "Verify coherence is maintained"

4. **docs/nram-v5-specification.md** (line 1015):
   > "Measurable novelty increase without coherence loss"

**Empirical Evidence** (from `artifacts/nram_v5_ablation_full/statistical_summary.json`):

| Profile | Coherence Effect Size (Cohen's d) | P-value | Interpretation |
|---------|-----------------------------------|---------|----------------|
| profile_normal | -1.16 | < 0.001 | **Large degradation** |
| profile_peak | -0.90 | < 0.001 | **Large degradation** |
| profile_psychedelic | -0.52 | < 0.001 | **Medium degradation** |
| concepts_injection | -1.62 | < 0.001 | **Very large degradation** |
| zero_intensity | -1.71 | < 0.001 | **Very large degradation** |
| disabled_nram | -1.81 | < 0.001 | **Very large degradation** |

**Baseline coherence**: 0.905 ± 0.037  
**Worst profile coherence**: 0.810 ± 0.064 (disabled_nram)  
**Coherence loss**: 10.5% reduction

**Verdict**: ❌ **DOCUMENTATION FALSE**

All NRAM profiles significantly reduce coherence. The documentation claim that "coherence is maintained" is contradicted by empirical evidence. The `coherence_floor` mechanism is not preventing coherence degradation.

---

### 1.2 Quality Preservation

**Documentation Claims**:

1. **docs/evaluation.md** (line 3):
   > "prove the persona changes style *without* degrading substance"

2. **docs/nram-fix-spec-v1.md** (line 579):
   > "Při chybějícím tokenizeru failnout, ne tiše degradovat" (Fail on missing tokenizer, don't silently degrade)

**Empirical Evidence**:

The ablation study shows significant degradation across multiple metrics:

| Metric | Baseline | Worst Profile | Degradation |
|--------|----------|---------------|-------------|
| Coherence | 0.905 | 0.810 | -10.5% |
| Novelty | 0.360 | 0.241 (normal) | -33.1% |
| Source Distance | 0.360 | 0.275 (disabled_nram) | -23.6% |

**Verdict**: ⚠️ **PARTIALLY FALSE**

Documentation claims quality is preserved, but empirical evidence shows significant degradation in coherence, novelty, and source distance. The system does not "change style without degrading substance" — it changes style AND degrades substance.

---

### 1.3 Novelty Increase Without Coherence Loss

**Documentation Claims**:

**docs/nram-v5-specification.md** (line 1015):
> "Measurable novelty increase without coherence loss"

**Empirical Evidence**:

| Profile | Novelty Effect (d) | Coherence Effect (d) | Trade-off |
|---------|-------------------|---------------------|-----------|
| profile_normal | -1.50 (decrease) | -1.16 (decrease) | Both degrade |
| profile_peak | -0.27 (decrease) | -0.90 (decrease) | Both degrade |
| profile_psychedelic | +0.29 (increase) | -0.52 (decrease) | Novelty ↑, Coherence ↓ |
| concepts_injection | -0.03 (no change) | -1.62 (decrease) | Coherence degrades |

**Verdict**: ❌ **DOCUMENTATION FALSE**

Only `profile_psychedelic` shows a novelty increase (d=+0.29), but this comes at the cost of coherence degradation (d=-0.52). No profile achieves "novelty increase without coherence loss."

---

## 2. Negative Control Defect

### 2.1 `disabled_nram` Control Failure

**Expected Behavior**: `disabled_nram` should match baseline exactly (d ≈ 0).

**Observed Behavior**: `disabled_nram` shows large effects:
- Novelty: d = -1.06, p < 0.001
- Coherence: d = -1.81, p < 0.001

**Root Cause**: The `enabled: False` flag is not properly disabling NRAM steering. The `_is_nram_enabled()` method in `core/engines/sglang_engine.py:521-526` should return False, but large effect sizes suggest NRAM is still active.

**Impact**: This defect invalidates the ability to distinguish between "NRAM not working" and "NRAM working correctly." The main study results remain valid for comparing different NRAM profiles, but the claim that "NRAM produces measurable effects" cannot be validated against a true baseline.

**Verdict**: ⚠️ **CRITICAL DEFECT**

---

## 3. Documentation Accuracy Assessment

### 3.1 Accurate Claims

✅ **docs/nram-steering.md** correctly describes:
- NRAM modifies logits before sampling (categories 3, 4, 5)
- Activation steering (category 6) is NOT used
- Persona is a property vector, not a character mask
- "Psychedelic" = associative distance, not randomness

✅ **docs/evaluation.md** correctly describes:
- A/B evaluation harness methodology
- Deterministic heuristics
- Success criteria (≥25% jargon reduction, ≥20% persona fidelity)
- Honesty policy (report failure honestly)

✅ **docs/implementation-evidence.md** correctly describes:
- Historical artifact status
- Forced-token proof methodology
- Test results (99 tests passed)

### 3.2 Inaccurate Claims

❌ **Coherence maintenance**: Documentation claims coherence is maintained, but evidence shows 10.5% degradation.

❌ **Quality preservation**: Documentation claims quality is preserved, but evidence shows significant degradation across multiple metrics.

❌ **Novelty without coherence loss**: Documentation claims this is achievable, but evidence shows only one profile (psychedelic) increases novelty, and it degrades coherence.

### 3.3 Missing Claims

⚠️ **No documentation of coherence degradation**: The ablation study reveals significant coherence loss, but no documentation acknowledges this trade-off.

⚠️ **No documentation of negative control defect**: The `disabled_nram` control failure is not documented in any user-facing documentation.

---

## 4. Recommendations

### 4.1 Immediate Actions (Critical)

1. **Fix `disabled_nram` defect**: Investigate why `enabled: False` does not disable NRAM steering. This is a correctness issue that affects all downstream analysis.

2. **Update documentation**: Revise claims about coherence maintenance and quality preservation to reflect empirical evidence.

3. **Add coherence degradation warning**: Document that NRAM profiles significantly reduce coherence (10.5% average loss).

### 4.2 Short-term Actions (Important)

4. **Implement coherence floor enforcement**: The `coherence_floor` parameter should actually prevent coherence degradation below a threshold.

5. **Add coherence metric to success criteria**: The evaluation harness should include coherence as a pass/fail criterion.

6. **Document trade-offs**: Create a "Known Limitations" section that explicitly states the coherence-novelty trade-off.

### 4.3 Long-term Actions (Nice-to-have)

7. **Implement coherence-preserving steering**: Research steering mechanisms that maintain coherence while increasing novelty.

8. **Add human evaluation**: Automated metrics may not capture all aspects of quality. Consider adding human evaluation.

9. **Create regression tests**: Add tests that verify coherence does not degrade below a threshold.

---

## 5. Conclusion

The documentation contains **critical discrepancies** when compared to empirical evidence:

1. **Coherence maintenance**: Claimed but not achieved (10.5% degradation)
2. **Quality preservation**: Claimed but not achieved (significant degradation across metrics)
3. **Novelty without coherence loss**: Claimed but not achieved (only one profile increases novelty, and it degrades coherence)

The `disabled_nram` negative control defect is a **critical correctness issue** that must be fixed before the system can be considered production-ready.

**Overall Assessment**: ⚠️ **DOCUMENTATION NOT TRUTHFUL**

The documentation makes claims that are contradicted by empirical evidence. This violates the "honesty policy" stated in `docs/evaluation.md`: "Report failure honestly. If any criterion is not met, the result is reported as a failure."

**Required Action**: Update documentation to reflect empirical evidence, fix the `disabled_nram` defect, and implement coherence-preserving steering mechanisms.

---

## Appendix: Statistical Evidence

### A.1 Full Statistical Summary

See `artifacts/nram_v5_ablation_full/statistical_summary.json` for complete statistical results.

### A.2 Effect Sizes

See `artifacts/nram_v5_ablation_full/effect_sizes.json` for Cohen's d values.

### A.3 Confidence Intervals

See `artifacts/nram_v5_ablation_full/confidence_intervals.json` for bootstrap 95% CIs.

### A.4 Raw Data

See `artifacts/nram_v5_ablation_full/ablation_results.csv` for per-generation results (828 rows).

---

**Report Generated**: 2026-07-29T06:35:00Z  
**Audit Method**: Comparison of documentation claims against empirical evidence from Phase 6 ablation study  
**Evidence Quality**: High (828 generations, 7 conditions, 24 prompts, 5 seeds, bootstrap CIs, Bonferroni correction)
