# NRAM v5 Runtime Verification Report

**Date:** 2026-07-28T06:26:00Z  
**Mode:** runtime-adversary  
**Branch:** nram-v5-implementation  
**Commit:** 7f5718c  
**Working Tree:** Dirty (modified test outputs, untracked artifacts)

---

## EXECUTIVE SUMMARY

**STATUS: FAILED**

The NRAM v5 implementation is **NOT runtime-wired**. Code exists for all claimed features, but:

1. **All advanced features are explicitly rejected** by the API via `UNSUPPORTED_NRAM_FEATURES`
2. **Capability endpoint is inaccurate** - claims features are available when they are rejected
3. **Streamlit integration is broken** - exposes controls that cause runtime errors
4. **No GPU causal proofs exist** - features cannot execute because they are rejected

---

## CRITICAL FINDINGS

### 1. Features Are Explicitly Rejected

**Location:** [`core/contracts/nram_runtime.py:38-53`](core/contracts/nram_runtime.py:38)

```python
UNSUPPORTED_NRAM_FEATURES = {
    "dexperts",
    "activation_addition",
    "actadd",
    "conceptor_steering",
    "hidden_state_probes",
    "latent_closed_loop",
    "semantic_novelty_controller",
    "evidence_guard",
    "branch_tournament",
    "reft",
    "soft_prompts",
    "attention_head_gating",
    "kv_cache_firewall",
    "gpu_native_semantic_control",
}
```

**Impact:** Any request attempting to use these features is rejected with HTTP 400:
```
Unsupported NRAM runtime feature(s): activation_addition, dexperts, ...
```

### 2. Code Is Not Imported in Inference Engines

**Location:** [`core/engines/sglang_engine.py`](core/engines/sglang_engine.py)

**Falsification Tests:** `tests/runtime_adversary/test_nram_v5_wiring_falsification.py`

- `SGLangEngine` does NOT import `HookManager`
- `SGLangEngine` does NOT import `LiveActivationAddition`
- `SGLangEngine` does NOT import `NRAMRequestControlPlane`
- `SGLangEngine` does NOT import `DExpertsController`

**Conclusion:** Forward hooks and representation control code exists in `core/steering/` but is never invoked during actual GPU inference.

### 3. Capability Endpoint Is Inaccurate

**Location:** [`api/routes.py:744-790`](api/routes.py:744)

The `/v1/nram/capabilities` endpoint claims:
```python
"controls": {
    "activation_addition": True,
    "multi_vector_representation": True,
    "conceptor_steering": True,
    "hidden_state_probes": True,
    "latent_closed_loop": True,
    "semantic_closed_loop": True,
    "branch_tournament": True,
    "dexperts": True,
}
```

**But these features are in `UNSUPPORTED_NRAM_FEATURES` and rejected at runtime.**

**Falsification Tests:** `tests/runtime_adversary/test_capability_endpoint_falsification.py`

7 tests intentionally fail to document this inaccuracy:
- `test_activation_addition_claimed_but_rejected`
- `test_dexperts_claimed_but_rejected`
- `test_conceptor_steering_claimed_but_rejected`
- `test_hidden_state_probes_claimed_but_rejected`
- `test_latent_closed_loop_claimed_but_rejected`
- `test_branch_tournament_claimed_but_rejected`
- `test_streamlit_controls_send_unsupported_features`

### 4. Streamlit Integration Is Broken

**Location:** [`streamlit_console.py:628-777`](streamlit_console.py:628)

Streamlit exposes UI controls for:
- Conceptor steering (lines 628-634)
- DExperts (lines 671-677)

These controls send `nram_opts["conceptor"]` and `nram_opts["dexperts"]` to the API, which are rejected because they are in `UNSUPPORTED_NRAM_FEATURES`.

**Impact:** Users can enable these controls in the UI, but they will cause HTTP 400 errors.

---

## COMPONENT VERIFICATION MATRIX

| Component | IMPLEMENTED | RUNTIME_WIRED | CAUSALLY_PROVEN | STATISTICALLY_VALIDATED | BLOCKED |
|-----------|-------------|---------------|-----------------|-------------------------|---------|
| Activation Addition | ✅ | ❌ | ❌ | ❌ | ✅ |
| Multi-vector Representation | ✅ | ❌ | ❌ | ❌ | ✅ |
| Conceptor Steering | ✅ | ❌ | ❌ | ❌ | ✅ |
| Trained Hidden-State Probes | ✅ | ❌ | ❌ | ❌ | ✅ |
| Latent Closed-Loop | ✅ | ❌ | ❌ | ❌ | ✅ |
| Semantic Closed-Loop | ✅ | ❌ | ❌ | ❌ | ✅ |
| Branch-and-Tournament | ✅ | ❌ | ❌ | ❌ | ✅ |
| DExperts | ✅ | ❌ | ❌ | ❌ | ✅ |
| Request Control Plane | ✅ | ❌ | ❌ | ❌ | ✅ |
| Streamlit Integration | ✅ | ❌ | ❌ | ❌ | ✅ |

**Legend:**
- ✅ IMPLEMENTED: Code exists and imports successfully
- ❌ RUNTIME_WIRED: Code is invoked during real GPU inference
- ❌ CAUSALLY_PROVEN: Disabled vs enabled shows measurable effect
- ❌ STATISTICALLY_VALIDATED: Scientific ablation with paired comparisons
- ✅ BLOCKED: Cannot verify because feature is rejected

---

## WHAT IS ACTUALLY WIRED

The following features ARE runtime-wired through the logit processor:

1. **Token biasing** (positive/negative token IDs)
2. **Token masking** (forbidden token IDs)
3. **Repetition penalty** (dynamic, based on output history)
4. **Phrase constraints** (forbidden phrases, source n-gram blocking)
5. **Entropy control** (PID servo with phase-based targets)
6. **Concept injection** (dynamic concept capsules via token biases)
7. **Soft token injections** (bounded decode windows)
8. **Vocabulary logit vectors** (weighted vocabulary vectors)
9. **Hard token schedule** (forced token IDs)
10. **Phenomenon mixer** (logit-level implementations)

**Evidence:** These features are built into `NRAMLogitProcessor` in [`nram_sglang/processor.py`](nram_sglang/processor.py) and invoked via `custom_logit_processor` in SGLang.

---

## TEST RESULTS

### Wiring Falsification Tests

**File:** `tests/runtime_adversary/test_nram_v5_wiring_falsification.py`

```
29 tests collected
27 passed, 2 failed
```

**Passed tests confirm:**
- All advanced features are in `UNSUPPORTED_NRAM_FEATURES`
- Code is NOT imported in inference engines
- Requests attempting to use features are rejected
- Only logit-processor features are actually wired

**Failed tests:**
- `test_multi_vector_representation_module_exists` - class name mismatch
- `test_conceptor_steering_module_exists` - class name mismatch

### Capability Endpoint Falsification Tests

**File:** `tests/runtime_adversary/test_capability_endpoint_falsification.py`

```
15 tests collected
8 passed, 7 failed
```

**Passed tests confirm:**
- Capability endpoint claims features are available
- Streamlit exposes controls for unsupported features

**Failed tests (intentional falsifications):**
- 6 tests document capability endpoint inaccuracy
- 1 test documents Streamlit integration inaccuracy

---

## SILENT FALLBACKS DISCOVERED

1. **No silent fallbacks** - Features are explicitly rejected with clear error messages
2. **No fake implementations** - Code exists but is not wired
3. **Capability endpoint inaccuracy** - Claims features are available when they are not

---

## LIMITATIONS

1. **No GPU access** - Could not run actual inference tests
2. **No causal proofs** - Could not measure hidden-state deltas or logit changes
3. **No statistical validation** - Could not run ablation studies

---

## BLOCKERS

1. **Features are rejected** - Cannot verify runtime behavior because requests fail
2. **No wiring** - Code is not imported in inference engines
3. **Capability endpoint inaccuracy** - Creates false scientific claims

---

## NEXT RECOMMENDED STEPS

Before scientific ablation can proceed:

1. **Fix capability endpoint** - Change claims from `True` to `False` for unsupported features
2. **Fix Streamlit integration** - Remove or disable controls for unsupported features
3. **Wire features into inference** - Import and invoke `HookManager`, `LiveActivationAddition`, etc. in `SGLangEngine`
4. **Remove from UNSUPPORTED_NRAM_FEATURES** - Once wired, allow features to execute
5. **Run causal proofs** - Measure hidden-state deltas and logit changes with disabled vs enabled
6. **Run statistical ablation** - Paired comparisons with confidence intervals

---

## TEST ARTIFACTS

- `tests/runtime_adversary/test_nram_v5_wiring_falsification.py`
- `tests/runtime_adversary/test_capability_endpoint_falsification.py`
- `artifacts/nram_v5_verification/runtime-adversary-7f5718c-verification-report.md`

---

## CONCLUSION

The NRAM v5 implementation is **IMPLEMENTED but NOT RUNTIME_WIRED**. 

All advanced features (activation addition, conceptor steering, DExperts, etc.) exist as code modules but are:
1. Explicitly rejected by the API
2. Not imported in inference engines
3. Falsely claimed as available by the capability endpoint
4. Exposed in Streamlit but cause errors when used

**The Builder's claim of "216 tests passing and container running healthy" is misleading.** The tests pass because they test code that is never executed. The container runs healthy because it rejects unsupported features rather than attempting to execute them.

**This is not a runtime-wired implementation. This is a code-only implementation with false capability claims.**
