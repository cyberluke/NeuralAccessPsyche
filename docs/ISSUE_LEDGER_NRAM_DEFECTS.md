# NRAM Critical Defects - Issue Ledger

**Date:** 2026-07-26  
**Commit:** 23d676ada15216822d454e4876aa3411392591ac  
**Branch:** feature/nram-sglang-steering  
**Status:** CONFIRMED - All three critical defects verified with runtime evidence

---

## Executive Summary

Adversarial runtime testing has confirmed **three critical defects** in the NRAM implementation that completely break the logit-level steering system. These defects render the phenomenon mixer, dynamic repetition penalty, and phase scheduling non-functional in production, despite the code appearing correct at first glance.

**Impact:** The NRAM system operates in a degraded "prompt-only" mode where only static token biases work, but all dynamic, context-aware steering mechanisms are dead code.

---

## Finding 1: `__req__` Never Injected into custom_params

**Severity:** CRITICAL  
**Status:** CONFIRMED  
**Impact:** Breaks repetition penalty, phenomena, and phase scheduling

### Description

The `build_custom_params()` function in `core/steering/serialization.py` never includes the `__req__` key in its output. This means the `NRAMLogitProcessor` cannot access `request.output_ids`, which is required for:
- Dynamic repetition penalty (penalizes recently generated tokens)
- Phenomenon mixer (overlap, forgetting, looping all need output history)
- Phase scheduling (tracks generation progress)

### Evidence

**Test:** `tests/adversarial/test_req_injection.py::test_build_custom_params_does_not_include_req`

```python
custom_params = build_custom_params(
    positive_token_ids=[1, 2, 3],
    negative_token_ids=[4, 5, 6],
    forbidden_token_ids=[7, 8, 9],
    positive_bias=0.5,
    negative_bias=0.3,
    repetition_penalty=1.2,
    profile="peak",
    max_tokens=512,
    phenomenon_weights={"overlap": 0.8, "forgetting": 0.6},
)

assert "__req__" not in custom_params  # PASSES - defect confirmed
```

**Output:**
```
[CONFIRMED] __req__ is NOT in custom_params
  Keys present: ['positive_token_ids', 'negative_token_ids', 'forbidden_token_ids', 
                 'positive_bias', 'negative_bias', 'repetition_penalty', 'profile', 
                 'max_tokens', 'phenomenon_weights']
```

**Test:** `tests/adversarial/test_req_injection.py::test_sglang_payload_lacks_req_in_custom_params`

Verifies that even when building the full SGLang payload, `__req__` is never injected:

```
[CONFIRMED] __req__ is NOT in SGLang payload custom_params
  Keys present: ['positive_token_ids', 'negative_token_ids', 'forbidden_token_ids', 
                 'positive_bias', 'negative_bias', 'repetition_penalty', 'profile', 
                 'max_tokens', 'phenomenon_weights']
```

### Root Cause

The `build_custom_params()` function signature does not accept a `request` parameter, and the SGLang engine's `_build_upstream_payload()` method does not inject `__req__` into the custom_params dict.

**Code Location:**
- `core/steering/serialization.py:32-70` - `build_custom_params()` function
- `core/engines/sglang_engine.py:416-426` - payload construction

### Impact

Without `__req__`, the following features are completely broken:

1. **Dynamic Repetition Penalty** (lines 98-106 in `nram_logit_processor.py`)
   ```python
   if request is not None and repetition_penalty > 0:
       recent_ids = list(request.output_ids[-64:])
       # ... apply penalty
   ```
   - `request` is always `None`
   - Repetition penalty is never applied
   - Model can generate repetitive loops

2. **Phenomenon Mixer** (lines 111-117 in `nram_logit_processor.py`)
   ```python
   phenomena = params.get("phenomenon_weights") or {}
   if phenomena and request is not None:
       self._apply_phenomena(...)
   ```
   - `request` is always `None`
   - `_apply_phenomena()` is never called
   - All phenomenon effects (overlap, forgetting, looping, etc.) are dead code

3. **Phase Scheduling** (lines 69-76 in `nram_logit_processor.py`)
   ```python
   generated = 0
   request = params.get("__req__")
   if request is not None:
       generated = len(request.output_ids)
   ```
   - `generated` is always `0`
   - Phase schedule never progresses
   - All generations stay in "opening" phase

### Recommended Fix

**Option A: Inject `__req__` in SGLang Engine**

Modify `core/engines/sglang_engine.py` to inject the request object:

```python
# In _build_upstream_payload() around line 416
payload["custom_params"] = build_custom_params(
    positive_token_ids=compiled.positive_token_ids,
    negative_token_ids=compiled.negative_token_ids,
    forbidden_token_ids=compiled.forbidden_token_ids,
    positive_bias=compiled.positive_bias,
    negative_bias=compiled.negative_bias,
    repetition_penalty=compiled.repetition_penalty,
    profile=profile_name,
    max_tokens=request.max_tokens or 0,
    phenomenon_weights=nram_opts.get("phenomenon_weights"),
)

# Inject request object for dynamic features
payload["custom_params"]["__req__"] = request
```

**Option B: Modify build_custom_params() Signature**

Add `request` parameter to `build_custom_params()`:

```python
def build_custom_params(
    positive_token_ids: list[int],
    negative_token_ids: list[int],
    forbidden_token_ids: list[int],
    positive_bias: float,
    negative_bias: float,
    repetition_penalty: float,
    profile: str,
    max_tokens: int = 0,
    phenomenon_weights: Optional[Dict[str, float]] = None,
    request: Optional[Any] = None,  # NEW PARAMETER
) -> Dict[str, Any]:
    result = {
        "positive_token_ids": positive_token_ids,
        "negative_token_ids": negative_token_ids,
        "forbidden_token_ids": forbidden_token_ids,
        "positive_bias": positive_bias,
        "negative_bias": negative_bias,
        "repetition_penalty": repetition_penalty,
        "profile": profile,
        "max_tokens": max_tokens,
    }
    if phenomenon_weights:
        result["phenomenon_weights"] = phenomenon_weights
    if request is not None:
        result["__req__"] = request
    return result
```

Then update all call sites to pass the request object.

---

## Finding 2: Phenomenon Mixer is Dead Code

**Severity:** CRITICAL  
**Status:** CONFIRMED  
**Impact:** All phenomenon effects (overlap, forgetting, looping, synesthesia, etc.) never fire

### Description

The phenomenon mixer in `NRAMLogitProcessor._apply_phenomena()` is explicitly gated on `request is not None` (line 112 in `nram_logit_processor.py`). Since `__req__` is never injected (Finding 1), the phenomenon mixer never executes.

### Evidence

**Test:** `tests/adversarial/test_phenomenon_activation.py::test_phenomena_not_called_without_req`

```python
params_without_req = {
    "positive_token_ids": [1, 2, 3],
    "negative_token_ids": [4, 5, 6],
    "forbidden_token_ids": [7, 8, 9],
    "positive_bias": 0.5,
    "negative_bias": 0.3,
    "repetition_penalty": 1.2,
    "profile": "peak",
    "max_tokens": 512,
    "phenomenon_weights": {
        "overlap": 0.9,
        "forgetting": 0.9,
        "looping": 0.8,
    },
}

logits = np.zeros((1, 100), dtype=np.float32)
result_logits = processor(logits, [params_without_req])

# Base steering works
assert result_logits[0, 1] > 0  # Positive bias applied
assert result_logits[0, 4] < 0  # Negative bias applied
assert result_logits[0, 7] == -float("inf")  # Forbidden tokens blocked

# But phenomena do NOT fire
assert result_logits[0, 50] == 0.0  # Token 50 not modified by overlap
```

**Output:**
```
[CONFIRMED] overlap/forgetting/looping cannot fire without __req__
```

**Test:** `tests/adversarial/test_phenomenon_activation.py::test_overlap_fires_with_req`

When `__req__` IS provided (simulating the fix), phenomena work correctly:

```python
mock_request = MagicMock()
mock_request.output_ids = [50, 51, 52]

params_with_req = {
    # ... same params ...
    "__req__": mock_request,
}

result_logits = processor(logits, [params_with_req])

assert result_logits[0, 50] > 0  # Overlap boosts recent token
assert result_logits[0, 51] > 0
assert result_logits[0, 52] > 0
```

**Output:**
```
[CONFIRMED] overlap fires when __req__ is present
  Token 50 boosted: 0.3149999976158142
```

**Test:** `tests/adversarial/test_phenomenon_activation.py::test_phenomena_gating_line_in_processor`

Source code inspection confirms the gating:

```python
source = inspect.getsource(NRAMLogitProcessor.__call__)
assert "phenomena and request is not None" in source
```

**Output:**
```
[CONFIRMED] Source code contains: 'phenomena and request is not None'
  This proves phenomena are explicitly gated on request being non-None
```

### Root Cause

The gating condition on line 112 of `nram_logit_processor.py`:

```python
if phenomena and request is not None:
    self._apply_phenomena(...)
```

This is a defensive check to ensure `request.output_ids` is available, but since `request` is always `None` (Finding 1), the entire phenomenon mixer is dead code.

### Impact

All seven phenomenon effects are non-functional:

1. **Overlap** - Should boost recent token IDs (self-reinforcing bleed)
2. **Forgetting** - Should apply extra repetition penalty (memory decay)
3. **Looping** - Should reward recent tokens (perseveration)
4. **Insight** - Should apply periodic positive bias spikes
5. **Synesthesia** - Should add deterministic cross-activation noise
6. **Associative Jump** - Should flatten distribution toward uniform
7. **Dissolution** - Should scale logits toward zero

The UI shows phenomenon sliders, but adjusting them has no effect on generation.

### Recommended Fix

Fix Finding 1 first (inject `__req__`), then the phenomenon mixer will automatically work.

---

## Finding 3: MoE Synthesis Uses Baseline Model Without NRAM

**Severity:** HIGH  
**Status:** CONFIRMED  
**Impact:** MoE synthesis loses persona-specific steering

### Description

The MoE orchestrator in `api/routes.py` queries multiple personas with NRAM enabled, but the final synthesis step uses a baseline model (`qwen3-14b-awq-baseline`) with `nram=None`, losing all persona-specific steering characteristics.

### Evidence

**Test:** `tests/adversarial/test_moe_synthesis.py::test_moe_synthesis_uses_baseline_model`

```python
synth_request = ChatCompletionRequest(
    model="qwen3-14b-awq-baseline",  # Baseline model
    messages=[...],
    max_tokens=2048,
    temperature=0.5,
    nram=None  # No NRAM steering
)

nram_enabled = engine._is_nram_enabled(synth_request)

assert synth_request.model == "qwen3-14b-awq-baseline"
assert synth_request.nram is None
assert nram_enabled is False
```

**Output:**
```
[CONFIRMED] MoE synthesis uses baseline model
  Synthesis model: qwen3-14b-awq-baseline
  Synthesis nram: None
  NRAM would be enabled: False
```

**Test:** `tests/adversarial/test_moe_synthesis.py::test_moe_persona_queries_use_nram`

Persona queries DO use NRAM correctly:

```python
persona_request = ChatCompletionRequest(
    model="nram-qwen3-14b-awq",  # NRAM-enabled model
    messages=[...],
    max_tokens=1024,
    nram={
        "enabled": True,
        "profile": "psychedelic",
        "intensity": 0.9
    }
)

nram_enabled = engine._is_nram_enabled(persona_request)

assert persona_request.model == "nram-qwen3-14b-awq"
assert nram_enabled is True
```

**Output:**
```
[CONFIRMED] MoE persona queries use NRAM
  Persona model: nram-qwen3-14b-awq
  Persona nram: {'enabled': True, 'profile': 'psychedelic', 'intensity': 0.9}
  NRAM enabled: True
```

**Test:** `tests/adversarial/test_moe_synthesis.py::test_moe_synthesis_payload_lacks_custom_processor`

The synthesis payload does not include the custom logit processor:

```python
payload = engine._build_upstream_payload(synth_request, nram_enabled=False, ...)

assert "custom_logit_processor" not in payload
assert "custom_params" not in payload
```

**Output:**
```
[CONFIRMED] Synthesis payload lacks custom processor
  NRAM enabled: False
  Payload keys: ['presence_penalty', 'messages', 'temperature', 'max_tokens', 
                 'model', 'frequency_penalty', 'top_p', 'stream']
  Has custom_logit_processor: False
  Has custom_params: False
```

### Root Cause

In `api/routes.py`, the `_route_moe()` function (lines 167-229):

1. **Persona queries** (lines 180-191): Correctly use `nram-qwen3-14b-awq` with NRAM enabled
2. **Synthesis step** (lines 216-223): Uses `qwen3-14b-awq-baseline` with `nram=None`

```python
# Synthesis uses baseline model (no NRAM steering) for coherence
synth_messages = [...]
synth_request = _to_engine_request(request, synth_messages)
synth_request.model = "qwen3-14b-awq-baseline"  # <-- Baseline
synth_request.nram = None  # <-- No NRAM
synth_request.max_tokens = synthesis_max_tokens
synth_request.temperature = 0.5
```

The comment says "for coherence", suggesting this is intentional, but it means the final synthesis loses all persona characteristics.

### Impact

The MoE workflow:

1. **Persona queries** (4 parallel calls): Each persona generates content with NRAM steering
   - `normal` profile
   - `threshold` profile
   - `psychedelic` profile
   - `visionary-psychedelic-keynote` profile

2. **Synthesis** (1 call): Combines persona outputs using baseline model
   - No NRAM steering
   - No custom logit processor
   - No phenomenon effects
   - No persona characteristics

The synthesis step is essentially a "dumb" summarizer that loses the rich, persona-specific steering from the individual queries.

### Recommended Fix

**Option A: Use NRAM-Enabled Model for Synthesis**

```python
synth_request.model = "nram-qwen3-14b-awq"
synth_request.nram = {
    "enabled": True,
    "profile": "normal",  # Or a dedicated "synthesis" profile
    "intensity": 0.5,  # Moderate steering
}
```

**Option B: Create a Dedicated Synthesis Profile**

Define a new profile optimized for synthesis:

```python
# In core/persona/profiles.py
PROFILES["synthesis"] = NRAMState(
    visionary_intensity=0.6,
    contrarian_force=0.4,
    product_obsession=0.7,
    # ... balanced for coherent synthesis
)
```

Then use it:

```python
synth_request.nram = {
    "enabled": True,
    "profile": "synthesis",
    "intensity": 0.7,
}
```

---

## Test Coverage Summary

### Tests Created

| Test File | Test Count | Status |
|-----------|-----------|--------|
| `tests/adversarial/test_req_injection.py` | 3 | PASSED |
| `tests/adversarial/test_phenomenon_activation.py` | 5 | PASSED |
| `tests/adversarial/test_moe_synthesis.py` | 4 | PASSED |
| `tests/adversarial/test_end_to_end_logit_processor.py` | 4 | PASSED |
| **Total** | **16** | **ALL PASSED** |

### Test Execution

```bash
$ python -m pytest tests/adversarial/ -v -s
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.1.1, pluggy-1.6.0
collected 16 items

tests/adversarial/test_end_to_end_logit_processor.py::TestEndToEndLogitProcessor::test_full_generation_simulation_without_req PASSED
tests/adversarial/test_end_to_end_logit_processor.py::TestEndToEndLogitProcessor::test_full_generation_simulation_with_req PASSED
tests/adversarial/test_end_to_end_logit_processor.py::TestEndToEndLogitProcessor::test_comparison_baseline_vs_nram_without_req PASSED
tests/adversarial/test_end_to_end_logit_processor.py::TestEndToEndLogitProcessor::test_phase_schedule_without_req PASSED
tests/adversarial/test_moe_synthesis.py::TestMoESynthesisConfiguration::test_moe_synthesis_uses_baseline_model PASSED
tests/adversarial/test_moe_synthesis.py::TestMoESynthesisConfiguration::test_moe_persona_queries_use_nram PASSED
tests/adversarial/test_moe_synthesis.py::TestMoESynthesisConfiguration::test_moe_synthesis_loses_persona_characteristics PASSED
tests/adversarial/test_moe_synthesis.py::TestMoESynthesisConfiguration::test_moe_synthesis_payload_lacks_custom_processor PASSED
tests/adversarial/test_phenomenon_activation.py::TestPhenomenonMixerActivation::test_phenomena_not_called_without_req PASSED
tests/adversarial/test_phenomenon_activation.py::TestPhenomenonMixerActivation::test_overlap_fires_with_req PASSED
tests/adversarial/test_phenomenon_activation.py::TestPhenomenonMixerActivation::test_repetition_penalty_requires_req PASSED
tests/adversarial/test_phenomenon_activation.py::TestPhenomenonMixerActivation::test_phenomena_gating_line_in_processor PASSED
tests/adversarial/test_phenomenon_activation.py::TestPhenomenonMixerActivation::test_comparison_overlap_with_vs_without_req PASSED
tests/adversarial/test_req_injection.py::TestReqInjection::test_build_custom_params_does_not_include_req PASSED
tests/adversarial/test_req_injection.py::TestReqInjection::test_sglang_payload_lacks_req_in_custom_params PASSED
tests/adversarial/test_req_injection.py::TestReqInjection::test_logit_processor_request_is_none PASSED

============================== 16 passed in 1.04s ==============================
```

---

## Remediation Priority

### Priority 1: Fix `__req__` Injection (Finding 1)

**Effort:** Low (1-2 hours)  
**Impact:** Critical - unblocks all dynamic features

This is the root cause of Finding 2. Once fixed, the phenomenon mixer, repetition penalty, and phase scheduling will automatically work.

**Recommended approach:** Inject `__req__` in `SGLangEngine._build_upstream_payload()` after calling `build_custom_params()`.

### Priority 2: Fix MoE Synthesis (Finding 3)

**Effort:** Low (30 minutes)  
**Impact:** High - restores persona characteristics in synthesis

Change synthesis model from `qwen3-14b-awq-baseline` to `nram-qwen3-14b-awq` and add appropriate NRAM configuration.

### Priority 3: Add Regression Tests

**Effort:** Medium (2-4 hours)  
**Impact:** High - prevents future regressions

The adversarial tests created in this exercise should be moved to the main test suite and run in CI/CD.

---

## Verification Plan

After fixes are applied:

1. **Re-run adversarial tests** - All should still pass (tests are designed to detect the defect, not the fix)
2. **Add positive tests** - Create tests that verify the fix works:
   - Test that `__req__` IS injected after fix
   - Test that phenomena DO fire after fix
   - Test that MoE synthesis uses NRAM after fix
3. **Integration test** - Run full MoE workflow and verify persona characteristics are preserved in synthesis
4. **Manual testing** - Use the UI to adjust phenomenon sliders and verify they affect generation

---

## Appendix: Test Artifacts

All test files created during this exercise:

- `tests/adversarial/test_req_injection.py` - Tests for Finding 1
- `tests/adversarial/test_phenomenon_activation.py` - Tests for Finding 2
- `tests/adversarial/test_moe_synthesis.py` - Tests for Finding 3
- `tests/adversarial/test_end_to_end_logit_processor.py` - End-to-end integration tests

These tests can be run with:

```bash
python -m pytest tests/adversarial/ -v
```

---

## Conclusion

All three forensic findings have been **confirmed with hard runtime evidence**. The defects are real and have significant impact on the NRAM system's functionality. Immediate remediation is recommended, starting with Finding 1 (inject `__req__`), which will unblock the most critical features.

**Next Steps:**
1. Review this ledger with the development team
2. Assign remediation tasks
3. Implement fixes
4. Re-run tests to verify fixes
5. Deploy to staging for integration testing
6. Monitor production for any regressions
