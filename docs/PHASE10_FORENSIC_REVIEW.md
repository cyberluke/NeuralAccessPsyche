# Phase 10: Forensic Review Report

**Reviewer**: Forensic Reviewer Mode  
**Date**: 2026-07-29  
**Commit**: cca310c  
**Scope**: Deep analysis of implementation defects, security vulnerabilities, and edge cases

---

## Executive Summary

The NRAM v5 implementation demonstrates **solid engineering fundamentals** with proper request isolation, comprehensive validation, and robust error handling. However, **three critical defects** were identified that undermine the scientific validity of the system:

1. **`disabled_nram` control failure** - Negative control shows large effects instead of matching baseline
2. **Coherence degradation** - All profiles significantly reduce coherence (d=-0.52 to -1.62)
3. **Concept injection ineffective** - Shows negligible effect on novelty

**Overall Assessment**: ⚠️ **PARTIALLY FUNCTIONAL** - Core mechanisms work but critical defects prevent production deployment.

---

## 1. Critical Defects

### 1.1 `disabled_nram` Control Failure (CRITICAL)

**Location**: `core/engines/sglang_engine.py:521-526`  
**Severity**: CRITICAL  
**Impact**: Invalidates negative control validation

**Observed Behavior**:
```
disabled_nram condition:
- Novelty: d = -1.06, p < 0.001
- Coherence: d = -1.81, p < 0.001
```

**Expected Behavior**: `disabled_nram` should match baseline exactly (d ≈ 0).

**Root Cause Analysis**:

The `_is_nram_enabled()` method checks:
```python
def _is_nram_enabled(self, request: ChatCompletionRequest) -> bool:
    if request.model in NRAM_ENABLED_ALIASES:
        nram_opts = request.nram or {}
        return nram_opts.get("enabled", True)
    return False
```

When `enabled: False` is passed, the method should return `False`. However, the large effect sizes suggest NRAM steering is still active.

**Investigation Required**:
1. Verify request flow from API to engine
2. Check if `nram_opts` is properly passed through `_route_persona()`
3. Validate that `enabled: False` actually disables the logit processor

**Impact**: This defect makes it impossible to distinguish between "NRAM not working" and "NRAM working correctly." All downstream scientific claims are compromised.

---

### 1.2 Coherence Degradation (HIGH)

**Location**: `core/persona/profiles.py`, `core/steering/serialization.py`  
**Severity**: HIGH  
**Impact**: All NRAM profiles degrade output quality

**Observed Behavior**:
```
profile_normal:      d = -1.16, p < 0.001
profile_peak:        d = -0.90, p < 0.001
profile_psychedelic: d = -0.52, p < 0.001
concepts_injection:  d = -1.62, p < 0.001
```

**Root Cause Analysis**:

The `coherence_floor` mechanism is defined in `NRAMState` but **not enforced** in the logit processor. The `_build_upstream_payload()` method compiles token biases but does not apply coherence constraints.

**Missing Implementation**:
- `coherence_floor` parameter is accepted but ignored
- No mechanism to prevent coherence degradation below threshold
- No feedback loop to adjust steering when coherence drops

**Impact**: Users expect NRAM to maintain coherence while increasing novelty. Current implementation trades coherence for novelty, violating the core value proposition.

---

### 1.3 Concept Injection Ineffective (MEDIUM)

**Location**: `core/engines/sglang_engine.py:650-688`  
**Severity**: MEDIUM  
**Impact**: Concept injection does not measurably influence output

**Observed Behavior**:
```
concepts_injection: d = -0.03, p = 0.728 (not significant)
```

**Root Cause Analysis**:

The `_build_concept_config()` method encodes concept tokens:
```python
def _build_concept_config(self, request, nram_opts):
    concepts = nram_opts.get("concepts", [])
    if not concepts or not self._tokenizer:
        return None
    
    encoded_concepts = []
    for concept in concepts:
        token_forms = concept.get("en_tokens", []) + concept.get("cs_tokens", []) + concept.get("synonyms", [])
        token_ids = []
        for form in token_forms:
            try:
                ids = self._tokenizer.encode(form, add_special_tokens=False)
                token_ids.extend(ids)
            except Exception:
                pass
        if token_ids:
            encoded_concepts.append({
                "concept_id": concept.get("concept_id", ""),
                "token_ids": list(set(token_ids)),
                "activation_phase": concept.get("activation_phase"),
                "max_uses": concept.get("max_uses", 0),
            })
    
    return {
        "concepts": encoded_concepts,
        "base_strength": nram_opts.get("concept_strength", 0.5),
    }
```

**Possible Issues**:
1. Concept tokens not properly compiled into token IDs
2. `concept_strength` parameter not applied correctly in logit processor
3. Selected concepts ("innovation", "breakthrough", "novel", "creative") do not measurably influence output semantics
4. No phase-based activation logic implemented

**Investigation Required**:
- Verify concept tokens are actually injected into logit processor
- Check if `base_strength` is applied during decoding
- Validate concept selection methodology

---

## 2. Security Analysis

### 2.1 Input Validation ✅ GOOD

**Location**: `api/routes.py:522-534`

```python
# Reject client-supplied processor injection attempts
nram_opts = request.nram or {}
_FORBIDDEN_FIELDS = {
    "custom_logit_processor", "serialized_processor",
    "processor_class", "python_code",
}
if _FORBIDDEN_FIELDS & set(nram_opts.keys()):
    return _openai_error(
        400,
        "Forbidden field in nram options: processor injection is not allowed.",
        "invalid_request_error",
        "forbidden_field",
    )
```

**Assessment**: ✅ **SECURE** - Client cannot inject custom processors or execute arbitrary code.

### 2.2 Request Isolation ✅ GOOD

**Location**: `nram_sglang/hooks/batch_context.py`

The batch context system properly isolates request-scoped state:
```python
def register_request(request_id: str, nram_opts: Dict[str, Any]) -> None:
    """Register request-scoped NRAM configuration."""
    _REQUEST_CONTEXT[request_id] = {
        "nram_opts": nram_opts,
        "interventions": [],
        "telemetry_enabled": nram_opts.get("telemetry_enabled", False),
        # ... other fields
    }

def cleanup_request(request_id: str) -> None:
    """Clean up request context after completion."""
    _REQUEST_CONTEXT.pop(request_id, None)
```

**Assessment**: ✅ **SECURE** - No cross-request contamination.

### 2.3 Tokenizer ID Validation ✅ GOOD

**Location**: `core/contracts/nram_runtime.py:validate_tokenizer_ids()`

```python
def validate_tokenizer_ids(nram_opts: Dict[str, Any], tokenizer: Any) -> None:
    """Validate that all configured token IDs are within tokenizer vocabulary."""
    vocab_size = len(tokenizer)
    
    for field in ["forbidden_token_ids", "forced_token_id"]:
        ids = nram_opts.get(field, [])
        if isinstance(ids, int):
            ids = [ids]
        for token_id in ids:
            if token_id < 0 or token_id >= vocab_size:
                raise ValueError(f"Token ID {token_id} out of range [0, {vocab_size})")
```

**Assessment**: ✅ **SECURE** - Prevents out-of-bounds token ID injection.

### 2.4 Forbidden Feature Rejection ✅ GOOD

**Location**: `core/engines/sglang_engine.py:858-865`

```python
requested_unsupported = sorted(
    key for key in _UNSUPPORTED_NRAM_FEATURES if nram_opts.get(key)
)
if requested_unsupported:
    raise SGLangEngineError(
        "Unsupported NRAM runtime feature(s): " + ", ".join(requested_unsupported),
        status_code=400,
    )
```

**Assessment**: ✅ **SECURE** - Explicitly rejects unsupported features with clear error messages.

---

## 3. Performance Analysis

### 3.1 Request Lifecycle Management ✅ GOOD

**Location**: `api/routes.py:226-257`

```python
async def _complete_with_lifecycle(engine, engine_request, raw_request=None):
    """Race non-stream completion against a real ASGI disconnect event."""
    task = asyncio.create_task(engine.complete(engine_request))
    if raw_request is None:
        return await task
    
    watcher = asyncio.create_task(_wait_for_disconnect(raw_request))
    try:
        done, _ = await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return await task
        if watcher.result():
            await engine.abort(engine_request.runtime_request_id, reason="client_disconnected")
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise HTTPException(status_code=499, detail={"code": "client_disconnected"})
        return await task
    except asyncio.CancelledError:
        if not task.done():
            await engine.abort(engine_request.runtime_request_id, reason="downstream_cancelled")
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        raise
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
```

**Assessment**: ✅ **EFFICIENT** - Proper cancellation handling prevents resource leaks.

### 3.2 Streaming Lifecycle ✅ GOOD

**Location**: `api/routes.py:260-323`

```python
async def _stream_with_lifecycle(engine, engine_request, raw_request):
    """Relay SSE while independently monitoring disconnect and closing upstream."""
    upstream = engine.stream(engine_request).__aiter__()
    disconnect = (
        asyncio.create_task(_wait_for_disconnect(raw_request))
        if raw_request is not None
        else None
    )
    # ... proper cleanup in finally block
```

**Assessment**: ✅ **EFFICIENT** - Proper streaming with disconnect detection.

### 3.3 Abort Mechanism ✅ GOOD

**Location**: `core/engines/sglang_engine.py:366-391`

```python
async def abort(self, scheduler_request_id: str, reason: str = "client_cancelled") -> None:
    """Abort one real SGLang request through the official 0.5.16 endpoint."""
    if not scheduler_request_id:
        return
    async with self._abort_lock:
        if scheduler_request_id in self._aborted_request_ids:
            return
        self._aborted_request_ids[scheduler_request_id] = None
        # Bounded history to prevent memory leak
        if len(self._aborted_request_ids) > 4096:
            self._aborted_request_ids.pop(next(iter(self._aborted_request_ids)))
    # ... abort call to SGLang
```

**Assessment**: ✅ **EFFICIENT** - Bounded abort history prevents memory leaks.

---

## 4. Error Handling Analysis

### 4.1 Comprehensive Error Types ✅ GOOD

**Location**: `api/routes.py:579-593`

```python
except SGLangEngineError as e:
    logger.error(f"SGLang engine error: {e.message}")
    return _openai_error(e.status_code, e.message, "inference_error", "upstream_inference_failed")

except InferenceError as e:
    logger.error(f"Inference error: {e.message}")
    return _openai_error(502, e.message, "inference_error", e.code)

except HTTPException:
    raise

except Exception as e:
    logger.error(f"Chat completion error: {str(e)}", exc_info=True)
    return _openai_error(500, str(e), "inference_error", "upstream_inference_failed")
```

**Assessment**: ✅ **ROBUST** - All error types properly handled with OpenAI-compatible responses.

### 4.2 Validation Errors ✅ GOOD

**Location**: `core/engines/sglang_engine.py:538-554`

```python
def validate(self, request: ChatCompletionRequest) -> None:
    """Preflight one request without planning, network I/O, or generation."""
    self._resolve_upstream_model(request)
    options = request.nram or {}
    requested_unsupported = sorted(
        key for key in _UNSUPPORTED_NRAM_FEATURES if options.get(key)
    )
    if requested_unsupported:
        raise SGLangEngineError(
            "Unsupported NRAM runtime feature(s): " + ", ".join(requested_unsupported),
            status_code=400,
        )
    
    if self._tokenizer is not None:
        try:
            validate_tokenizer_ids(options, self._tokenizer)
        except ValueError as exc:
            raise SGLangEngineError(str(exc), status_code=400) from exc
```

**Assessment**: ✅ **ROBUST** - Preflight validation catches errors before generation.

---

## 5. Telemetry and Observability

### 5.1 State Hashing ✅ GOOD

**Location**: `core/engines/sglang_engine.py:407-456`

```python
def _applied_state_hash(self, request, payload) -> str:
    """Hash the complete behaviorally relevant, resolved request state."""
    processor_params = dict(payload.get("custom_params") or {})
    processor_params.pop("request_id", None)
    processor_params.pop("config_hash", None)
    processor_params.pop("applied_state_schema", None)
    applied = {
        "schema": "nram.applied-state.v1",
        "route": request.route_kind,
        "public_model": request.public_model or request.model,
        "actual_base_model": payload.get("model"),
        "model_artifact": {
            "configured_model": self._model,
            "snapshot": os.environ.get("NRAM_MODEL_SNAPSHOT", "unknown"),
        },
        "tokenizer_artifact": {
            "identity": os.environ.get("NRAM_TOKENIZER_ID", "unknown"),
            "hash": os.environ.get("NRAM_TOKENIZER_HASH", "unknown"),
            "vocab_size": int(getattr(self._tokenizer, "vocab_size", 0) or len(self._tokenizer)),
        },
        "messages": payload.get("messages", []),
        "sampling": {
            key: payload.get(key)
            for key in ("max_tokens", "temperature", "top_p", "seed", "stop",
                       "frequency_penalty", "presence_penalty", "stream")
        },
        "grammar_or_response_format": payload.get("response_format"),
        "processor": processor_params,
        "defaults_version": "nram.sglang.defaults.v1",
    }
    normalized = self._normalize_hash_value(applied)
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

**Assessment**: ✅ **EXCELLENT** - Comprehensive state hashing enables reproducibility and debugging.

### 5.2 Correlation Tracking ✅ GOOD

**Location**: `core/engines/sglang_engine.py:487-519`

```python
@staticmethod
def _correlation(request, payload, *, sampled=None) -> Optional[Dict[str, Any]]:
    params = payload.get("custom_params")
    if params is None:
        return {
            "scheduler_request_id": payload["rid"],
            "public_model": request.public_model or request.model,
            "actual_base_model": payload["model"],
            "processor_intervened": False,
            "sampled_token_ids": [],
            "sampled_tokens": [],
            "sample_join": "not_requested",
        }
    sampled = sampled or []
    return {
        "request_id": params["request_id"],
        "scheduler_request_id": payload["rid"],
        "config_hash": params["config_hash"],
        "applied_state_schema": params["applied_state_schema"],
        "public_model": request.public_model or request.model,
        "actual_base_model": payload["model"],
        "processor_intervened": True,
        "sampled_token_ids": [int(item[1]) for item in sampled if len(item) >= 2],
        "sampled_tokens": [str(item[2]) for item in sampled if len(item) >= 3],
        "sample_join": "sglang_meta_info" if sampled else (
            "unavailable_for_streaming" if payload.get("stream") else "not_requested"
        ),
    }
```

**Assessment**: ✅ **EXCELLENT** - Full correlation tracking enables end-to-end debugging.

---

## 6. Edge Cases and Boundary Conditions

### 6.1 Reasoning Content Stripping ✅ GOOD

**Location**: `core/engines/sglang_engine.py:209-249`

```python
_REASONING_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>(?:(?!<think>).)*$", re.DOTALL)

_REASONING_HEURISTIC_RE = re.compile(
    r"^(?:Okay,?|Hmm,?|Let me|Let's|I need to|I'm trying|I should|So,?|"
    r"First,?|Alright,?|Well,?|The user|I think|I wonder|I'll|Now,?|"
    # Czech reasoning preambles (strong NRAM steering → Czech CoT)
    r"Potřebuji|Musím|Uživatel|Podívám|Zkusím|Pojďme|Nejprve|"
    r"Zamyslím|Přemýšlím|Dobře,?|Takže,?|Hmm|Aha,?|No,?|"
    r"Tak,?|Podívejme|Pojď|Uvažuj|Analyzuj|Zvaž)",
    re.IGNORECASE,
)

def strip_reasoning(text: str) -> str:
    """Remove <think> tags AND heuristic chain-of-thought from final text."""
    text = _REASONING_RE.sub("", text).strip()
    text = _UNCLOSED_THINK_RE.sub("", text).strip()
    
    # Iteratively strip leading reasoning preambles (max 3 sentences)
    for _ in range(3):
        if not _REASONING_HEURISTIC_RE.match(text):
            break
        parts = re.split(r"(?<=[.!?…])\s+", text, maxsplit=1)
        if len(parts) > 1 and len(parts[1].strip()) > 20:
            text = parts[1].strip()
        else:
            break
    
    return text
```

**Assessment**: ✅ **ROBUST** - Handles both tagged and untagged reasoning content, including Czech language edge cases.

### 6.2 Streaming State Machine ✅ GOOD

**Location**: `core/engines/sglang_engine.py:252-331`

```python
class _StreamState:
    """Per-stream state machine that strips <think></think> blocks from SSE chunks."""
    
    def __init__(self, public_model: str, correlation: Optional[Dict[str, Any]] = None) -> None:
        self._public_model = public_model
        self._correlation = correlation
        self._buffer = ""
    
    def feed(self, data_line: str) -> List[str]:
        """Consume one raw `data: {...}` line; yield sanitized `data: ...` lines."""
        # ... handles partial tags split across chunk boundaries
    
    def _extract_visible(self) -> str:
        """Return content outside <think></think> blocks, buffering partial tags."""
        # ... holds back trailing partial close-tag to avoid emitting it piecewise
```

**Assessment**: ✅ **ROBUST** - Properly handles partial tags and chunk boundaries.

---

## 7. Recommendations

### 7.1 Critical (Must Fix Before Release)

1. **Fix `disabled_nram` defect**: Investigate why `enabled: False` does not disable NRAM steering. This is a correctness issue that affects all downstream analysis.

2. **Implement coherence floor enforcement**: The `coherence_floor` parameter should actually prevent coherence degradation below a threshold. Add validation in the logit processor.

3. **Verify concept injection pipeline**: Ensure concept tokens are properly compiled and applied. Add telemetry to track concept activation.

### 7.2 Important (Should Fix)

4. **Add regression tests**: Add tests that verify coherence does not degrade below a threshold.

5. **Document trade-offs**: Create a "Known Limitations" section that explicitly states the coherence-novelty trade-off.

6. **Implement DExperts**: Layer 3 is incomplete without DExperts. This requires expert/anti-expert model artifacts.

### 7.3 Nice-to-Have (Future Work)

7. **Implement Layer 4 (Representation Control)**: ActAdd, Conceptor, ReFT require hidden-state access and trained artifacts.

8. **Implement Layer 5 (Generation Search)**: Branch-and-tournament requires distinct branch runtime and scorers.

9. **Implement Layer 6 (Closed-Loop)**: Real-time evaluation and feedback loop require embedding model and control logic.

---

## 8. Conclusion

The NRAM v5 implementation demonstrates **solid engineering fundamentals** with proper security, request isolation, and error handling. However, **three critical defects** prevent production deployment:

1. **`disabled_nram` control failure** - Invalidates negative control validation
2. **Coherence degradation** - All profiles significantly reduce coherence
3. **Concept injection ineffective** - Shows negligible effect

**Recommendation**: Fix critical defects, then proceed to Phase 11 (Runtime Adversary) and Phase 12 (Release Gate) before considering release.

---

**Review Completed**: 2026-07-29T06:55:00Z  
**Reviewer**: Forensic Reviewer Mode  
**Evidence Base**: Code review of `api/routes.py`, `core/engines/sglang_engine.py`, ablation study artifacts
