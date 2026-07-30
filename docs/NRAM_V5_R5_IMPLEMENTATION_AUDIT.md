# NRAM V5 R5 Implementation Audit

**Commit:** `e60bc53`
**Branch:** `nram-v5-implementation`
**Date:** 2026-07-30
**Auditor:** Builder

## Executive Summary

The current DExperts implementation (R4) has **fundamental architectural defects** that violate the DExperts paper requirements and prevent scientific validity. This audit documents where raw base logits enter the custom processor, how SGLang applies sampling, and the critical missing components.

---

## 1. Where Raw Base Logits Enter Custom Processor

**Location:** [`nram_sglang/processor.py:96-364`](../nram_sglang/processor.py:96)

**Entry Point:**
```python
def __call__(self, logits: Any, custom_param_list: Optional[List[Dict[str, Any]]] = None) -> Any:
```

**What happens:**
1. SGLang calls the custom logit processor with `logits` tensor (shape: `[batch_size, vocab_size]`)
2. These are **raw logits from the base model's final linear layer** BEFORE any sampling
3. The processor iterates over `custom_param_list` (one dict per batch row)
4. Each dict contains per-request parameters from `build_custom_params()` in `core/steering/serialization.py`

**Critical observation:** The `logits` tensor is **mutable** — modifications directly affect what SGLang samples next.

---

## 2. Where SGLang Applies Temperature, Top-K, Top-P, Sampling

**SGLang Sampling Pipeline (external to custom processor):**

```
Base Model Forward Pass
    ↓
Raw Logits (vocab_size)
    ↓
[Custom Logit Processor runs HERE] ← We modify logits
    ↓
Temperature scaling
    ↓
Top-K filtering
    ↓
Top-P (nucleus) filtering
    ↓
Softmax → probabilities
    ↓
Multinomial sampling → selected token
```

**Critical:** Our custom processor runs **BEFORE** temperature/top-k/top-p. This means:
- We can apply DExperts formula to raw logits
- SGLang's sampler will then apply its own temperature and filtering
- We CANNOT control what happens after our processor returns

**Implication for canonical decoding order:**
The DExperts paper specifies:
1. Compute base support from unmodified base logits (BEFORE expert perturbation)
2. Apply expert perturbation: `z_combined = z_base + alpha * (z_expert - z_anti_expert)`
3. Mask tokens outside base support
4. Allow sampler to apply temperature and final selection

**Current implementation defect:** The processor applies DExperts formula but does NOT compute base support first. It directly modifies logits without truncation.

---

## 3. Custom Logit Processor Execution Order

**Location:** [`nram_sglang/processor.py:96-364`](../nram_sglang/processor.py:96)

**Execution order within one batch row:**

1. **Parameter extraction** (lines 114-146)
   - Extract positive/negative/forbidden token IDs
   - Extract bias values (positive_bias, negative_bias, repetition_penalty, corporate_jargon_penalty)

2. **Phase schedule computation** (lines 148-163)
   - Compute generation progress (0.0 to 1.0)
   - Apply phase-dependent scaling multipliers

3. **Layer 2: Phrase constraints** (lines 166-172)
   - Forbidden phrase masking
   - Source n-gram blocking

4. **Base steering** (lines 175-203)
   - Apply positive token bias
   - Apply negative token bias
   - Apply forbidden token masking (-inf)
   - Apply dynamic repetition penalty
   - Apply corporate jargon penalty

5. **Layer 3: Entropy control** (lines 206-212)
   - PID controller for entropy targeting
   - Phase-based target entropy
   - Temperature scaling via logit division

6. **Layer 3: Concept injection** (lines 215-221)
   - Dynamic concept capsules
   - Phase-dependent strength curve

7. **Soft injections** (lines 223-227)
   - Time-windowed token biases

8. **Logit vectors** (lines 229-233)
   - Vocabulary-space vector addition

9. **Phenomenon mixer** (lines 236-244)
   - overlap, forgetting, looping, insight, synesthesia, associative_jump, dissolution

10. **Layer 4: DExperts toxicity steering** (lines 247-277)
    - Extract input_ids from request object
    - Call `self._dexperts_runtime.apply_dexperts()`
    - Replace logits with combined logits

11. **Coherence floor enforcement** (lines 280-299)
    - Measure trigram coherence
    - Reduce steering if coherence too low

12. **Forced token control** (lines 301-320)
    - Apply forced token if enabled
    - Mask all other tokens to -inf

13. **Safety checks** (lines 322-332)
    - Check for NaN/+Inf
    - Check for all-masked rows

14. **Telemetry emission** (lines 334-352)
    - Emit bounded JSON event with before/after logits

15. **Request state cleanup** (lines 355-362)
    - Release DExperts state when request finishes

**Critical observation:** DExperts runs at step 10, AFTER base steering, entropy control, concept injection, and phenomenon mixer. This means:
- DExperts formula is applied to **already-modified logits**, not raw base logits
- This violates the canonical decoding order from the DExperts paper

---

## 4. How Request Identity Maps to SGLang Batch Rows

**Request identity flow:**

1. **HTTP request arrives** at FastAPI endpoint (`api/routes.py:510`)
2. **Runtime request ID generated:** `f"nram-scheduler-{uuid.uuid4().hex}"` (line 619)
3. **Engine request created** via `_to_engine_request()` (line 596)
4. **SGLang receives request** via HTTP POST to SGLang server
5. **SGLang assigns internal request ID** (scheduler request ID, `req_pool_indices`)
6. **Custom logit processor receives** `params` dict with:
   - `params["request_id"]` = our runtime request ID (from `build_custom_params()`)
   - `params["__req__"]` = SGLang's internal request object (has `.rid`, `.output_ids`, etc.)

**State keying:**
- **HTTP request ID** = `nram-scheduler-{uuid}` (our external ID)
- **SGLang request ID** = `request.rid` (internal scheduler ID)
- **req_pool_indices** = SGLang's batch row index (changes per step)

**Current implementation:** DExperts state is keyed by `params["request_id"]` (our external ID), which is correct.

---

## 5. State Keying (HTTP Request ID vs SGLang Request ID vs req_pool_indices)

**Current state keying:**

```python
# In processor.py line 253
request_id = str(params.get("request_id", ""))

# In dexperts_runtime.py line 124
def get_or_create_state(self, request_id: str) -> DExpertsRuntimeState:
    if request_id not in self._request_states:
        self._request_states[request_id] = DExpertsRuntimeState(request_id=request_id)
    return self._request_states[request_id]
```

**Problem:** State is keyed by our external request ID, which is stable across the request lifetime. However:
- `req_pool_indices` can be reused by SGLang after request completion
- If state is not properly released, a new request could inherit old state

**Current cleanup:**
```python
# In processor.py lines 357-362
if request is not None and self._dexperts_runtime is not None:
    finished = getattr(request, "finished", False)
    if finished:
        request_id = str(params.get("request_id", ""))
        if request_id:
            self._dexperts_runtime.release_state(request_id)
```

**Defect:** Cleanup only happens when `request.finished == True`. If request is cancelled, times out, or raises exception, state may leak.

---

## 6. Request Completion/Cancellation State Removal

**Current state removal triggers:**
1. `request.finished == True` (normal completion)

**Missing triggers:**
1. Request cancellation (client disconnect)
2. Request timeout
3. Request exception
4. Request abortion (SGLang scheduler abort)
5. Stale state cleanup (periodic sweep)

**Required invariants:**
- State must be removed on: completion, EOS, cancellation, timeout, disconnect, exception, abortion
- State storage must be bounded and observable
- Stale-state cleanup as final safeguard

---

## 7. Prompt Prefill vs Decode Step Handling

**SGLang execution phases:**

1. **Prefill phase:** Process entire prompt in one forward pass
   - `input_ids` = full prompt token IDs
   - `output_ids` = [] (empty)
   - Custom logit processor called ONCE for last prompt token

2. **Decode phase:** Generate tokens autoregressively
   - `input_ids` = [last_generated_token] (single token)
   - `output_ids` = [all_previously_generated_tokens]
   - Custom logit processor called ONCE per generated token

**Current implementation defect:**
```python
# In processor.py lines 256-260
input_ids = None
attention_mask = None
if request is not None:
    input_ids = getattr(request, "input_ids", None)
    attention_mask = getattr(request, "attention_mask", None)
```

**Problem:** `request.input_ids` is the **full prompt** during prefill, but only the **last generated token** during decode. The DExperts runtime receives different `input_ids` shapes depending on phase, but doesn't distinguish between them.

**Required:** DExperts must maintain separate KV caches for expert and anti-expert, and feed them the **same token history** (prompt + all generated tokens so far).

---

## 8. Expert/Anti-Expert KV Cache Separation (CURRENTLY MISSING)

**Current implementation:**

```python
# In dexperts_runtime.py lines 106-119
self.expert_adapter = PeftModel.from_pretrained(
    self.base_model,
    expert_adapter_path,
    adapter_name="expert",
)

self.anti_expert_adapter = PeftModel.from_pretrained(
    self.base_model,
    anti_expert_adapter_path,
    adapter_name="anti_expert",
)
```

**Critical defect:** Both adapters wrap the **same base model instance**. When you call `expert_adapter(input_ids)`, it uses the base model's KV cache. When you call `anti_expert_adapter(input_ids)`, it **overwrites** the same KV cache.

**This violates the DExperts paper requirement:** Expert and anti-expert must have **separate KV-cache trees**.

**Required architecture:**
```python
@dataclass
class DExpertsRequestState:
    runtime_request_id: str
    sglang_request_index: int
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    expert_past_key_values: Any  # Separate KV cache
    anti_expert_past_key_values: Any  # Separate KV cache
    expert_position: int
    anti_expert_position: int
    last_token_id: int | None
    step: int
    created_at: float
    last_accessed_at: float
    finished: bool
```

**Invariants:**
1. Expert and anti-expert must have separate KV-cache trees
2. Both paths receive exactly the same token history
3. Positions remain equal at every step
4. History = original prompt + every generated token
5. State identity bound to real SGLang request/batch identity
6. Reused request-pool index never inherits old state
7. State removed on: completion, EOS, cancellation, timeout, disconnect, exception, abortion
8. State storage bounded and observable
9. Stale-state cleanup as final safeguard

---

## 9. LoRA Adapter Switching Concurrency Safety

**Current implementation:**

```python
# In dexperts_runtime.py lines 156-160
self.expert_adapter.set_adapter("expert")
outputs = self.expert_adapter(input_ids=input_ids, attention_mask=attention_mask)
```

**Critical defect:** `set_adapter()` is **not thread-safe**. If two requests are processed concurrently in the same batch, they will race to set the adapter, causing:
- Request A sets adapter to "expert"
- Request B sets adapter to "anti_expert" (overwrites A)
- Request A runs forward pass with wrong adapter

**Required:** Explicit execution lock around adapter switching and forward pass:

```python
with self._execution_lock:
    # Activate expert adapter
    self.backbone.set_adapter("expert")
    expert_outputs = self.backbone(input_ids, past_key_values=expert_kv)
    expert_kv = expert_outputs.past_key_values
    
    # Activate anti-expert adapter
    self.backbone.set_adapter("anti_expert")
    anti_expert_outputs = self.backbone(input_ids, past_key_values=anti_expert_kv)
    anti_expert_kv = anti_expert_outputs.past_key_values
```

---

## 10. Prompt Inclusion (Full vs Generated-Only)

**DExperts paper requirement:** Expert and anti-expert must process the **same token history** as the base model.

**Token history = original prompt + all generated tokens so far**

**Current implementation defect:**
```python
# In processor.py lines 256-260
input_ids = getattr(request, "input_ids", None)
```

During decode, `request.input_ids` is only the **last generated token**, not the full history.

**Required:** Maintain full token history in request state:
```python
state.prompt_token_ids = [...]  # From prefill
state.generated_token_ids = [...]  # Accumulated during decode
full_history = state.prompt_token_ids + state.generated_token_ids
```

---

## 11. Ablation API Type (Raw Completions vs Chat Completions)

**Current implementation:** Uses OpenAI-compatible chat completions API with system/developer prompts.

**Scientific requirement:** For mechanism proof, need **raw text completion** (NO system/developer/safety prompts, NO chat template).

**Required:** Add endpoint or parameter to bypass chat template and send raw text directly to model.

---

## 12. System/Developer Prompt Injection

**Current implementation:**
```python
# In sglang_engine.py
developer_instruction = policy.developer_instruction
messages = [{"role": "system", "content": developer_instruction}] + request.messages
```

**Problem:** System/developer prompts inject safety steering that conflicts with DExperts toxicity reduction.

**Required for Regime A (mechanism proof):** Raw completion with NO system/developer prompts.

---

## 13. Semantic Coherence Runtime Status

**Current implementation:** [`core/steering/semantic_coherence.py`](../core/steering/semantic_coherence.py)

**Status:** Uses real embedding model (Qwen3-Embedding-0.6B), computes cosine similarity.

**Defects:**
1. No startup preload or warm-up command
2. No readiness state distinct from process health
3. No explicit model path and revision
4. No protection against hidden network download during run
5. No model/tokenizer hashes in manifest

---

## Summary of Critical Defects

| # | Defect | Severity | Impact |
|---|--------|----------|--------|
| 1 | No separate KV caches for expert/anti-expert | CRITICAL | Violates DExperts paper requirement |
| 2 | No proper state machine with token history | CRITICAL | Cannot maintain separate KV positions |
| 3 | No canonical decoding order (base support truncation) | HIGH | Violates paper algorithm |
| 4 | No tokenizer compatibility gate | HIGH | Cannot fail closed on mismatch |
| 5 | No mechanistic proof (teacher-forced logit analysis) | HIGH | Only generation-level ablation |
| 6 | LoRA adapter switching not thread-safe | CRITICAL | Concurrent requests corrupt state |
| 7 | State cleanup incomplete (missing cancellation/timeout) | HIGH | State leak on error paths |
| 8 | No context bounds enforcement | MEDIUM | Can exceed model context length |
| 9 | Scientific null result (Qwen3-14B-AWQ too well-aligned) | HIGH | Cannot demonstrate toxicity reduction |
| 10 | No raw completion mode for mechanism proof | HIGH | Cannot bypass safety steering |

---

## Required Implementation (Phases 2-8)

See implementation contract for detailed specifications.

**Phase 2:** Correct DExperts state machine with separate KV caches
**Phase 3:** Shared-backbone LoRA execution with explicit lock
**Phase 4:** Tokenizer compatibility gate (fail closed)
**Phase 5:** Canonical DExperts decoding order (base support truncation)
**Phase 6:** Context bounds (4096 tokens for DExperts)
**Phase 7:** Two scientific regimes (0.6B base for mechanism proof, 14B for aligned-model floor)
**Phase 8:** Semantic coherence runtime (startup preload, readiness state, model hashes)

---

## Files to Modify

1. `core/steering/dexperts_runtime.py` — Complete rewrite with correct state machine
2. `nram_sglang/processor.py` — Integrate new DExperts runtime, add base support truncation
3. `core/steering/semantic_coherence.py` — Add startup preload, readiness state, model hashes
4. `api/routes.py` — Add raw completion endpoint for mechanism proof
5. `scripts/check_tokenizer_compatibility.py` — New tokenizer compatibility gate
6. `config/experiments/dexperts_qwen06b.yaml` — Regime A configuration
7. `config/experiments/dexperts_qwen14b_awq.yaml` — Regime B configuration
8. `docker-compose.dexperts-proof.yml` — Docker compose for proof regime
9. `tests/gpu/test_dexperts_causal.py` — Update tests for new architecture
10. `tests/runtime_adversary/test_dexperts_runtime_falsification.py` — Update falsification tests

---

## Metrics to Expose

- `active_dexperts_requests` — Number of active DExperts requests
- `created_states_total` — Total states created since startup
- `cleaned_states_total` — Total states cleaned (completion/cancellation/timeout)
- `stale_states_total` — Total stale states detected and cleaned
- `request_state_collisions_total` — Total state identity collisions
- `expert_cache_position` — Current KV cache position for expert
- `anti_expert_cache_position` — Current KV cache position for anti-expert
- `history_token_count` — Current token history length

---

## Telemetry Per Token

- request identity
- SGLang batch row/request-pool index
- token position
- alpha
- base support size
- expert/anti-expert cache positions
- finite-logit counts
- ||z_expert - z_anti||
- ||z_combined - z_base||
- KL(base || combined) on retained support
- selected token ID
- selected-token base log-probability
- selected-token combined log-probability
- adapter identities

---

**END OF AUDIT**
