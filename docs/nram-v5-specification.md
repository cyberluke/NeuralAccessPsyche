# NRAM v5 Architecture Specification

> **Design document, not implementation evidence.** Advanced representation,
> DExperts, tournament, ReFT, head-gating, and KV-cache sections below are
> unimplemented design targets unless `NRAM_V5_RUNTIME_WIRING.md` explicitly
> marks a bounded component proven.

**Version:** 5.0.0-draft  
**Date:** 2026-07-27  
**Status:** HISTORICAL DESIGN; IMPLEMENTATION STATUS RECORDED SEPARATELY

---

## Executive Summary

NRAM v5 transforms the system from a simple token-bias layer into a **multi-layer inference-control stack** that operates across symbolic, logit, latent, and search dimensions simultaneously.

The architecture addresses critical defects identified in v4 (persona routing bypass, missing tokenizer injection, dead phenomenon mixer) while introducing advanced steering mechanisms: phrase-aware constraints, entropy control, activation addition, and closed-loop evaluation.

---

## 1. Architecture Overview

### 1.1 Six-Layer Inference-Control Stack

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 6: CLOSED-LOOP EVALUATION                             │
│ Real-time novelty, coherence, source-distance monitoring    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: GENERATION SEARCH                                  │
│ Branch-and-tournament, divergent speculative decoding       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: REPRESENTATION CONTROL                             │
│ Activation Addition, Conceptors, ReFT, soft prefixes        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: LOGIT CONTROL                                      │
│ DExperts, Entropy Servo, Dynamic Concept Injection          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: STRUCTURAL CONTROL                                 │
│ Grammar, TokenTrie, Phrase Masking, Source Guard            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: EVIDENCE / CONCEPT PLANE                           │
│ Facts, Concept Capsules, Forbidden Frames, Taxonomy         │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Design Principles

1. **Layered Intervention**: Operations compose from safe (symbolic) to experimental (latent)
2. **Closed-Loop Control**: Every layer can receive feedback from evaluation
3. **SGLang-Native**: Leverage existing mechanisms (forward hooks, custom logit processors, grammar constraints)
4. **No Silent Degradation**: Fail fast if critical components are missing
5. **Per-Request Isolation**: All state is request-scoped, no cross-contamination

---

## 2. Critical Defects (SPRINT 0)

### 2.1 Defect Inventory

| ID | Severity | Component | Issue | Impact |
|----|----------|-----------|-------|--------|
| D1 | CRITICAL | `api/routes.py` | `persona-*` aliases not in `NRAM_ENABLED_ALIASES` | NRAM never activates for persona models |
| D2 | CRITICAL | `compose.yaml` | `nram-api` container lacks tokenizer mount | Token bias compiler returns None |
| D3 | CRITICAL | `core/steering/serialization.py` | `build_custom_params()` never includes `__req__` | Phenomenon mixer is dead code |
| D4 | HIGH | `api/routes.py` | `intensity` slider not mapped to NRAMState | Decorative UI control |
| D5 | HIGH | `core/engines/sglang_engine.py` | State computed twice (developer prompt vs logit policy) | Inconsistent steering |

### 2.2 Fix Specifications

#### D1: Persona Routing Fix

**Location:** `api/routes.py:125-150` (`_route_persona()`)

**Current Behavior:**
```python
async def _route_persona(request: ChatCompletionRequest, model: str) -> Dict[str, Any]:
    profile = _PERSONA_MODEL_MAP[model]
    nram_opts = {"enabled": True, "profile": profile, "intensity": intensity}
    # BUG: request.model remains "persona-peak", not in NRAM_ENABLED_ALIASES
    response = await engine.complete(engine_request)
```

**Fix:**
```python
async def _route_persona(request: ChatCompletionRequest, model: str) -> Dict[str, Any]:
    profile = _PERSONA_MODEL_MAP[model]
    
    # CRITICAL: Internally route to actual NRAM model
    request.model = "nram-qwen3-14b-awq"
    request.nram = {
        "enabled": True,
        "profile": profile,
        "intensity": intensity,
        **(request.nram or {}),
    }
    
    engine_request = _to_engine_request(request, request.messages)
    response = await engine.complete(engine_request)
    
    # Return public model name in response
    result = response.model_dump()
    result["model"] = model  # "persona-peak"
    return result
```

**Test:** `tests/contract/test_persona_routing.py::test_persona_activates_nram`

#### D2: Tokenizer Mount Fix

**Location:** `compose.yaml`

**Current:**
```yaml
nram-api:
  environment:
    NRAM_ENGINE: sglang
    SGLANG_BASE_URL: http://sglang:30000/v1
    # MISSING: NRAM_TOKENIZER_PATH
```

**Fix:**
```yaml
nram-api:
  environment:
    NRAM_ENGINE: sglang
    SGLANG_BASE_URL: http://sglang:30000/v1
    SGLANG_MODEL: nram-qwen3-14b-awq
    NRAM_TOKENIZER_PATH: /models
    
  volumes:
    - ${MODEL_PATH:-./models}:/models:ro
    - workflow-data:/app/data
```

**Startup Guard:** Add to `core/engines/registry.py`:
```python
def validate_tokenizer_availability():
    if sglang_enabled() and get_tokenizer() is None:
        raise RuntimeError(
            "NRAM_ENGINE=sglang requires NRAM_TOKENIZER_PATH. "
            "Refusing to run in silent prompt-only mode."
        )
```

#### D3: `__req__` Injection Fix

**Location:** `core/steering/serialization.py:27-65`

**Current:**
```python
def build_custom_params(...) -> Dict[str, Any]:
    result = {
        "positive_token_ids": positive_token_ids,
        # ... other fields
        # MISSING: __req__
    }
    return result
```

**Fix:**
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
    request: Optional[Any] = None,  # NEW: accept request object
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
        "__req__": request,  # NEW: inject request for output_ids access
    }
    if phenomenon_weights:
        result["phenomenon_weights"] = phenomenon_weights
    return result
```

**Caller Update:** In `core/engines/sglang_engine.py:_build_upstream_payload()`:
```python
custom_params = build_custom_params(
    positive_token_ids=bias_compiler.positive_ids,
    negative_token_ids=bias_compiler.negative_ids,
    forbidden_token_ids=bias_compiler.forbidden_ids,
    positive_bias=state.visionary_intensity,
    negative_bias=state.corporate_jargon_penalty,
    repetition_penalty=state.repetition_penalty,
    profile=profile_name,
    max_tokens=request.max_tokens or 512,
    phenomenon_weights=request.nram.get("phenomenon_weights"),
    request=request,  # NEW: pass request object
)
```

#### D4: Intensity Slider Fix

**Location:** `api/routes.py:133-143`

**Current:** `intensity` is set but never used to compute final NRAMState.

**Fix:** Already addressed by `resolve_request_state()` in `sglang_engine.py:147-185`. Ensure `_route_persona()` calls it:

```python
async def _route_persona(request: ChatCompletionRequest, model: str) -> Dict[str, Any]:
    profile = _PERSONA_MODEL_MAP[model]
    intensity = intensity_map.get(profile, 0.5)
    
    request.model = "nram-qwen3-14b-awq"
    request.nram = {
        "enabled": True,
        "profile": profile,
        "intensity": intensity,
        **(request.nram or {}),
    }
    
    # State is resolved inside engine.complete() via resolve_request_state()
    engine_request = _to_engine_request(request, request.messages)
    response = await engine.complete(engine_request)
    ...
```

#### D5: State Unification

**Location:** `core/engines/sglang_engine.py:complete()`

**Current:** Developer instruction uses one state, logit policy uses another.

**Fix:** Compute state ONCE at the beginning of `complete()`:

```python
async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
    # SINGLE source of truth
    profile_name = request.nram.get("profile", DEFAULT_PROFILE) if request.nram else DEFAULT_PROFILE
    state = resolve_request_state(request.nram, profile_name)
    
    # Use same state for everything
    policy = compile_policy(state, max_tokens=request.max_tokens or 512, profile_name=profile_name)
    developer_instruction = policy.developer_instruction
    
    payload = self._build_upstream_payload(request, state, policy)
    ...
```

---

## 3. Layer 1: Evidence / Concept Plane

### 3.1 Concept Capsules

**Purpose:** Pre-computed semantic units injected before generation.

**Data Structure:**
```python
@dataclass
class ConceptCapsule:
    """A semantic unit with multiple lexical forms and activation rules."""
    
    concept_id: str  # "ambient_computing"
    
    # Lexical forms
    en_tokens: list[str]  # ["ambient", "computing"]
    cs_tokens: list[str]  # ["ambientní", "výpočetní"]
    synonyms: list[str]   # ["ubiquitous", "everywhere"]
    
    # Embedding centroid (computed from lexical forms)
    embedding: Optional[np.ndarray] = None  # shape: (hidden_dim,)
    
    # Activation rules
    activation_phase: Optional[str] = None  # "revelation", "synthesis", None=always
    max_uses: int = 3  # Maximum times this concept can appear
    min_distance: float = 0.3  # Minimum semantic distance from source
    
    # Metadata
    forbidden_frames: list[str] = []  # ["better chatbot", "list of features"]
    revelation_tokens: list[str] = []  # ["disappears", "continuity"]
```

**Capsule Registry:**
```python
class ConceptCapsuleRegistry:
    """Manages concept capsules for a generation request."""
    
    def __init__(self):
        self.capsules: dict[str, ConceptCapsule] = {}
        self.usage_counts: dict[str, int] = {}
    
    def load_from_config(self, config: dict):
        """Load capsules from request config."""
        for capsule_data in config.get("required_concepts", []):
            capsule = ConceptCapsule(**capsule_data)
            self.capsules[capsule.concept_id] = capsule
            self.usage_counts[capsule.concept_id] = 0
    
    def get_active_capsules(self, phase: str, generated_tokens: int) -> list[ConceptCapsule]:
        """Return capsules eligible for injection at current phase."""
        active = []
        for capsule in self.capsules.values():
            if self.usage_counts[capsule.concept_id] >= capsule.max_uses:
                continue
            if capsule.activation_phase and capsule.activation_phase != phase:
                continue
            active.append(capsule)
        return active
    
    def record_usage(self, concept_id: str):
        self.usage_counts[concept_id] = self.usage_counts.get(concept_id, 0) + 1
```

**Integration Point:** `core/steering/concept_capsules.py`

### 3.2 Forbidden Frames

**Purpose:** Prevent specific semantic frames from appearing.

**Implementation:**
```python
FORBIDDEN_FRAMES_EN = [
    "better chatbot",
    "centralized assistant",
    "list of features",
    "revolutionary AI",
    "game-changing",
]

FORBIDDEN_FRAMES_CS = [
    "lepší chatbot",
    "centralizovaný asistent",
    "seznam funkcí",
    "revoluční AI",
]
```

**Enforcement:** Compiled into `forbidden_token_ids` via `TokenBiasCompiler`.

---

## 4. Layer 2: Structural Control

### 4.1 TokenTrieConstraint (Phrase-Aware Masking)

**Purpose:** Prevent or enforce multi-token phrases during decoding.

**Data Structure:**
```python
class TokenTrie:
    """Trie for tracking multi-token sequences."""
    
    def __init__(self):
        self.root: dict[int, dict] = {}  # token_id -> children
    
    def insert(self, token_ids: list[int]):
        """Insert a sequence into the trie."""
        node = self.root
        for token_id in token_ids:
            if token_id not in node:
                node[token_id] = {}
            node = node[token_id]
        node["$"] = True  # End marker
    
    def matches_prefix(self, recent_tokens: list[int]) -> list[int]:
        """Return token IDs that would complete a forbidden phrase."""
        candidates = []
        node = self.root
        for token_id in recent_tokens:
            if token_id not in node:
                return []  # No match
            node = node[token_id]
        
        # Collect all possible next tokens
        for next_token, children in node.items():
            if next_token != "$":
                candidates.append(next_token)
        
        return candidates
```

**Phrase Constraint:**
```python
class TokenTrieConstraint:
    """Applies phrase-level masking during decoding."""
    
    def __init__(self, forbidden_phrases: list[str], tokenizer):
        self.forbidden_trie = TokenTrie()
        self.allowed_trie = TokenTrie()  # Optional: enforce specific phrases
        
        for phrase in forbidden_phrases:
            token_ids = tokenizer.encode(phrase)
            self.forbidden_trie.insert(token_ids)
    
    def apply(self, logits, recent_tokens: list[int], batch_index: int):
        """Mask tokens that would complete a forbidden phrase."""
        forbidden_next = self.forbidden_trie.matches_prefix(recent_tokens)
        for token_id in forbidden_next:
            logits[batch_index, token_id] = -float("inf")
```

**Integration:** Applied in `NRAMLogitProcessor.__call__()` before other logit operations.

### 4.2 Anti-Copy Gate (Source N-Gram Masker)

**Purpose:** Prevent copying of n-grams from source document.

**Implementation:**
```python
class SourceNgramBlocker:
    """Blocks tokens that would complete source n-grams."""
    
    def __init__(self, source_text: str, tokenizer, n: int = 8):
        self.n = n
        self.source_trie = TokenTrie()
        
        # Extract all n-grams from source
        source_tokens = tokenizer.encode(source_text)
        for i in range(len(source_tokens) - n + 1):
            ngram = source_tokens[i:i+n]
            self.source_trie.insert(ngram)
    
    def apply(self, logits, output_ids: list[int], batch_index: int):
        """Mask tokens that would complete a source n-gram."""
        recent = output_ids[-(self.n-1):]
        forbidden_next = self.source_trie.matches_prefix(recent)
        for token_id in forbidden_next:
            logits[batch_index, token_id] = -float("inf")
```

**Key Insight:** This prevents copying **during decoding**, not after. The model never sees the copied token as a valid option.

### 4.3 Grammar Constraints

**Purpose:** Enforce structural formats (JSON, DSL, section markers).

**Integration:** Use SGLang's xgrammar integration alongside custom constraints.

```python
# In SGLang payload
payload["guided_decoding"] = {
    "backend": "xgrammar",
    "json_schema": output_schema,  # Optional
}
```

**Note:** Grammar constraints operate independently from phrase masking. Both can be active simultaneously.

---

## 5. Layer 3: Logit Control

### 5.1 Entropy Servo (PID Controller)

**Purpose:** Dynamically adjust temperature to maintain target entropy per phase.

**Data Structure:**
```python
class EntropyController:
    """PID controller for entropy targeting."""
    
    def __init__(
        self,
        target_entropy: float,
        kp: float = 0.5,
        ki: float = 0.1,
        kd: float = 0.05,
        scale_min: float = 0.6,
        scale_max: float = 1.8,
    ):
        self.target = target_entropy
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.scale_min = scale_min
        self.scale_max = scale_max
        
        self.integral = 0.0
        self.prev_error = 0.0
    
    def adjust(self, logits, batch_index: int):
        """Adjust logits to move toward target entropy."""
        probs = torch.softmax(logits[batch_index], dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        
        error = self.target - entropy
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        
        scale = self.kp * error + self.ki * self.integral + self.kd * derivative
        scale = max(self.scale_min, min(self.scale_max, scale))
        
        # Apply scale as temperature adjustment
        logits[batch_index] = logits[batch_index] / scale
        
        return entropy  # For telemetry
```

**Phase Profile:**
```python
ENTROPY_PHASES = {
    "extraction": 2.5,      # Low entropy — factual
    "questioning": 4.0,     # Medium entropy — exploratory
    "divergence": 6.0,      # High entropy — creative
    "synthesis": 4.5,       # Medium entropy — converging
    "formulation": 3.0,     # Low entropy — precise
}
```

**Integration:** Called in `NRAMLogitProcessor.__call__()` after base steering.

### 5.2 DExperts (Decoding-Time Experts)

**Purpose:** Combine base model with expert and anti-expert distributions.

**Formula:**
```
z_final = z_base + α * (z_expert - z_base) - β * (z_anti - z_base)
```

Simplified:
```
z_final = z_base + α * z_expert - β * z_anti
```

**Implementation:**
```python
class DExpertsController:
    """Decoding-time expert combination."""
    
    def __init__(
        self,
        base_model: str,
        expert_model: str,
        anti_expert_model: str,
        alpha: float = 0.5,
        beta: float = 0.3,
    ):
        self.base_model = base_model
        self.expert_model = expert_model
        self.anti_expert_model = anti_expert_model
        self.alpha = alpha
        self.beta = beta
    
    def combine_logits(
        self,
        base_logits: torch.Tensor,
        expert_logits: torch.Tensor,
        anti_logits: torch.Tensor,
        batch_index: int,
    ):
        """Combine logits from three models."""
        combined = (
            base_logits[batch_index]
            + self.alpha * expert_logits[batch_index]
            - self.beta * anti_logits[batch_index]
        )
        return combined
```

**Practical Consideration:** Running three 14B models in parallel is expensive. Start with:
- Base: Qwen3-14B
- Expert: Qwen3-1.5B fine-tuned on product manifests
- Anti-expert: Qwen3-1.5B fine-tuned on corporate jargon

**Integration:** Requires SGLang to support multiple model instances or LoRA switching.

### 5.3 Dynamic Concept Injection

**Purpose:** Inject concept tokens based on phase and context.

**Three Injection Types:**

#### Hard Injection
```python
# Force specific tokens (structure, section markers)
if phase == "section_start":
    logits[batch_index, section_token_id] = 10.0  # Near-certain
```

**Use Cases:**
- JSON structure enforcement
- Section markers
- Phase transitions

**Warning:** Overuse destroys grammar. Use sparingly.

#### Soft Injection
```python
# Time-windowed bias
if phase == "revelation":
    strength = phase_strength_curve(generated, max_tokens)
    logits[batch_index, concept_token_ids] += strength
```

**Strength Curve:**
```python
def phase_strength_curve(generated: int, max_tokens: int) -> float:
    """Ramp up and down within phase."""
    progress = generated / max_tokens
    if progress < 0.2:
        return progress / 0.2 * 0.9  # 0.0 → 0.9
    elif progress < 0.8:
        return 0.9
    else:
        return (1.0 - progress) / 0.2 * 0.9  # 0.9 → 0.0
```

#### Latent Injection
```python
# Inject concept as hidden-state vector (Sprint 2)
# See Section 6: Representation Control
```

---

## 6. Layer 4: Representation Control

### 6.1 Activation Addition (ActAdd)

**Purpose:** Steer model behavior by adding contrastive vectors to hidden states.

**Theory:**
```
v = E[h_B - h_A]  # Difference vector from contrastive pairs
h'_l = h_l + α * v_l  # Add vector at layer l
```

**Implementation:**
```python
class ActivationAddition:
    """Apply contrastive activation vectors."""
    
    def __init__(self, vector_path: str, layer: int, alpha: float = 0.7):
        self.vector = torch.load(vector_path)  # shape: (hidden_dim,)
        self.layer = layer
        self.alpha = alpha
    
    def apply(self, hidden_states: torch.Tensor, layer_idx: int):
        """Add vector to hidden states at target layer."""
        if layer_idx == self.layer:
            hidden_states += self.alpha * self.vector
        return hidden_states
```

**Vector Collection:**
```python
# Contrastive pairs
PAIRS = [
    ("novelty", "novel_product_manifest.txt", "generic_corporate_text.txt"),
    ("concrete", "specific_mechanism.txt", "abstract_buzzwords.txt"),
    ("human", "human_need_story.txt", "feature_list.txt"),
]

def collect_activation_vector(model, tokenizer, pair_type: str, positive: str, negative: str, layer: int):
    """Collect activation difference vector from contrastive pair."""
    pos_tokens = tokenizer.encode(positive)
    neg_tokens = tokenizer.encode(negative)
    
    with torch.no_grad():
        pos_outputs = model(torch.tensor([pos_tokens]), output_hidden_states=True)
        neg_outputs = model(torch.tensor([neg_tokens]), output_hidden_states=True)
    
    pos_hidden = pos_outputs.hidden_states[layer].mean(dim=1)
    neg_hidden = neg_outputs.hidden_states[layer].mean(dim=1)
    
    vector = pos_hidden - neg_hidden
    torch.save(vector, f"vectors/{pair_type}_vs_{negative}.pt")
```

**SGLang Integration:** Use `--forward-hooks` mechanism:
```json
[
  {
    "name": "nram_actadd_novelty",
    "target_modules": ["model.layers.20"],
    "hook_factory": "nram_hooks.activation:make_steering_hook",
    "config": {
      "strength": 0.7,
      "vector_path": "/vectors/novelty_vs_generic.pt"
    }
  }
]
```

### 6.2 Conceptor Steering

**Purpose:** Multi-dimensional concept projection using conceptor matrices.

**Theory:**
```
h' = C * h  # Conceptor projection
h' = h + α * C_visionary * h - β * C_corporate * h
```

**Implementation:**
```python
class ConceptorSteering:
    """Multi-dimensional concept steering."""
    
    def __init__(self, conceptor_matrix: torch.Tensor, alpha: float = 0.5):
        self.C = conceptor_matrix  # shape: (hidden_dim, hidden_dim)
        self.alpha = alpha
    
    def apply(self, hidden_states: torch.Tensor):
        """Apply conceptor projection."""
        projected = torch.matmul(hidden_states, self.C.T)
        hidden_states += self.alpha * (projected - hidden_states)
        return hidden_states
```

**Advantage:** Can represent multi-dimensional concepts (novelty + coherence + concreteness) as a subspace rather than a single vector.

### 6.3 ReFT (Representation Fine-Tuning)

**Purpose:** Learn small intervention matrices for specific layers.

**Theory:**
```
h'_l = h_l + R_l * h_l  # R_l is low-rank intervention matrix
```

**Implementation:**
```python
class ReFTIntervention:
    """Low-rank representation intervention."""
    
    def __init__(self, intervention_matrix: torch.Tensor, layer: int):
        self.R = intervention_matrix  # shape: (hidden_dim, rank)
        self.layer = layer
    
    def apply(self, hidden_states: torch.Tensor, layer_idx: int):
        """Apply intervention at target layer."""
        if layer_idx == self.layer:
            intervention = torch.matmul(hidden_states, self.R.T)
            hidden_states += intervention
        return hidden_states
```

**Catalog:**
```
reforge/visionary-product.reft
reforge/ruthless-cto.reft
reforge/evidence-first.reft
reforge/anti-paraphrase.reft
```

**Note:** Requires custom forward hook or SGLang patch for per-request activation.

### 6.4 Soft Prompt Capsules

**Purpose:** Learn virtual embedding tokens prepended to input.

**Implementation:**
```python
class SoftPromptCapsule:
    """Learned soft prompt prefix."""
    
    def __init__(self, capsule_embeddings: torch.Tensor):
        self.embeddings = capsule_embeddings  # shape: (num_tokens, hidden_dim)
    
    def prepend(self, input_embeds: torch.Tensor):
        """Prepend capsule to input embeddings."""
        batch_size = input_embeds.shape[0]
        capsule_batch = self.embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([capsule_batch, input_embeds], dim=1)
```

**Training:** Optimize capsule embeddings to maximize difference from baseline while maintaining coherence.

---

## 7. Layer 5: Generation Search

### 7.1 Branch-and-Tournament Decoding

**Purpose:** Explore multiple conceptual trajectories and select best.

**Algorithm:**
```python
def branch_and_tournament_decode(
    model,
    prompt: str,
    num_branches: int = 4,
    branch_length: int = 50,
    evaluator: Callable,
):
    """Generate multiple branches and select best."""
    
    branches = []
    for i in range(num_branches):
        # Generate branch with different sampling
        branch = model.generate(
            prompt,
            max_tokens=branch_length,
            temperature=0.8 + i * 0.1,  # Increasing creativity
        )
        branches.append(branch)
    
    # Evaluate each branch
    scores = []
    for branch in branches:
        score = evaluator(branch)  # novelty, coherence, source_distance
        scores.append(score)
    
    # Select winner
    winner_idx = torch.argmax(torch.tensor(scores))
    return branches[winner_idx]
```

**Evaluator Metrics:**
- Novelty: semantic distance from source
- Coherence: perplexity under language model
- Source Distance: n-gram overlap with source
- Factual Risk: hallucination probability

**SGLang Integration:** Leverage speculative decoding infrastructure (EAGLE, EAGLE3) but with different objective: conceptual diversity, not speed.

### 7.2 Divergent Speculative Decoding

**Purpose:** Draft model generates multiple conceptual trajectories, target model verifies.

**Difference from Standard Speculative Decoding:**
- Standard: draft model predicts next N likely tokens
- Divergent: draft model predicts N different conceptual directions

**Implementation:**
```python
def divergent_speculative_decode(
    draft_model,
    target_model,
    prompt: str,
    num_directions: int = 4,
):
    """Draft model generates diverse continuations, target verifies."""
    
    # Draft model generates diverse beams
    draft_outputs = draft_model.generate(
        prompt,
        num_beams=num_directions,
        num_return_sequences=num_directions,
        diversity_penalty=2.0,  # Encourage diversity
    )
    
    # Target model verifies each direction
    verified = []
    for draft in draft_outputs:
        # Target model continues from draft
        continuation = target_model.generate(draft, max_tokens=20)
        verified.append(continuation)
    
    # Select best verified continuation
    return select_best(verified)
```

---

## 8. Layer 6: Closed-Loop Evaluation

### 8.1 Real-Time Metrics

**Metrics Computed Every N Tokens:**
```python
class RealTimeEvaluator:
    """Online evaluation of generation quality."""
    
    def __init__(self, tokenizer, embedding_model):
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.source_embeddings = None  # Pre-computed
    
    def evaluate(self, generated_text: str, source_text: str, phase: str) -> dict:
        """Compute real-time metrics."""
        
        # 1. Source similarity (cosine similarity of embeddings)
        gen_emb = self.embedding_model.encode(generated_text)
        src_emb = self.embedding_model.encode(source_text)
        source_similarity = cosine_similarity(gen_emb, src_emb)
        
        # 2. Novelty (1 - source_similarity)
        novelty = 1.0 - source_similarity
        
        # 3. Coherence (perplexity under language model)
        coherence = compute_perplexity(generated_text)
        
        # 4. Factual preservation (overlap with source facts)
        factual_overlap = compute_fact_overlap(generated_text, source_text)
        
        # 5. N-gram overlap (copying detection)
        ngram_overlap = compute_ngram_overlap(generated_text, source_text, n=8)
        
        return {
            "novelty": novelty,
            "coherence": coherence,
            "source_similarity": source_similarity,
            "factual_preservation": factual_overlap,
            "ngram_overlap": ngram_overlap,
        }
```

### 8.2 Feedback Loop

**Purpose:** Adjust steering parameters based on evaluation.

**Implementation:**
```python
class ClosedLoopController:
    """Adjusts steering based on real-time evaluation."""
    
    def __init__(self, target_novelty: float = 0.7, target_coherence: float = 0.8):
        self.target_novelty = target_novelty
        self.target_coherence = target_coherence
    
    def adjust(self, metrics: dict, current_params: dict) -> dict:
        """Adjust parameters based on metrics."""
        
        adjusted = current_params.copy()
        
        # If novelty too low, increase concept injection
        if metrics["novelty"] < self.target_novelty:
            adjusted["concept_injection_strength"] *= 1.2
            adjusted["entropy_target"] *= 1.1
        
        # If coherence too low, reduce entropy
        if metrics["coherence"] < self.target_coherence:
            adjusted["entropy_target"] *= 0.9
            adjusted["concept_injection_strength"] *= 0.8
        
        # If source similarity too high, activate anti-copy
        if metrics["source_similarity"] > 0.6:
            adjusted["anti_copy_strength"] = 1.0
        
        return adjusted
```

**Integration:** Called every 24 tokens in `NRAMLogitProcessor.__call__()`.

---

## 9. Implementation Plan

### 9.1 Sprint 0: Critical Fixes

**Goal:** Make existing NRAM actually work.

**Tasks:**
1. Fix persona routing (D1)
2. Fix tokenizer mount (D2)
3. Fix `__req__` injection (D3)
4. Fix intensity mapping (D4)
5. Unify state computation (D5)
6. Add integration tests for each fix

**Deliverables:**
- All persona models activate NRAM
- Token bias compiler has tokenizer
- Phenomenon mixer is functional
- Intensity slider affects output
- Single state used everywhere

### 9.2 Sprint 1: Basic Inference Control

**Goal:** Implement Layers 1-3 with core features.

**Tasks:**
1. Implement `ConceptCapsule` and registry
2. Implement `TokenTrieConstraint` for phrase masking
3. Implement `SourceNgramBlocker` for anti-copy
4. Implement `EntropyController` with PID
5. Implement dynamic concept injection (hard + soft)
6. Create A/B test harness with metrics
7. Add Czech lexemes to token bias

**Deliverables:**
- Phrase-aware masking prevents clichés
- Source copying blocked at decode time
- Entropy controlled per phase
- Concepts injected dynamically
- A/B test shows measurable difference

### 9.3 Sprint 2: Activation Steering

**Goal:** Implement Layer 4 with hidden-state interventions.

**Tasks:**
1. Set up SGLang forward hooks
2. Collect hidden states from contrastive prompts
3. Identify optimal layers for ActAdd
4. Implement first ActAdd vector (novelty_vs_paraphrase)
5. Verify coherence is maintained
6. Add multi-vector controller

**Deliverables:**
- Forward hooks operational
- At least one validated ActAdd vector
- Measurable novelty increase without coherence loss

### 9.4 Sprint 3: Advanced Techniques

**Goal:** Implement Layers 5-6 with search and closed-loop.

**Tasks:**
1. Implement conceptor steering
2. Implement hidden-state probes
3. Implement closed-loop controller
4. Implement DExperts with small expert models
5. Implement branch-and-tournament decoding

**Deliverables:**
- Multi-dimensional concept steering
- Real-time evaluation and adjustment
- Expert/anti-expert combination
- Divergent search operational

### 9.5 Research Branch (Optional)

**Goal:** Experimental features for future exploration.

**Tasks:**
1. Per-request ReFT interventions
2. KV-cache firewall (semantic airlock)
3. GPU-native NRAM controller
4. Attention-head gating

---

## 10. Testing Strategy

### 10.1 Unit Tests

**Coverage:**
- Each layer component has unit tests
- Mock SGLang for isolated testing
- Test data structures (Trie, Capsule, Controller)

**Example:**
```python
def test_token_trie_blocks_forbidden_phrase():
    trie = TokenTrie()
    trie.insert([1, 2, 3])
    
    assert trie.matches_prefix([1, 2]) == [3]
    assert trie.matches_prefix([1, 4]) == []
```

### 10.2 Integration Tests

**Coverage:**
- Full request flow from API to SGLang
- Persona routing activates NRAM
- Logit processor receives `__req__`
- Phenomenon mixer applies effects

**Example:**
```python
async def test_persona_peak_activates_nram():
    request = ChatCompletionRequest(
        model="persona-peak",
        messages=[{"role": "user", "content": "test"}],
    )
    response = await engine.complete(request)
    
    assert response.nram_telemetry is not None
    assert response.nram_telemetry.profile == "peak"
```

### 10.3 A/B Tests

**Methodology:**
- Same seed, same prompt, different NRAM configs
- Measure: novelty, coherence, source_similarity, ngram_overlap
- Human preference rating (optional)

**Metrics:**
```python
METRICS = {
    "novelty": "1 - cosine_similarity(gen_emb, src_emb)",
    "coherence": "perplexity under language model",
    "source_similarity": "cosine_similarity(gen_emb, src_emb)",
    "ngram_overlap": "fraction of 8-grams from source",
    "factual_preservation": "overlap with source facts",
}
```

### 10.4 Adversarial Tests

**Purpose:** Verify system doesn't break under edge cases.

**Tests:**
- Empty prompt
- Very long prompt
- Non-UTF8 characters
- Concurrent requests with different profiles
- Missing tokenizer (should fail fast)

---

## 11. Performance Considerations

### 11.1 Computational Cost

| Component | Cost | Mitigation |
|-----------|------|------------|
| TokenTrie | O(n) per token, n = phrase length | Use hash map for O(1) lookup |
| EntropyController | O(vocab_size) per step | Compute on GPU |
| ActAdd | O(hidden_dim) per layer | Vector addition is fast |
| DExperts | 3x model inference | Use smaller expert models (1.5B) |
| Branch-and-tournament | 4x generation | Limit branch length |
| Closed-loop evaluation | Embedding model call every 24 tokens | Batch evaluations |

### 11.2 Memory Footprint

| Component | Memory | Notes |
|-----------|--------|-------|
| Concept capsules | ~1MB per capsule | Embeddings + metadata |
| Token tries | ~100KB per trie | Sparse structure |
| ActAdd vectors | ~10KB per vector | Single hidden_dim vector |
| Conceptor matrices | ~10MB per conceptor | hidden_dim x hidden_dim |
| ReFT interventions | ~1MB per intervention | Low-rank matrices |

### 11.3 Latency Budget

**Target:** < 100ms overhead per token

**Breakdown:**
- TokenTrie: < 1ms
- EntropyController: < 5ms
- ActAdd: < 2ms
- Closed-loop evaluation: < 50ms (every 24 tokens = ~2ms/token)

---

## 12. Deployment Checklist

### 12.1 Pre-Deployment

- [ ] All Sprint 0 fixes verified
- [ ] Tokenizer mounted in nram-api container
- [ ] Integration tests passing
- [ ] A/B harness operational
- [ ] Telemetry logging enabled

### 12.2 Configuration

```yaml
# compose.yaml
nram-api:
  environment:
    NRAM_ENGINE: sglang
    SGLANG_BASE_URL: http://sglang:30000/v1
    NRAM_TOKENIZER_PATH: /models
    NRAM_ENABLE_FORWARD_HOOKS: true
    NRAM_ENABLE_HIDDEN_STATES: true

sglang:
  command: >
    python -m sglang.launch_server
    --model-path /models/Qwen3-14B-AWQ
    --enable-forward-hooks
    --enable-return-hidden-states
```

### 12.3 Monitoring

**Metrics to Track:**
- NRAM activation rate (should be 100% for persona models)
- Phenomenon mixer activation (should be > 0 for non-normal profiles)
- Entropy distribution per phase
- Novelty/coherence scores over time
- Source n-gram overlap (should be < 10%)

---

## 13. Future Extensions

### 13.1 Multi-Model Support

- Support different base models (Llama, Mistral, DeepSeek)
- Model-specific ActAdd vectors
- Automatic layer selection per model

### 13.2 User-Defined Concepts

- UI for creating custom concept capsules
- Upload contrastive pairs for ActAdd
- Train custom ReFT interventions

### 13.3 Distributed Inference

- Shard expert models across GPUs
- Distributed branch-and-tournament
- GPU-resident semantic controller

---

## 14. References

1. **DExperts:** [Decoding-Time Controlled Text Generation with Experts and Anti-Experts](https://aclanthology.org/2021.acl-long.522/)
2. **Activation Addition:** [Steering Language Models Without Optimization](https://research-information.bris.ac.uk/en/publications/activation-addition-steering-language-models-without-optimization/)
3. **Conceptor Steering:** [Steering LLMs using Conceptors](https://arxiv.org/abs/2410.16314)
4. **ReFT:** [Representation Finetuning for Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/75008a0fba53bf13b0bb3b7bff986e0e-Abstract-Conference.html)
5. **SGLang Forward Hooks:** [Server Arguments Documentation](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md)

---

## 15. Approval

**Architecture Review:**
- [ ] Technical feasibility confirmed
- [ ] Performance budget acceptable
- [ ] Testing strategy adequate
- [ ] Deployment plan clear

**Sign-off:**
- Architect: _________________
- Date: _________________

---

**END OF SPECIFICATION**
