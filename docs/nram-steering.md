# NRAM Steering — What NRAM Actually Controls

This document is deliberately precise about **categories**. Steering techniques are often blurred together in marketing ("we steer the model!"); here we separate them and state exactly which ones NeuralAccessPsyche uses and which it does not.

## The seven categories — kept distinct

| # | Category | Mechanism | When it acts | Used by NRAM? |
|---|----------|-----------|--------------|:---:|
| 1 | **Prompt steering** | Instruction text in the system/developer prompt | Before generation starts (input) | ✅ |
| 2 | **Structured-output constraints** | Grammar / JSON-schema-constrained decoding | During decoding (constrains allowed tokens) | ✅ (planner only) |
| 3 | **Hard token masking** | Set specific token logits to `-inf` so they can never be sampled | Before sampling (per step) | ✅ |
| 4 | **Soft logit biasing** | Add/subtract a bounded value to specific token logits | Before sampling (per step) | ✅ |
| 5 | **Dynamic repetition penalties** | Penalize tokens already in the recent output | Before sampling (per step, state-dependent) | ✅ |
| 6 | **Activation steering** | Add a vector to hidden-state activations inside the model | Inside the model forward pass | ❌ |
| 7 | **Post-generation visualization** | Inspect/visualize state *after* tokens are produced | After generation | ✅ (dashboards only) |

The critical line is between **1–5 (which act on inputs or on logits before sampling)** and **6 (which edits internal activations)**. NeuralAccessPsyche does categories 1–5 and 7. It does **not** do activation steering.

## NRAM = real pre-sampling control (categories 3, 4, 5)

The defining property of NRAM is that it **modifies logits _before_ token sampling**. This is not a prompt effect and not post-hoc filtering — the sampler literally sees different logits than the base model produced.

Inside SGLang, `NRAMLogitProcessor.__call__` runs every decoding step, per batch row:

```python
# Soft positive bias: make persona-aligned tokens more likely
logits[batch_index, positive_ids] += positive_bias      # bounded ≤ 1.2

# Soft negative bias: make corporate-filler tokens less likely
logits[batch_index, negative_ids]  -= negative_bias      # bounded ≤ 2.5

# Hard mask: forbidden jargon tokens can never be sampled
logits[batch_index, forbidden_ids] = -float("inf")

# Dynamic repetition penalty against the recent output window
logits[batch_index, recent_ids]    -= repetition_penalty # bounded ≤ 2.0
```

Because this executes inside the SGLang server's decoding loop, the changes are applied **before** `softmax` and **before** a token is drawn. The processor:

- runs with **no network, filesystem, or app-service access**;
- **validates and bounds** every parameter (out-of-range token IDs are silently dropped; floats are clamped, NaN/Inf rejected);
- applies **per-batch-row isolation** so one request's bias never leaks into another's.

### Proof: the forced-token test

Pre-sampling control is proven by a **forced-token test**: bias one target token's logit high enough (or mask every competitor to `-inf`) and confirm the model emits that token deterministically, even when the base model would not. If steering were only a prompt effect, such a guarantee would be impossible — prompts bias probabilities softly and never force an exact token. A successful forced-token test demonstrates that NRAM reaches the logits the sampler actually consumes.

> `NRAMController.get_capabilities()` reports `pre_sampling_logit_modification` as `False` until the forced-token test has been run and passes. It is flipped to `True` only on that evidence — not asserted ahead of time.

## Prompt steering and the rhetorical plan (categories 1, 2)

Prompt steering and structured output are **separate, complementary** mechanisms — they are *not* the logit processor.

- **Prompt steering (1):** the persona compiler emits a `developer_instruction` (an original-keynote system prompt that explicitly forbids impersonation, quotes, and jargon). This is ordinary instruction text.
- **Structured-output constraints (2):** the **rhetorical planner** runs a hidden, schema-constrained `llguidance` call that returns a `VisionaryPlan` as strict JSON. The plan is converted to a prompt fragment for the final generation and is **never shown to the user**. This constrained-decoding call does **not** use the NRAM logit processor and runs at a low temperature (0.3–0.6).

These two shape the model through its *input*; categories 3–5 shape it through its *logits*. Both are in play for an NRAM-enabled request.

## Activation steering (category 6) — NOT used

**Activation steering** means adding a direction vector to the model's hidden-state activations during the forward pass. NeuralAccessPsyche does **not** do this. SGLang's verified capability set reports `hidden_state_access = False`, so there is no supported path to edit activations here. Any claim that NRAM "steers activations" would be false for this system — NRAM is a **logit-layer** technique.

## Post-generation visualization (category 7) — observability, not control

The WebSocket dashboards (`/v1/visualize`, `/v1/nram/explorer`, `/ws/nram-state`) **observe** NRAM state after the fact. Visualization does not influence generation — it is read-only telemetry, kept strictly separate from the steering path.

## The persona is a set of properties, NOT a cosplay

The `visionary-psychedelic-keynote` persona is defined by **cognitive and rhetorical properties** — `visionary_intensity`, `contrarian_force`, `associative_distance`, `coherence_floor`, etc. It is **not** an imitation of any real or public figure.

The developer instruction states this explicitly:

> *"Do not impersonate or identify as a real person. Do not quote or imitate famous keynote phrases."*

The persona describes a **style of thinking and arguing**, reproduced through token biases and prompt structure. It is not a character mask of a named individual.

## "Psychedelic" = associative distance, NOT random nonsense

The word *psychedelic* here is a precise parameter, not a license for incoherence.

- **`associative_distance`** controls how far apart the linked concepts may sit — i.e. **cross-domain connections** (tying a hardware detail to a human ritual, a sensory metaphor to a product claim).
- Higher associative distance means **more unexpected-but-meaningful links**, constrained by **`coherence_floor`**, which sets the minimum coherence the output must retain.

So "psychedelic" means **novel, cross-domain associations that still serve the argument** — explicitly *not* random word salad, mysticism, or empty hype (all of which the developer instruction forbids). The `coherence_floor` and the structured plan are designed to keep conceptual novelty from degrading factual coherence.

**Known Limitation**: Phase 6 ablation study (commit ba52bbe) shows the `coherence_floor` mechanism does not prevent coherence degradation. All NRAM profiles significantly reduce coherence (Cohen's d = -0.52 to -1.62, p < 0.001). See `docs/DOCUMENTATION_AUDIT_REPORT.md` for full analysis.

## Summary

NRAM's contribution is **pre-sampling logit control**: hard masking, soft biasing, and dynamic repetition penalties applied to logits inside SGLang **before** each sampling step, proven by a forced-token test. Prompt steering and a hidden structured plan complement it at the input layer. Activation steering is **not** part of the system, and visualization is read-only. The persona is a property vector, and "psychedelic" is bounded associative distance under a coherence floor — not randomness.
