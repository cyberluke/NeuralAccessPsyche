# NRAM v5 Runtime Wiring

## Scope and status vocabulary

This document describes the bounded remediation runtime at starting SHA
`7aa12854800327ea905e608b9ececc3e3f5fd758` plus the dirty working-tree changes
listed in the run manifest. It does not upgrade CPU helper classes into runtime
features. Advanced-component status uses exactly `IMPLEMENTED_AND_PROVEN`,
`IMPLEMENTED_NOT_PROVEN`, `PARTIAL`, `DESIGN_ONLY`, or `NOT_IMPLEMENTED`.

## Live path

`POST /v1/chat/completions` -> `api.routes.create_chat_completion` ->
`SGLangEngine._build_upstream_payload` -> JSON `custom_params` plus dill class
serialization -> SGLang 0.5.16 `Req.__init__` trusted `__req__` merge ->
`Sampler._preprocess_logits` -> `nram_sglang.processor.NRAMLogitProcessor` ->
SGLang sampling on the RTX 4090.

The application request object is never serialized. Installed source at
`/sgl-workspace/sglang/python/sglang/srt/managers/schedule_batch.py:813-819`
shallow-copies sampling parameters and injects the server-side scheduler
request. This is an official hook in the pinned image. It corrects the Phase-0
negative finding that the hook was absent; no SGLang source patch is used.

## Wiring matrix

| Component | Entry point | Runtime hook | Input signal | Intervention | Telemetry | Controlled proof | Status |
|---|---|---|---|---|---|---|---|
| Persona/request routing | Public chat API/persona aliases | API route + engine | alias, profile, intensity | prompt policy + shared processor | correlation IDs and processor events | contract tests; live forced path | `IMPLEMENTED_AND_PROVEN` |
| Tokenizer handling | API engine startup | Qwen tokenizer at `/models` | pinned tokenizer files | compile text to Qwen token IDs | container/version/hash records | API/server token 11064 equality | `IMPLEMENTED_AND_PROVEN` |
| Evidence ledger | workflow routes | no decode hook | stored workflow evidence | storage/display only | database events | no causal decode proof | `PARTIAL` |
| Concept capsules | `nram.concepts` | custom logit processor | bilingual forms/synonyms, phase, max uses | bounded vocabulary-logit delta | token rank/probability/logit/delta | live GPU local-effect test | `IMPLEMENTED_AND_PROVEN` for vocabulary logits only |
| Grammar | `response_format` | SGLang xgrammar | JSON schema/regex/EBNF | grammar vocabulary mask | SGLang + NRAM events | live JSON grammar plus phrase mask | `IMPLEMENTED_AND_PROVEN` |
| Phrase trie/suffix blocker | `nram.forbidden_phrases` | processor with server `Req.output_ids` | real output suffix | completion-token `-inf` mask | masked ID/pre/post logits | bilingual, overlap, negative, disabled, stream | `IMPLEMENTED_AND_PROVEN` |
| Source n-gram blocker | `nram.source_text` | same | exact output suffix and source 8-grams | continuation-token `-inf` mask | masked ID/pre/post logits | live exact 8-token continuation/control | `IMPLEMENTED_AND_PROVEN` |
| Hard token injection | `forced_token_*`, `hard_token_schedule` | same | step and configured Qwen ID | all-other-token mask | requested/applied ID, mask count | public same-seed force/control | `IMPLEMENTED_AND_PROVEN` |
| Soft token injection | `soft_token_injections` | same | decode window and Qwen IDs | bounded local additive delta | window, IDs, exact deltas/ranks/probabilities | live GPU window and inactive-window control | `IMPLEMENTED_AND_PROVEN` |
| Entropy PID | `entropy_*` | same | actual next-token logits + per-Req state | bounded logit scale | entropy before/after, target/error/P/I/D/clamp | live low/medium/high and reset | `IMPLEMENTED_AND_PROVEN` for one-step distribution movement |
| Semantic novelty controller | rejected request key | none | unavailable embedding/runtime interval signal | none | explicit HTTP 400 | negative contract | `NOT_IMPLEMENTED` |
| Hallucination/evidence guard | helper classes only | no decode hook | keyword/regex helper input | post-processing helper only | helper tests | no labeled scientific evaluation | `DESIGN_ONLY` as detector claim |
| Sparse vocabulary logit vectors | `vocabulary_logit_vectors` | custom processor | explicit sparse Qwen-vocabulary entries | normalized/clipped signed sum | vector ID/norm/coefficient/delta | live sign and zero controls | `IMPLEMENTED_AND_PROVEN` |
| DExperts | rejected request key | none | no expert/anti-expert artifacts | none | explicit HTTP 400 | negative contract | `NOT_IMPLEMENTED` |
| Activation addition/ActAdd | rejected request key | none | no activation artifact or model hook | none | explicit HTTP 400 | negative contract | `DESIGN_ONLY` CPU helpers |
| Conceptor steering | rejected request key | none | no live hidden states | none | explicit HTTP 400 | negative contract | `DESIGN_ONLY` CPU helpers |
| Hidden-state probes | rejected request key | none | no dataset/probe/model hook | none | explicit HTTP 400 | negative contract | `DESIGN_ONLY` CPU helpers |
| Closed-loop latent steering | rejected request key | none | no aligned hidden signal | none | explicit HTTP 400 | negative contract | `NOT_IMPLEMENTED` |
| Soft prompts/ReFT | rejected request key | none | no trained artifact/hook | none | explicit HTTP 400 | negative contract | `NOT_IMPLEMENTED` |
| Branch-and-tournament | rejected request key | none | no distinct branch runtime/scorers | none | explicit HTTP 400 | negative contract | `NOT_IMPLEMENTED` |
| Attention-head analysis/gating | rejected request key | none | no head telemetry/hook | none | explicit HTTP 400 | negative contract | `NOT_IMPLEMENTED` |
| KV-cache intervention/firewall | rejected request key | none | no selective cache control | none | explicit HTTP 400 | negative contract | `NOT_IMPLEMENTED` |
| GPU-native semantic control | rejected request key | none | no semantic GPU kernel | none | explicit HTTP 400 | negative contract | `NOT_IMPLEMENTED` |

The public `nram-moe-orchestrator` is a sequential prose-orchestration workflow.
It is not DExperts, model-level MoE, or branch-and-tournament generation.

