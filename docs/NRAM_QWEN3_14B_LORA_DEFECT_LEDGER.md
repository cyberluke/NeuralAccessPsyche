# Defect ledger

| ID | Finding | Status |
|---|---|---|
| D-001 | GPU/Docker/SGLang runtime gates were not available in the initial CPU phase | RESOLVED by local Docker RTX evidence; retained as historical provenance |
| D-002 | Training base and private registry digests require a real build/push | OPEN; local image is proven, registry digest remains unavailable |
| D-003 | Gradient evidence was sampled after Trainer cleared gradients | RESOLVED in `nram_training/trainer.py`; callback captures the pre-optimizer-step tensors |
| D-004 | Repeated SGLang LoRA flags overwrote prior values in direct launch | RESOLVED; Compose uses one flag followed by the complete target/path lists |
| D-005 | Deterministic zero-versus-small-delta logprob effect was not detected | OPEN HIGH; supported pinned SGLang scoring path returns identical zero/small values; native `/generate` returned HTTP 500 |
| D-006 | Historical Civil Comments membership cannot be reconstructed | OPEN HIGH; original script persisted text only after shuffled sampling/deduplication and no row IDs or split manifest exist; status is `DATASET_PROVENANCE_UNRECOVERABLE` |
| D-007 | Strong-delta public probe does not yield clean numerical activation proof | OPEN HIGH; identities route and CSGMV loads, but strong equals zero and repeated base inherits post-adapter result, consistent with cache/state coupling; status `INCONCLUSIVE`, not inactive |
| D-008 | Tiny BF16 Qwen3-14B RTX gate is not applicable | RESOLVED AS NOT_APPLICABLE; unquantized Qwen3-14B BF16 parent alone exceeds RTX 4090 VRAM; no retry performed |
| D-009 | Cold-server strong-delta remains numerically equal to base | OPEN HIGH; `CACHE_CONTAMINATION=RULED_OUT`, but `FIRST_TOKEN_LORA_SENSITIVITY=INVALID` because all probes measured the saturated `<think>` template token; corrected 16-token content comparison is required |
| D-010 | First-token synthetic sensitivity probe used the wrong chat-template request form | OPEN MEDIUM; top-level `chat_template_kwargs: {"enable_thinking": false}` was not applied, so the observed `<think>` token is not valid LoRA sensitivity evidence |
