# NRAM Qwen3-14B LoRA runtime evidence

Evidence collected locally on 2026-07-31 from the pinned runtime and the
focused evidence commits. The final deterministic sensitivity gate is recorded
as failed; request reachability is not treated as proof of computation change.

## Training smoke

- Docker image: `neuralaccesspsyche-nram-training:local-a1810b7`
- GPU invocation: `docker run --gpus all --ipc=host --shm-size=8g`
- Device: NVIDIA GeForce RTX 4090, 24,146,083,840 bytes reported by CUDA
- Test model: `Qwen/Qwen3-0.6B` only; marked `TEST_ONLY;NOT_FOR_SCIENTIFIC_USE`
- Roles: `nontoxic` and `toxic`, isolated output directories
- Both roles: CUDA true, base parameters frozen, finite nonzero LoRA gradients
- Nontoxic initial run: global step 2, finite loss `7.4832611083984375`,
  checkpoint `checkpoint-2`
- Nontoxic resume run: resumed from `checkpoint-2`, global step 3, finite loss
  `1.669153054555257`, checkpoint `checkpoint-3`
- Toxic run: global step 3, finite loss `1.753562291463216`

The generated checkpoints and adapters remain local under
`artifacts/training/rtx-smoke/` and are intentionally excluded from the Git
evidence commit.

## Pinned SGLang AWQ + LoRA

- Runtime image: `neuralaccesspsyche-nram-sglang:0.5.16-nram`
- Base image digest: `sha256:2a4e0bfde6eb75a2b6ccdda0296494747bb8643e6a88da14c1e6c87035d49cda`
- Model snapshot: `Qwen/Qwen3-14B-AWQ` revision
  `31c69efc29464b6bb0aee1398b5a7b50a99340c3`
- Launch parsed all seven target modules and both read-only fixtures:
  `zero` and `small-delta`
- Model load: AWQ 4-bit, 9.47 GB weight memory, no OOM
- KV allocation: 34,493 tokens, 2.63 GB each for K and V
- Decode graph capture: batch sizes 1, 2, and 3 completed without OOM
- `/health` passed; `/v1/models` exposed base, `zero`, and `small-delta`
- Base, zero, and small-delta requests returned finite `OK.` responses with
  `enable_thinking=false`
- Three concurrent requests for base, zero, and small-delta completed without
  HTTP errors or cross-request model identity contamination

### Deterministic logprob gate

- Probe artifact: `artifacts/training/rtx-logprob-probe.json`
- Base versus zero: `0.0` (within `0.0001` tolerance)
- Repeated base: `0.0` (within tolerance)
- Base versus small-delta: `0.0` (no effect detected)
- Zero versus small-delta: `0.0` (no effect detected)
- All returned values were finite.
- Verdict: `SMALL_DELTA_LOGPROB_GATE=FAIL`; the pinned OpenAI-compatible path
  did not prove that the synthetic nonzero adapter changed the scored token.
  The native `/generate` attempt returned HTTP 500, and completion echo
  logprobs are rejected by this SGLang build.

The first direct probe intentionally exposed that repeated argparse flags
overwrite values. The corrected Compose contract was verified in the parsed
`ServerArgs`; the rejected first probe is retained as defect provenance, not
as acceptance evidence.

## Limitations

- Training model, tokenizer, config, and chat-template revision:
  `40c069824f4251a91eefaf281ebe4c544efd3e18`.
- Civil Comments repository revision:
  `f2970eb3a55777454c94069077cc8d9b5866312d`.
- Historical dataset membership is `DATASET_PROVENANCE_UNRECOVERABLE`: the
  original script persisted text only after shuffled sampling and deduplication,
  with no row IDs, split membership, or calibration manifest.
- Registry publication is `REGISTRY_PUBLICATION=NOT_CONFIGURED`; local image
  readiness is independent of registry publication.
- Peak allocator telemetry was not persisted by the current trainer/runtime
  harness; startup memory and no-OOM evidence are recorded above.
- This evidence does not authorize the prohibited 50-prompt causal ablation.

## RTX follow-up generation

- Strong fixture: `artifacts/training/fixtures/strong-delta`, deterministic rank 2,
  seed `20260731`, scale `8.0`, finite norm `21424.61201497445`.
- Offline zero/strong tensor verification passed; hashes differ and both
  fixtures match Qwen3-14B dimensions, targets, and model revision.
- Eight fixed prompts were exercised in base → zero → strong-delta → base order.
- `LORA_LOADED=PASS`, `LORA_ROUTED=PASS`,
  `OPENAI_ENDPOINT_OBSERVABLE=PASS`, `NATIVE_ENDPOINT_OBSERVABLE=UNAVAILABLE`.
- `LORA_NUMERICALLY_ACTIVE=INCONCLUSIVE`: strong-delta matched zero on all
  exposed scalars, while repeated base matched the post-adapter result,
  indicating shared cache/state coupling. This is not classified as inactive.
- Raw request/response bodies: `artifacts/training/rtx-strong-lora-probe.json`.
- Tiny BF16 14B gate: `NOT_APPLICABLE`; the unquantized Qwen3-14B BF16
  parent alone exceeds RTX 4090 VRAM. No BF16 retry was performed.

### Final cache-disabled diagnostic

- Temporary overlay: `compose.lora.diagnostic.yaml`; production Compose files
  were not modified and were restored after the probe.
- Disabled radix/prefix cache, CUDA graphs, chunked prefix cache, HiCache,
  LMCache, and CPU offload; retained one running request and overlap disabled.
- Eight base → zero → strong-delta → base sequences were finite and identity-clean.
  Base equaled zero and repeated base on every prompt; strong-delta equaled base
  on every prompt.
- Three cold-server probes independently produced the same selected token and
  logprob for base, zero, and strong-delta. This rules out cross-request cache
  contamination, but does not prove inactive LoRA because the pinned public
  runtime exposes no A/B pointer checksum or forward scalar checksum.
- Source inspection confirms `lora_id` reaches scheduler requests, is included
  in the batch/cache key, and feeds LoRA `weight_indices` into CSGMV A/B kernels.
- Evidence: `artifacts/training/rtx-cache-disabled-lora-summary.json` and the
  raw response/server-log artifacts listed there.
