# NRAM Qwen3-14B LoRA runtime evidence

Evidence collected locally on 2026-07-31 from commit `a1810b7` plus the
runtime fixes in the focused evidence commit.

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

The first direct probe intentionally exposed that repeated argparse flags
overwrite values. The corrected Compose contract was verified in the parsed
`ServerArgs`; the rejected first probe is retained as defect provenance, not
as acceptance evidence.

## Limitations

- Registry image digest, immutable Qwen3-14B training revision, and dataset
  manifest are unresolved and remain `UNAVAILABLE`.
- Peak allocator telemetry was not persisted by the current trainer/runtime
  harness; startup memory and no-OOM evidence are recorded above.
- This evidence does not authorize the prohibited 50-prompt causal ablation.
