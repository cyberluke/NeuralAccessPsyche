# Status: canonical
# Version: 1
# Source: user mission specification received 2026-07-31
# Scope: NRAM repository only; Xeno-SGLang is out of scope

# NRAM Local RTX Phase — Prepare and Validate Qwen3-14B LoRA Pipeline

Work only in the NRAM repository. Xeno-SGLang is out of scope.

## Existing runtime

Preserve:

```text
SGLang: 0.5.16
image:
lmsysorg/sglang@sha256:2a4e0bfde6eb75a2b6ccdda0296494747bb8643e6a88da14c1e6c87035d49cda

runtime source:
fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1

local AWQ model:
Qwen/Qwen3-14B-AWQ
snapshot:
31c69efc29464b6bb0aee1398b5a7b50a99340c3
```

Do not launch the 50-prompt causal ablation.

## Target architecture

Prepare training of two new adapters:

```text
Qwen3-14B nontoxic LoRA = expert
Qwen3-14B toxic LoRA    = anti-expert
```

Training parent:

```text
Qwen/Qwen3-14B
```

Deployment:

```text
Qwen/Qwen3-14B-AWQ + external LoRA adapters
```

Existing 0.6B adapters must not be attached to the 14B model.

## 1. Inspect before editing

Inspect:

```bash
git status --short
git rev-parse HEAD
rg -n "dexperts|lora|adapter|Qwen3|SGLang" .
docker compose config
docker image inspect neuralaccesspsyche-nram-sglang:0.5.16-nram
docker run --rm \
  lmsysorg/sglang@sha256:2a4e0bfde6eb75a2b6ccdda0296494747bb8643e6a88da14c1e6c87035d49cda \
  python3 -m sglang.launch_server --help
```

Preserve unrelated untracked R5/R6 artifacts.

## 2. Implement a dedicated training image

Do not reuse the SGLang image for training.

Add a separate pinned Linux AMD64 image containing:

```text
PyTorch with CUDA
Transformers
PEFT
TRL or existing trainer
Accelerate
safetensors
huggingface_hub
dataset dependencies
FlashAttention 2 or verified SDPA fallback
```

Requirements:

```text
RTX 4090 / L4: SM89
H100 / H200:   SM90
Python and every dependency pinned
base image pinned by digest
no runtime pip install
no compilation on paid cloud GPU
CUDA-specific imports lazy in CPU mode
```

Do not put model weights, datasets, adapters, caches or credentials into the image.

Add BuildKit/buildx scripts producing:

```text
private Docker Hub image
SBOM
build provenance
resolved image digest
source commit OCI label
```

Use an immutable tag:

```text
<namespace>/nram-training:qwen3-14b-lora-<git-sha>
```

## 3. Implement one production trainer

Provide a CLI equivalent to:

```bash
python -m nram_training doctor
python -m nram_training inspect-data
python -m nram_training smoke
python -m nram_training train --adapter nontoxic
python -m nram_training train --adapter toxic
python -m nram_training validate
python -m nram_training package
```

Both adapters must use the same training implementation and hyperparameter schema.

Each adapter training must:

* load a clean immutable Qwen3-14B base;
* freeze all base parameters;
* train only its own LoRA parameters;
* use the exact Qwen3 chat template with thinking disabled;
* mask loss to continuation/assistant tokens;
* use sequence packing;
* persist checkpoints atomically;
* support `--resume auto`;
* write into a completely separate output directory;
* record model, tokenizer, dataset, code and image hashes.

Inspect existing 0.6B training artifacts before selecting LoRA rank, alpha, target modules and learning parameters. Preserve the existing scientific intent where valid. Do not silently invent unrelated settings.

## 4. Local RTX training smoke

Use the actual production trainer with a small compatible model:

```text
Qwen/Qwen3-0.6B
```

Do not use a mocked network.

Run a tiny synthetic fixture for both roles and verify:

```text
CUDA works inside the final container
selected attention backend works
loss is finite
LoRA gradients are non-zero
base parameters remain frozen
only LoRA parameters change
checkpoint save works
checkpoint resume advances global step
adapter save/reload preserves outputs
adapter disable restores base behavior
both adapter jobs use isolated output directories
```

The synthetic fixture must be visibly marked:

```text
TEST_ONLY
NOT_FOR_SCIENTIFIC_USE
```

## 5. Validate AWQ + multi-LoRA locally before cloud training

The H100 training must not begin until the current pinned SGLang image proves it can load LoRA against the existing Qwen3-14B-AWQ model.

Add a generator for deterministic synthetic 14B LoRA runtime fixtures using the intended rank and target-module shapes:

```text
synthetic-zero adapter
synthetic-small-delta adapter
```

Do not load or train the full BF16 14B model merely to create these fixtures. Use model config/meta initialization or directly construct validated adapter tensors.

Mark all generated fixtures as non-scientific and keep them outside production adapter paths.

Create `compose.lora.yaml` that overlays the current compose configuration:

```text
mount synthetic adapters read-only
enable LoRA using flags verified against SGLang 0.5.16
max LoRAs per batch = 3
max running requests = 3 initially
preserve AWQ model and tokenizer paths
```

Test on RTX 4090:

```text
AWQ base
AWQ + zero LoRA
AWQ + small-delta LoRA
base/zero parity
small-delta changes output or logits
both adapters load simultaneously
base and two adapter requests can share one batch
no shape mismatch
no request contamination
no NaN/Inf
no OOM
```

If pinned SGLang cannot execute dense Qwen3-14B AWQ + LoRA:

* do not silently upgrade it;
* produce a minimal reproduction;
* record the exact unsupported layer, kernel or flag;
* stop before H100 training;
* propose the smallest runtime change separately.

## 6. Fix deployment configuration assumptions

The existing compose has:

```text
MAX_RUNNING_REQUESTS=1
```

Keep this default for the current runtime proof, but the LoRA overlay must start with:

```text
MAX_RUNNING_REQUESTS=3
source microbatch=1
```

Then test source microbatches:

```text
1 → 2 → 4
```

Do not start with eight source sequences / 24 requests.

Correct the misleading memory comment:

```text
24 GB - 17.3 GB = approximately 6.7 GB
```

not 7.8 GB.

Do not change measured runtime memory values without collecting fresh telemetry.

## 7. Prepare private Docker Hub artifacts

If Docker Hub is already authenticated and the private namespace is known:

1. Build the training image.
2. Run the RTX smoke inside the exact image.
3. Push the image.
4. Resolve its registry digest.
5. Optionally build and push the current thin NRAM SGLang wrapper.
6. Never push weights, datasets or adapters.

If credentials are unavailable, do not request or print them. Prepare exact commands for the user.

Create:

```text
artifacts/training/training-image-lock.json
artifacts/training/inference-image-lock.json
artifacts/training/remote-handoff.json
docs/NRAM_QWEN3_14B_LIGHTNING_RUNBOOK.md
```

`remote-handoff.json` must be sufficient for a new coding-agent session with no conversation history. Include:

```text
repository URL
branch
training code SHA
required files
training image reference and digest
inference image reference and digest
model repositories and immutable revisions
dataset manifest path and hash
expected persistent paths
exact CPU bootstrap command
exact 2×H100 preflight command
exact 2×H100 production command
local gates and results
remaining blockers
```

## 8. Local acceptance gate

Before declaring the local phase complete:

```text
CPU tests pass
training image builds
RTX 0.6B training smoke passes for both roles
checkpoint/resume passes
AWQ + synthetic 14B LoRA loads in pinned SGLang
three-way base/expert/anti runtime probe passes
private image digest is recorded or push commands are ready
Lightning runbook is complete
```

Commit in scoped commits. Stage explicit files only.

Return:

```text
STARTING_SHA
FINAL_SHA
COMMITS
WORKTREE_STATUS

TRAINING_IMAGE
TRAINING_IMAGE_DIGEST
RTX_TRAINING_SMOKE
CHECKPOINT_RESUME
SGLANG_0516_AWQ_LORA
THREE_WAY_RUNTIME_PROBE

TRAINING_SOURCE_REVISION
AWQ_REVISION
TOKENIZER_REVISION
LORA_CONFIG
DATASET_MANIFEST_HASH

REMOTE_HANDOFF_PATH
LIGHTNING_RUNBOOK_PATH
EXACT_NEXT_ACTION
```
