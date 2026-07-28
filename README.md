# NeuralAccessPsyche

An **NRAM (Neural Random Access Memory) controlled local inference system**. NeuralAccessPsyche exposes an **OpenAI-compatible FastAPI server** in front of a **local SGLang inference server** running Qwen3-14B-AWQ on an NVIDIA RTX 4090 (Windows 11 → WSL2 → Docker). Supported NRAM controls use a custom logit processor that modifies logits before sampling. Advanced representation, DExperts, semantic closed-loop, and tournament mechanisms are not implemented.

## What makes this different

Most "persona" or "style" systems work by writing a clever system prompt. NeuralAccessPsyche does that too, but it *also* reaches into the decoding loop:

- A persona is a set of **cognitive / rhetorical properties** (visionary intensity, associative distance, corporate-jargon penalty, …), **not a public-figure cosplay**.
- Those properties are compiled deterministically into **token-level biases, hard masks, and repetition penalties**.
- A serialized `NRAMLogitProcessor` runs **inside SGLang** and adjusts logits **before** the sampler draws a token.

## Architecture (high level)

```
OpenAI-compatible client
        │
        ▼
NeuralAccessPsyche FastAPI (:8000)
   ├─ request validation + forbidden-field rejection
   ├─ persona policy compiler   (NRAMState → SteeringPolicy)
   ├─ versioned applied-state compiler and hash
   └─ token bias compiler       (lexemes → concrete token IDs)
        │  injects trusted custom_logit_processor + custom_params
        ▼
SGLang inference server (internal network, :30000, not published to host)
   └─ NRAMLogitProcessor  →  modifies logits BEFORE sampling  →  GPU decoding
        │
        ▼
   SSE token stream back through FastAPI to the client
```

The immutable base model is **Qwen/Qwen3-14B-AWQ**, snapshot `31c69efc29464b6bb0aee1398b5a7b50a99340c3`, served as `nram-qwen3-14b-awq` through the pinned SGLang runtime.

For the full request flow, see [`docs/architecture.md`](docs/architecture.md).

## Quickstart

```bash
# 1. Start the stack (FastAPI gateway + SGLang inference server)
docker compose up --build

# 2. Validate the GPU is visible inside the container (one-time)
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi

# 3. Talk to the OpenAI-compatible endpoint
#    (see the Python example below)
```

The FastAPI gateway listens on `http://localhost:8000`. The SGLang server runs on an internal Docker network at `sglang:30000` and is **not** published to the host — see [`docs/security.md`](docs/security.md).

> The local GGUF model is **mounted read-only** from the Windows drive (WSL path `/mnt/e/...`), not downloaded by the container. See [`docs/windows-wsl2-setup.md`](docs/windows-wsl2-setup.md).

## Model identity and virtual routes

`GET /v1/models` enumerates only the immutable loaded identity:

| Alias | NRAM steering | Description |
|-------|:---:|-------------|
| `nram-qwen3-14b-awq` | request dependent | Actual loaded Qwen checkpoint. An `nram` object enables the bounded processor path. |

The accepted `qwen3-14b-awq-baseline` and `persona-*` names are documented virtual routes over that same base checkpoint; they are not separate loaded models and are therefore not returned by `/v1/models`. Responses expose both public route identity and `actual_base_model` correlation.

List the loaded identity at runtime:

```bash
curl http://localhost:8000/v1/models -H "Authorization: Bearer YOUR_TOKEN"
```

## OpenAI Python client example

The `nram` options are passed through the OpenAI SDK's `extra_body`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="YOUR_CONFIGURED_NRAM_API_KEY",
)

response = client.chat.completions.create(
    model="nram-qwen3-14b-awq",
    messages=[
        {"role": "user", "content": "Introduce a new tool for personal knowledge work."},
    ],
    temperature=0.7,
    max_tokens=512,
    stream=True,
    extra_body={
        "nram": {
            "enabled": True,
            "profile": "visionary-psychedelic-keynote",
            "visionary_intensity": 0.95,
            "associative_distance": 0.72,
            "corporate_jargon_penalty": 0.92,
            "coherence_floor": 0.80,
        }
    },
)

for chunk in response:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

Every `nram` field is optional; omit it to use the profile default. The full field list and semantics are in [`docs/openai-compatibility.md`](docs/openai-compatibility.md).

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/architecture.md`](docs/architecture.md) | End-to-end request flow + Mermaid diagram |
| [`docs/windows-wsl2-setup.md`](docs/windows-wsl2-setup.md) | WSL2 + Docker Desktop + NVIDIA GPU passthrough |
| [`docs/openai-compatibility.md`](docs/openai-compatibility.md) | Supported request fields + the `nram` extension object |
| [`docs/nram-steering.md`](docs/nram-steering.md) | What NRAM actually controls (pre-sampling logit steering) |
| [`docs/security.md`](docs/security.md) | Trust boundary & rejected client fields |
| [`docs/evaluation.md`](docs/evaluation.md) | A/B evaluation harness & success criteria |
| [`docs/runtime-baseline.md`](docs/runtime-baseline.md) | Recorded machine/runtime baseline |

## Repository layout

```
api/            FastAPI router + middleware (auth, rate limiting)
core/
  contracts/    Pydantic contracts (OpenAI + NRAM schemas)
  engines/      SGLang engine (+ legacy fallback)
  persona/      profiles and policy compiler
  steering/     NRAMLogitProcessor, tokenizer bias compiler, policy bounds
  nram_controller/  NRAM state controller + metrics
utils/          validators, auth, rate limiter, api logger
templates/      visualization dashboards
tests/          contract + unit tests
```
