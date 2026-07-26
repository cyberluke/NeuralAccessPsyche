# NeuralAccessPsyche

An **NRAM (Neural Random Access Memory) controlled local inference system**. NeuralAccessPsyche exposes an **OpenAI-compatible FastAPI server** in front of a **local SGLang inference server** running a quantized GGUF model on an NVIDIA RTX 4090 (Windows 11 → WSL2 → Docker). NRAM steers generation by injecting a **custom logit processor that modifies logits _before_ token sampling** — real pre-sampling control, not a prompt trick.

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
   ├─ NRAM controller           (state + deterministic policy hash)
   ├─ rhetorical planner        (llguidance structured plan)
   └─ token bias compiler       (lexemes → concrete token IDs)
        │  injects trusted custom_logit_processor + custom_params
        ▼
SGLang inference server (internal network, :30000, not published to host)
   └─ NRAMLogitProcessor  →  modifies logits BEFORE sampling  →  GPU decoding
        │
        ▼
   SSE token stream back through FastAPI to the client
```

The base model is **DeepSeek-R1-Distill-Qwen-7B (Q4_K_M GGUF)**, served through SGLang.

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

## Model aliases

Two primary aliases are exposed for the local 7B model:

| Alias | NRAM steering | Description |
|-------|:---:|-------------|
| `deepseek-r1-qwen-7b-baseline` | off | Unsteered baseline — the raw local model. Use for A/B comparison. |
| `nram-deepseek-r1-qwen-7b` | on | Same model with the NRAM persona policy + logit processor active. |

List them at runtime:

```bash
curl http://localhost:8000/v1/models -H "Authorization: Bearer YOUR_TOKEN"
```

## OpenAI Python client example

The `nram` options are passed through the OpenAI SDK's `extra_body`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="YOUR_TOKEN",  # any Bearer token longer than 10 chars
)

response = client.chat.completions.create(
    model="nram-deepseek-r1-qwen-7b",
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
  persona/      profiles, policy compiler, rhetorical planner
  steering/     NRAMLogitProcessor, tokenizer bias compiler, policy bounds
  nram_controller/  NRAM state controller + metrics
utils/          validators, auth, rate limiter, api logger
templates/      visualization dashboards
tests/          contract + unit tests
```
