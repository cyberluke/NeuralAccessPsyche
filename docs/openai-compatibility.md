# OpenAI Compatibility

NeuralAccessPsyche exposes an OpenAI-compatible API at `/v1`. Point any OpenAI SDK at it and it behaves like the standard Chat Completions API — plus one extension object, `nram`, for steering.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="YOUR_TOKEN",
)
```

**Authentication:** send `Bearer <NRAM_API_KEY>` in the `Authorization` header. The local configured token is compared exactly with a constant-time comparison; arbitrary long strings are rejected.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (streaming + non-streaming) |
| `/v1/models` | GET | List immutable loaded model identities (not virtual routes) |

## Supported request fields

These standard OpenAI fields are accepted on `/v1/chat/completions`:

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `model` | string | required | Model alias (see below) |
| `messages` | array | required | Chat messages (`role` + `content`) |
| `temperature` | float | `1.0` | Sampling temperature |
| `top_p` | float | `1.0` | Nucleus sampling |
| `max_tokens` | int | `100` | Max generated tokens |
| `stream` | bool | `false` | SSE streaming when `true` |
| `stop` | string[] | `null` | Stop sequences |
| `seed` | int | `null` | Sampler seed |
| `frequency_penalty` | float | `0.0` | Standard OpenAI penalty |
| `presence_penalty` | float | `0.0` | Standard OpenAI penalty |
| `response_format` | object | `null` | e.g. JSON schema / strict JSON |
| `tools` | array | `null` | Tool definitions |
| `tool_choice` | any | `null` | Tool selection mode |
| `nram` | object | `null` | **Extension** — NRAM steering options |

### Model aliases

| Alias | NRAM | Upstream model |
|-------|:---:|----------------|
| `nram-qwen3-14b-awq` | request dependent | loaded Qwen3-14B-AWQ checkpoint |
| `qwen3-14b-awq-baseline` | off | virtual unmodified route over the same loaded checkpoint |
| `persona-*` | on | virtual profile route over `nram-qwen3-14b-awq` |

`/v1/models` returns only `nram-qwen3-14b-awq`; baseline/persona names are virtual routes, not immutable model identities. Responses and SSE chunks expose `public_model`, `actual_base_model`, and a server-generated `scheduler_request_id`. Baseline requests do not attach a processor.

## The `nram` extension object

Pass `nram` via the OpenAI SDK's `extra_body`. Every field is optional.

| Field | Type | Default | Range | Meaning |
|-------|------|---------|-------|---------|
| `enabled` | bool | `true` | — | Turn NRAM steering on/off for an `nram-*` alias |
| `profile` | string | `"normal"` | known profile | Persona profile name |
| `visionary_intensity` | float | `0.85` | 0.0–1.0 | Drive toward bold, future-oriented framing |
| `contrarian_force` | float | `0.75` | 0.0–1.0 | Willingness to challenge accepted assumptions |
| `product_obsession` | float | `0.90` | 0.0–1.0 | Density of concrete product detail |
| `human_focus` | float | `0.90` | 0.0–1.0 | Grounding in human need/experience |
| `rhetorical_compression` | float | `0.75` | 0.0–1.0 | Preference for short declarative phrasing |
| `associative_distance` | float | `0.65` | 0.0–1.0 | Cross-domain / "psychedelic" linking distance |
| `theatricality` | float | `0.72` | 0.0–1.0 | Dramatic staging of the argument |
| `emotional_voltage` | float | `0.68` | 0.0–1.0 | Emotional intensity |
| `coherence_floor` | float | `0.78` | 0.0–1.0 | Minimum coherence — novelty must not break factual coherence |
| `novelty_target` | float | `0.70` | 0.0–1.0 | Target conceptual novelty |
| `repetition_penalty` | float | `0.45` | 0.0–1.0 | Dynamic penalty on recently emitted tokens |
| `corporate_jargon_penalty` | float | `0.85` | 0.0–1.0 | Suppression of corporate filler language |

The `profile` selects a base `NRAMState`; any other fields you pass act as **per-request overrides** on top of that profile. Defaults shown above are the `NRAMState` defaults; the built-in `visionary-psychedelic-keynote` profile uses higher values (e.g. `visionary_intensity=0.95`, `corporate_jargon_penalty=0.92`).

> Unknown fields, unknown profiles, malformed nested shapes, non-finite values, out-of-range values, non-integer token IDs, tokenizer-undefined IDs, explicit force/mask conflicts, and unsupported advanced mechanisms are rejected with a 4xx response before generation. Values are not silently coerced or clamped at the public boundary.

### Forbidden fields

The API **rejects** requests whose `nram` object contains any of these fields (see [`security.md`](security.md)):

```
custom_logit_processor
serialized_processor
processor_class
python_code
```

These are reserved for the server. A request containing them returns an `invalid_request_error` / `forbidden_field`.

## Full example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="YOUR_TOKEN",
)

stream = client.chat.completions.create(
    model="nram-qwen3-14b-awq",
    messages=[
        {"role": "user", "content": "Introduce a new tool for personal knowledge work."},
    ],
    temperature=0.7,
    top_p=1.0,
    max_tokens=512,
    stream=True,
    extra_body={
        "nram": {
            "enabled": True,
            "profile": "visionary-psychedelic-keynote",
            "visionary_intensity": 0.95,
            "contrarian_force": 0.86,
            "associative_distance": 0.72,
            "theatricality": 0.84,
            "coherence_floor": 0.80,
            "corporate_jargon_penalty": 0.92,
        }
    },
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

### A/B comparison

To compare steered vs. unsteered output on identical input, issue the same request against both aliases:

```python
for alias in ("qwen3-14b-awq-baseline", "nram-qwen3-14b-awq"):
    r = client.chat.completions.create(
        model=alias,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,      # greedy, reproducible
        seed=1234,
        max_tokens=512,
    )
    print(alias, "->", r.choices[0].message.content)
```
