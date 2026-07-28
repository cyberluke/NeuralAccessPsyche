# Security

NeuralAccessPsyche runs a custom logit processor **inside** the inference server. That capability is powerful and therefore tightly constrained. This document defines the trust boundary.

## Threat model

The public API is OpenAI-compatible and reachable by clients. The dangerous operations — constructing and serializing a logit processor, and reaching SGLang's custom-processor endpoint — must **never** be controllable by a client. Arbitrary client-supplied code running inside the inference server would be a remote-code-execution risk.

## Rejected client fields

The public API **rejects** any request whose `nram` object contains these fields. They are reserved exclusively for the server:

| Field | Why it is rejected |
|-------|--------------------|
| `custom_logit_processor` | Serialized processor — would let a client inject code into SGLang |
| `custom_params.__req__` | Internal request handle — only valid inside SGLang, never client-supplied |
| `serialized_processor` | Same as `custom_logit_processor` |
| `processor_class` | Would let a client choose an arbitrary class to instantiate |
| `python_code` | Direct code injection |

Rejection is enforced in two layers:

- **API layer** (`api/routes.py`): requests containing a forbidden field return an OpenAI-shaped error:

  ```json
  {
    "error": {
      "message": "Forbidden field in nram options: processor injection is not allowed.",
      "type": "invalid_request_error",
      "param": null,
      "code": "forbidden_field"
    }
  }
  ```

- **Engine layer** (`core/engines/sglang_engine.py`): the upstream payload builder checks the same set (plus `custom_params` and `__req__`) and raises before anything is sent to SGLang.

## Only the server builds the processor

The trusted processor is serialized **only** by the server, at engine initialization:

```python
# core/steering/serialization.py — server-side only
def serialize_processor(processor_class: type) -> str:
    import dill
    return json.dumps({"callable": dill.dumps(processor_class).hex()})
```

`SGLangEngine.__init__` serializes the known `NRAMLogitProcessor` class once. Clients can never substitute a different class or payload. The only per-request data a client influences is the **bounded numeric parameters** (token IDs + bias magnitudes), all of which are validated and clamped inside the processor.

## The processor itself is sandboxed

`NRAMLogitProcessor` is written to be safe to run inside SGLang:

- **No** dependency on application services
- **No** network access
- **No** filesystem access
- **No** arbitrary request-controlled imports
- All token IDs validated against the vocabulary size (out-of-range dropped)
- All floats clamped to safe ranges; NaN/Inf rejected

## SGLang is not exposed publicly

- **SGLang's custom-logit-processor endpoint is never exposed to clients.** Only the FastAPI gateway talks to SGLang, and only with server-constructed payloads built from an explicit **allowlist** of forwardable fields:

  ```
  model, messages, temperature, top_p, max_tokens, stream, stop,
  seed, frequency_penalty, presence_penalty, response_format
  ```

- **The SGLang port is not published to the host.** SGLang listens on the internal Docker network at `sglang:30000`. There is no host port mapping, so it is unreachable from outside the compose network. All traffic must pass through the authenticated FastAPI gateway on port `8000`.

## The model is read-only

The GGUF model is mounted into the SGLang container **read-only**:

```yaml
volumes:
  - /mnt/e/_MODELS/.../DeepSeek-R1-Distill-Qwen-7B-GGUF:/models:ro
```

The `:ro` flag means the container cannot modify or replace the model weights. The model is a local file mounted from the host — it is never downloaded or written by the runtime.

## Authentication & rate limiting

- **Auth:** `/v1` routes require a `Bearer` token (`TokenAuthMiddleware` + `utils/auth.py`).
- **Rate limiting:** a time-window limiter (`RateLimitMiddleware`, default 60 req/min) guards against abuse.
- Visualization/explorer routes are exempt from auth for local observability, but they are read-only telemetry and cannot influence generation.

## Summary of the boundary

| Asset | Protection |
|-------|-----------|
| Processor construction/serialization | Server-only |
| Client processor injection | Rejected (API + engine layers) |
| SGLang custom-processor endpoint | Never exposed publicly |
| SGLang port | Not published to host (internal network only) |
| Upstream payload fields | Explicit allowlist |
| GGUF model | Mounted read-only, never downloaded |
| Public API | Bearer auth + rate limiting |
