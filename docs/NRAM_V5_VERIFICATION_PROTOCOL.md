# NRAM v5 Verification Protocol

## Fixed runtime

- SGLang: `0.5.16`, source `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`.
- Base image: `lmsysorg/sglang@sha256:2a4e0bfde6eb75a2b6ccdda0296494747bb8643e6a88da14c1e6c87035d49cda`.
- Model/tokenizer: local `Qwen/Qwen3-14B-AWQ` snapshot `31c69efc29464b6bb0aee1398b5a7b50a99340c3`, mounted read-only.
- Concurrency: one; overlap scheduling disabled; prefill/decode CUDA graphs disabled.
- Public API: `http://127.0.0.1:8000/v1/chat/completions`.

## Ordered gates

1. `python -m pytest --collect-only -q`
2. `python -m pytest tests/unit tests/contract tests/adversarial tests/integration -q`
3. `python -m pytest tests/gpu -q -s`
4. `python -m pytest tests -q --junitxml=<run>/test-results.xml`
5. `python -m ruff check .`
6. `python -m mypy nram_sglang/processor.py core/steering/serialization.py core/engines/sglang_engine.py api/routes.py`
7. `docker compose config --quiet`
8. `docker compose build sglang nram-api`
9. Cold restart: `docker compose up -d --force-recreate --wait --wait-timeout 600 sglang`, followed by API recreation.
10. Serialization diagnostics and round trips in both containers.

Lint and type checking are gates, not implied passes. A command missing its
declared optional dependency is recorded `UNAVAILABLE`, not passed.

## Causal proof design

All live proofs use fixed prompt, model, tokenizer, seed, temperature, top-p,
and maximum tokens. Enabled and disabled requests differ only in the named
intervention. The processor emits bounded `nram.processor.step.v1` events with
request ID, configuration hash, invocation count, scheduler request ID,
pre/post top-k, direct changed-token values, masks, PID terms, and local control
details. The public response carries correlation metadata only; it does not
synthesize causal telemetry.

Forced and structural proofs use hard schedules solely as deterministic test
fixtures. Phrase/source masks must defeat a scheduled completion token, proving
mask order before sampling. This fixture does not claim natural-language quality.

## Statistics policy

No confidence interval, hypothesis test, or multiple-comparison result is
reported unless the paired run has a preregistered sample size sufficient for
that analysis. Missing/blocked conditions produce machine-readable status rows,
never zero-filled pseudo-results.

