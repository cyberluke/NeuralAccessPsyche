# NRAM v5 Reproducibility

## Identity

- Canonical specification: `docs/NRAM_V5_SCIENTIFIC_VALIDATION_SPEC.md`.
- Specification provenance SHA: `7aa12854800327ea905e608b9ececc3e3f5fd758`; the specification file is uncommitted at that provenance point.
- Baseline ancestor: `6b92283d8fee480fe6355faed6fa24a2f0d75a56`.
- Runtime image/model/tokenizer identities are recorded in the versioned manifest and environment files.

## Build and launch

```text
docker compose config --quiet
docker compose build sglang nram-api
docker compose up -d --force-recreate --wait --wait-timeout 600 sglang
docker compose up -d --force-recreate --no-deps nram-api
```

The SGLang entrypoint fails before model load if version `0.5.16`, the official
server-side `__req__` merge, or the dill class contract is absent.

## Replay

```text
python -m pytest tests/gpu/test_forced_token_proof.py -q -s
python -m pytest tests/gpu/test_structural_controls.py -q -s
python -m pytest tests/gpu/test_logit_controls.py -q -s
```

Each test creates a unique public request ID and verifies matching
`NRAM_PROCESSOR_EVENT` lines from `docker logs nram-sglang`. Configuration hashes
are canonical SHA-256 hashes of the exact JSON intervention configuration.

## Artifact policy

The final versioned run contains environment, manifest, commands, launch state,
dependency versions, serialization diagnostics, JUnit, runtime events,
token-level rows, generation metrics, a human-evaluation template, raw outputs,
plot metadata, and failure records. Representation CSV rows are explicitly
`NOT_IMPLEMENTED`; they are not scientific measurements.

