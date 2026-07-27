# NRAM v5 Phase 0 Intake

## Scope and verdict

This document records a bounded evidence-collection pass at Git SHA `7aa12854800327ea905e608b9ececc3e3f5fd758` on branch `main`. The tree was dirty before Phase 0. No production implementation or existing test was modified by this task.

Scientific verdict for the observed state: `PARTIALLY FUNCTIONAL`.

That verdict is intentionally narrow:

- The live SGLang server, gateway, baseline generation, stable processor serialization, and a controlled forced-token intervention function on the real RTX 4090.
- Output-history-dependent logit controls are not wired to real scheduler request state.
- Representation control, DExperts, closed-loop control, and several advanced claims are definition-only or absent.
- The complete repository gate does not pass.
- No causal ablation study, confidence interval, representation intervention, or paper-grade evidence bundle exists.

This Phase-0 document is intake evidence, not proof that the governing specification is implemented.

## Requirement-to-code mapping before repair

| Requirement slice | Affected/current evidence files | Runtime entry point | Required test/proof | Observable result |
|---|---|---|---|---|
| Canonical source of truth | `docs/NRAM_V5_SCIENTIFIC_VALIDATION_SPEC.md` | Documentation only | Text/provenance inspection | Complete governing text retained; no implementation claim |
| Request/persona routing | `api/routes.py`, `core/engines/sglang_engine.py` | `POST /v1/chat/completions` | Alias routing plus live request | Alias resolves to loaded Qwen model and intended NRAM path |
| Processor serialization | `core/steering/serialization.py`, `nram_sglang/processor.py`, SGLang installed source | Engine initialization and SGLang batch construction | API/server round-trip, live HTTP request, cold restart | Stable class imports and zero-arg construction succeed |
| Request-scoped processor state | `core/steering/serialization.py`, `nram_sglang/processor.py`, installed `sampling_batch_info.py` and `sampler.py` | SGLang sampler | Live output-history forced behavior and telemetry | Scheduler state reaches processor without unsafe client object serialization |
| Structural masks | `nram_sglang/processor.py`, `core/engines/sglang_engine.py` | Custom processor before sampling | Real-tokenizer Czech/English trie and source continuation proofs | Only completion token is masked and source continuation is prevented |
| Static/logit control | `nram_sglang/processor.py` | Custom processor before softmax/sample | Pre/post logits and forced-token controls | Intended token logit/rank/probability changes locally |
| Entropy control | `nram_sglang/processor.py`, `core/steering/entropy_controller.py` | Custom processor | Per-step entropy/PID telemetry and paired targets | Distribution moves stably toward target and state resets |
| Concept capsules | `nram_sglang/processor.py`, `core/steering/concept_capsules.py` | Custom processor | Bilingual same-seed local effect | Intended concepts move without broad distortion |
| DExperts | `core/steering/dexperts.py` | No live entry point found | Three real same-position distributions and equation | Expert and anti-expert have measurable signed effects |
| Representation control | `core/steering/activation_addition.py`, `core/steering/forward_hooks.py`, `core/steering/conceptor_steering.py`, `core/steering/hidden_state_probes.py` | No SGLang model hook found | Real layer hook, norms, downstream logit delta | Hidden state changes at named loaded layer |
| Semantic/evidence guard | `core/steering/hallucination_guard.py` | No decoding integration found | Labeled evaluation and threshold-caused next intervention | Threshold changes subsequent generation |
| Branch-and-tournament | `core/agentic/document_innovation_workflow.py` | Workflow stage | Multiple candidates, raw scores, deterministic selection | All branches exist and scorer weight changes winner |
| Complete gates | `pyproject.toml`, `package.json`, `tests/` | CI/developer shell | Unit, contract, integration, GPU, Streamlit, lint, type, build | Zero unexplained failures and no false passes |
| Reproducibility | `artifacts/nram_v5_verification/` | Experiment runner | Manifest completeness and replay | Results trace to tree/image/model/config/prompt/seed/raw output |

## Repository baseline

| Field | Observed value |
|---|---|
| Worktree | `D:/_SATIN_AI/NeuralAccessPsyche` |
| Branch | `main` |
| Upstream | `origin/main` |
| Ahead/behind | `+0/-0` |
| Starting SHA | `7aa12854800327ea905e608b9ececc3e3f5fd758` |
| Starting subject | `Enhance logit processing` |
| Governing baseline | `6b92283d8fee480fe6355faed6fa24a2f0d75a56` |
| Baseline subject | `feat: add verified SGLang engine baseline` |
| Baseline ancestry | Baseline is an ancestor of starting SHA |
| Starting tree | Dirty: unstaged tracked changes and untracked files; no staged files |
| Pre-existing tracked changes | 10 files, 458 insertions, 54 deletions |
| Pre-existing status paths | 142 |
| Phase-0 production changes | None |
| Commit created | None |

The complete pre-existing path inventory is preserved in `artifacts/nram_v5_verification/phase0-20260727T180644Z-7aa1285/pre-existing-worktree.txt`.

## Environment baseline

### Host and containers

| Field | Observed value |
|---|---|
| Host OS | Microsoft Windows 11 Pro Insider Preview, 10.0.26300, build 26300.8935 |
| WSL | 2.7.10.0; kernel 6.18.33.2-microsoft-standard-WSL2 |
| Distribution | Ubuntu 26.04 LTS |
| Docker Desktop | 4.83.0 (234302) |
| Docker Engine | 29.6.2 |
| Docker Compose | v5.3.1 |
| Docker context | `desktop-linux` |
| GPU | NVIDIA GeForce RTX 4090, UUID `GPU-2511a5a1-c9f7-42bd-112f-0fcae20e106a` |
| Driver | 591.86 |
| CUDA reported by NVIDIA-SMI | 13.1 |
| Host NVCC | 12.4.99 |
| VRAM at inventory | 23,028 MiB total; 16,582 MiB used; 6,027 MiB free |
| Existing SGLang service | `nram-sglang`, healthy; no duplicate started |
| Existing API service | `nram-api`, running; no duplicate started |

### SGLang image and package

| Field | Observed value |
|---|---|
| Declared compose image | `lmsysorg/sglang:latest` |
| Resolved image/digest | `lmsysorg/sglang@sha256:2a4e0bfde6eb75a2b6ccdda0296494747bb8643e6a88da14c1e6c87035d49cda` |
| Image label | `lmsysorg/sglang:v0.5.16` |
| SGLang package | 0.5.16 |
| SGLang source commit | `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1` |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu130 |
| dill | 0.4.1 |
| cloudpickle | 3.1.2 |
| transformers | 5.12.1 |
| tokenizers | 0.22.2 |

### Model and tokenizer

| Field | Observed value |
|---|---|
| Model repository inferred from mount | `Qwen/Qwen3-14B-AWQ` |
| Snapshot | `31c69efc29464b6bb0aee1398b5a7b50a99340c3` |
| Mount | `/models`, read-only |
| Served model | `nram-qwen3-14b-awq` |
| Architecture | `Qwen3ForCausalLM` |
| Quantization | AWQ 4-bit, group size 128, GEMM |
| Layers / hidden size | 40 / 5120 |
| Config vocabulary | 151,936 |
| Weight shards | Two shards, each approximately 4.99 GB |
| Aggregate hash of sorted per-file SHA-256 manifest | `1fc86763f6e47c0e4d534f914268b54be1221afb8e82d6f9c82d4fd84e621dd3` |
| Tokenizer class | `Qwen2Tokenizer` |
| Tokenizer base vocabulary / length | 151,643 / 151,669 |
| `tokenizer.json` SHA-256 | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |

### Exact launch state

The observed process command is preserved in `server-launch.txt`. Important active settings:

- `--enable-custom-logit-processor`
- `--disable-overlap-schedule`
- `--max-running-requests 1`
- `--disable-prefill-cuda-graph`
- `--disable-decode-cuda-graph`
- `--grammar-backend xgrammar`
- `--enable-hierarchical-cache`
- model and tokenizer both `/models`

Every active flag appears in the installed SGLang 0.5.16 executable help. The compose file's old DeepSeek/GGUF launch comment is stale relative to the actual AWQ process.

## Installed serialization contract

Installed source: `/sgl-workspace/sglang/python/sglang/srt/sampling/custom_logit_processor.py`, SHA-256 `5f502753c4b06df62f9df24f6b9cdbad18a02b7cae1e63757d9836b56899cd77`.

Verified contract:

1. `to_str()` returns JSON containing `dill.dumps(cls).hex()`.
2. `_cache_from_str()` decodes JSON and hex, then calls `dill.loads`.
3. `from_str()` invokes the resulting class with a zero-argument constructor.
4. `SamplingBatchInfo` separately copies each request's JSON `custom_params`.
5. `Sampler._preprocess_logits()` applies custom processors before argmax, temperature scaling, softmax, and sampling.
6. The sampler passes the selected request parameter dictionaries to the processor.
7. Installed SGLang does not inject a scheduler request object under `__req__`.

Current processor serialization:

| Field | Value |
|---|---|
| Module | `nram_sglang.processor` |
| Qualname | `NRAMLogitProcessor` |
| Raw dill bytes | 59 |
| Hex characters | 118 |
| JSON payload characters | 134 |
| Raw SHA-256 | `ff2da67fd87cedd80269ddf9bca0e09844d8785a243202412514890556e642e2` |
| API/server source SHA-256 | `ea56fcee0a127abb04f47a3b50add54f16294b0b7cdf8a4f51d17b0eb8bd80a7` in both |
| API raw round trip | Passed |
| SGLang `from_str()` round trip | Passed |
| Zero-argument constructor | Passed |

The pickle contains only a stable global module/name reference. Small payload size is not interpreted as a SGLang limit.

### Historical serialization failure

The pre-existing `sglang_logs.txt` preserves the original full failure. The scheduler attempted to unpickle a class under module `core` and raised:

`ModuleNotFoundError: No module named 'core'`

The direct cause occurs in `CustomLogitProcessor.from_str()` during `SamplingBatchInfo.from_schedule_batch()`. Scheduler shutdown, cancelled requests, and timeout traces are secondary effects.

The current stable `nram_sglang.processor` path removes that exact cause in the currently running images. This is supported by matching source hashes, matching dill versions, round trips in both containers, a successful gateway NRAM request, and the real forced-token result. A cold restart proof was not run during this bounded pass.

## Test results

### Defined/observed gates

| Gate | Result |
|---|---|
| Default `pytest --collect-only` from repository root | Hung in import-time live generation from `test_ab_quick.py`; terminated after bounded observation; exit 1 |
| `tests/` collection with only pytest-asyncio | 282 collected; exit 0 |
| Host Python 3.13 unit suite | 113 passed, 7 failed because declared `openai` package is absent |
| Frozen isolated environment without ephemeral PyTorch | 4 skipped and 3 collection errors; no pass inferred |
| Non-GPU suite with complete ephemeral test dependencies | 275 passed, 5 failed; exit 1 |
| Repository GPU suite | 2 failed in its direct run; exit 1 |
| Unified `tests/` JUnit run | 273 passed, 9 failed, 0 skipped, 0 errors, 2 warnings; exit 1 |
| `npm test` | Fails: `Error: no test specified`; exit 1 |
| `docker compose config --quiet` | Passed; exit 0 |
| Lint | No configured command; unavailable, not passed |
| Type checking | No configured command; unavailable, not passed |
| Node/package build | No configured command; unavailable, not passed |
| Streamlit integration suite | No repository-defined automated command; unavailable, not passed |

JUnit: `artifacts/nram_v5_verification/phase0-20260727T180644Z-7aa1285/test-results.xml`.

### Reconstruction of the four reported failures

The current worktree has 282 tests. The historical claim was 269/273. The difference is explained by drift:

- `tests/integration/test_hallucination_guard.py` is an untracked pre-task file containing nine passing tests; removing those gives 273.
- `tests/adversarial/test_req_injection.py` was modified before this task to invert two valid negative controls into failing unsafe-injection assertions.
- Running all async suites before unit tests exposes three event-loop order failures; the narrower historical order did not.

The four historically applicable failures are therefore:

| # | Exact node | Reproduced result | Root cause/current interpretation |
|---|---|---|---|
| 1 | `tests/adversarial/test_remediation_verification.py::TestDefect1Remediation::test_req_is_injected_into_custom_params` | Failed: `__req__` absent | Test expects an unsupported client-to-server object injection. Installed SGLang does not inject `__req__`; this also proves request-history controls are not wired. Repair must use a supported scheduler signal or pinned server hook, not JSON object injection. |
| 2 | `tests/adversarial/test_remediation_verification.py::TestDefect3Remediation::test_moe_synthesis_payload_has_custom_processor` | Failed: `__req__` absent | Same contract mismatch on MoE synthesis. Static processor presence is not real MoE or request-history state. |
| 3 | `tests/adversarial/test_remediation_verification.py::TestIntegrationVerification::test_full_nram_pipeline_works` | Failed: `__req__` absent | The claimed full pipeline depends on nonexistent state injection. The test's expected implementation is invalid, while the missing behavior is real. |
| 4 | `tests/gpu/test_forced_token_proof.py::test_forced_token` | Failed: `httpx.ConnectError` | Test targets unpublished host port 30000 and stale model alias. It also has stale payload/path assumptions and does not provide required telemetry. It is not a valid acceptance test. |

Phase 0 does not resolve these failures because production and test repair is explicitly out of scope.

### Additional current failures

| Count | Cause |
|---:|---|
| 2 | Pre-task edits in `test_req_injection.py` now demand JSON serialization of `__req__`, conflicting with the installed contract and security boundary. |
| 3 | `test_existing_llm_handler.py` uses `asyncio.get_event_loop()` after pytest-asyncio has cleared the current loop; suite-order-sensitive test design. |

## Real runtime evidence

### Vanilla determinism

Two sequential requests used:

- baseline alias `qwen3-14b-awq-baseline`
- prompt `Reply with one concise sentence defining deterministic inference.`
- seed 271828
- temperature 0.0
- top-p 1.0
- max tokens 24
- explicit empty tools
- no NRAM options

Both returned HTTP 200 and identical content with SHA-256 `c0f0a0d4fed0200c99a06dcb0af2ccb40f162704a73bc008a752af0d25a75b7b`. Envelope hashes differed because response IDs differed. This proves content-level deterministic replay for this one prompt/settings pair only.

### Gateway NRAM smoke

A request to `nram-qwen3-14b-awq` returned HTTP 200 and content `Precise, deliberate, transformative.` This proves the alias reaches the live server and the current class can deserialize. API-produced telemetry does not prove processor invocation or effect.

### Forced-token control

The controlled proof used the real shared `NRAMLogitProcessor` from the API container against internal SGLang on the RTX 4090:

| Field | Disabled control | Controlled |
|---|---|---|
| Prompt | `Say one word.` | Same |
| Seed | 271828 | Same |
| Temperature / top-p | 1.0 / 1.0 | Same |
| Max tokens | 1 | 1 |
| Processor | Disabled | Enabled |
| Mask | None | All 151,936 config IDs except 11064 |
| HTTP status | 200 | 200 |
| Output | `Hello` | ` proof` |

Token 11064 decodes to ` proof` with the live tokenizer. Configuration hash: `699fbcca3a76e66e6be54fd11e3e64cbcbae12bc66a3573af9e4e6ce05f66b93`.

This proves class deserialization and pre-sampling custom-logit execution. It does not provide pre/post top-k, exact numeric logit deltas, invocation count, or processor-emitted request telemetry; the full forced-token acceptance criterion remains unmet.

## Runtime wiring matrix

| Component | Entry point | Runtime hook | Input signal | Intervention | Telemetry | Controlled proof | Status |
|---|---|---|---|---|---|---|---|
| Persona/request routing | `POST /v1/chat/completions` | API route and SGLang engine | Model alias and NRAM options | Select route/profile | API logs/response | Contract mocks; one live NRAM smoke | `IMPLEMENTED_NOT_PROVEN` |
| Tokenizer handling | API startup/engine | `TokenBiasCompiler` | `/models` tokenizer | Compile lexemes/concepts/phrases | Compiler diagnostics only | Hash and identity recorded; stale GPU test found | `IMPLEMENTED_NOT_PROVEN` |
| Evidence ledger | Workflow/database routes | No decoding hook identified | Workflow evidence | Storage/display | Database events | No controlled decode proof | `NOT_WIRED` |
| Concept capsules | Engine payload | Custom logit processor | Client concept forms compiled to IDs | Static token bias with ramp | API count only | CPU tests only | `IMPLEMENTED_NOT_PROVEN` |
| Grammar | Response format path | SGLang grammar backend | JSON schema/regex/EBNF | Vocabulary constraint | None captured | Contract mocks only | `IMPLEMENTED_NOT_PROVEN` |
| Phrase trie | Engine payload | Custom logit processor | Forbidden phrase token IDs plus output suffix | Mask completion token | None | CPU tests only; live request state absent | `NOT_WIRED` |
| Source n-gram blocker | Engine payload | Custom logit processor | Source n-grams plus output suffix | Mask 8-token continuation | None | CPU tests only; live request state absent | `NOT_WIRED` |
| Hard token injection | Engine payload | Custom logit processor | Forbidden/static token IDs | `-inf` mask | None | Real forced-token proof | `PROVEN` |
| Soft token injection | Engine payload | Custom logit processor | Positive/negative IDs and biases | Add/subtract bounded logits | None | Unit tests only | `IMPLEMENTED_NOT_PROVEN` |
| Entropy controller | Engine payload | Simplified processor scaling | Current logits and phase config | Divide logits by proportional scale | None | CPU tests only; no PID/runtime state proof | `IMPLEMENTED_NOT_PROVEN` |
| Semantic novelty controller | None found | None | None | None | None | None | `NOT_IMPLEMENTED` |
| Hallucination/evidence guard | Definition/helper only | No generation hook | Keyword/regex checks | Post-output flags/sanitize | Helper result | CPU tests only | `NOT_WIRED` |
| Multi-vector logit controller | Definition only | None | CPU tensors | Tensor combination | None | CPU tests only | `NOT_WIRED` |
| DExperts | Definition only | None | Mock logits/models | Static tensor equation | None | Mock CPU tests only | `NOT_WIRED` |
| Activation addition | Definitions and generic hooks | Not attached to SGLang loaded model | CPU hidden tensors | Vector addition | None | CPU tests only | `NOT_WIRED` |
| Conceptor steering | Definition only | None | CPU hidden tensors | Projection helper | None | CPU tests only | `NOT_WIRED` |
| Hidden-state probes | Definition only | None | CPU hidden tensors | Score helper | None | CPU tests only | `NOT_WIRED` |
| Closed-loop latent steering | Definition only | None | Probe helper output | Controller state helper | None | CPU tests only | `NOT_WIRED` |
| Soft prompts/ReFT | No runtime symbols | None | None | None | None | None | `NOT_IMPLEMENTED` |
| Branch-and-tournament | Agent workflow judge stage | Ordinary sequential workflow calls | Expert prose | Judge prompt selects concepts | Workflow result | No divergent candidate/state proof | `NOT_WIRED` |
| Attention-head analysis/gating | No runtime symbols | None | None | None | None | None | `NOT_IMPLEMENTED` |
| KV-cache intervention/firewall | Hierarchical cache is infrastructure only | No intervention hook | None | None | SGLang cache logs only | None | `NOT_IMPLEMENTED` |

## Findings

| Severity | Confidence | Location | Finding and evidence |
|---|---:|---|---|
| Critical | High | `core/steering/serialization.py:47-51`, installed SGLang sampler | Repository states SGLang injects `__req__`; installed 0.5.16 source proves it does not. Output-history controls are dormant. |
| Critical | High | Representation/DExperts controller files | Definitions have no non-test runtime imports or live model hook. Claims of ActAdd, conceptors, probes, closed loop, multi-vector, and DExperts are not live features. |
| High | High | `tests/gpu/test_forced_token_proof.py` | GPU proof targets stale unpublished endpoint/model/path and is not telemetry-complete. It cannot support acceptance claims. |
| High | High | `test_ab_quick.py` | Module performs live requests at import, causing default pytest collection to execute generation and appear hung. |
| High | High | `pyproject.toml`, `uv.lock`, tests | Lock/manifest drift: dill declared but missing from frozen environment; PyTorch required by tests but undeclared. |
| High | High | `compose.yaml` | Image remains `latest`; current digest is known but not declared. Rebuild is not reproducible. |
| High | High | `compose.yaml` comments, entrypoint comments | Stale DeepSeek/GGUF documentation conflicts with active Qwen3 AWQ process. |
| High | High | `api/routes.py:427-443` | “NRAM telemetry” is synthesized from request/response fields and does not prove processor invocation or measured effects. |
| Medium | High | `nram_sglang/processor.py:251-269` | Claimed PID entropy controller is proportional-only and stateless; no integral/derivative terms or telemetry. |
| Medium | High | `core/agentic/document_innovation_workflow.py` | “Tournament” is a judge prompt over expert analyses, not configured divergent branch generation from junction state. |
| Medium | High | Unified test run | Three unit tests are suite-order-sensitive because they assume a current asyncio event loop. |
| Medium | High | Tokenizer evidence | Repository GPU test loads stale DeepSeek tokenizer with different vocabulary from live Qwen snapshot. |
| Medium | Medium | API/SGLang dependencies | API and SGLang use different transformers/tokenizers versions; exact tokenizer file hashes match the mounted artifact, but library behavior compatibility needs contract coverage. |

## Atomic acceptance inventory

Statuses are limited to `PROVEN`, `IMPLEMENTED_NOT_PROVEN`, `NOT_WIRED`, `NOT_IMPLEMENTED`, `BLOCKED`, and `UNKNOWN`.

### Intake and Phase 1

| ID | Atomic criterion | Status | Evidence/note |
|---|---|---|---|
| INTAKE-001 | Canonical governing specification captured completely with provenance | `PROVEN` | Canonical file created with starting SHA/dirty state |
| INTAKE-002 | No production behavior or existing test changed in Phase 0 | `PROVEN` | Task-created path inventory only |
| P1-001 | Exact branch, SHA, and dirty state recorded | `PROVEN` | Git artifact |
| P1-002 | Baseline `6b92283` ancestry recorded without reset | `PROVEN` | Merge-base exit 0 |
| P1-003 | Host OS recorded | `PROVEN` | Environment artifact |
| P1-004 | WSL/Ubuntu recorded | `PROVEN` | Environment artifact |
| P1-005 | Docker Desktop/Engine/Compose recorded | `PROVEN` | Environment artifact |
| P1-006 | GPU, UUID, driver, CUDA, VRAM recorded | `PROVEN` | NVIDIA-SMI host/container |
| P1-007 | Host/API/SGLang/test Python recorded | `PROVEN` | Dependency artifact |
| P1-008 | SGLang version/source commit recorded | `PROVEN` | Package plus image labels |
| P1-009 | dill versions recorded | `PROVEN` | API/server both 0.4.1 |
| P1-010 | Model ID/snapshot/artifact hashes recorded | `PROVEN` | Full 9.4 GB hash pass |
| P1-011 | Tokenizer ID and hashes recorded | `PROVEN` | Tokenizer metadata and hashes |
| P1-012 | Docker image IDs/digests recorded | `PROVEN` | Image inspection |
| P1-013 | Exact launch command recorded | `PROVEN` | Process list |
| P1-014 | Relevant environment recorded with secrets omitted | `PROVEN` | Allowlisted values only |
| P1-015 | Installed executable flags confirmed | `PROVEN` | SGLang `--help`, exit 0 |
| P1-016 | Concurrency one active | `PROVEN` | Process command |
| P1-017 | Overlap scheduling disabled | `PROVEN` | Process command and server args |
| P1-018 | Fixed seed/sampling recorded for runtime checks | `PROVEN` | Runtime evidence |
| P1-019 | Repeated vanilla deterministic content | `PROVEN` | Two equal outputs/hashes |

### Phase 2 serialization and forced token

| ID | Atomic criterion | Status | Evidence/note |
|---|---|---|---|
| P2-001 | Installed `to_str` contract verified | `PROVEN` | Installed source lines 36-39 |
| P2-002 | Installed `from_str` zero-arg behavior verified | `PROVEN` | Installed source lines 41-44 |
| P2-003 | Class-vs-instance distinction verified | `PROVEN` | 59-byte global class reference |
| P2-004 | Stable top-level import path | `PROVEN` | `nram_sglang.processor.NRAMLogitProcessor` |
| P2-005 | Package exists in API and SGLang containers | `PROVEN` | Matching hashes |
| P2-006 | Python/SGLang/dill versions captured | `PROVEN` | Dependency artifact |
| P2-007 | Raw/hex/payload lengths captured | `PROVEN` | 59/118/134 |
| P2-008 | Pickletools disassembly captured | `PROVEN` | Serialization artifact |
| P2-009 | API raw round trip | `PROVEN` | Successful |
| P2-010 | SGLang `from_str` round trip | `PROVEN` | Successful |
| P2-011 | Exact HTTP payload live deserialization | `PROVEN` | Gateway NRAM and forced request HTTP 200 |
| P2-012 | Historical traceback preserved | `PROVEN` | `sglang_logs.txt` |
| P2-013 | Historical unavailable-`core` root cause removed | `PROVEN` | Stable path plus live forced request |
| P2-014 | Request state reaches processor through supported mechanism | `NOT_WIRED` | Installed version passes only JSON custom params |
| P2-015 | Cold restart repeat | `UNKNOWN` | Not performed in bounded pass |
| P2-016 | Disabled forced-token control | `PROVEN` | Control output `Hello` |
| P2-017 | Live forced sampled token | `PROVEN` | Controlled output token 11064 |
| P2-018 | Pre/post top-k and exact logit deltas | `NOT_IMPLEMENTED` | No telemetry |
| P2-019 | Invocation count/request-ID/config telemetry | `NOT_IMPLEMENTED` | No processor events |

### Phases 3-5 structural and logit controls

| ID | Atomic criterion | Status | Evidence/note |
|---|---|---|---|
| P3-001 | Complete runtime wiring matrix exists | `PROVEN` | This document |
| P4-001 | Czech multi-token forbidden phrase live proof | `NOT_WIRED` | Request history absent |
| P4-002 | English multi-token forbidden phrase live proof | `NOT_WIRED` | Request history absent |
| P4-003 | Trie suffix/only-completion behavior | `IMPLEMENTED_NOT_PROVEN` | Processor helper and CPU tests only |
| P4-004 | Overlap/tokenizer variants/streaming phrase proof | `NOT_WIRED` | No live proof |
| P4-005 | Phrase masked IDs/logits telemetry | `NOT_IMPLEMENTED` | No telemetry |
| P4-006 | Exact 8-token source continuation prevention | `NOT_WIRED` | Request history absent |
| P4-007 | N-gram boundaries/punctuation/negative/disabled controls | `NOT_WIRED` | No live proof |
| P4-008 | Mask is applied before sampling | `IMPLEMENTED_NOT_PROVEN` | Generic forced mask proven; n-gram path not proven |
| P4-009 | Overlap before/after metric | `NOT_IMPLEMENTED` | No evaluation result |
| P4-010 | Grammar composes with phrase/source masks | `UNKNOWN` | Not tested live |
| P5-001 | Actual next-token entropy measured | `IMPLEMENTED_NOT_PROVEN` | Processor computes entropy; no telemetry |
| P5-002 | PID P/I/D terms implemented/logged | `NOT_IMPLEMENTED` | Runtime processor is proportional-only |
| P5-003 | Low/medium/high target controlled runs | `NOT_IMPLEMENTED` | No experiment |
| P5-004 | Stable target movement/no NaN | `UNKNOWN` | No live evidence |
| P5-005 | Distinct from fixed temperature | `UNKNOWN` | No paired test |
| P5-006 | Entropy state reset per request | `UNKNOWN` | Runtime control is stateless; intended criterion unmet |
| P5-007 | Dynamic concepts include bilingual forms/synonyms | `IMPLEMENTED_NOT_PROVEN` | Engine compiles forms to token IDs |
| P5-008 | Concept phase/ramp/max-use behavior | `IMPLEMENTED_NOT_PROVEN` | Ramp exists; max-use not on live path |
| P5-009 | Concept rank/probability/logit pre/post | `NOT_IMPLEMENTED` | No telemetry |
| P5-010 | Concept same-seed local controlled effect | `NOT_IMPLEMENTED` | No live ablation |
| P5-011 | Hard injection direct logit proof | `PROVEN` | Real forced mask/token |
| P5-012 | Soft injection window/schedule direct proof | `IMPLEMENTED_NOT_PROVEN` | Static bias unit tests only |
| P5-013 | Multi-vector identity/coefficient/norm/delta/clipping | `NOT_WIRED` | Definition/CPU tests only |
| P5-014 | Multi-vector sign reversal and zero controls | `NOT_WIRED` | No live hook |

### Phases 6-10 advanced controls

| ID | Atomic criterion | Status | Evidence/note |
|---|---|---|---|
| P6-001 | Distinct base/expert/anti-expert distributions | `NOT_WIRED` | No live models/hooks |
| P6-002 | Documented DExperts equation on live tensors | `NOT_WIRED` | Mock tensor helper only |
| P6-003 | Real checkpoint/adapter execution | `NOT_IMPLEMENTED` | None configured |
| P6-004 | Tokenizer/vocabulary/decode alignment | `NOT_IMPLEMENTED` | No experts |
| P6-005 | Zero coefficient reproduces base | `NOT_WIRED` | CPU mock only |
| P6-006 | Alpha/beta signed effects | `NOT_WIRED` | CPU mock only |
| P6-007 | DExperts latency/VRAM metadata | `NOT_IMPLEMENTED` | No run |
| P7-001 | Verified SGLang hidden-state/model hook | `NOT_IMPLEMENTED` | None found |
| P7-002 | Loaded Qwen named module paths enumerated | `NOT_IMPLEMENTED` | None captured |
| P7-003 | ActAdd artifact provenance/hash/dataset/layer/norm | `NOT_IMPLEMENTED` | No valid artifact evidence |
| P7-004 | Live `h'=h+alpha*v` hook proof | `NOT_WIRED` | Generic CPU hook only |
| P7-005 | ActAdd absent/zero/positive/negative/control-vector controls | `NOT_WIRED` | No live model hook |
| P7-006 | Multi-depth real-layer ActAdd tests | `NOT_IMPLEMENTED` | No mapped layers |
| P7-007 | Real conceptor subspace/projection | `NOT_WIRED` | Definition only |
| P7-008 | Conceptor construction/aperture/composition controls | `NOT_WIRED` | CPU tests only |
| P7-009 | Probe dataset/splits/leakage controls | `NOT_IMPLEMENTED` | None documented |
| P7-010 | Probe AUROC/precision/recall/F1/calibration | `NOT_IMPLEMENTED` | None reported |
| P7-011 | Closed-loop aligned causal chain | `NOT_WIRED` | Definition only |
| P8-001 | Block-level semantic evaluation at intervals | `NOT_IMPLEMENTED` | None on decode path |
| P8-002 | Embedding model/version/scores/threshold/action | `NOT_IMPLEMENTED` | None recorded |
| P8-003 | Threshold changes subsequent decoding | `NOT_WIRED` | No closed loop |
| P8-004 | Honest chunk stop/evaluate/resume | `NOT_IMPLEMENTED` | None |
| P8-005 | Evidence-guard labeled dataset | `NOT_IMPLEMENTED` | Keyword/regex helper only |
| P8-006 | Evidence confusion matrix/precision/recall/F1/FPR/FNR | `NOT_IMPLEMENTED` | None |
| P9-001 | Multiple distinct branches per junction | `NOT_IMPLEMENTED` | Workflow judge is not branch generation |
| P9-002 | Branch prompts/seeds/sampling retained | `NOT_IMPLEMENTED` | None |
| P9-003 | Required branch scorers executed/raw scores | `NOT_IMPLEMENTED` | None |
| P9-004 | Deterministic winner arithmetic | `NOT_IMPLEMENTED` | None |
| P9-005 | Continue from selected branch state | `NOT_IMPLEMENTED` | None |
| P9-006 | Disabled trajectory and weight-change fixture | `NOT_IMPLEMENTED` | None |
| P10-001 | ReFT status | `NOT_IMPLEMENTED` | No symbols |
| P10-002 | Soft-prompt capsules status | `NOT_IMPLEMENTED` | No symbols |
| P10-003 | Attention-head analysis/gating status | `NOT_IMPLEMENTED` | No symbols |
| P10-004 | KV-cache imprint/forgetting/firewall status | `NOT_IMPLEMENTED` | Hierarchical cache is infrastructure only |
| P10-005 | GPU-native semantic-control status | `NOT_IMPLEMENTED` | No implementation found |

### Phases 11-13 and final evidence

| ID | Atomic criterion | Status | Evidence/note |
|---|---|---|---|
| P11-001 | Same-seed paired harness for all named conditions | `NOT_IMPLEMENTED` | Existing A/B scripts do not meet design |
| P11-002 | Full-minus-each-major-component harness | `NOT_IMPLEMENTED` | None |
| P11-003 | Fixed bilingual/multi-task prompt suite | `IMPLEMENTED_NOT_PROVEN` | Some prompts exist; governing suite absent |
| P11-004 | Required generation metrics | `NOT_IMPLEMENTED` | Partial metrics only |
| P11-005 | Direct intermediate-effect metrics | `NOT_IMPLEMENTED` | No runtime events |
| P11-006 | Multiple seeds/raw outputs | `IMPLEMENTED_NOT_PROVEN` | Historical outputs exist without complete traceability |
| P11-007 | n/mean/median/dispersion/paired effects/CIs | `NOT_IMPLEMENTED` | None |
| P11-008 | Paired significance/multiple comparison handling | `NOT_IMPLEMENTED` | None |
| P11-009 | Negative result retention/no cherry-picking | `UNKNOWN` | No complete protocol/results |
| P12-001 | Unit suite zero failures | `BLOCKED` | Unified run fails |
| P12-002 | Contract suite zero failures | `PROVEN` | 29 passed in non-GPU run |
| P12-003 | Adversarial suite zero failures | `BLOCKED` | Five current failures |
| P12-004 | Live SGLang GPU suite | `BLOCKED` | Repository GPU test invalid/failing; separate partial proof succeeds |
| P12-005 | Streamlit/API integration suite | `UNKNOWN` | API smoke passed; no Streamlit gate |
| P12-006 | Serialization round trips | `PROVEN` | API and SGLang containers |
| P12-007 | Deterministic replay | `PROVEN` | One fixed vanilla case |
| P12-008 | Type checking | `UNKNOWN` | No command defined |
| P12-009 | Lint | `UNKNOWN` | No command defined |
| P12-010 | Package/production build | `UNKNOWN` | No build command defined; compose config passes |
| P12-011 | Original four failure repairs/reruns | `BLOCKED` | Reproduced; repair out of Phase-0 scope |
| P12-012 | Zero unexplained failures | `BLOCKED` | Nine unified failures |
| P13-001 | Versioned artifact directory | `PROVEN` | Phase-0 run directory exists |
| P13-002 | Environment/manifest/commands/launch/dependencies | `PROVEN` | Present |
| P13-003 | Test results XML | `PROVEN` | Present with failures |
| P13-004 | Serialization diagnostics | `PROVEN` | Present |
| P13-005 | Runtime events JSONL | `NOT_IMPLEMENTED` | Processor emits no events |
| P13-006 | Token-level ablation CSV | `NOT_IMPLEMENTED` | No telemetry table |
| P13-007 | Representation ablation CSV | `NOT_IMPLEMENTED` | No live representation control |
| P13-008 | Generation metrics/human template/raw/plots/failures | `NOT_IMPLEMENTED` | Future verification work |
| P13-009 | Required five final documentation files | `NOT_IMPLEMENTED` | Intentionally not fabricated in Phase 0 |
| FINAL-001 | Exact SHA/tree/environment/model evidence | `PROVEN` | Phase-0 artifacts |
| FINAL-002 | Complete counts and four-failure resolution | `BLOCKED` | Counts/reproduction proven; resolution not in scope |
| FINAL-003 | Serialization root cause/proof | `PROVEN` | Historical root cause and current path proof |
| FINAL-004 | Wiring matrix/direct evidence | `IMPLEMENTED_NOT_PROVEN` | Matrix exists; most direct proofs absent |
| FINAL-005 | Ablation effects/confidence intervals | `NOT_IMPLEMENTED` | None |
| FINAL-006 | Exact reproduction commands/raw paths/limitations | `IMPLEMENTED_NOT_PROVEN` | Phase-0 commands/limitations exist; full study absent |
| FINAL-007 | Scientific validation verdict above partial | `BLOCKED` | Required evidence absent |
| FINAL-008 | arXiv artifact readiness | `BLOCKED` | Required evidence absent |

## Limitations and blockers

1. The tree was heavily dirty before intake; historical results cannot be attributed to a single clean commit without preserving or committing user work.
2. The four reported failures can be reconstructed, but current drift creates nine unified failures.
3. Cold restart serialization was not run because Phase 0 did not restart an existing healthy service.
4. No hidden-state hook or model-module enumeration was available through the active server runtime.
5. No processor-level telemetry provides pre/post logits, request ID, invocation count, or configuration hash.
6. The separate forced-token request is real GPU evidence but not the complete Phase-2 acceptance proof.
7. Default pytest discovery is unsafe due to import-time live generation in `test_ab_quick.py`.
8. Project dependency metadata cannot create a complete test environment without ephemeral dill, cloudpickle, PyTorch, pytest, and pytest-asyncio additions.
9. Lint, type-check, production build, and Streamlit integration commands are not defined; they are unavailable, not passed.
10. Image declaration, stale comments, and tokenizer test paths are not reproducible without repair.

## Bounded implementation sequence

1. **Stabilize the tree and gates.** Preserve user work in a coherent commit/worktree decision; move import-time A/B execution behind a main guard; define test dependencies; make async tests order-independent; restore negative contract tests that reject client request-object injection. Gate: deterministic 282-test collection and exact failure inventory.
2. **Pin runtime identity.** Declare the observed SGLang digest/version, reconcile active AWQ comments/configuration, and pin compatible Python/dill/transformers/tokenizers. Gate: clean compose config, image digest verification, API/server package contract.
3. **Repair supported request-state access.** Do not serialize `__req__` from clients. Inspect official 0.5.16 extension points for output-token history; if absent, implement the smallest version-pinned server-side patch that supplies scheduler state to the trusted processor. Gate: cold restart, exact HTTP round trip, concurrency one, request isolation, invocation telemetry.
4. **Add processor telemetry and forced proof.** Emit request ID, config hash, invocation count, pre/post top-k, exact deltas, sampled token, and disabled control. Gate: live RTX 4090 deterministic forced-token proof through the public API.
5. **Repair structural controls vertically.** Implement trie/source blocker against real output history and tokenizer variants, then grammar composition. Gate: Czech/English positive, negative, overlap, punctuation, streaming, disabled controls on GPU.
6. **Repair and prove logits controls.** Replace proportional “PID” claims with a documented controller or rename truthfully; prove entropy targets, soft/hard injection, capsules, and multi-vector behavior with direct telemetry and reset tests.
7. **Decide advanced scope before code.** Mark DExperts, ActAdd, conceptors, probes, closed loop, ReFT, head gating, and KV interventions unsupported unless official hooks and experiment designs are selected. Do not treat CPU helper tests as implementation.
8. **Implement one representation slice only after hook verification.** Enumerate actual loaded Qwen modules, create provenance-hashed vectors, add request-isolated hooks, and prove hidden-state/downstream effects before adding conceptors/probes.
9. **Implement real DExperts and tournament only as separate vertical slices.** Require real expert distributions/equation and explicit branch generation/scoring/selection respectively.
10. **Build causal ablation and reproducibility harness.** Pre-register prompts, seeds, conditions, metrics, CIs/statistics, negative results, and raw artifact schema. Gate: full stack plus leave-one-out results with traceability.
11. **Run complete release-science gates.** Unit, contract, adversarial, API, Streamlit, live GPU, serialization restart, deterministic replay, lint, type, build, and artifact validation must all pass before any stronger scientific verdict.

No commit should be created until the bounded repairs and required gates pass in a coherent tree.
