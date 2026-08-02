# NRAM v5 R5 readiness and scientific controls

## Independent regimes

Regime A (06b) and Regime B (14b) are **independent scientific regimes**.
Their contrast must not be interpreted as a pure model-scale effect. They
also differ in model family/configuration, quantization, request endpoint,
prompt/template policy, serving instance, and adapter deployment context.
Any cross-regime comparison is descriptive only unless those confounds are
separately controlled.

## Runtime commands

The causal script now has a fail-closed preflight path that does not load the
toxicity evaluator or start the experiment:

```powershell
python scripts/dexperts_causal_ablation.py --regime 14b --preflight
python scripts/dexperts_causal_ablation.py --regime 06b --preflight
```

The implementation is in [`scripts/dexperts_r5_readiness.py`](../scripts/dexperts_r5_readiness.py)
and writes `artifacts/dexperts/r5/preflight_<regime>.json`.
Health alone is never treated as proof of model revision, tokenizer
compatibility, adapter behavior, or output schema. Those checks remain
`BLOCKED` until the server exposes the required evidence and deterministic
control probes pass.

## 06b infrastructure plan — do not launch yet

Pinned model and tokenizer revision:

```text
Qwen/Qwen3-0.6B-Base
revision: da87bfb608c14b7cf20ba1ce41287e8de496c0cd
tokenizer: same pinned revision
```

Exact isolated launch command (CUDA/PowerShell or WSL shell):

```text
python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B-Base --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd --host 127.0.0.1 --port 30001 --dtype float16 --tp 1 --context-length 2048
```

Before launch, the SGLang image must contain the NRAM/DExperts integration
and load the following adapter directories:

```text
expert:      artifacts/dexperts/adapters/nontoxic
anti-expert: artifacts/dexperts/adapters/toxic
```

Both adapter configs declare the same 0.6B base revision and LoRA target
modules. The current adapter roots contain `adapter_model.safetensors`; the
preflight records their SHA-256 tree hashes. The current hashes observed
locally are:

```text
nontoxic: f2b5aef5961cc1c852820d0dd17359e6320f4330350f5ea607f1460f37c7628e
toxic:    889f528baa1070b7c5f5b7720b3ae0cfdfa5349f493abfd656fe515a5f3c0471
```

The adapters declare the pinned 06b base, so compatibility passes for 06b.
Compatibility is intentionally blocked for 14b until adapter transfer to the
14b model is explicitly validated; adapter names alone are not proof.

Port isolation is 30001 for 06b; the existing 14b service remains on 30000.
Health procedure:

```powershell
Invoke-WebRequest http://127.0.0.1:30001/health
Invoke-WebRequest http://127.0.0.1:30001/v1/models
python scripts/dexperts_causal_ablation.py --regime 06b --preflight --api-url http://127.0.0.1:30001/v1/completions
```

The preflight must report the pinned model identity/revision, tokenizer
fingerprint, both adapter hashes/roles, output schema, and activation,
deactivation, and swapped-adapter control verdicts before any 06b run.

Estimated VRAM: the pinned 0.6B model in FP16 is approximately 1.5--2.5 GiB
for weights plus runtime overhead; reserve at least 6 GiB GPU VRAM for
CUDA graphs, KV cache, adapters, and batching. This is an estimate, not a
measurement; record peak VRAM during the first approved preflight smoke.

## Batching and parity

[`scripts/dexperts_r5_batch_runner.py`](../scripts/dexperts_r5_batch_runner.py)
length-buckets source sequences, supports microbatch sizes 8/16/32 and a
token budget, and records throughput, retry count, effective batch size and
server-reported peak VRAM. It never persists full-vocabulary logits.
Teacher-forced mechanistic prefill must be implemented by the server-side
runtime and return only target-token delta, LR, KL, JS and TV scalars.

[`scripts/dexperts_r5_batch_parity.py`](../scripts/dexperts_r5_batch_parity.py)
requires two runtime fixtures containing the same 16 sequence IDs in the same
order, compares batch 1 vs batch 8, and fails if scalar differences exceed
the configured BF16/FP32 tolerance or control verdicts differ.

## Offline gates and authorization

Run the corrected power/gate recomputation without invoking SGLang:

```powershell
python scripts/dexperts_r5_readiness.py --offline
```

The existing evaluator artifact has ROC-AUC 0.7533 and correct class ordering,
but ECE 0.2986 and `all_pass=false`; evaluator calibration is therefore not
accepted as complete. The existing corrected adapter discrimination artifact
passes ROC-AUC 0.8411, lower bootstrap CI 0.8012, and both expected LR signs.
The Phase 4 artifact does not contain source text, so exact/near-duplicate
leakage cannot be certified and remains blocked.

Consequently, causal ablation is **not authorized**. No 50-prompt ablation is
run and no 06b server is provisioned. The two-prompt-per-class 14b smoke is
also withheld until runtime preflight, batch parity, evaluator calibration,
and leakage audit pass.
