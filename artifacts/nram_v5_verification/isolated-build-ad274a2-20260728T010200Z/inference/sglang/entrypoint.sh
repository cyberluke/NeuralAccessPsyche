#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# SGLang entrypoint for NeuralAccessPsyche (NRAM).
#
# Launches SGLang 0.5.16 backed by the LOCAL Qwen3-14B-AWQ snapshot
# mounted read-only at /models by docker compose. The model is NEVER
# downloaded or copied here.
#
# Every flag below can be overridden via a matching environment variable so
# the compose file (or a human) can tune the launch without editing this
# script. Defaults match the NRAM runtime contract.
# ---------------------------------------------------------------------------
set -euo pipefail

# --- Tunable launch parameters (env-overridable) ---------------------------
MODEL_PATH="${MODEL_PATH:-/models}"
# Directory holding the proper HF tokenizer (tokenizer.json / tokenizer_config.json).
# Defaults to /models — the same read-only mount as the model. For GGUF models
# this fixes the llguidance vocab-size mismatch. For safetensors/AWQ models
# the tokenizer lives alongside the weights.
TOKENIZER_PATH="${TOKENIZER_PATH:-/models}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nram-qwen3-14b-awq}"
# Model type: gguf | awq | auto. GGUF needs explicit --load-format/--quantization;
# AWQ/safetensors auto-detect from config.json.
MODEL_TYPE="${MODEL_TYPE:-awq}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
QUANTIZATION="${QUANTIZATION:-awq}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.72}"
GRAMMAR_BACKEND="${GRAMMAR_BACKEND:-xgrammar}"
# Reasoning parser: split the model's reasoning chain into a separate
# reasoning_content field so the main content contains only the final answer.
# Use deepseek-r1 for DeepSeek-R1 models, qwen3 for Qwen3 models, none to disable.
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
# Thinking mode control (Qwen3-specific):
# ENABLE_THINKING=true  → model generates  reasoning (slower, better quality)
# ENABLE_THINKING=false → model skips reasoning entirely (faster, direct answers)
ENABLE_THINKING="${ENABLE_THINKING:-false}"
# Hierarchical cache: offload KV cache to system RAM (128 GB DDR5) when VRAM is
# insufficient for the full context. Enables 32k context on a 24 GB GPU.
HICACHE_RATIO="${HICACHE_RATIO:-2.0}"
HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through_selective}"
HOST="${SGLANG_HOST:-0.0.0.0}"
PORT="${SGLANG_PORT:-30000}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"

# Fail before model loading when the pinned runtime contract drifts. The
# output-history hook is an official in-process SGLang mechanism; no request
# object is accepted from JSON custom_params.
python3 - <<'PY'
from importlib.metadata import version
from pathlib import Path

expected = "0.5.16"
actual = version("sglang")
if actual != expected:
    raise SystemExit(f"NRAM requires sglang=={expected}; found {actual}")

root = Path("/sgl-workspace/sglang/python/sglang/srt")
schedule = (root / "managers/schedule_batch.py").read_text()
processor = (root / "sampling/custom_logit_processor.py").read_text()
if '"__req__": self' not in schedule:
    raise SystemExit("Pinned SGLang server-side __req__ history hook is absent")
if "dill.dumps(cls).hex()" not in processor or "_cache_from_str(json_str)()" not in processor:
    raise SystemExit("Pinned SGLang dill class-serialization contract drifted")
PY

echo "[nram-sglang] Launching SGLang server"
echo "[nram-sglang]   model-path        = ${MODEL_PATH}"
echo "[nram-sglang]   tokenizer-path    = ${TOKENIZER_PATH}"
echo "[nram-sglang]   served-model-name = ${SERVED_MODEL_NAME}"
echo "[nram-sglang]   load-format       = ${LOAD_FORMAT}"
echo "[nram-sglang]   quantization      = ${QUANTIZATION}"
echo "[nram-sglang]   context-length    = ${CONTEXT_LENGTH}"
echo "[nram-sglang]   mem-fraction      = ${MEM_FRACTION_STATIC}"
echo "[nram-sglang]   grammar-backend   = ${GRAMMAR_BACKEND}"
echo "[nram-sglang]   reasoning-parser  = ${REASONING_PARSER}"
echo "[nram-sglang]   hicache-ratio     = ${HICACHE_RATIO}"
echo "[nram-sglang]   hicache-write     = ${HICACHE_WRITE_POLICY}"
echo "[nram-sglang]   host:port         = ${HOST}:${PORT}"

# ---------------------------------------------------------------------------
# IMPORTANT: every flag here MUST be verified against the chosen image via
#   python3 -m sglang.launch_server --help
# before launch. SGLang renames flags between releases. If a flag has been
# renamed, update the flag name but DO NOT remove the intended capability
# (custom logit processor, grammar backend, etc.).
# ---------------------------------------------------------------------------

# Build the launch arguments array. GGUF-specific flags are only added when
# LOAD_FORMAT=gguf; AWQ/safetensors models auto-detect from config.json.
LAUNCH_ARGS=(
    --model-path "${MODEL_PATH}"
    --tokenizer-path "${TOKENIZER_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --context-length "${CONTEXT_LENGTH}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --enable-custom-logit-processor
    --grammar-backend "${GRAMMAR_BACKEND}"
    --disable-overlap-schedule
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --enable-hierarchical-cache
    --hicache-ratio "${HICACHE_RATIO}"
    --hicache-write-policy "${HICACHE_WRITE_POLICY}"
    --reasoning-parser "${REASONING_PARSER}"
    --host "${HOST}"
    --port "${PORT}"
)

# CUDA graph capture restores decode throughput (~30+ tok/s vs <1 tok/s without
# graphs). The PREVIOUS hang was caused by capturing a LARGE decode batch list
# (bs=1,2,4,8,12,16,24) with insufficient free VRAM. The fix:
#   - keep PREFILL cuda graph disabled (it auto-requires ~4 GB we don't have,
#     and prefill of short prompts is not the bottleneck);
#   - enable DECODE cuda graph with a SMALL, env-configurable batch list so
#     capture is quick and low-VRAM. bs=1..8 covers interactive single-stream
#     use (the console + API mostly run 1-4 concurrent generations).
# Set CUDA_GRAPH_BS_DECODE="" to fully disable decode graphs (safe fallback).
CUDA_GRAPH_BS_DECODE="${CUDA_GRAPH_BS_DECODE-}"
# Prefill graph needs ~4 GB scratch; disable to guarantee reliable startup.
LAUNCH_ARGS+=(--disable-prefill-cuda-graph)
if [ -n "${CUDA_GRAPH_BS_DECODE}" ]; then
    # shellcheck disable=SC2206  # intentional word-splitting of the bs list
    LAUNCH_ARGS+=(--cuda-graph-bs-decode ${CUDA_GRAPH_BS_DECODE})
else
    LAUNCH_ARGS+=(--disable-decode-cuda-graph)
fi

# GGUF-specific flags: only for GGUF models (DeepSeek-R1-Distill-Qwen-7B-GGUF).
# AWQ/safetensors models auto-detect quantization from config.json.
if [ "${MODEL_TYPE}" = "gguf" ]; then
    LAUNCH_ARGS+=(--load-format gguf --quantization gguf)
fi

# Thinking mode control (Qwen3-specific):
# ENABLE_THINKING=false → skip reasoning entirely (faster, direct answers)
# ENABLE_THINKING=true  → full  reasoning (slower, better quality)
if [ "${ENABLE_THINKING}" = "false" ]; then
    LAUNCH_ARGS+=(--default-chat-template-kwargs '{"enable_thinking": false}')
fi

exec python3 -m sglang.launch_server "${LAUNCH_ARGS[@]}" "$@"
