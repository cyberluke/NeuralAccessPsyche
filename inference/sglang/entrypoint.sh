#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# SGLang entrypoint for NeuralAccessPsyche (NRAM).
#
# Launches the SGLang OpenAI-compatible server backed by a LOCAL GGUF model
# mounted read-only at /models by docker compose. The model is NEVER
# downloaded or copied here.
#
# Every flag below can be overridden via a matching environment variable so
# the compose file (or a human) can tune the launch without editing this
# script. Defaults match the NRAM runtime contract.
# ---------------------------------------------------------------------------
set -euo pipefail

# --- Tunable launch parameters (env-overridable) ---------------------------
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf}"
# Directory holding the proper HF tokenizer (tokenizer.json / tokenizer_config.json).
# Defaults to /models — the same read-only mount as the model. For GGUF models
# this fixes the llguidance vocab-size mismatch. For safetensors/AWQ models
# the tokenizer lives alongside the weights.
TOKENIZER_PATH="${TOKENIZER_PATH:-/models}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nram-deepseek-r1-qwen-7b}"
# Model type: gguf | awq | auto. GGUF needs explicit --load-format/--quantization;
# AWQ/safetensors auto-detect from config.json.
MODEL_TYPE="${MODEL_TYPE:-gguf}"
LOAD_FORMAT="${LOAD_FORMAT:-gguf}"
QUANTIZATION="${QUANTIZATION:-gguf}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
GRAMMAR_BACKEND="${GRAMMAR_BACKEND:-llguidance}"
# Reasoning parser: split the model's reasoning chain into a separate
# reasoning_content field so the main content contains only the final answer.
# Use deepseek-r1 for DeepSeek-R1 models, qwen3 for Qwen3 models, none to disable.
REASONING_PARSER="${REASONING_PARSER:-deepseek-r1}"
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
    --enable-hierarchical-cache
    --hicache-ratio "${HICACHE_RATIO}"
    --hicache-write-policy "${HICACHE_WRITE_POLICY}"
    --reasoning-parser "${REASONING_PARSER}"
    --host "${HOST}"
    --port "${PORT}"
)

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
