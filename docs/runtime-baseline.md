# Runtime Baseline

Recorded: 2026-07-25  
Branch: `feature/nram-sglang-steering`  
Starting commit: `eaeae0228ed28b9fdf6175b62d9fb2b46b492011`

## Machine Environment

| Item | Value |
|------|-------|
| Windows version | 10.0.26300.8935 |
| WSL version | 2.7.10.0 |
| WSL kernel | 6.18.33.2-2 |
| Ubuntu version | 26.04 LTS (Resolute Raccoon) |
| Docker Desktop version | 4.83.0 (234302) |
| Docker Engine version | 29.6.2 |
| Docker Compose version | v5.3.1 |
| NVIDIA driver version | 591.86 |
| CUDA version (nvidia-smi) | 13.1 |
| GPU | NVIDIA GeForce RTX 4090 |
| Total VRAM | 23028 MiB |
| Available VRAM at baseline | ~10004 MiB (12463 MiB in use by desktop apps) |

## Docker GPU Validation

Image used: `nvidia/cuda:12.8.1-base-ubuntu24.04`  
Digest: `sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2`  
Result: **PASSED** — RTX 4090 visible inside container.

## WSL Docker Integration Note

Docker CLI is **not available** inside WSL2 Ubuntu (`docker` command not found).  
Docker Desktop WSL2 integration for Ubuntu must be enabled in Docker Desktop settings:  
Settings → Resources → WSL Integration → enable Ubuntu.  
All Docker commands run from Windows PowerShell work correctly.

## Local Model

| Item | Value |
|------|-------|
| File | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` |
| Path | `E:\_MODELS\huggingface\hub\DeepSeek-R1-Distill-Qwen-7B-GGUF\` |
| Size | 4.36 GB |
| WSL mount path | `/mnt/e/_MODELS/huggingface/hub/DeepSeek-R1-Distill-Qwen-7B-GGUF/` |

## Git Baseline

```
commit eaeae0228ed28b9fdf6175b62d9fb2b46b492011
HEAD -> main, origin/main, origin/HEAD
Saved progress at the end of the loop
```

## Confirmed Defects (Phase 1)

| # | File | Defect | Status |
|---|------|--------|--------|
| 1 | `core/llm_handler.py` | `process_with_guidance()` returns dict; content stored as dict not string | Confirmed |
| 2 | `core/llm_handler.py` | `get_token_metrics()` is async but called without `await` | Confirmed |
| 3 | `core/llm_handler.py` | `.split()` called on dict (consequence of #1) | Confirmed |
| 4 | `core/config_suggester.py` | Sync `OpenAI` client used with `await` | Confirmed |
| 5 | `api/routes.py` | `request.model` field ignored; hardcoded `gpt-4o` | Confirmed |
| 6 | `api/routes.py` | `request.stream` field ignored; always non-streaming | Confirmed |
| 7 | `core/nram.py` | `np.softmax()` called; NumPy has no `softmax` attribute | Confirmed |
| 8 | `core/llm_handler.py` | Errors converted to fake HTTP 200 chat completions | Confirmed |
| 9 | `core/nram.py` | `process_messages()` randomly corrupts user message content | Confirmed |

## SGLang Image (to be pinned after Phase 4)

```
SGLANG_IMAGE=lmsysorg/sglang:latest  # replace with digest after first successful start
```
