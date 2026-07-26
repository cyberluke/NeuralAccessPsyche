"""Engine registry — feature-flag-gated selection between legacy and SGLang engines.

The SGLang engine is only constructed when NRAM_ENGINE=sglang. This keeps the
legacy path (and its tests) fully intact and avoids import-time side effects
(httpx client creation, dill serialization) unless explicitly enabled.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Aliases served by the SGLang engine (local models).
SGLANG_ALIASES = {
    "nram-gpt-oss-20b",
    "gpt-oss-20b-baseline",
    "deepseek-r1-qwen-7b-baseline",
    "nram-deepseek-r1-qwen-7b",
    "nram-qwen3-14b-awq",
    "qwen3-14b-awq-baseline",
}

_sglang_engine = None
_sglang_init_failed = False


def sglang_enabled() -> bool:
    """Feature flag: is the SGLang engine active?"""
    return os.environ.get("NRAM_ENGINE", "legacy").lower() == "sglang"


def should_route_to_sglang(model: str) -> bool:
    """Return True only when the flag is on AND the model is an SGLang alias."""
    return sglang_enabled() and model in SGLANG_ALIASES


def get_sglang_engine():
    """Lazily construct the SGLang engine. Returns None if construction fails.

    Construction requires `dill` (for processor serialization) and a reachable
    tokenizer. Without tokenizer, NRAM logit steering is completely disabled.
    We fail-fast if NRAM is enabled but tokenizer is missing.
    """
    global _sglang_engine, _sglang_init_failed

    if _sglang_engine is not None:
        return _sglang_engine
    if _sglang_init_failed:
        return None

    base_url = os.environ.get("SGLANG_BASE_URL", "http://sglang:30000/v1")
    model = os.environ.get("SGLANG_MODEL", "nram-deepseek-r1-qwen-7b")

    tokenizer = _load_tokenizer()
    
    # P0: Fail-fast if NRAM is enabled but tokenizer is missing
    if sglang_enabled() and tokenizer is None:
        error_msg = (
            "CRITICAL: NRAM_ENGINE=sglang requires NRAM_TOKENIZER_PATH. "
            "Refusing to run in silent prompt-only mode. "
            "Set NRAM_TOKENIZER_PATH=/models in compose.yaml and ensure the model "
            "directory is mounted into the nram-api container."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    try:
        from core.engines.sglang_engine import SGLangEngine

        _sglang_engine = SGLangEngine(
            base_url=base_url,
            model=model,
            tokenizer=tokenizer,
        )
        logger.info(
            f"SGLang engine constructed: base_url={base_url}, model={model}, "
            f"tokenizer={'yes' if tokenizer else 'no'}"
        )
        return _sglang_engine
    except Exception as e:  # pragma: no cover - depends on runtime env
        _sglang_init_failed = True
        logger.error(f"Failed to construct SGLang engine: {e}", exc_info=True)
        return None


def _load_tokenizer():
    """Load tokenizer for token-aware bias compilation.
    
    CRITICAL: If NRAM_TOKENIZER_PATH is set but tokenizer fails to load,
    we raise an error instead of silently degrading. Without tokenizer,
    NRAM logit steering is completely disabled.
    """
    tokenizer_path = os.environ.get("NRAM_TOKENIZER_PATH")
    if not tokenizer_path:
        logger.warning(
            "NRAM_TOKENIZER_PATH not set. Token-level steering disabled. "
            "Set NRAM_TOKENIZER_PATH=/models to enable logit processor."
        )
        return None
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        logger.info(f"✓ Tokenizer loaded successfully from {tokenizer_path}")
        return tokenizer
    except Exception as e:
        # P0: Fail-fast instead of silent degradation
        error_msg = (
            f"CRITICAL: Tokenizer load failed from {tokenizer_path}: {e}\n"
            f"NRAM logit steering is DISABLED. "
            f"Check NRAM_TOKENIZER_PATH and model files."
        )
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


def reset_engine_cache() -> None:
    """Reset cached engine state (used by tests)."""
    global _sglang_engine, _sglang_init_failed
    _sglang_engine = None
    _sglang_init_failed = False
