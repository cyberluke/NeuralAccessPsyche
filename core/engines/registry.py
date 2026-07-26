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
    tokenizer is optional — without one, token biasing is skipped but the engine
    still proxies requests. We never raise at import time.
    """
    global _sglang_engine, _sglang_init_failed

    if _sglang_engine is not None:
        return _sglang_engine
    if _sglang_init_failed:
        return None

    base_url = os.environ.get("SGLANG_BASE_URL", "http://sglang:30000/v1")
    model = os.environ.get("SGLANG_MODEL", "nram-deepseek-r1-qwen-7b")

    tokenizer = _load_tokenizer()

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
    """Best-effort tokenizer load. Returns None if unavailable.

    The tokenizer is needed for token-aware bias compilation. If transformers or
    the model tokenizer is not present locally, we return None and the engine
    proxies requests without token biasing (steering falls back to prompt/plan).
    """
    tokenizer_path = os.environ.get("NRAM_TOKENIZER_PATH")
    if not tokenizer_path:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as e:  # pragma: no cover - depends on runtime env
        logger.warning(f"Tokenizer load failed ({tokenizer_path}): {e}")
        return None


def reset_engine_cache() -> None:
    """Reset cached engine state (used by tests)."""
    global _sglang_engine, _sglang_init_failed
    _sglang_engine = None
    _sglang_init_failed = False
