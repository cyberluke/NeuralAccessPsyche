"""Runtime gate for the optional DExperts integration."""
from __future__ import annotations

import os


def dexperts_enabled() -> bool:
    """Return whether DExperts is explicitly enabled at runtime."""
    return os.getenv("NRAM_DEXPERTS_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
