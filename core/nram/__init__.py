"""NRAM package — legacy NRAM class + new phenomena/session/telemetry modules.

Re-exports the legacy NRAM class so that both:
  - `from core.nram import NRAM` (legacy consumers in api/routes.py, tests)
  - `from core.nram.session import ...` (new NRAM REST API code)
resolve correctly without import collisions.
"""
from __future__ import annotations

# Re-export legacy NRAM class for backward compatibility
from core.nram.legacy import NRAM

__all__ = ["NRAM"]
