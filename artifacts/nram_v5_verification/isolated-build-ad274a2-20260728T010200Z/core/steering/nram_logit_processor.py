"""Compatibility import for the shared SGLang-side NRAM processor.

The canonical implementation lives in the stable :mod:`nram_sglang` package
that is installed in both API and inference containers.  Keeping this alias
prevents unit tests and older application imports from exercising a divergent
processor implementation.
"""

from nram_sglang.processor import NRAMLogitProcessor

__all__ = ["NRAMLogitProcessor"]
