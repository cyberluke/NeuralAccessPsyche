"""NRAM custom logit processor for SGLang.

This processor is serialized with dill and runs INSIDE the SGLang server.
It must have NO dependency on application services, NO network access,
NO filesystem access, and NO arbitrary request-controlled imports.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class NRAMLogitProcessor:
    """Custom logit processor that applies NRAM token biases before sampling.

    Applied per-batch-row with strict isolation. All parameters are validated
    and bounded. Out-of-range token IDs are silently ignored.
    """

    def __call__(self, logits: Any, custom_param_list: Optional[List[Dict[str, Any]]] = None) -> Any:
        if not custom_param_list:
            return logits

        for batch_index, params in enumerate(custom_param_list):
            if not params:
                continue

            positive_ids = self._safe_ids(
                params.get("positive_token_ids", []),
                logits.shape[-1],
            )
            negative_ids = self._safe_ids(
                params.get("negative_token_ids", []),
                logits.shape[-1],
            )
            forbidden_ids = self._safe_ids(
                params.get("forbidden_token_ids", []),
                logits.shape[-1],
            )

            positive_bias = self._bounded_float(
                params.get("positive_bias", 0.0),
                minimum=0.0,
                maximum=1.2,
            )
            negative_bias = self._bounded_float(
                params.get("negative_bias", 0.0),
                minimum=0.0,
                maximum=2.5,
            )
            repetition_penalty = self._bounded_float(
                params.get("repetition_penalty", 0.0),
                minimum=0.0,
                maximum=2.0,
            )

            # Phase 13: dynamic per-token schedule based on generation progress.
            generated = 0
            request = params.get("__req__")
            if request is not None:
                try:
                    generated = len(request.output_ids)
                except (AttributeError, TypeError):
                    generated = 0

            schedule = self._phase_schedule(params, generated)
            positive_bias = positive_bias * schedule["positive_scale"]
            negative_bias = negative_bias * schedule["negative_scale"]
            repetition_penalty = repetition_penalty * schedule["repetition_scale"]

            if positive_ids and positive_bias > 0:
                logits[batch_index, positive_ids] += positive_bias

            if negative_ids and negative_bias > 0:
                logits[batch_index, negative_ids] -= negative_bias

            if forbidden_ids:
                logits[batch_index, forbidden_ids] = -float("inf")

            # Dynamic repetition penalty using request output history
            if request is not None and repetition_penalty > 0:
                try:
                    recent_ids = list(request.output_ids[-64:])
                except (AttributeError, TypeError):
                    recent_ids = []

                recent_ids = self._safe_ids(recent_ids, logits.shape[-1])
                if recent_ids:
                    logits[batch_index, recent_ids] -= repetition_penalty

        return logits

    @staticmethod
    def _phase_schedule(params: Dict[str, Any], generated: int) -> Dict[str, float]:
        """Compute per-phase scaling multipliers.

        Schedule (as a fraction of max_tokens):
          0-15%   opening       moderate contrarian force
          15-40%  contradiction increased negative corporate bias
          40-70%  revelation    increased product/human-focus bias
          70-90%  implication   increased associative distance
          90-100% final turn    stronger repetition penalty

        All multipliers are bounded and deterministic. No global state is mutated;
        everything is derived from request-scoped params.
        """
        max_tokens = params.get("max_tokens")
        try:
            max_tokens = int(max_tokens)
        except (TypeError, ValueError):
            max_tokens = 0

        if max_tokens <= 0 or generated <= 0:
            return {"positive_scale": 1.0, "negative_scale": 1.0, "repetition_scale": 1.0}

        progress = min(1.0, generated / max_tokens)

        if progress < 0.15:
            return {"positive_scale": 1.0, "negative_scale": 1.0, "repetition_scale": 1.0}
        if progress < 0.40:
            # contradiction: boost negative corporate bias
            return {"positive_scale": 1.0, "negative_scale": 1.3, "repetition_scale": 1.0}
        if progress < 0.70:
            # revelation: boost positive product/human bias
            return {"positive_scale": 1.3, "negative_scale": 1.0, "repetition_scale": 1.0}
        if progress < 0.90:
            # implication: balanced, slightly more positive (associative)
            return {"positive_scale": 1.15, "negative_scale": 1.0, "repetition_scale": 1.1}
        # final turn: stronger repetition penalty, dampen entropy
        return {"positive_scale": 0.9, "negative_scale": 1.0, "repetition_scale": 1.5}

    @staticmethod
    def _safe_ids(values: Any, vocab_size: int) -> List[int]:
        """Validate and deduplicate token IDs against vocabulary size."""
        result: List[int] = []
        if not isinstance(values, (list, tuple)):
            return result

        for value in values:
            try:
                token_id = int(value)
            except (TypeError, ValueError):
                continue

            if 0 <= token_id < vocab_size:
                result.append(token_id)

        return sorted(set(result))

    @staticmethod
    def _bounded_float(value: Any, minimum: float, maximum: float) -> float:
        """Clamp a value to [minimum, maximum], returning minimum on invalid input."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return minimum

        if math.isnan(number) or math.isinf(number):
            return minimum

        return max(minimum, min(maximum, number))
