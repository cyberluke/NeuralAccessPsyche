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

    Phenomenon mixer — each phenomenon maps to a concrete logit operation:
      overlap          → boost recent output token IDs (self-reinforcing bleed)
      forgetting       → extra repetition penalty on recent tokens (memory decay)
      looping          → REWARD recent tokens (perseveration / thought loops)
      associative_jump → scale logit row toward uniform (flatten → random leaps)
      synesthesia      → deterministic cross-activation noise on scattered vocab
      dissolution      → scale logits toward zero (dissolve all steering structure)
      insight          → periodic positive_bias spikes at 25/50/75% of generation
    """

    def __call__(self, logits: Any, custom_param_list: Optional[List[Dict[str, Any]]] = None) -> Any:
        if not custom_param_list:
            return logits

        vocab_size = logits.shape[-1]

        for batch_index, params in enumerate(custom_param_list):
            if not params:
                continue

            positive_ids = self._safe_ids(
                params.get("positive_token_ids", []),
                vocab_size,
            )
            negative_ids = self._safe_ids(
                params.get("negative_token_ids", []),
                vocab_size,
            )
            forbidden_ids = self._safe_ids(
                params.get("forbidden_token_ids", []),
                vocab_size,
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

            max_tokens = self._safe_int(params.get("max_tokens", 0))
            progress = (generated / max_tokens) if max_tokens > 0 else 0.0

            # ---------------------------------------------------------------
            # Base steering (existing)
            # ---------------------------------------------------------------
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

                recent_ids = self._safe_ids(recent_ids, vocab_size)
                if recent_ids:
                    logits[batch_index, recent_ids] -= repetition_penalty

            # ---------------------------------------------------------------
            # Phenomenon mixer — logit-level implementations
            # ---------------------------------------------------------------
            phenomena = params.get("phenomenon_weights") or {}
            if phenomena and request is not None:
                self._apply_phenomena(
                    logits, batch_index, vocab_size, phenomena,
                    request, generated, max_tokens, progress,
                    positive_ids, positive_bias,
                )

        return logits

    # ------------------------------------------------------------------
    # Phenomenon mixer — each phenomenon is a concrete logit operation
    # ------------------------------------------------------------------
    def _apply_phenomena(
        self,
        logits: Any,
        batch_index: int,
        vocab_size: int,
        phenomena: Dict[str, Any],
        request: Any,
        generated: int,
        max_tokens: int,
        progress: float,
        positive_ids: List[int],
        positive_bias: float,
    ) -> None:
        """Apply all active phenomena as logit-level operations.

        Order matters:
        1. overlap/forgetting/looping — operate on recent output tokens
        2. insight — periodic boost on positive_ids
        3. synesthesia — deterministic cross-activation noise
        4. associative_jump — flatten entire row (scales all prior biases)
        5. dissolution — scale entire row toward zero (strongest last)
        """
        row = logits[batch_index]

        # Recent output token IDs (shared by overlap/forgetting/looping)
        try:
            output_history = list(request.output_ids)
        except (AttributeError, TypeError):
            output_history = []

        # ------------------------------------------------------------------
        # 1. OVERLAP — boost recent output tokens (self-reinforcing bleed)
        #    Fragments from previous thoughts bleed into current generation.
        #    Window: last 32 tokens, bias scaled by weight.
        # ------------------------------------------------------------------
        overlap_w = self._bounded_float(phenomena.get("overlap", 0.0), 0.0, 1.0)
        if overlap_w > 0.05 and output_history:
            recent = self._safe_ids(output_history[-32:], vocab_size)
            if recent:
                row[recent] += overlap_w * 0.35

        # ------------------------------------------------------------------
        # 2. FORGETTING — extra repetition penalty (working memory decay)
        #    Model "forgets" what it just said → stronger avoidance of recent
        #    tokens. This is ADDITIVE to the base repetition_penalty.
        #    Window: last 64 tokens.
        # ------------------------------------------------------------------
        forgetting_w = self._bounded_float(phenomena.get("forgetting", 0.0), 0.0, 1.0)
        if forgetting_w > 0.05 and output_history:
            recent = self._safe_ids(output_history[-64:], vocab_size)
            if recent:
                row[recent] -= forgetting_w * 1.8

        # ------------------------------------------------------------------
        # 3. LOOPING — REWARD recent tokens (perseveration / thought loops)
        #    Opposite of forgetting: model gets stuck repeating itself.
        #    Window: last 16 tokens (tight = tight loop).
        # ------------------------------------------------------------------
        looping_w = self._bounded_float(phenomena.get("looping", 0.0), 0.0, 1.0)
        if looping_w > 0.05 and output_history:
            recent = self._safe_ids(output_history[-16:], vocab_size)
            if recent:
                row[recent] += looping_w * 0.9

        # ------------------------------------------------------------------
        # 4. INSIGHT — periodic positive_bias spikes
        #    At ~25%, ~50%, ~75% of generation, boost the positive lexeme
        #    tokens to create "AHA!" moments of clarity.
        # ------------------------------------------------------------------
        insight_w = self._bounded_float(phenomena.get("insight", 0.0), 0.0, 1.0)
        if insight_w > 0.05 and positive_ids and positive_bias > 0:
            # Spike windows centered at 0.25, 0.50, 0.75
            for target in (0.25, 0.50, 0.75):
                if abs(progress - target) < 0.04:
                    row[positive_ids] += insight_w * 0.6
                    break  # only one spike per step

        # ------------------------------------------------------------------
        # 5. SYNESTHESIA — deterministic cross-activation noise
        #    Simulates cross-sensory blending by adding small positive bias
        #    to a deterministic scattered subset of the vocabulary.
        #    The pattern shifts with generation step (evolving cross-activation).
        # ------------------------------------------------------------------
        syn_w = self._bounded_float(phenomena.get("synesthesia", 0.0), 0.0, 1.0)
        if syn_w > 0.05:
            # Deterministic scattered selection: stride through vocab with
            # an offset that shifts per generation step.
            stride = max(1, vocab_size // 180)
            offset = (generated * 7) % stride
            cross_ids = list(range(offset, vocab_size, stride))
            # Filter to safe IDs inline (avoid full vocab scan)
            cross_ids = [t for t in cross_ids if 0 <= t < vocab_size]
            if cross_ids:
                row[cross_ids] += syn_w * 0.12

        # ------------------------------------------------------------------
        # 6. ASSOCIATIVE_JUMP — flatten distribution toward uniform
        #    Scales the ENTIRE logit row by (1 - weight * factor), reducing
        #    the effect of ALL prior biases → more random, distant leaps.
        # ------------------------------------------------------------------
        aj_w = self._bounded_float(phenomena.get("associative_jump", 0.0), 0.0, 1.0)
        if aj_w > 0.05:
            scale = 1.0 - aj_w * 0.45
            row[:] = row * scale

        # ------------------------------------------------------------------
        # 7. DISSOLUTION — scale logits toward zero (dissolve structure)
        #    The strongest effect: collapses all steering toward a flat
        #    distribution. Applied LAST so it overrides everything.
        # ------------------------------------------------------------------
        diss_w = self._bounded_float(phenomena.get("dissolution", 0.0), 0.0, 1.0)
        if diss_w > 0.05:
            scale = 1.0 - diss_w * 0.55
            row[:] = row * scale

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

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

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """Convert value to int, returning default on invalid input."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
