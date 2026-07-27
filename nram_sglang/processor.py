"""NRAM custom logit processor for SGLang.

This processor is serialized with dill and runs INSIDE the SGLang server.
It must have NO dependency on application services, NO network access,
NO filesystem access, and NO arbitrary request-controlled imports.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
except ImportError:
    class CustomLogitProcessor:  # type: ignore[no-redef]
        """API-side compatibility base; SGLang supplies the real class."""

        pass


class NRAMLogitProcessor(CustomLogitProcessor):
    """Custom logit processor that applies NRAM token biases before sampling.

    Applied per-batch-row with strict isolation. All parameters are validated
    and bounded. Out-of-range token IDs are silently ignored.

    NRAM v5 multi-layer inference control:
      Layer 2: Phrase constraints (forbidden phrases, source n-gram blocking)
      Layer 3: Entropy control (PID servo with phase-based targets)
      Layer 3: Concept injection (dynamic concept capsules)
    
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

            telemetry_enabled = bool(params.get("telemetry_enabled", False))
            row_before = logits[batch_index].clone() if telemetry_enabled else None
            masked_ids: List[int] = []
            entropy_event: Optional[Dict[str, Any]] = None
            soft_event: List[Dict[str, Any]] = []
            vector_event: List[Dict[str, Any]] = []
            concept_event: List[Dict[str, Any]] = []

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
            # Layer 2: Phrase constraints (forbidden phrases, source n-gram blocking)
            # ---------------------------------------------------------------
            phrase_config = params.get("phrase_constraint_config")
            if phrase_config and request is not None:
                masked_ids.extend(self._apply_phrase_constraints(
                    logits, batch_index, vocab_size, phrase_config, request
                ))

            # ---------------------------------------------------------------
            # Base steering (existing)
            # ---------------------------------------------------------------
            if positive_ids and positive_bias > 0:
                logits[batch_index, positive_ids] += positive_bias

            if negative_ids and negative_bias > 0:
                logits[batch_index, negative_ids] -= negative_bias

            if forbidden_ids:
                logits[batch_index, forbidden_ids] = -float("inf")
                masked_ids.extend(forbidden_ids)

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
            # Layer 3: Entropy control (PID servo with phase-based targets)
            # ---------------------------------------------------------------
            entropy_config = params.get("entropy_config")
            if entropy_config:
                entropy_event = self._apply_entropy_control(
                    logits, batch_index, entropy_config, progress, request
                )

            # ---------------------------------------------------------------
            # Layer 3: Concept injection (dynamic concept capsules)
            # ---------------------------------------------------------------
            concept_config = params.get("concept_config")
            if concept_config:
                concept_event = self._apply_concept_injection(
                    logits, batch_index, vocab_size, concept_config, progress, request
                )

            soft_config = params.get("soft_injection_config")
            if soft_config:
                soft_event = self._apply_soft_injections(
                    logits, batch_index, vocab_size, soft_config, generated
                )

            vector_config = params.get("logit_vector_config")
            if vector_config:
                vector_event = self._apply_logit_vectors(
                    logits, batch_index, vocab_size, vector_config
                )

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

            # Explicit scientific forced-token control. It is disabled unless
            # both fields are supplied, and is applied last before sampling.
            forced_token_id = self._safe_int(params.get("forced_token_id"), -1)
            force_enabled = bool(params.get("forced_token_enabled", False))
            hard_config = params.get("hard_injection_config") or {}
            hard_schedule = hard_config.get("token_ids") or []
            hard_start = max(0, self._safe_int(hard_config.get("start_step"), 0))
            hard_index = generated - hard_start
            if not force_enabled and 0 <= hard_index < len(hard_schedule):
                forced_token_id = self._safe_int(hard_schedule[hard_index], -1)
                force_enabled = True
            requested_forced_token_id = forced_token_id if force_enabled else None
            forced_applied = False
            if force_enabled and 0 <= forced_token_id < vocab_size:
                forced_value = logits[batch_index, forced_token_id].clone()
                if bool(forced_value.isfinite().item()):
                    forced_applied = True
            if forced_applied:
                logits[batch_index, :] = -float("inf")
                logits[batch_index, forced_token_id] = forced_value

            if telemetry_enabled and row_before is not None:
                self._emit_telemetry(
                    row_before=row_before,
                    row_after=logits[batch_index],
                    params=params,
                    request=request,
                    generated=generated,
                    positive_ids=positive_ids,
                    negative_ids=negative_ids,
                    masked_ids=masked_ids,
                    entropy_event=entropy_event,
                    soft_event=soft_event,
                    vector_event=vector_event,
                    concept_event=concept_event,
                    forced_token_id=forced_token_id if forced_applied else None,
                    requested_forced_token_id=requested_forced_token_id,
                    vocab_size=vocab_size,
                )

        return logits

    # ------------------------------------------------------------------
    # Layer 2: Phrase constraints
    # ------------------------------------------------------------------
    def _apply_phrase_constraints(
        self,
        logits: Any,
        batch_index: int,
        vocab_size: int,
        phrase_config: Dict[str, Any],
        request: Any,
    ) -> List[int]:
        """Apply phrase-level masking and source n-gram blocking.
        
        Args:
            logits: Logit tensor
            batch_index: Batch index
            vocab_size: Vocabulary size
            phrase_config: Phrase constraint configuration
            request: Request object with output_ids
        """
        try:
            output_ids = list(request.output_ids)
        except (AttributeError, TypeError):
            return []

        masked_ids: List[int] = []
        
        # Forbidden phrase masking
        forbidden_phrases = phrase_config.get("forbidden_phrase_ids", [])
        if forbidden_phrases:
            for phrase_token_ids in forbidden_phrases:
                if not phrase_token_ids:
                    continue
                prefix_length = len(phrase_token_ids) - 1
                if len(output_ids) >= prefix_length:
                    recent = output_ids[-prefix_length:] if prefix_length else []
                    if recent == phrase_token_ids[:-1]:
                        # Next token would complete forbidden phrase
                        forbidden_next = phrase_token_ids[-1]
                        if 0 <= forbidden_next < vocab_size:
                            logits[batch_index, forbidden_next] = -float("inf")
                            masked_ids.append(forbidden_next)
        
        # Source n-gram blocking
        source_ngram_ids = phrase_config.get("source_ngram_ids", [])
        source_ngram_size = phrase_config.get("source_ngram_size", 8)
        if source_ngram_ids and len(output_ids) >= source_ngram_size - 1:
            recent = output_ids[-(source_ngram_size - 1):]
            for ngram in source_ngram_ids:
                if len(ngram) == source_ngram_size:
                    if recent == ngram[:-1]:
                        forbidden_next = ngram[-1]
                        if 0 <= forbidden_next < vocab_size:
                            logits[batch_index, forbidden_next] = -float("inf")
                            masked_ids.append(forbidden_next)

        return sorted(set(masked_ids))

    # ------------------------------------------------------------------
    # Layer 3: Entropy control
    # ------------------------------------------------------------------
    def _apply_entropy_control(
        self,
        logits: Any,
        batch_index: int,
        entropy_config: Dict[str, Any],
        progress: float,
        request: Any,
    ) -> Dict[str, Any]:
        """Apply entropy control via temperature scaling.
        
        Args:
            logits: Logit tensor
            batch_index: Batch index
            entropy_config: Entropy control configuration
            progress: Generation progress (0.0 to 1.0)
        """
        # Phase-based target entropy
        phase_targets = entropy_config.get("phase_targets", {
            "extraction": 2.5,
            "questioning": 4.0,
            "divergence": 6.0,
            "synthesis": 4.5,
            "formulation": 3.0,
        })
        
        # Determine current phase
        if progress < 0.2:
            phase = "extraction"
        elif progress < 0.4:
            phase = "questioning"
        elif progress < 0.6:
            phase = "divergence"
        elif progress < 0.8:
            phase = "synthesis"
        else:
            phase = "formulation"
        
        target_entropy = phase_targets.get(phase, 4.0)
        
        # Compute current entropy from the actual next-token distribution.
        row = logits[batch_index]
        entropy = self._entropy(row)

        # PID state lives on SGLang's per-request Req object. No processor-level
        # mutable state is shared between requests.
        error = target_entropy - entropy
        kp = self._bounded_float(entropy_config.get("kp", 0.35), 0.0, 2.0)
        ki = self._bounded_float(entropy_config.get("ki", 0.02), 0.0, 0.5)
        kd = self._bounded_float(entropy_config.get("kd", 0.05), 0.0, 1.0)
        integral_limit = self._bounded_float(
            entropy_config.get("integral_limit", 20.0), 0.0, 100.0
        )
        state = getattr(request, "_nram_entropy_pid", None) if request is not None else None
        if not isinstance(state, dict):
            state = {"integral": 0.0, "previous_error": 0.0}
        integral = max(
            -integral_limit,
            min(integral_limit, float(state["integral"]) + error),
        )
        derivative = error - float(state["previous_error"])
        p_term = kp * error
        i_term = ki * integral
        d_term = kd * derivative
        scale = 1.0 + p_term + i_term + d_term

        scale_min = self._bounded_float(entropy_config.get("scale_min", 0.6), 0.05, 1.0)
        scale_max = self._bounded_float(entropy_config.get("scale_max", 1.8), 1.0, 5.0)
        scale = max(scale_min, min(scale_max, scale))
        if request is not None:
            request._nram_entropy_pid = {
                "integral": integral,
                "previous_error": error,
            }

        logits[batch_index] = logits[batch_index] / scale
        entropy_after = self._entropy(logits[batch_index])
        return {
            "target": target_entropy,
            "error": error,
            "p": p_term,
            "i": i_term,
            "d": d_term,
            "integral": integral,
            "derivative": derivative,
            "applied_scale": scale,
            "entropy_before": entropy,
            "entropy_after": entropy_after,
            "clamped": scale in (scale_min, scale_max),
            "scale_min": scale_min,
            "scale_max": scale_max,
        }

    # ------------------------------------------------------------------
    # Layer 3: Concept injection
    # ------------------------------------------------------------------
    def _apply_concept_injection(
        self,
        logits: Any,
        batch_index: int,
        vocab_size: int,
        concept_config: Dict[str, Any],
        progress: float,
        request: Any,
    ) -> List[Dict[str, Any]]:
        """Apply dynamic concept injection.
        
        Args:
            logits: Logit tensor
            batch_index: Batch index
            vocab_size: Vocabulary size
            concept_config: Concept injection configuration
            progress: Generation progress (0.0 to 1.0)
        """
        concepts = concept_config.get("concepts", [])
        if not concepts:
            return []
        events: List[Dict[str, Any]] = []
        
        # Phase-dependent strength curve
        if progress < 0.2:
            phase_strength = progress / 0.2 * 0.9
        elif progress < 0.8:
            phase_strength = 0.9
        else:
            phase_strength = (1.0 - progress) / 0.2 * 0.9
        
        base_strength = concept_config.get("base_strength", 0.5)
        injection_strength = base_strength * phase_strength
        
        for concept in concepts:
            # Check phase activation
            activation_phase = concept.get("activation_phase")
            if activation_phase:
                # Simple phase matching
                if progress < 0.2 and activation_phase != "extraction":
                    continue
                elif progress < 0.4 and activation_phase != "questioning":
                    continue
                elif progress < 0.6 and activation_phase != "divergence":
                    continue
                elif progress < 0.8 and activation_phase != "synthesis":
                    continue
                elif activation_phase != "formulation":
                    continue
            
            # Apply bias to concept tokens
            concept_token_ids = concept.get("token_ids", [])
            max_uses = self._safe_int(concept.get("max_uses"), 0)
            if max_uses > 0 and request is not None:
                try:
                    history = list(request.output_ids)
                except (AttributeError, TypeError):
                    history = []
                uses = sum(1 for token_id in history if token_id in concept_token_ids)
                if uses >= max_uses:
                    continue
            for token_id in concept_token_ids:
                if 0 <= token_id < vocab_size:
                    logits[batch_index, token_id] += injection_strength
            safe_token_ids = self._safe_ids(concept_token_ids, vocab_size)
            if safe_token_ids and injection_strength:
                events.append(
                    {
                        "concept_id": str(concept.get("concept_id", ""))[:64],
                        "token_ids": safe_token_ids[:64],
                        "delta": injection_strength,
                        "max_uses": max_uses,
                    }
                )
        return events

    def _apply_soft_injections(
        self,
        logits: Any,
        batch_index: int,
        vocab_size: int,
        config: Dict[str, Any],
        generated: int,
    ) -> List[Dict[str, Any]]:
        """Apply bounded token-logit deltas inside explicit decode windows."""
        events: List[Dict[str, Any]] = []
        for item in (config.get("injections") or [])[:32]:
            start = max(0, self._safe_int(item.get("start_step"), 0))
            end = max(start, self._safe_int(item.get("end_step"), 2**31 - 1))
            if generated < start or generated >= end:
                continue
            token_ids = self._safe_ids(item.get("token_ids", []), vocab_size)
            bias = self._bounded_signed_float(item.get("bias", 0.0), 5.0)
            if token_ids and bias:
                logits[batch_index, token_ids] += bias
                events.append(
                    {
                        "token_ids": token_ids[:32],
                        "bias": bias,
                        "start_step": start,
                        "end_step": end,
                    }
                )
        return events

    def _apply_logit_vectors(
        self,
        logits: Any,
        batch_index: int,
        vocab_size: int,
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Combine sparse vocabulary-space vectors; never hidden-state vectors."""
        events: List[Dict[str, Any]] = []
        row = logits[batch_index]
        clip = self._bounded_float(config.get("clip", 5.0), 0.0, 10.0)
        for vector in (config.get("vectors") or [])[:16]:
            coefficient = self._bounded_signed_float(vector.get("coefficient", 0.0), 4.0)
            entries = vector.get("entries") or []
            parsed: List[Tuple[int, float]] = []
            for entry in entries[:256]:
                token_id = self._safe_int(entry.get("token_id"), -1)
                weight = self._bounded_signed_float(entry.get("weight", 0.0), 10.0)
                if 0 <= token_id < vocab_size and weight:
                    parsed.append((token_id, weight))
            raw_norm = math.sqrt(sum(weight * weight for _, weight in parsed))
            normalize = bool(vector.get("normalize", False))
            denominator = raw_norm if normalize and raw_norm > 0 else 1.0
            deltas = []
            for token_id, weight in parsed:
                delta = coefficient * weight / denominator
                delta = max(-clip, min(clip, delta)) if clip > 0 else 0.0
                if delta:
                    row[token_id] += delta
                    deltas.append({"token_id": token_id, "delta": delta})
            events.append(
                {
                    "vector_id": str(vector.get("vector_id", "unnamed"))[:64],
                    "coefficient": coefficient,
                    "normalized": normalize,
                    "raw_norm": raw_norm,
                    "clip": clip,
                    "deltas": deltas[:64],
                }
            )
        return events

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
            # Target ~180 tokens affected (or 10% of vocab if smaller)
            target_count = min(180, max(1, vocab_size // 10))
            stride = max(2, vocab_size // target_count)
            # Non-linear offset function to ensure different patterns for
            # different generation steps (avoids cycling when generated is
            # a multiple of stride).
            offset = (generated * 7 + generated // 3) % stride
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
    # Processor-originated telemetry
    # ------------------------------------------------------------------

    @staticmethod
    def _entropy(row: Any) -> float:
        """Return finite Shannon entropy for one raw-logit row."""
        finite = row.isfinite()
        if not bool(finite.any().item()):
            return 0.0
        safe_row = row[finite]
        probs = safe_row.softmax(dim=-1)
        entropy = -(probs * probs.clamp_min(1e-20).log()).sum()
        value = float(entropy.item())
        return value if math.isfinite(value) else 0.0

    @staticmethod
    def _json_logit(value: Any) -> Optional[float]:
        number = float(value.item() if hasattr(value, "item") else value)
        return number if math.isfinite(number) else None

    def _emit_telemetry(
        self,
        *,
        row_before: Any,
        row_after: Any,
        params: Dict[str, Any],
        request: Any,
        generated: int,
        positive_ids: List[int],
        negative_ids: List[int],
        masked_ids: List[int],
        entropy_event: Optional[Dict[str, Any]],
        soft_event: List[Dict[str, Any]],
        vector_event: List[Dict[str, Any]],
        concept_event: List[Dict[str, Any]],
        forced_token_id: Optional[int],
        requested_forced_token_id: Optional[int],
        vocab_size: int,
    ) -> None:
        """Emit one bounded JSON event from the actual processor invocation."""
        invocation = int(getattr(request, "_nram_invocation_count", 0)) + 1
        if request is not None:
            request._nram_invocation_count = invocation
        max_steps = max(0, min(128, self._safe_int(params.get("telemetry_max_steps"), 16)))
        if invocation > max_steps:
            return

        top_k = max(1, min(20, self._safe_int(params.get("telemetry_top_k"), 5)))
        k = min(top_k, vocab_size)
        before_values, before_ids = row_before.topk(k)
        after_values, after_ids = row_after.topk(k)
        candidate_ids = set(positive_ids) | set(negative_ids) | set(masked_ids)
        candidate_ids.update(int(v) for v in before_ids.tolist())
        candidate_ids.update(int(v) for v in after_ids.tolist())
        if forced_token_id is not None:
            candidate_ids.add(forced_token_id)
        for item in soft_event:
            candidate_ids.update(item.get("token_ids", []))
        for item in vector_event:
            candidate_ids.update(delta["token_id"] for delta in item.get("deltas", []))
        for item in concept_event:
            candidate_ids.update(item.get("token_ids", []))

        before_probs = row_before.softmax(dim=-1)
        after_probs = row_after.softmax(dim=-1)

        changes = []
        for token_id in sorted(candidate_ids):
            before = self._json_logit(row_before[token_id])
            after = self._json_logit(row_after[token_id])
            if before != after:
                delta = None if before is None or after is None else after - before
                changes.append(
                    {
                        "token_id": token_id,
                        "before": before,
                        "after": after,
                        "delta": delta,
                        "rank_before": int((row_before > row_before[token_id]).sum().item()) + 1,
                        "rank_after": int((row_after > row_after[token_id]).sum().item()) + 1,
                        "probability_before": self._json_logit(before_probs[token_id]),
                        "probability_after": self._json_logit(after_probs[token_id]),
                    }
                )
        changes = changes[:64]

        event = {
            "schema": "nram.processor.step.v1",
            "request_id": str(params.get("request_id", "unknown"))[:128],
            "config_hash": str(params.get("config_hash", "unknown"))[:128],
            "invocation_count": invocation,
            "generated_tokens_before_sample": generated,
            "scheduler_request_id": str(getattr(request, "rid", "unknown"))[:128],
            "pre_top_k": [
                {"token_id": int(t), "logit": self._json_logit(v)}
                for t, v in zip(before_ids.tolist(), before_values)
            ],
            "post_top_k": [
                {"token_id": int(t), "logit": self._json_logit(v)}
                for t, v in zip(after_ids.tolist(), after_values)
            ],
            "changed": changes,
            "masked_token_ids": sorted(set(masked_ids))[:64],
            "forced_token_id": forced_token_id,
            "requested_forced_token_id": requested_forced_token_id,
            "hard_injection_blocked_by_mask": (
                requested_forced_token_id is not None and forced_token_id is None
            ),
            "mask_count": (vocab_size - 1) if forced_token_id is not None else len(set(masked_ids)),
            "entropy_pid": entropy_event,
            "soft_injections": soft_event,
            "vocabulary_logit_vectors": vector_event,
            "concept_capsules": concept_event,
        }
        print("NRAM_PROCESSOR_EVENT " + json.dumps(event, sort_keys=True), flush=True)

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
        max_tokens = NRAMLogitProcessor._safe_int(params.get("max_tokens"), 0)

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
    def _bounded_signed_float(value: Any, magnitude: float) -> float:
        """Clamp a signed value, returning zero for invalid input."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(number) or math.isinf(number):
            return 0.0
        return max(-magnitude, min(magnitude, number))

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """Convert value to int, returning default on invalid input."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
