"""
Telemetry emission for NRAM hooks.

This module provides structured telemetry events that prove hooks execute
inside the real model forward path. Events are emitted from within the
actual Qwen decoder layer forward pass.

Critical: Telemetry must originate from inside the hook, not from
synthetic events in the logit processor or engine.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class HookTelemetry:
    """
    Telemetry emitter for hook invocations.
    
    Emits structured events that prove:
    - Hook was called from inside real forward pass
    - Hidden states were modified (or not, if alpha=0)
    - Intervention parameters were applied correctly
    - Request-scoped state is isolated
    """
    
    @staticmethod
    def emit_hook_invocation(
        request_id: str,
        hook_id: str,
        module_name: str,
        layer_index: int,
        phase: str,  # "prefill" or "decode"
        decode_step: int,
        invocation_count: int,
        hidden_shape: tuple,
        hidden_dtype: str,
        hidden_device: str,
        hidden_norm_before: float,
        hidden_norm_after: float,
        intervention_delta_norm: float,
        active_intervention_ids: list,
    ) -> Dict[str, Any]:
        """
        Emit a hook invocation telemetry event.
        
        This event proves the hook executed inside the real forward path
        and modified (or preserved) the hidden states.
        
        Args:
            request_id: Request identifier
            hook_id: Unique hook identifier
            module_name: Name of the module (e.g., "model.layers.20")
            layer_index: Layer index
            phase: "prefill" or "decode"
            decode_step: Current decode step (0 for prefill)
            invocation_count: Total invocations for this request
            hidden_shape: Shape of hidden states tensor
            hidden_dtype: Data type of hidden states
            hidden_device: Device of hidden states
            hidden_norm_before: L2 norm before intervention
            hidden_norm_after: L2 norm after intervention
            intervention_delta_norm: L2 norm of the delta (h' - h)
            active_intervention_ids: List of active intervention IDs
        
        Returns:
            Telemetry event dict
        """
        event = {
            "schema": "nram.hook.invocation.v1",
            "timestamp": time.time(),
            "request_id": request_id,
            "hook_id": hook_id,
            "module_name": module_name,
            "layer_index": layer_index,
            "phase": phase,
            "decode_step": decode_step,
            "invocation_count": invocation_count,
            "hidden_shape": list(hidden_shape),
            "hidden_dtype": hidden_dtype,
            "hidden_device": hidden_device,
            "hidden_norm_before": hidden_norm_before,
            "hidden_norm_after": hidden_norm_after,
            "intervention_delta_norm": intervention_delta_norm,
            "active_intervention_ids": active_intervention_ids,
        }
        
        # Emit as structured log
        print("NRAM_HOOK_EVENT " + json.dumps(event, sort_keys=True), flush=True)
        
        return event
    
    @staticmethod
    def emit_intervention_applied(
        request_id: str,
        intervention_id: str,
        intervention_type: str,
        layer_index: int,
        strength: float,
        delta_norm: float,
    ) -> Dict[str, Any]:
        """
        Emit an intervention-applied telemetry event.
        
        Proves a specific intervention was applied to hidden states.
        
        Args:
            request_id: Request identifier
            intervention_id: Intervention identifier
            intervention_type: Type of intervention
            layer_index: Layer where applied
            strength: Applied strength (alpha)
            delta_norm: L2 norm of the delta
        
        Returns:
            Telemetry event dict
        """
        event = {
            "schema": "nram.hook.intervention.v1",
            "timestamp": time.time(),
            "request_id": request_id,
            "intervention_id": intervention_id,
            "intervention_type": intervention_type,
            "layer_index": layer_index,
            "strength": strength,
            "delta_norm": delta_norm,
        }
        
        print("NRAM_INTERVENTION_EVENT " + json.dumps(event, sort_keys=True), flush=True)
        
        return event
    
    @staticmethod
    def emit_closed_loop_update(
        request_id: str,
        loop_type: str,  # "latent" or "semantic"
        step: int,
        probe_score: Optional[float],
        target: Optional[float],
        error: Optional[float],
        alpha_before: float,
        alpha_after: float,
    ) -> Dict[str, Any]:
        """
        Emit a closed-loop update telemetry event.
        
        Proves the controller updated alpha based on probe scores.
        
        Args:
            request_id: Request identifier
            loop_type: "latent" or "semantic"
            step: Current step
            probe_score: Score from probe (if available)
            target: Target score
            error: Error (target - score)
            alpha_before: Alpha before update
            alpha_after: Alpha after update
        
        Returns:
            Telemetry event dict
        """
        event = {
            "schema": "nram.hook.closed_loop.v1",
            "timestamp": time.time(),
            "request_id": request_id,
            "loop_type": loop_type,
            "step": step,
            "probe_score": probe_score,
            "target": target,
            "error": error,
            "alpha_before": alpha_before,
            "alpha_after": alpha_after,
        }
        
        print("NRAM_CLOSED_LOOP_EVENT " + json.dumps(event, sort_keys=True), flush=True)
        
        return event
