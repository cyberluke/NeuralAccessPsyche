"""
Live Activation Addition for NRAM v5.

Implements actual hidden-state modification during model forward pass:
  h' = h + alpha * v

Executes inside the real Qwen forward path with full telemetry.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from core.steering.representation_config import (
    NRAMRepresentationConfig,
    RepresentationIntervention,
    RepresentationRequestState,
    RepresentationTelemetry,
    InterventionKind,
    TokenScope,
)

logger = logging.getLogger(__name__)


@dataclass
class ActivationAdditionState:
    """Per-request state for activation addition."""
    vectors: Dict[str, torch.Tensor] = field(default_factory=dict)
    alphas: Dict[str, float] = field(default_factory=dict)
    layer_names: Dict[str, str] = field(default_factory=dict)
    hook_counts: Dict[str, int] = field(default_factory=dict)
    
    def reset(self) -> None:
        """Reset all state."""
        self.vectors.clear()
        self.alphas.clear()
        self.layer_names.clear()
        self.hook_counts.clear()


class LiveActivationAddition:
    """
    Live activation addition controller.
    
    Applies h' = h + alpha * v during model forward pass.
    
    Requirements:
    - Execute inside real Qwen forward path
    - Support selected decoder layers
    - Support prefill-only, decode-only, or both
    - Move vectors to correct device and dtype
    - Handle tensor, tuple, and model-specific outputs
    - Avoid in-place modifications
    - Preserve gradients-disabled inference behavior
    - Support positive, negative, and zero strength
    - Support token-step strength schedule
    - Reset all per-request state
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.enabled = True
        self._state = ActivationAdditionState()
        self._telemetry: List[RepresentationTelemetry] = []
    
    def register_vector(
        self,
        intervention_id: str,
        vector: torch.Tensor,
        alpha: float,
        layer_name: str,
    ) -> None:
        """Register an activation vector for intervention."""
        self._state.vectors[intervention_id] = vector.to(self.device)
        self._state.alphas[intervention_id] = alpha
        self._state.layer_names[intervention_id] = layer_name
        self._state.hook_counts[intervention_id] = 0
        logger.info(
            f"Registered activation vector: {intervention_id} "
            f"at layer {layer_name} with alpha={alpha}"
        )
    
    def apply(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_name: str,
        request_state: RepresentationRequestState,
        phase: str = "decode",
    ) -> Tuple[torch.Tensor, Optional[RepresentationTelemetry]]:
        """
        Apply activation addition to hidden states.
        
        Args:
            hidden_states: Shape (batch_size, seq_len, hidden_dim)
            layer_name: Current layer name
            request_state: Per-request state
            phase: "prefill" or "decode"
        
        Returns:
            (modified_hidden_states, telemetry_or_none)
        """
        if not self.enabled:
            return hidden_states, None
        
        # Get interventions for this layer
        interventions = request_state.config.get_interventions_for_layer(layer_name)
        if not interventions:
            return hidden_states, None
        
        # Filter by step and phase
        step = request_state.step_counter
        active_interventions = [
            iv for iv in interventions
            if iv.is_active_at_step(step, phase)
        ]
        
        if not active_interventions:
            return hidden_states, None
        
        # Compute norm before
        hidden_norm_before = hidden_states.norm().item()
        
        # Apply each intervention
        modified = hidden_states
        total_delta_norm = 0.0
        last_telemetry = None
        
        for iv in active_interventions:
            if iv.kind != InterventionKind.ACTIVATION_ADDITION:
                continue
            
            vector = self._state.vectors.get(iv.intervention_id)
            if vector is None:
                continue
            
            alpha = self._state.alphas.get(iv.intervention_id, iv.strength)
            
            # Zero strength is exact no-op
            if abs(alpha) < 1e-10:
                continue
            
            # Move vector to correct device and dtype
            vector = vector.to(modified.device, dtype=modified.dtype)
            
            # Broadcast vector to match hidden states shape
            # vector shape: (hidden_dim,) -> (1, 1, hidden_dim)
            vector_expanded = vector.unsqueeze(0).unsqueeze(0)
            
            # Compute delta: alpha * v
            delta = alpha * vector_expanded
            
            # Apply: h' = h + alpha * v (non-in-place)
            modified = modified + delta
            
            # Compute telemetry
            vector_norm = vector.norm().item()
            delta_norm = delta.norm().item()
            total_delta_norm += delta_norm
            
            # Cosine similarity between hidden and vector
            hidden_flat = modified.reshape(-1, modified.shape[-1])
            vector_flat = vector.unsqueeze(0).expand(hidden_flat.shape[0], -1)
            cosine = torch.nn.functional.cosine_similarity(
                hidden_flat, vector_flat, dim=1
            ).mean().item()
            
            # Update hook count
            self._state.hook_counts[iv.intervention_id] = (
                self._state.hook_counts.get(iv.intervention_id, 0) + 1
            )
            
            # Create telemetry
            hidden_norm_after = modified.norm().item()
            last_telemetry = RepresentationTelemetry(
                request_id=request_state.request_id,
                decode_step=step,
                layer_name=layer_name,
                intervention_id=iv.intervention_id,
                alpha=alpha,
                hidden_norm_before=hidden_norm_before,
                hidden_norm_after=hidden_norm_after,
                delta_norm=delta_norm,
                vector_norm=vector_norm,
                cosine_hidden_vector=cosine,
                hook_invocation_count=self._state.hook_counts[iv.intervention_id],
                prefill_or_decode=phase,
            )
            
            # Record telemetry
            request_state.record_telemetry(last_telemetry.to_dict())
        
        # Update request state
        request_state.hook_invocation_count += 1
        
        return modified, last_telemetry
    
    def reset(self) -> None:
        """Reset all per-request state."""
        self._state.reset()
        self._telemetry.clear()
        logger.info("Reset activation addition state")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "enabled": self.enabled,
            "num_vectors": len(self._state.vectors),
            "vectors": {
                k: {
                    "layer": self._state.layer_names.get(k),
                    "alpha": self._state.alphas.get(k),
                    "hook_count": self._state.hook_counts.get(k, 0),
                    "norm": v.norm().item() if v is not None else 0.0,
                }
                for k, v in self._state.vectors.items()
            },
            "total_hook_invocations": sum(self._state.hook_counts.values()),
        }


class ActivationAdditionHook:
    """
    Forward hook for activation addition.
    
    Registered on transformer layers to apply activation steering
    during the forward pass.
    """
    
    def __init__(
        self,
        controller: LiveActivationAddition,
        layer_name: str,
        layer_index: int,
        request_state: RepresentationRequestState,
    ):
        self.controller = controller
        self.layer_name = layer_name
        self.layer_index = layer_index
        self.request_state = request_state
        self.enabled = True
        self.current_phase = "decode"
        self.invocation_count = 0
    
    def __call__(
        self,
        module: nn.Module,
        input_args: tuple,
        output: Any,
    ) -> Any:
        """
        Hook function called during forward pass.
        
        Handles tuple outputs (common in transformer layers).
        """
        if not self.enabled or not self.controller.enabled:
            return output
        
        self.invocation_count += 1
        
        # Determine phase based on sequence length
        # Prefill: longer sequence, Decode: single token
        if isinstance(output, tuple):
            hidden_states = output[0]
            seq_len = hidden_states.shape[1] if hidden_states.dim() >= 2 else 1
        else:
            hidden_states = output
            seq_len = hidden_states.shape[1] if hidden_states.dim() >= 2 else 1
        
        # Heuristic: prefill if seq_len > 1, decode if seq_len == 1
        phase = "prefill" if seq_len > 1 else "decode"
        self.current_phase = phase
        
        # Apply activation addition
        modified_hidden, telemetry = self.controller.apply(
            hidden_states,
            layer_name=self.layer_name,
            request_state=self.request_state,
            phase=phase,
        )
        
        # Return modified output
        if isinstance(output, tuple):
            return (modified_hidden,) + output[1:]
        else:
            return modified_hidden
    
    def enable(self) -> None:
        """Enable this hook."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable this hook."""
        self.enabled = False
