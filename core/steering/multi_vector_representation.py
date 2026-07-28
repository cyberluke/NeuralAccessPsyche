"""
Multi-vector representation controller for NRAM v5.

Supports multiple simultaneous axes and layers with:
- Vector normalization
- Coefficient schedules
- Weighted combination
- Orthogonalization option
- Norm budget
- Per-layer clamp
- Conflict telemetry
- Deterministic ordering
- Independent enable/disable controls
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from core.steering.representation_config import (
    CombinationMode,
    NRAMRepresentationConfig,
    RepresentationIntervention,
    RepresentationRequestState,
    RepresentationTelemetry,
    InterventionKind,
)

logger = logging.getLogger(__name__)


@dataclass
class VectorEntry:
    """A registered vector with its metadata."""
    intervention_id: str
    vector: torch.Tensor
    layer_name: str
    strength: float
    enabled: bool = True
    schedule: Optional[List[Tuple[int, float]]] = None  # (step, alpha) pairs


class MultiVectorRepresentationController:
    """
    Unified controller for multiple simultaneous representation interventions.
    
    Supports:
    - sum: simple weighted sum
    - normalized_sum: normalize each vector before summing
    - orthogonalized_sum: Gram-Schmidt orthogonalization
    - norm_budgeted_sum: clamp total norm to budget
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.entries: Dict[str, VectorEntry] = {}
        self.enabled = True
        self._invocation_count = 0
    
    def register_vector(
        self,
        intervention_id: str,
        vector: torch.Tensor,
        layer_name: str,
        strength: float,
        schedule: Optional[List[Tuple[int, float]]] = None,
    ) -> None:
        """Register a vector for intervention."""
        self.entries[intervention_id] = VectorEntry(
            intervention_id=intervention_id,
            vector=vector.to(self.device),
            layer_name=layer_name,
            strength=strength,
            schedule=schedule,
        )
        logger.info(f"Registered vector: {intervention_id} at {layer_name}")
    
    def remove_vector(self, intervention_id: str) -> None:
        """Remove a vector."""
        if intervention_id in self.entries:
            del self.entries[intervention_id]
    
    def enable_vector(self, intervention_id: str, enabled: bool = True) -> None:
        """Enable or disable a vector."""
        if intervention_id in self.entries:
            self.entries[intervention_id].enabled = enabled
    
    def _get_effective_alpha(
        self, entry: VectorEntry, step: int
    ) -> float:
        """Get effective alpha for a vector at a given step."""
        if not entry.enabled:
            return 0.0
        if entry.schedule is None:
            return entry.strength
        # Linear interpolation between schedule points
        schedule = sorted(entry.schedule, key=lambda x: x[0])
        if step <= schedule[0][0]:
            return schedule[0][1]
        if step >= schedule[-1][0]:
            return schedule[-1][1]
        for i in range(len(schedule) - 1):
            if schedule[i][0] <= step <= schedule[i + 1][0]:
                t = (step - schedule[i][0]) / (schedule[i + 1][0] - schedule[i][0])
                return schedule[i][1] + t * (schedule[i + 1][1] - schedule[i][1])
        return entry.strength
    
    def apply(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_name: str,
        request_state: RepresentationRequestState,
    ) -> Tuple[torch.Tensor, Optional[RepresentationTelemetry]]:
        """
        Apply all active vectors for a given layer.
        
        Returns (modified_hidden_states, telemetry_or_none).
        """
        if not self.enabled:
            return hidden_states, None
        
        self._invocation_count += 1
        step = request_state.step_counter
        mode = request_state.config.combination_mode
        
        # Collect active vectors for this layer
        active: List[Tuple[str, torch.Tensor, float]] = []
        for entry in self.entries.values():
            if entry.layer_name != layer_name or not entry.enabled:
                continue
            alpha = self._get_effective_alpha(entry, step)
            if abs(alpha) < 1e-10:
                continue
            active.append((entry.intervention_id, entry.vector, alpha))
        
        if not active:
            return hidden_states, None
        
        hidden_norm_before = hidden_states.norm().item()
        
        # Combine vectors according to mode
        combined = self._combine(active, mode, hidden_states.shape[-1])
        
        # Apply norm budget if configured
        if request_state.config.norm_budget is not None:
            combined_norm = combined.norm()
            if combined_norm > request_state.config.norm_budget:
                combined = combined * (request_state.config.norm_budget / combined_norm)
        
        # Apply to hidden states (non-in-place)
        modified = hidden_states + combined.unsqueeze(0).unsqueeze(0)
        
        hidden_norm_after = modified.norm().item()
        delta_norm = combined.norm().item()
        
        # Create telemetry for first intervention
        first_id = active[0][0]
        first_vec = active[0][1]
        first_alpha = active[0][2]
        
        hidden_flat = modified.reshape(-1, modified.shape[-1])
        vec_flat = first_vec.unsqueeze(0).expand(hidden_flat.shape[0], -1)
        cosine = torch.nn.functional.cosine_similarity(
            hidden_flat, vec_flat, dim=1
        ).mean().item()
        
        telemetry = RepresentationTelemetry(
            request_id=request_state.request_id,
            decode_step=step,
            layer_name=layer_name,
            intervention_id=first_id,
            alpha=first_alpha,
            hidden_norm_before=hidden_norm_before,
            hidden_norm_after=hidden_norm_after,
            delta_norm=delta_norm,
            vector_norm=first_vec.norm().item(),
            cosine_hidden_vector=cosine,
            hook_invocation_count=self._invocation_count,
            prefill_or_decode="decode" if hidden_states.shape[1] == 1 else "prefill",
        )
        
        request_state.record_telemetry(telemetry.to_dict())
        request_state.hook_invocation_count += 1
        
        return modified, telemetry
    
    def _combine(
        self,
        active: List[Tuple[str, torch.Tensor, float]],
        mode: CombinationMode,
        hidden_dim: int,
    ) -> torch.Tensor:
        """Combine vectors according to mode."""
        if mode == CombinationMode.SUM:
            return self._combine_sum(active)
        elif mode == CombinationMode.NORMALIZED_SUM:
            return self._combine_normalized_sum(active)
        elif mode == CombinationMode.ORTHOGONALIZED_SUM:
            return self._combine_orthogonalized_sum(active)
        elif mode == CombinationMode.NORM_BUDGETED_SUM:
            return self._combine_sum(active)  # Budget applied after
        else:
            return self._combine_sum(active)
    
    def _combine_sum(
        self, active: List[Tuple[str, torch.Tensor, float]]
    ) -> torch.Tensor:
        """Simple weighted sum."""
        combined = torch.zeros_like(active[0][1])
        for _, vec, alpha in active:
            combined = combined + alpha * vec
        return combined
    
    def _combine_normalized_sum(
        self, active: List[Tuple[str, torch.Tensor, float]]
    ) -> torch.Tensor:
        """Normalize each vector before summing."""
        combined = torch.zeros_like(active[0][1])
        for _, vec, alpha in active:
            norm = vec.norm()
            if norm > 1e-8:
                combined = combined + alpha * (vec / norm)
        return combined
    
    def _combine_orthogonalized_sum(
        self, active: List[Tuple[str, torch.Tensor, float]]
    ) -> torch.Tensor:
        """Gram-Schmidt orthogonalization before summing."""
        if len(active) == 1:
            return active[0][2] * active[0][1]
        
        orthogonalized = []
        for _, vec, alpha in active:
            v = vec.clone()
            for u in orthogonalized:
                proj = (v @ u) / (u @ u + 1e-10)
                v = v - proj * u
            orthogonalized.append(v)
        
        combined = torch.zeros_like(active[0][1])
        for i, (_, _, alpha) in enumerate(active):
            combined = combined + alpha * orthogonalized[i]
        return combined
    
    def reset(self) -> None:
        """Reset invocation count."""
        self._invocation_count = 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "enabled": self.enabled,
            "num_vectors": len(self.entries),
            "vectors": {
                k: {
                    "layer": v.layer_name,
                    "strength": v.strength,
                    "enabled": v.enabled,
                    "norm": v.vector.norm().item(),
                }
                for k, v in self.entries.items()
            },
            "invocation_count": self._invocation_count,
        }
