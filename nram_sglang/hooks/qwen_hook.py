"""
Qwen decoder layer forward hook for NRAM hidden-state interventions.

This hook is registered on Qwen3 decoder layers via model.named_modules()
and executes INSIDE the actual model forward pass to modify hidden states.

Architecture:
- Hook is called by PyTorch after each decoder layer forward pass
- Reads request-scoped configuration from thread-local context
- Applies interventions (activation addition, conceptor, etc.)
- Emits telemetry proving execution inside real forward path

Critical: This is NOT a logit processor. It modifies hidden states,
not logits. The modification happens BEFORE logits are computed.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import logging

import torch
import torch.nn as nn

from nram_sglang.hooks.request_context import get_current_context, NRAMHookContext, InterventionConfig
from nram_sglang.hooks.telemetry import HookTelemetry

logger = logging.getLogger(__name__)


class QwenDecoderHook:
    """
    Forward hook for Qwen decoder layers.
    
    Registered on decoder layer modules. Called by PyTorch after each
    layer's forward pass. Modifies the output hidden states based on
    request-scoped intervention configuration.
    
    Lifecycle:
    - Registered once per layer at server startup
    - Active for all requests (but reads request-specific config)
    - No-op when no context is active (baseline requests)
    """
    
    def __init__(self, layer_index: int, module_name: str):
        """
        Initialize the hook.
        
        Args:
            layer_index: Index of this decoder layer (0-based)
            module_name: Dotted name of the module (e.g., "model.layers.20")
        """
        self.layer_index = layer_index
        self.module_name = module_name
        self.hook_id = f"qwen_decoder_{layer_index}"
        self._invocation_count = 0
        self._enabled = True
    
    def __call__(
        self,
        module: nn.Module,
        input_args: Tuple[Any, ...],
        output: Any,
    ) -> Any:
        """
        Hook function called by PyTorch after layer forward pass.
        
        Args:
            module: The decoder layer module
            input_args: Input to the layer (hidden states, attention mask, etc.)
            output: Output from the layer (hidden states)
        
        Returns:
            Modified output with interventions applied, or original output if no interventions
        """
        if not self._enabled:
            return output
        
        # Get current request context (thread-local)
        context = get_current_context()
        if context is None:
            # No active request - baseline pass, no intervention
            return output
        
        # Record invocation
        context.record_hook_invocation()
        self._invocation_count += 1
        
        # Get interventions for this layer
        # Determine phase (prefill vs decode) based on input shape
        phase = self._determine_phase(input_args)
        interventions = context.get_interventions_for_layer(self.layer_index, phase)
        
        if not interventions:
            # No interventions for this layer/phase - pass through
            return output
        
        # Extract hidden states from output
        hidden_states, output_structure = self._extract_hidden_states(output)
        if hidden_states is None:
            logger.warning(f"Could not extract hidden states from layer {self.module_name}")
            return output
        
        # Record pre-intervention state
        hidden_norm_before = float(hidden_states.norm().item())
        original_shape = hidden_states.shape
        original_dtype = str(hidden_states.dtype)
        original_device = str(hidden_states.device)
        
        # Apply interventions
        modified_hidden_states = hidden_states.clone()
        total_delta_norm = 0.0
        active_intervention_ids = []
        
        for intervention in interventions:
            delta, delta_norm = self._apply_intervention(
                modified_hidden_states, intervention, context
            )
            if delta is not None:
                modified_hidden_states = modified_hidden_states + delta
                total_delta_norm += delta_norm
                active_intervention_ids.append(intervention.intervention_id)
        
        hidden_norm_after = float(modified_hidden_states.norm().item())
        
        # Reconstruct output with modified hidden states
        modified_output = self._reconstruct_output(output, output_structure, modified_hidden_states)
        
        # Emit telemetry
        if context.telemetry_enabled:
            decode_step = self._get_decode_step(context)
            HookTelemetry.emit_hook_invocation(
                request_id=context.request_id,
                hook_id=self.hook_id,
                module_name=self.module_name,
                layer_index=self.layer_index,
                phase=phase,
                decode_step=decode_step,
                invocation_count=context.hook_invocation_count,
                hidden_shape=original_shape,
                hidden_dtype=original_dtype,
                hidden_device=original_device,
                hidden_norm_before=hidden_norm_before,
                hidden_norm_after=hidden_norm_after,
                intervention_delta_norm=total_delta_norm,
                active_intervention_ids=active_intervention_ids,
            )
        
        return modified_output
    
    def _determine_phase(self, input_args: Tuple[Any, ...]) -> str:
        """
        Determine if this is prefill or decode phase.
        
        SGLang's Qwen3DecoderLayer.forward signature:
        (positions, hidden_states, forward_batch, residual, post_residual_addition)
        
        hidden_states shape: (num_tokens, hidden_dim) where num_tokens > 1 for prefill
        """
        if not input_args or len(input_args) < 2:
            return "decode"
        
        # Second arg is hidden_states in SGLang's signature
        hidden_states = input_args[1]
        if isinstance(hidden_states, torch.Tensor):
            # SGLang flattens batch and sequence: (num_tokens, hidden_dim)
            # prefill has multiple tokens, decode has 1 token
            num_tokens = hidden_states.shape[0]
            return "prefill" if num_tokens > 1 else "decode"
        
        return "decode"
    
    def _extract_hidden_states(self, output: Any) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        Extract hidden states tensor from layer output.
        
        Qwen decoder layers can return:
        - torch.Tensor (hidden states only)
        - Tuple[torch.Tensor, ...] (hidden states + cache + etc.)
        - BaseModelOutputWithPast (dataclass with .last_hidden_state)
        
        Returns:
            Tuple of (hidden_states_tensor, output_structure_info)
        """
        structure = {"type": "unknown"}
        
        # Case 1: Direct tensor
        if isinstance(output, torch.Tensor):
            structure["type"] = "tensor"
            return output, structure
        
        # Case 2: Tuple (hidden_states, past_key_value, ...)
        if isinstance(output, tuple) and len(output) > 0:
            if isinstance(output[0], torch.Tensor):
                structure["type"] = "tuple"
                structure["length"] = len(output)
                return output[0], structure
        
        # Case 3: Dataclass/object with last_hidden_state or hidden_states attribute
        if hasattr(output, "last_hidden_state"):
            structure["type"] = "model_output"
            structure["attr"] = "last_hidden_state"
            return output.last_hidden_state, structure
        
        if hasattr(output, "hidden_states"):
            structure["type"] = "model_output"
            structure["attr"] = "hidden_states"
            return output.hidden_states, structure
        
        logger.warning(f"Unknown output structure for {self.module_name}: {type(output)}")
        return None, structure
    
    def _reconstruct_output(
        self,
        original_output: Any,
        structure: Dict[str, Any],
        modified_hidden_states: torch.Tensor,
    ) -> Any:
        """
        Reconstruct the output with modified hidden states.
        
        Preserves the original output structure (tuple, dataclass, etc.)
        while replacing the hidden states tensor.
        """
        output_type = structure.get("type")
        
        if output_type == "tensor":
            return modified_hidden_states
        
        if output_type == "tuple":
            # Replace first element of tuple
            return (modified_hidden_states,) + original_output[1:]
        
        if output_type == "model_output":
            # Create a shallow copy with modified attribute
            attr_name = structure.get("attr")
            if hasattr(original_output, "__class__"):
                # Try to create new instance with modified field
                try:
                    # For dataclasses, use replace()
                    from dataclasses import replace
                    return replace(original_output, **{attr_name: modified_hidden_states})
                except Exception:
                    pass
                
                # Fallback: set attribute directly (not ideal but works)
                # This mutates the original, which may be shared
                # For single-user concurrency=1 this is acceptable
                setattr(original_output, attr_name, modified_hidden_states)
                return original_output
        
        # Fallback: return modified tensor directly
        return modified_hidden_states
    
    def _apply_intervention(
        self,
        hidden_states: torch.Tensor,
        intervention: InterventionConfig,
        context: NRAMHookContext,
    ) -> Tuple[Optional[torch.Tensor], float]:
        """
        Apply a single intervention to hidden states.
        
        Args:
            hidden_states: Current hidden states (batch, seq_len, hidden_dim)
            intervention: Intervention configuration
            context: Request context
        
        Returns:
            Tuple of (delta_tensor, delta_norm)
        """
        intervention_type = intervention.intervention_type
        strength = intervention.strength
        params = intervention.parameters
        
        delta = None
        delta_norm = 0.0
        
        if intervention_type == "activation_addition":
            delta, delta_norm = self._apply_activation_addition(
                hidden_states, strength, params
            )
        elif intervention_type == "conceptor":
            delta, delta_norm = self._apply_conceptor(
                hidden_states, strength, params
            )
        elif intervention_type == "probe":
            # Probes read hidden states but don't modify them
            # They emit scores for closed-loop controllers
            self._apply_probe(hidden_states, intervention, context)
            return None, 0.0
        else:
            logger.warning(f"Unknown intervention type: {intervention_type}")
            return None, 0.0
        
        # Emit intervention telemetry
        if context.telemetry_enabled and delta is not None:
            HookTelemetry.emit_intervention_applied(
                request_id=context.request_id,
                intervention_id=intervention.intervention_id,
                intervention_type=intervention_type,
                layer_index=self.layer_index,
                strength=strength,
                delta_norm=delta_norm,
            )
        
        return delta, delta_norm
    
    def _apply_activation_addition(
        self,
        hidden_states: torch.Tensor,
        strength: float,
        params: Dict[str, Any],
    ) -> Tuple[Optional[torch.Tensor], float]:
        """
        Apply activation addition: h' = h + alpha * v

        Args:
            hidden_states: Current hidden states. SGLang's Qwen3 decoder layer
                passes 2D flattened tensors (num_tokens, hidden_dim); some
                alternative runners may pass 3D (batch, seq_len, hidden_dim).
            strength: Alpha (scaling factor)
            params: Must contain "vector" (torch.Tensor of shape (hidden_dim,))

        Returns:
            Tuple of (delta, delta_norm)
        """
        vector = params.get("vector")
        if vector is None:
            return None, 0.0

        if not isinstance(vector, torch.Tensor):
            logger.error("Activation addition vector is not a tensor")
            return None, 0.0

        if vector.dim() != 1:
            logger.error(
                f"Activation addition vector must be 1D (hidden_dim,), got shape {vector.shape}"
            )
            return None, 0.0

        # Ensure vector is on same device/dtype as hidden states
        vector = vector.to(device=hidden_states.device, dtype=hidden_states.dtype)

        # Broadcast vector to match hidden states shape.
        # SGLang Qwen3 decoder layer: (num_tokens, hidden_dim) -> (1, hidden_dim)
        # 3D fallback: (batch, seq_len, hidden_dim) -> (1, 1, hidden_dim)
        if hidden_states.dim() == 2:
            vector_expanded = vector.unsqueeze(0)
        elif hidden_states.dim() == 3:
            vector_expanded = vector.unsqueeze(0).unsqueeze(0)
        else:
            logger.error(
                f"Unexpected hidden_states rank {hidden_states.dim()} for ActAdd"
            )
            return None, 0.0

        # Compute delta
        delta = strength * vector_expanded

        # Compute delta norm
        delta_norm = float(delta.norm().item())

        return delta, delta_norm
    
    def _apply_conceptor(
        self,
        hidden_states: torch.Tensor,
        strength: float,
        params: Dict[str, Any],
    ) -> Tuple[Optional[torch.Tensor], float]:
        """
        Apply conceptor steering: h' = h + alpha * C(h)
        
        Where C is the conceptor operator (subspace projection).
        
        Args:
            hidden_states: Current hidden states. SGLang's Qwen3 decoder layer
                passes 2D flattened tensors (num_tokens, hidden_dim); some
                alternative runners may pass 3D (batch, seq_len, hidden_dim).
            strength: Alpha (scaling factor)
            params: Must contain "basis_vectors" and "singular_values"
        
        Returns:
            Tuple of (delta, delta_norm)
        """
        # Low-rank conceptor: use basis vectors
        basis = params.get("basis_vectors")  # (rank, hidden_dim)
        singular_values = params.get("singular_values")  # (rank,)
        aperture = params.get("aperture", 1.0)
        
        if basis is None or singular_values is None:
            return None, 0.0
        
        if not isinstance(basis, torch.Tensor) or not isinstance(singular_values, torch.Tensor):
            return None, 0.0
        
        # Ensure on correct device/dtype
        basis = basis.to(device=hidden_states.device, dtype=hidden_states.dtype)
        singular_values = singular_values.to(device=hidden_states.device, dtype=hidden_states.dtype)
        
        # Compute conceptor: C = R(R + aperture^-2 I)^-1
        # Where R = basis^T @ diag(singular_values^2) @ basis
        # For low-rank: C ≈ basis^T @ diag(sv^2 / (sv^2 + aperture^-2)) @ basis
        
        # Compute conceptor weights
        sv_squared = singular_values ** 2
        aperture_sq_inv = 1.0 / (aperture ** 2)
        conceptor_weights = sv_squared / (sv_squared + aperture_sq_inv)
        
        # Project hidden states onto basis
        # hidden_states: (num_tokens, hidden_dim) or (batch, seq_len, hidden_dim)
        # basis: (rank, hidden_dim)
        # projection: (num_tokens, rank) or (batch, seq_len, rank)
        projection = torch.matmul(hidden_states, basis.T)
        
        # Apply conceptor weights - handle both 2D and 3D cases
        if projection.dim() == 2:
            # 2D: (num_tokens, rank) * (rank,) -> (num_tokens, rank)
            weighted_projection = projection * conceptor_weights.unsqueeze(0)
        else:
            # 3D: (batch, seq_len, rank) * (rank,) -> (batch, seq_len, rank)
            weighted_projection = projection * conceptor_weights.unsqueeze(0).unsqueeze(0)
        
        # Project back
        # (num_tokens, rank) @ (rank, hidden_dim) -> (num_tokens, hidden_dim)
        # or (batch, seq_len, rank) @ (rank, hidden_dim) -> (batch, seq_len, hidden_dim)
        reconcepted = torch.matmul(weighted_projection, basis)
        
        # Delta is the difference
        delta = strength * (reconcepted - hidden_states)
        delta_norm = float(delta.norm().item())
        
        return delta, delta_norm
    
    def _apply_probe(
        self,
        hidden_states: torch.Tensor,
        intervention: InterventionConfig,
        context: NRAMHookContext,
    ) -> None:
        """
        Apply a trained probe to hidden states.
        
        Probes are linear classifiers that score hidden states.
        They don't modify hidden states but emit scores for closed-loop control.
        
        Args:
            hidden_states: Current hidden states
            intervention: Probe configuration
            context: Request context
        """
        params = intervention.parameters
        weights = params.get("weights")  # (hidden_dim,)
        bias = params.get("bias", 0.0)
        
        if weights is None or not isinstance(weights, torch.Tensor):
            return
        
        # Ensure on correct device/dtype
        weights = weights.to(device=hidden_states.device, dtype=hidden_states.dtype)
        
        # Compute probe score: score = h @ weights + bias
        # Average over batch and sequence dimensions
        # (batch, seq_len, hidden_dim) @ (hidden_dim,) -> (batch, seq_len)
        scores = torch.matmul(hidden_states, weights) + bias
        
        # Average score
        avg_score = float(scores.mean().item())
        
        # Emit probe score event
        if context.telemetry_enabled:
            probe_event = {
                "schema": "nram.hook.probe.v1",
                "timestamp": time.time(),
                "request_id": context.request_id,
                "probe_id": intervention.intervention_id,
                "layer_index": self.layer_index,
                "score": avg_score,
            }
            context.emit_telemetry_event(probe_event)
            print("NRAM_PROBE_EVENT " + json.dumps(probe_event, sort_keys=True), flush=True)
        
        # Update closed-loop state if configured
        if context.latent_loop_state is not None:
            # Store probe score for latent closed-loop controller
            context.latent_loop_state[f"probe_{intervention.intervention_id}_score"] = avg_score
    
    def _get_decode_step(self, context: NRAMHookContext) -> int:
        """Get current decode step from context."""
        # Decode step is tracked by invocation count for this layer
        # This is approximate; real decode step would come from scheduler
        return context.hook_invocation_count
    
    def enable(self) -> None:
        """Enable this hook."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable this hook."""
        self._enabled = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get hook status."""
        return {
            "hook_id": self.hook_id,
            "layer_index": self.layer_index,
            "module_name": self.module_name,
            "enabled": self._enabled,
            "invocation_count": self._invocation_count,
        }


# Import json for probe telemetry
import json
