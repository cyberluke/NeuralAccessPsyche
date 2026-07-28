"""
Factory for registering NRAM hooks on Qwen model layers.

This module provides the interface to register forward hooks on the actual
Qwen decoder layers via model.named_modules(). The hooks execute INSIDE
the model forward pass to modify hidden states.

Architecture:
- Factory inspects model.named_modules() to find decoder layers
- Registers QwenDecoderHook on each target layer
- Returns HookManager for lifecycle management
- Hooks are active for all requests but read request-specific config

Critical: This is the bridge between SGLang's model loading and NRAM's
hidden-state interventions. Without this, no real interventions execute.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import torch
import torch.nn as nn

from nram_sglang.hooks.qwen_hook import QwenDecoderHook
from nram_sglang.hooks.request_context import context_manager, NRAMHookContext, InterventionConfig

logger = logging.getLogger(__name__)


class NRAMHookFactory:
    """
    Factory for creating and managing NRAM hooks on transformer models.
    
    Inspects the model architecture to find decoder layers and registers
    forward hooks that modify hidden states during the forward pass.
    """
    
    def __init__(self):
        self._hooks: Dict[str, QwenDecoderHook] = {}
        self._hook_handles: Dict[str, Any] = {}
        self._model: Optional[nn.Module] = None
    
    def register_hooks_on_model(
        self,
        model: nn.Module,
        target_layers: Optional[List[int]] = None,
    ) -> int:
        """
        Register forward hooks on decoder layers.
        
        Args:
            model: The transformer model (e.g., Qwen3ForCausalLM)
            target_layers: Specific layer indices to hook (None = all decoder layers)
        
        Returns:
            Number of hooks registered
        """
        self._model = model
        registered_count = 0
        
        # Find decoder layers using common patterns
        decoder_layers = self._find_decoder_layers(model)
        
        if not decoder_layers:
            logger.error("No decoder layers found in model")
            return 0
        
        logger.info(f"Found {len(decoder_layers)} decoder layers")
        
        # Filter by target_layers if specified
        if target_layers is not None:
            decoder_layers = [
                (name, idx, module)
                for name, idx, module in decoder_layers
                if idx in target_layers
            ]
            logger.info(f"Filtered to {len(decoder_layers)} target layers")
        
        # Register hook on each layer
        for layer_name, layer_index, layer_module in decoder_layers:
            try:
                hook = QwenDecoderHook(layer_index=layer_index, module_name=layer_name)
                handle = layer_module.register_forward_hook(hook)
                
                self._hooks[layer_name] = hook
                self._hook_handles[layer_name] = handle
                registered_count += 1
                
                logger.info(f"Registered hook on layer {layer_name} (index {layer_index})")
                
            except Exception as e:
                logger.error(f"Failed to register hook on {layer_name}: {e}")
        
        logger.info(f"Successfully registered {registered_count} hooks")
        return registered_count
    
    def _find_decoder_layers(self, model: nn.Module) -> List[Tuple[str, int, nn.Module]]:
        """
        Find decoder layers in the model using named_modules().
        
        Supports common transformer architectures:
        - Qwen: model.layers.N
        - LLaMA: model.layers.N
        - GPT: transformer.h.N
        
        Returns:
            List of (layer_name, layer_index, layer_module) tuples
        """
        decoder_layers = []
        
        # Pattern 1: model.layers.N (Qwen, LLaMA)
        pattern1 = re.compile(r"^model\.layers\.(\d+)$")
        
        # Pattern 2: transformer.h.N (GPT-style)
        pattern2 = re.compile(r"^transformer\.h\.(\d+)$")
        
        # Pattern 3: gpt_neox.layers.N (GPT-NeoX)
        pattern3 = re.compile(r"^gpt_neox\.layers\.(\d+)$")
        
        for name, module in model.named_modules():
            # Check pattern 1
            match = pattern1.match(name)
            if match:
                layer_index = int(match.group(1))
                decoder_layers.append((name, layer_index, module))
                continue
            
            # Check pattern 2
            match = pattern2.match(name)
            if match:
                layer_index = int(match.group(1))
                decoder_layers.append((name, layer_index, module))
                continue
            
            # Check pattern 3
            match = pattern3.match(name)
            if match:
                layer_index = int(match.group(1))
                decoder_layers.append((name, layer_index, module))
                continue
        
        # Sort by layer index
        decoder_layers.sort(key=lambda x: x[1])
        
        return decoder_layers
    
    def remove_all_hooks(self) -> None:
        """Remove all registered hooks."""
        for layer_name, handle in self._hook_handles.items():
            try:
                handle.remove()
                logger.info(f"Removed hook from {layer_name}")
            except Exception as e:
                logger.error(f"Failed to remove hook from {layer_name}: {e}")
        
        self._hooks.clear()
        self._hook_handles.clear()
        logger.info("Removed all hooks")
    
    def get_hook(self, layer_name: str) -> Optional[QwenDecoderHook]:
        """Get a hook by layer name."""
        return self._hooks.get(layer_name)
    
    def get_all_hooks(self) -> Dict[str, QwenDecoderHook]:
        """Get all registered hooks."""
        return self._hooks.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all hooks."""
        return {
            "registered_hooks": len(self._hooks),
            "hooks": {
                name: hook.get_status()
                for name, hook in self._hooks.items()
            },
        }
    
    def enable_all_hooks(self) -> None:
        """Enable all hooks."""
        for hook in self._hooks.values():
            hook.enable()
    
    def disable_all_hooks(self) -> None:
        """Disable all hooks."""
        for hook in self._hooks.values():
            hook.disable()


class HookManager:
    """
    High-level manager for NRAM hooks.
    
    Provides a simple interface for:
    - Registering hooks on a model
    - Creating request contexts
    - Installing/removing contexts
    - Querying hook status
    """
    
    def __init__(self):
        self.factory = NRAMHookFactory()
        self.context_manager = context_manager
    
    def register_on_model(
        self,
        model: nn.Module,
        target_layers: Optional[List[int]] = None,
    ) -> int:
        """Register hooks on a model. Returns number of hooks registered."""
        return self.factory.register_hooks_on_model(model, target_layers)
    
    def create_request_context(self, request_id: str) -> NRAMHookContext:
        """Create a new request context."""
        return self.context_manager.create_context(request_id)
    
    def install_request_context(self, request_id: str) -> None:
        """Install a request context as current thread-local context."""
        self.context_manager.install_context(request_id)
    
    def remove_request_context(self, request_id: str) -> None:
        """Remove a request context."""
        self.context_manager.remove_context(request_id)
    
    def get_request_context(self, request_id: str) -> Optional[NRAMHookContext]:
        """Get a request context by ID."""
        return self.context_manager.get_context(request_id)
    
    def cleanup_stale_contexts(self, max_age_seconds: float = 300.0) -> int:
        """Clean up stale contexts. Returns count removed."""
        return self.context_manager.cleanup_stale_contexts(max_age_seconds)
    
    def remove_all_hooks(self) -> None:
        """Remove all registered hooks."""
        self.factory.remove_all_hooks()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of hooks and contexts."""
        return {
            "hooks": self.factory.get_status(),
            "active_contexts": len(self.context_manager._active_contexts),
        }


# Global hook manager instance
hook_manager = HookManager()


def make_nram_hook(config: Dict[str, Any]) -> "SGLangCompatibleNRAMHook":
    """
    Factory function for SGLang's --forward-hooks mechanism.
    
    CRITICAL: SGLang's register_forward_hooks() registers the SAME hook instance
    on ALL matched modules. The hook must determine which layer it's on dynamically
    from the module argument passed to __call__.
    
    Args:
        config: Configuration dict from the hook spec, e.g.:
            {"enabled": True}
    
    Returns:
        SGLangCompatibleNRAMHook instance that determines layer dynamically
    
    Example:
        --forward-hooks '[{"name":"nram_actadd","target_modules":["model.layers.*"],"hook_factory":"nram_sglang.hooks.factory:make_nram_hook","config":{"enabled":true}}]'
    """
    enabled = config.get("enabled", True)
    
    # Create a hook that determines layer index from module at call time
    hook = SGLangCompatibleNRAMHook(enabled=enabled)
    
    logger.info(f"Created SGLang-compatible NRAM hook with config: {config}")
    return hook


class SGLangCompatibleNRAMHook:
    """
    NRAM hook compatible with SGLang's forward hook registration pattern.
    
    SGLang registers ONE hook instance on ALL matched modules. This hook
    determines which layer it's on by inspecting the module argument at
    call time, not at registration time.
    
    The hook extracts layer index from the module's class name or attributes,
    then looks up request-scoped interventions from a global context manager
    (not thread-local, which is incompatible with async batching).
    """
    
    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._invocation_count = 0
        self._layer_cache: Dict[int, int] = {}  # module_id -> layer_index
    
    def __call__(
        self,
        module: nn.Module,
        input_args: Tuple[Any, ...],
        output: Any,
    ) -> Any:
        """
        Hook called by PyTorch after layer forward pass.
        
        Determines layer index from module, then applies interventions
        from the current request context.
        """
        if not self._enabled:
            return output
        
        # Determine which layer this is
        layer_index = self._get_layer_index(module)
        module_name = self._get_module_name(module)
        
        # Get current request context (global, not thread-local)
        from nram_sglang.hooks.request_context import get_current_context_for_batch
        context = get_current_context_for_batch()
        
        if context is None:
            # No active request - baseline pass, no intervention
            return output
        
        # Record invocation
        context.record_hook_invocation()
        self._invocation_count += 1
        
        # Determine phase (prefill vs decode) from input shape
        phase = self._determine_phase(input_args)
        interventions = context.get_interventions_for_layer(layer_index, phase)
        
        if not interventions:
            # No interventions for this layer/phase - pass through
            return output
        
        # Extract hidden states from output
        hidden_states, output_structure = self._extract_hidden_states(output)
        if hidden_states is None:
            logger.warning(f"Could not extract hidden states from layer {module_name}")
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
            from nram_sglang.hooks.telemetry import HookTelemetry
            decode_step = self._get_decode_step(context)
            HookTelemetry.emit_hook_invocation(
                request_id=context.request_id,
                hook_id=f"nram_hook_layer_{layer_index}",
                module_name=module_name,
                layer_index=layer_index,
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
    
    def _get_layer_index(self, module: nn.Module) -> int:
        """
        Determine layer index from module.
        
        Tries multiple strategies:
        1. Check if module has layer_idx attribute (Qwen3DecoderLayer)
        2. Extract from module name (model.layers.N)
        3. Cache by module id
        """
        module_id = id(module)
        if module_id in self._layer_cache:
            return self._layer_cache[module_id]
        
        # Strategy 1: Check for layer_idx attribute
        if hasattr(module, "layer_idx"):
            layer_index = module.layer_idx
            self._layer_cache[module_id] = layer_index
            return layer_index
        
        # Strategy 2: Extract from module name
        module_name = self._get_module_name(module)
        import re
        match = re.search(r"layers?\.(\d+)", module_name)
        if match:
            layer_index = int(match.group(1))
            self._layer_cache[module_id] = layer_index
            return layer_index
        
        # Fallback: unknown layer
        return -1
    
    def _get_module_name(self, module: nn.Module) -> str:
        """Get module name from module attributes."""
        if hasattr(module, "_module_name"):
            return module._module_name
        return module.__class__.__name__
    
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
        """Extract hidden states tensor from layer output."""
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
            structure["type"] = "dataclass"
            return output.last_hidden_state, structure
        if hasattr(output, "hidden_states"):
            structure["type"] = "dataclass"
            return output.hidden_states, structure
        
        return None, structure
    
    def _reconstruct_output(
        self,
        original_output: Any,
        structure: Dict[str, Any],
        modified_hidden_states: torch.Tensor,
    ) -> Any:
        """Reconstruct output with modified hidden states."""
        if structure["type"] == "tensor":
            return modified_hidden_states
        
        if structure["type"] == "tuple":
            # Replace first element of tuple
            return (modified_hidden_states,) + original_output[1:]
        
        if structure["type"] == "dataclass":
            # Modify dataclass attribute
            if hasattr(original_output, "last_hidden_state"):
                original_output.last_hidden_state = modified_hidden_states
            elif hasattr(original_output, "hidden_states"):
                original_output.hidden_states = modified_hidden_states
            return original_output
        
        return original_output
    
    def _apply_intervention(
        self,
        hidden_states: torch.Tensor,
        intervention: "InterventionConfig",
        context: "NRAMHookContext",
    ) -> Tuple[Optional[torch.Tensor], float]:
        """
        Apply a single intervention to hidden states.
        
        Returns:
            Tuple of (delta_tensor, delta_norm) or (None, 0.0) if not applicable
        """
        if intervention.intervention_type == "activation_addition":
            # Load activation vector from parameters
            vector_path = intervention.parameters.get("vector_path")
            strength = intervention.strength
            
            if vector_path and Path(vector_path).exists():
                vector = torch.load(vector_path, map_location=hidden_states.device)
                # Broadcast vector to match hidden states shape
                if vector.dim() == 1:
                    vector = vector.unsqueeze(0).unsqueeze(0)
                delta = strength * vector.to(hidden_states.dtype)
                delta_norm = float(delta.norm().item())
                return delta, delta_norm
        
        # Unknown intervention type - no-op
        return None, 0.0
    
    def _get_decode_step(self, context: "NRAMHookContext") -> int:
        """Get current decode step from context."""
        # Simple counter based on hook invocations
        return context.hook_invocation_count
    
    def disable(self) -> None:
        """Disable the hook."""
        self._enabled = False
    
    def enable(self) -> None:
        """Enable the hook."""
        self._enabled = True


def register_nram_hooks_on_model(model: nn.Module, config: Optional[Dict[str, Any]] = None) -> int:
    """
    Register NRAM hooks on all decoder layers of a model.
    
    This is the main entry point for SGLang integration. It should be called
    after the model is loaded to register hooks on all decoder layers.
    
    Args:
        model: The loaded transformer model (e.g., Qwen3ForCausalLM)
        config: Optional configuration dict
    
    Returns:
        Number of hooks registered
    """
    manager = HookManager()
    count = manager.register_on_model(model)
    logger.info(f"Registered {count} NRAM hooks on model")
    return count
