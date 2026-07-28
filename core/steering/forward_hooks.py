"""
Forward hooks for SGLang to apply activation steering during inference.

This module provides hooks that can be registered with SGLang's forward pass
to apply activation vectors at specific layers during generation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)


class ActivationSteeringHook:
    """
    Forward hook that applies activation steering to hidden states.
    
    This hook is registered on transformer layers and modifies their output
    by adding contrastive activation vectors.
    """
    
    def __init__(
        self,
        controller,  # ActivationAdditionController
        layer_name: str,
        layer_index: int
    ):
        self.controller = controller
        self.layer_name = layer_name
        self.layer_index = layer_index
        self.enabled = True
        self.current_phase = None
        
    def __call__(
        self,
        module: nn.Module,
        input_args: tuple,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Hook function called during forward pass.
        
        Args:
            module: The layer module
            input_args: Input to the layer
            output: Output from the layer (hidden states)
        
        Returns:
            Modified output with activation steering applied
        """
        if not self.enabled or not self.controller.enabled:
            return output
        
        # Apply activation vectors
        modified_output = self.controller.apply_to_hidden_states(
            hidden_states=output,
            layer=self.layer_index,
            phase=self.current_phase
        )
        
        return modified_output
    
    def set_phase(self, phase: Optional[str]) -> None:
        """Set current generation phase for phase-aware steering."""
        self.current_phase = phase
    
    def enable(self) -> None:
        """Enable this hook."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable this hook."""
        self.enabled = False


class HookManager:
    """
    Manages forward hooks for activation steering.
    
    Handles registration, removal, and lifecycle of hooks on model layers.
    """
    
    def __init__(self):
        self.hooks: Dict[str, ActivationSteeringHook] = {}
        self.hook_handles: Dict[str, Any] = {}
        
    def register_hook(
        self,
        model: nn.Module,
        controller,  # ActivationAdditionController
        layer_name: str,
        layer_index: int
    ) -> ActivationSteeringHook:
        """
        Register a forward hook on a specific layer.
        
        Args:
            model: The transformer model
            controller: ActivationAdditionController instance
            layer_name: Name of the layer (e.g., "transformer.h.20")
            layer_index: Index of the layer
        
        Returns:
            The registered hook
        """
        # Get the target module
        target_module = self._get_module_by_name(model, layer_name)
        if target_module is None:
            raise ValueError(f"Layer not found: {layer_name}")
        
        # Create hook
        hook = ActivationSteeringHook(controller, layer_name, layer_index)
        
        # Register forward hook
        handle = target_module.register_forward_hook(hook)
        
        # Store references
        self.hooks[layer_name] = hook
        self.hook_handles[layer_name] = handle
        
        logger.info(f"Registered hook on layer {layer_name} (index {layer_index})")
        
        return hook
    
    def remove_hook(self, layer_name: str) -> None:
        """Remove a hook from a layer."""
        if layer_name in self.hook_handles:
            self.hook_handles[layer_name].remove()
            del self.hook_handles[layer_name]
            del self.hooks[layer_name]
            logger.info(f"Removed hook from layer {layer_name}")
    
    def remove_all_hooks(self) -> None:
        """Remove all registered hooks."""
        for layer_name in list(self.hook_handles.keys()):
            self.remove_hook(layer_name)
    
    def set_phase_all(self, phase: Optional[str]) -> None:
        """Set phase for all hooks."""
        for hook in self.hooks.values():
            hook.set_phase(phase)
    
    def enable_all(self) -> None:
        """Enable all hooks."""
        for hook in self.hooks.values():
            hook.enable()
    
    def disable_all(self) -> None:
        """Disable all hooks."""
        for hook in self.hooks.values():
            hook.disable()
    
    def _get_module_by_name(self, model: nn.Module, name: str) -> Optional[nn.Module]:
        """Get a module by its dotted name."""
        parts = name.split('.')
        module = model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all hooks."""
        return {
            name: {
                "enabled": hook.enabled,
                "layer_index": hook.layer_index,
                "phase": hook.current_phase
            }
            for name, hook in self.hooks.items()
        }


def find_transformer_layers(model: nn.Module) -> List[Dict[str, Any]]:
    """
    Find all transformer layers in a model.
    
    Returns list of dicts with layer info:
    - name: Layer name
    - index: Layer index
    - type: Layer type
    """
    layers = []
    
    # Common patterns for transformer layers
    layer_patterns = [
        ("transformer.h", "GPT-style"),
        ("model.layers", "LLaMA-style"),
        ("encoder.layer", "BERT-style"),
        ("gpt_neox.layers", "GPT-NeoX"),
    ]
    
    for pattern, style in layer_patterns:
        if hasattr(model, pattern.split('.')[0]):
            base = model
            for part in pattern.split('.'):
                if hasattr(base, part):
                    base = getattr(base, part)
                else:
                    break
            else:
                # Found the layer container
                if hasattr(base, '__iter__'):
                    for idx, layer in enumerate(base):
                        layers.append({
                            "name": f"{pattern}.{idx}",
                            "index": idx,
                            "type": style,
                            "module": layer
                        })
    
    return layers


def auto_register_hooks(
    model: nn.Module,
    controller,  # ActivationAdditionController
    layer_indices: Optional[List[int]] = None,
    style: str = "auto"
) -> HookManager:
    """
    Automatically register hooks on specified layers.
    
    Args:
        model: The transformer model
        controller: ActivationAdditionController instance
        layer_indices: Specific layer indices to hook (None = all)
        style: Model style ("auto", "gpt", "llama", "bert")
    
    Returns:
        HookManager with registered hooks
    """
    manager = HookManager()
    layers = find_transformer_layers(model)
    
    if not layers:
        logger.warning("No transformer layers found")
        return manager
    
    # Filter by indices if specified
    if layer_indices is not None:
        layers = [l for l in layers if l["index"] in layer_indices]
    
    # Register hooks
    for layer_info in layers:
        try:
            manager.register_hook(
                model=model,
                controller=controller,
                layer_name=layer_info["name"],
                layer_index=layer_info["index"]
            )
        except Exception as e:
            logger.error(f"Failed to register hook on {layer_info['name']}: {e}")
    
    logger.info(f"Registered {len(manager.hooks)} hooks")
    return manager
