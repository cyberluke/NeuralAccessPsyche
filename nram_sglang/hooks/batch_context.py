"""
Batch-aware request context for NRAM hooks.

SGLang batches multiple sequences per forward pass. Thread-local storage
cannot distinguish which request a tensor row belongs to. This module
provides batch-aware context that maps tensor rows to request IDs using
SGLang's forward batch metadata.

Architecture:
- BatchContextManager maintains registry of active request contexts
- Forward hooks receive forward_batch metadata with rid_to_index mapping
- Device-resident masks enable efficient per-request intervention
- Supports both prefill (flattened [total_tokens, hidden_dim]) and
  decode ([batch, hidden_dim]) phases

Critical: This replaces thread-local storage for SGLang compatibility.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import logging

import torch

from nram_sglang.hooks.request_context import (
    NRAMHookContext,
    InterventionConfig,
    context_manager as legacy_context_manager,
)

logger = logging.getLogger(__name__)


@dataclass
class BatchRequestState:
    """Per-request state within a batch."""
    request_id: str
    context: NRAMHookContext
    batch_index: int  # Position in current batch
    token_indices: List[int]  # Token positions in flattened prefill tensor
    decode_step: int = 0
    created_at: float = field(default_factory=time.time)
    
    def advance_decode_step(self) -> None:
        """Advance decode step counter."""
        self.decode_step += 1


class BatchContextManager:
    """
    Manages request contexts with batch-aware lookup.
    
    Replaces thread-local storage for SGLang compatibility. Maintains
    registry of active contexts and provides batch-row to request mapping.
    """
    
    def __init__(self):
        self._active_requests: Dict[str, BatchRequestState] = {}
        self._batch_index_to_request: Dict[int, str] = {}
        self._lock = __import__('threading').Lock()
        self._current_batch_size: int = 0
        
    def register_request(
        self,
        request_id: str,
        context: NRAMHookContext,
        batch_index: int,
        token_indices: Optional[List[int]] = None,
    ) -> BatchRequestState:
        """
        Register a request in the current batch.
        
        Args:
            request_id: Unique request identifier
            context: Request-scoped hook context
            batch_index: Position in current batch (0-based)
            token_indices: Token positions for prefill phase
        
        Returns:
            BatchRequestState for this request
        """
        with self._lock:
            if request_id in self._active_requests:
                logger.warning(f"Request {request_id} already registered, updating")
                self._cleanup_request(request_id)
            
            state = BatchRequestState(
                request_id=request_id,
                context=context,
                batch_index=batch_index,
                token_indices=token_indices or [],
            )
            
            self._active_requests[request_id] = state
            self._batch_index_to_request[batch_index] = request_id
            self._current_batch_size = max(self._current_batch_size, batch_index + 1)
            
            logger.debug(
                f"Registered request {request_id} at batch_index={batch_index}, "
                f"tokens={len(token_indices or [])}"
            )
            
            return state
    
    def get_request_by_batch_index(self, batch_index: int) -> Optional[BatchRequestState]:
        """Get request state by batch index."""
        with self._lock:
            request_id = self._batch_index_to_request.get(batch_index)
            if request_id is None:
                return None
            return self._active_requests.get(request_id)
    
    def get_request_by_id(self, request_id: str) -> Optional[BatchRequestState]:
        """Get request state by request ID."""
        with self._lock:
            return self._active_requests.get(request_id)
    
    def get_context_by_batch_index(self, batch_index: int) -> Optional[NRAMHookContext]:
        """Get hook context by batch index."""
        state = self.get_request_by_batch_index(batch_index)
        return state.context if state else None
    
    def build_request_mask(
        self,
        tensor_shape: torch.Size,
        request_id: str,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Build a device-resident mask for a specific request.
        
        For decode phase: mask is [batch_size] with 1.0 at request's batch_index
        For prefill phase: mask is [total_tokens] with 1.0 at request's token_indices
        
        Args:
            tensor_shape: Shape of hidden states tensor
            request_id: Request to build mask for
            device: Target device
        
        Returns:
            Boolean mask tensor, or None if request not found
        """
        state = self.get_request_by_id(request_id)
        if state is None:
            return None
        
        if len(tensor_shape) == 2:
            # Prefill: [total_tokens, hidden_dim]
            total_tokens = tensor_shape[0]
            mask = torch.zeros(total_tokens, dtype=torch.bool, device=device)
            if state.token_indices:
                mask[state.token_indices] = True
            return mask
        elif len(tensor_shape) == 1:
            # Decode: [batch_size, hidden_dim] flattened to [batch_size]
            batch_size = tensor_shape[0]
            mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            if 0 <= state.batch_index < batch_size:
                mask[state.batch_index] = True
            return mask
        else:
            logger.warning(f"Unexpected tensor shape for mask: {tensor_shape}")
            return None
    
    def build_all_masks(
        self,
        tensor_shape: torch.Size,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Build masks for all active requests.
        
        Returns:
            Dict mapping request_id to mask tensor
        """
        masks = {}
        with self._lock:
            for request_id, state in self._active_requests.items():
                mask = self.build_request_mask(tensor_shape, request_id, device)
                if mask is not None:
                    masks[request_id] = mask
        return masks
    
    def advance_decode_step(self, request_id: str) -> None:
        """Advance decode step for a request."""
        state = self.get_request_by_id(request_id)
        if state:
            state.advance_decode_step()
    
    def get_decode_step(self, request_id: str) -> int:
        """Get current decode step for a request."""
        state = self.get_request_by_id(request_id)
        return state.decode_step if state else 0
    
    def complete_request(self, request_id: str) -> None:
        """Mark request as complete and clean up."""
        self._cleanup_request(request_id)
        logger.info(f"Completed request {request_id}")
    
    def _cleanup_request(self, request_id: str) -> None:
        """Internal cleanup of request state."""
        with self._lock:
            state = self._active_requests.pop(request_id, None)
            if state:
                self._batch_index_to_request.pop(state.batch_index, None)
                state.context.reset()
    
    def clear_batch(self) -> None:
        """Clear all requests in current batch."""
        with self._lock:
            for state in self._active_requests.values():
                state.context.reset()
            self._active_requests.clear()
            self._batch_index_to_request.clear()
            self._current_batch_size = 0
            logger.debug("Cleared batch context")
    
    def get_active_request_ids(self) -> Set[str]:
        """Get set of active request IDs."""
        with self._lock:
            return set(self._active_requests.keys())
    
    def get_batch_size(self) -> int:
        """Get current batch size."""
        with self._lock:
            return self._current_batch_size
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of batch context manager."""
        with self._lock:
            return {
                "active_requests": len(self._active_requests),
                "batch_size": self._current_batch_size,
                "requests": {
                    rid: {
                        "batch_index": state.batch_index,
                        "token_count": len(state.token_indices),
                        "decode_step": state.decode_step,
                    }
                    for rid, state in self._active_requests.items()
                },
            }


# Global batch context manager
batch_context_manager = BatchContextManager()


def register_nram_request(
    request_id: str,
    interventions: List[InterventionConfig],
    batch_index: int,
    token_indices: Optional[List[int]] = None,
    telemetry_enabled: bool = False,
) -> NRAMHookContext:
    """
    Register an NRAM request in the batch context.
    
    This is the primary entry point for setting up request-scoped
    interventions before generation begins.
    
    Args:
        request_id: Unique request identifier
        interventions: List of intervention configurations
        batch_index: Position in current batch
        token_indices: Token positions for prefill phase
        telemetry_enabled: Whether to emit telemetry
    
    Returns:
        NRAMHookContext for this request
    """
    # Create legacy context for compatibility
    context = legacy_context_manager.create_context(request_id)
    context.telemetry_enabled = telemetry_enabled
    
    for intervention in interventions:
        context.add_intervention(intervention)
    
    # Register in batch context
    batch_context_manager.register_request(
        request_id=request_id,
        context=context,
        batch_index=batch_index,
        token_indices=token_indices,
    )
    
    logger.info(
        f"Registered NRAM request {request_id} with {len(interventions)} interventions"
    )
    
    return context


def complete_nram_request(request_id: str) -> None:
    """
    Complete an NRAM request and clean up state.
    
    Called after generation completes (success, failure, or abort).
    """
    batch_context_manager.complete_request(request_id)
    legacy_context_manager.remove_context(request_id)


def get_context_for_batch_index(batch_index: int) -> Optional[NRAMHookContext]:
    """
    Get hook context for a specific batch index.
    
    This is called by forward hooks to access request-scoped configuration.
    Replaces thread-local get_current_context().
    
    Args:
        batch_index: Position in current batch
    
    Returns:
        NRAMHookContext or None if no request at this index
    """
    return batch_context_manager.get_context_by_batch_index(batch_index)
