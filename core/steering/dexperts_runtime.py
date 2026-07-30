"""
True DExperts runtime for SGLang logit processor.

Implements the DExperts equation at the logit level:
    z_combined = z_base + alpha * (z_expert - z_anti_expert)

This runs INSIDE the SGLang custom logit processor, receiving logits
from the base model and applying expert/anti-expert deltas.

Architecture:
- LoRA adapters are loaded at server startup
- Per-token: run base forward (already done by SGLang), then compute
  expert and anti-expert logits using the same input_ids
- Apply DExperts formula to modify logits before sampling
- Maintain separate KV state for expert/anti-expert via adapter switching

Critical: This is NOT three separate API calls. It operates on the
logits tensor directly inside the SGLang process.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class DExpertsRuntimeState:
    """Per-request DExperts state."""
    request_id: str = ""
    alpha: float = 1.0
    enabled: bool = False
    step_count: int = 0
    telemetry_events: List[Dict[str, Any]] = field(default_factory=list)
    
    def reset(self) -> None:
        self.step_count = 0
        self.telemetry_events.clear()


class DExpertsRuntime:
    """
    True DExperts runtime for SGLang.
    
    Loads LoRA adapters and applies the DExperts formula:
        z_combined = z_base + alpha * (z_expert - z_anti_expert)
    
    The base logits come from SGLang's normal forward pass.
    Expert and anti-expert logits are computed by switching LoRA adapters
    and running forward on the same input_ids.
    """
    
    def __init__(
        self,
        base_model: Any,
        tokenizer: Any,
        device: str = "cuda",
    ):
        """
        Initialize DExperts runtime.
        
        Args:
            base_model: The base language model (already loaded by SGLang)
            tokenizer: The tokenizer
            device: Device to use
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        
        # LoRA adapters (loaded lazily)
        self.expert_adapter = None
        self.anti_expert_adapter = None
        
        # Adapter paths
        self.expert_adapter_path: Optional[str] = None
        self.anti_expert_adapter_path: Optional[str] = None
        
        # State
        self._loaded = False
        self._request_states: Dict[str, DExpertsRuntimeState] = {}
        
        logger.info("DExperts runtime initialized (adapters not yet loaded)")
    
    def load_adapters(
        self,
        expert_adapter_path: str,
        anti_expert_adapter_path: str,
    ) -> None:
        """
        Load LoRA adapters for expert and anti-expert.
        
        Args:
            expert_adapter_path: Path to non-toxic expert adapter
            anti_expert_adapter_path: Path to toxic anti-expert adapter
        """
        from peft import PeftModel
        
        logger.info(f"Loading expert adapter from {expert_adapter_path}")
        self.expert_adapter = PeftModel.from_pretrained(
            self.base_model,
            expert_adapter_path,
            adapter_name="expert",
        )
        self.expert_adapter_path = expert_adapter_path
        
        logger.info(f"Loading anti-expert adapter from {anti_expert_adapter_path}")
        self.anti_expert_adapter = PeftModel.from_pretrained(
            self.base_model,
            anti_expert_adapter_path,
            adapter_name="anti_expert",
        )
        self.anti_expert_adapter_path = anti_expert_adapter_path
        
        self._loaded = True
        logger.info("DExperts adapters loaded successfully")
    
    def get_or_create_state(self, request_id: str) -> DExpertsRuntimeState:
        """Get or create per-request state."""
        if request_id not in self._request_states:
            self._request_states[request_id] = DExpertsRuntimeState(request_id=request_id)
        return self._request_states[request_id]
    
    def release_state(self, request_id: str) -> None:
        """Release per-request state after completion/abort."""
        if request_id in self._request_states:
            del self._request_states[request_id]
            logger.debug(f"Released DExperts state for request {request_id}")
    
    @torch.no_grad()
    def compute_expert_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute logits using expert adapter.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            Logits from expert model (last token position)
        """
        if not self._loaded or self.expert_adapter is None:
            raise RuntimeError("Expert adapter not loaded")
        
        # Switch to expert adapter
        self.expert_adapter.set_adapter("expert")
        
        # Forward pass
        outputs = self.expert_adapter(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits[:, -1, :]  # Last token logits
    
    @torch.no_grad()
    def compute_anti_expert_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute logits using anti-expert adapter.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            Logits from anti-expert model (last token position)
        """
        if not self._loaded or self.anti_expert_adapter is None:
            raise RuntimeError("Anti-expert adapter not loaded")
        
        # Switch to anti-expert adapter
        self.anti_expert_adapter.set_adapter("anti_expert")
        
        # Forward pass
        outputs = self.anti_expert_adapter(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits[:, -1, :]  # Last token logits
    
    @torch.no_grad()
    def apply_dexperts(
        self,
        base_logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        request_id: str = "",
        position: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply DExperts formula to base logits.
        
        z_combined = z_base + alpha * (z_expert - z_anti_expert)
        
        Args:
            base_logits: Logits from base model (already computed by SGLang)
            input_ids: Input token IDs (for expert/anti-expert forward)
            attention_mask: Attention mask
            alpha: Steering strength
            request_id: Request identifier for telemetry
            position: Token position for telemetry
        
        Returns:
            Tuple of (combined_logits, telemetry_dict)
        """
        if not self._loaded:
            logger.warning("DExperts not loaded, returning base logits")
            return base_logits, {"applied": False, "reason": "not_loaded"}
        
        start_time = time.perf_counter()
        
        # Compute expert and anti-expert logits
        expert_logits = self.compute_expert_logits(input_ids, attention_mask)
        anti_expert_logits = self.compute_anti_expert_logits(input_ids, attention_mask)
        
        # Apply DExperts formula
        # z_combined = z_base + alpha * (z_expert - z_anti_expert)
        expert_delta = expert_logits - anti_expert_logits
        combined_logits = base_logits + alpha * expert_delta
        
        # Compute telemetry
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Top tokens for telemetry
        base_top_ids = torch.topk(base_logits, k=5, dim=-1).indices[0].tolist()
        expert_top_ids = torch.topk(expert_logits, k=5, dim=-1).indices[0].tolist()
        anti_expert_top_ids = torch.topk(anti_expert_logits, k=5, dim=-1).indices[0].tolist()
        
        # Norms for telemetry
        expert_minus_anti_norm = float(torch.norm(expert_delta).item())
        combined_delta_norm = float(torch.norm(combined_logits - base_logits).item())
        
        telemetry = {
            "request_id": request_id,
            "position": position,
            "alpha": alpha,
            "base_top_token_ids": base_top_ids,
            "expert_top_token_ids": expert_top_ids,
            "anti_expert_top_token_ids": anti_expert_top_ids,
            "expert_minus_anti_norm": expert_minus_anti_norm,
            "combined_delta_norm": combined_delta_norm,
            "latency_ms": latency_ms,
            "backend": "sglang_lora",
            "device": self.device,
            "applied": True,
        }
        
        # Update request state
        if request_id:
            state = self.get_or_create_state(request_id)
            state.step_count += 1
            state.telemetry_events.append(telemetry)
        
        return combined_logits, telemetry
    
    def get_status(self) -> Dict[str, Any]:
        """Get runtime status."""
        return {
            "loaded": self._loaded,
            "device": self.device,
            "expert_adapter_path": self.expert_adapter_path,
            "anti_expert_adapter_path": self.anti_expert_adapter_path,
            "active_requests": len(self._request_states),
        }
