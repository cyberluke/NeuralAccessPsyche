"""
DExperts (Democratized Experts) implementation for NRAM v5.

STATUS: NOT_IMPLEMENTED

DExperts combines multiple expert models at inference time:
- Base model: General-purpose language model
- Expert model: Specialized for desired behavior (e.g., creativity, technical accuracy)
- Anti-expert model: Specialized to avoid undesired behavior (e.g., corporate jargon, hallucination)

The final logits are computed as:
logits = base_logits + alpha * (expert_logits - base_logits) - beta * (anti_expert_logits - base_logits)

This allows dynamic steering without fine-tuning the base model.

IMPLEMENTATION STATUS:
----------------------
DExperts is NOT IMPLEMENTED in the current runtime due to VRAM constraints.

The RTX 4090 (24GB VRAM) cannot simultaneously load:
- Base model (Qwen3-14B-AWQ): ~14GB
- Expert model: ~14GB
- Anti-expert model: ~14GB

Sequential execution (making separate API calls for expert/anti-expert) would require
significant architectural changes to the logit processor, which runs inside SGLang and
cannot make HTTP calls back to itself.

ALTERNATIVE APPROACHES:
-----------------------
1. Use smaller expert models (e.g., 7B or 3B) that fit in remaining VRAM
2. Use activation addition or conceptor steering instead (already implemented)
3. Wait for multi-GPU support or model quantization advances
4. Implement prompt-based expert guidance (already available via persona profiles)

This module is retained for future implementation but should not be used in production.
"""

import torch
import torch.nn.functional as F
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Protocol, runtime_checkable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class DExpertsLogits:
    """Result from DExperts next-token logits computation."""
    combined_logits: torch.Tensor
    base_logits: torch.Tensor
    expert_logits: Optional[torch.Tensor] = None
    anti_expert_logits: Optional[torch.Tensor] = None
    alpha: float = 1.0
    beta: float = 0.5
    selected_token_id: Optional[int] = None
    latency_ms: float = 0.0
    peak_vram_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "selected_token_id": self.selected_token_id,
            "latency_ms": self.latency_ms,
            "peak_vram_mb": self.peak_vram_mb,
            "has_expert": self.expert_logits is not None,
            "has_anti_expert": self.anti_expert_logits is not None,
        }


@dataclass
class DExpertsState:
    """Per-request state for DExperts."""
    request_id: str = ""
    expert_name: Optional[str] = None
    alpha: float = 1.0
    beta: float = 0.5
    step_count: int = 0
    telemetry_events: List[Dict[str, Any]] = field(default_factory=list)
    
    def reset(self) -> None:
        self.step_count = 0
        self.telemetry_events.clear()


@runtime_checkable
class DExpertsBackend(Protocol):
    """Protocol for DExperts backend implementations."""
    
    def next_token_logits(
        self,
        input_ids: torch.Tensor,
        state: DExpertsState,
    ) -> DExpertsLogits:
        """Compute next-token logits using DExperts formula."""
        ...


class ExpertType(Enum):
    """Types of expert models."""
    CREATIVITY = "creativity"
    TECHNICAL = "technical"
    FORMAL = "formal"
    CASUAL = "casual"
    VISIONARY = "visionary"
    ANALYTICAL = "analytical"
    CUSTOM = "custom"


@dataclass
class ExpertConfig:
    """Configuration for an expert model."""
    name: str
    expert_type: ExpertType
    model_path: str
    alpha: float = 1.0  # Weight for expert
    beta: float = 0.5   # Weight for anti-expert (if applicable)
    active_phases: Optional[List[str]] = None
    description: str = ""


class DExpertsController:
    """
    Controller for DExperts inference-time steering.
    
    Manages base, expert, and anti-expert models and combines their outputs
    to achieve desired generation behavior.
    """
    
    def __init__(
        self,
        base_model: Any,
        base_tokenizer: Any,
        device: str = "cuda"
    ):
        """
        Initialize DExperts controller.
        
        Args:
            base_model: Base language model (already loaded)
            base_tokenizer: Base tokenizer
            device: Device to use
        """
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.device = device
        
        self.experts: Dict[str, ExpertConfig] = {}
        self.expert_models: Dict[str, Any] = {}
        self.anti_expert_models: Dict[str, Any] = {}
        
        self.current_phase: Optional[str] = None
        self.global_alpha: float = 1.0
        self.global_beta: float = 0.5
    
    def register_expert(
        self,
        name: str,
        expert_type: ExpertType,
        model_path: str,
        alpha: float = 1.0,
        beta: float = 0.5,
        active_phases: Optional[List[str]] = None,
        description: str = ""
    ) -> None:
        """
        Register an expert model.
        
        Args:
            name: Unique identifier for this expert
            expert_type: Type of expert (creativity, technical, etc.)
            model_path: Path to the expert model
            alpha: Weight for expert contribution
            beta: Weight for anti-expert contribution
            active_phases: Phases when this expert is active
            description: Human-readable description
        """
        config = ExpertConfig(
            name=name,
            expert_type=expert_type,
            model_path=model_path,
            alpha=alpha,
            beta=beta,
            active_phases=active_phases,
            description=description
        )
        self.experts[name] = config
        logger.info(f"Registered expert '{name}' of type {expert_type.value}")
    
    def load_expert_model(self, name: str) -> None:
        """
        Load an expert model from disk.
        
        Args:
            name: Name of the expert to load
        """
        if name not in self.experts:
            raise ValueError(f"Expert '{name}' not registered")
        
        config = self.experts[name]
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading expert model from {config.model_path}")
            
            # Load expert model
            expert_model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            expert_model = expert_model.to(self.device)
            expert_model.eval()
            
            self.expert_models[name] = expert_model
            
            logger.info(f"Loaded expert model '{name}'")
            
        except Exception as e:
            logger.error(f"Failed to load expert model '{name}': {e}")
            raise
    
    def load_anti_expert_model(self, name: str, model_path: str) -> None:
        """
        Load an anti-expert model.
        
        Args:
            name: Name of the expert this anti-expert is paired with
            model_path: Path to the anti-expert model
        """
        try:
            from transformers import AutoModelForCausalLM
            
            logger.info(f"Loading anti-expert model from {model_path}")
            
            anti_expert_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            anti_expert_model = anti_expert_model.to(self.device)
            anti_expert_model.eval()
            
            self.anti_expert_models[name] = anti_expert_model
            
            logger.info(f"Loaded anti-expert model for '{name}'")
            
        except Exception as e:
            logger.error(f"Failed to load anti-expert model for '{name}': {e}")
            raise
    
    def set_phase(self, phase: Optional[str]) -> None:
        """Set current generation phase."""
        self.current_phase = phase
    
    def set_global_weights(self, alpha: float, beta: float) -> None:
        """Set global weights for expert/anti-expert contribution."""
        self.global_alpha = alpha
        self.global_beta = beta
    
    def get_active_experts(self) -> List[Tuple[str, ExpertConfig]]:
        """Get experts active for current phase."""
        active = []
        for name, config in self.experts.items():
            if config.active_phases is None or self.current_phase in config.active_phases:
                active.append((name, config))
        return active
    
    @torch.no_grad()
    def get_base_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get logits from base model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            Logits from base model
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits[:, -1, :]  # Last token logits
    
    @torch.no_grad()
    def get_expert_logits(
        self,
        expert_name: str,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get logits from expert model.
        
        Args:
            expert_name: Name of the expert
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            Logits from expert model
        """
        if expert_name not in self.expert_models:
            raise ValueError(f"Expert model '{expert_name}' not loaded")
        
        expert_model = self.expert_models[expert_name]
        
        outputs = expert_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits[:, -1, :]
    
    @torch.no_grad()
    def get_anti_expert_logits(
        self,
        expert_name: str,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Get logits from anti-expert model.
        
        Args:
            expert_name: Name of the expert this anti-expert is paired with
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            Logits from anti-expert model, or None if not available
        """
        if expert_name not in self.anti_expert_models:
            return None
        
        anti_expert_model = self.anti_expert_models[expert_name]
        
        outputs = anti_expert_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits[:, -1, :]
    
    def combine_logits(
        self,
        base_logits: torch.Tensor,
        expert_logits: torch.Tensor,
        anti_expert_logits: Optional[torch.Tensor],
        alpha: float,
        beta: float
    ) -> torch.Tensor:
        """
        Combine logits using DExperts formula.
        
        Formula:
        logits = base + alpha * (expert - base) - beta * (anti_expert - base)
        
        Args:
            base_logits: Logits from base model
            expert_logits: Logits from expert model
            anti_expert_logits: Logits from anti-expert model (optional)
            alpha: Weight for expert contribution
            beta: Weight for anti-expert contribution
        
        Returns:
            Combined logits
        """
        # Expert contribution
        expert_delta = expert_logits - base_logits
        combined = base_logits + alpha * expert_delta
        
        # Anti-expert contribution (if available)
        if anti_expert_logits is not None:
            anti_expert_delta = anti_expert_logits - base_logits
            combined = combined - beta * anti_expert_delta
        
        return combined
    
    @torch.no_grad()
    def get_combined_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        expert_name: Optional[str] = None
    ) -> torch.Tensor:
        """
        Get combined logits from base, expert, and anti-expert models.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            expert_name: Specific expert to use (if None, uses active experts)
        
        Returns:
            Combined logits
        """
        # Get base logits
        base_logits = self.get_base_logits(input_ids, attention_mask)
        
        # Determine which expert(s) to use
        if expert_name is not None:
            experts_to_use = [(expert_name, self.experts[expert_name])]
        else:
            experts_to_use = self.get_active_experts()
        
        if not experts_to_use:
            # No active experts, return base logits
            return base_logits
        
        # Combine logits from all active experts
        combined_logits = base_logits.clone()
        
        for name, config in experts_to_use:
            # Get expert logits
            expert_logits = self.get_expert_logits(name, input_ids, attention_mask)
            
            # Get anti-expert logits (if available)
            anti_expert_logits = self.get_anti_expert_logits(name, input_ids, attention_mask)
            
            # Combine using DExperts formula
            expert_combined = self.combine_logits(
                base_logits=base_logits,
                expert_logits=expert_logits,
                anti_expert_logits=anti_expert_logits,
                alpha=config.alpha * self.global_alpha,
                beta=config.beta * self.global_beta
            )
            
            # Accumulate (simple averaging for multiple experts)
            combined_logits = (combined_logits + expert_combined) / 2
        
        return combined_logits
    
    def unload_expert_model(self, name: str) -> None:
        """
        Unload an expert model to free memory.
        
        Args:
            name: Name of the expert to unload
        """
        if name in self.expert_models:
            del self.expert_models[name]
            torch.cuda.empty_cache()
            logger.info(f"Unloaded expert model '{name}'")
        
        if name in self.anti_expert_models:
            del self.anti_expert_models[name]
            torch.cuda.empty_cache()
            logger.info(f"Unloaded anti-expert model for '{name}'")
    
    def get_status(self) -> Dict:
        """Get controller status."""
        return {
            "device": self.device,
            "current_phase": self.current_phase,
            "global_alpha": self.global_alpha,
            "global_beta": self.global_beta,
            "total_experts": len(self.experts),
            "loaded_experts": len(self.expert_models),
            "loaded_anti_experts": len(self.anti_expert_models),
            "active_experts": len(self.get_active_experts()),
            "experts": {
                name: {
                    "type": config.expert_type.value,
                    "alpha": config.alpha,
                    "beta": config.beta,
                    "loaded": name in self.expert_models,
                    "anti_expert_loaded": name in self.anti_expert_models,
                    "active_phases": config.active_phases,
                    "description": config.description
                }
                for name, config in self.experts.items()
            }
        }


# Pre-defined expert configurations
CREATIVITY_EXPERT_CONFIG = {
    "name": "creativity",
    "expert_type": ExpertType.CREATIVITY,
    "description": "Enhances creative and novel outputs"
}

TECHNICAL_EXPERT_CONFIG = {
    "name": "technical",
    "expert_type": ExpertType.TECHNICAL,
    "description": "Enhances technical accuracy and precision"
}

FORMAL_EXPERT_CONFIG = {
    "name": "formal",
    "expert_type": ExpertType.FORMAL,
    "description": "Enhances formal and professional tone"
}

VISIONARY_EXPERT_CONFIG = {
    "name": "visionary",
    "expert_type": ExpertType.VISIONARY,
    "description": "Enhances visionary and forward-thinking outputs"
}


def create_creativity_expert(
    controller: DExpertsController,
    model_path: str,
    anti_expert_path: Optional[str] = None,
    alpha: float = 1.0,
    beta: float = 0.5
) -> None:
    """
    Create and register a creativity expert.
    
    Args:
        controller: DExpertsController instance
        model_path: Path to creativity expert model
        anti_expert_path: Path to anti-expert model (e.g., boring/corporate text)
        alpha: Weight for expert
        beta: Weight for anti-expert
    """
    controller.register_expert(
        name="creativity",
        expert_type=ExpertType.CREATIVITY,
        model_path=model_path,
        alpha=alpha,
        beta=beta,
        description="Enhances creative and novel outputs"
    )
    
    controller.load_expert_model("creativity")
    
    if anti_expert_path:
        controller.load_anti_expert_model("creativity", anti_expert_path)


def create_technical_expert(
    controller: DExpertsController,
    model_path: str,
    anti_expert_path: Optional[str] = None,
    alpha: float = 1.0,
    beta: float = 0.5
) -> None:
    """
    Create and register a technical expert.
    
    Args:
        controller: DExpertsController instance
        model_path: Path to technical expert model
        anti_expert_path: Path to anti-expert model (e.g., vague/inaccurate text)
        alpha: Weight for expert
        beta: Weight for anti-expert
    """
    controller.register_expert(
        name="technical",
        expert_type=ExpertType.TECHNICAL,
        model_path=model_path,
        alpha=alpha,
        beta=beta,
        description="Enhances technical accuracy and precision"
    )
    
    controller.load_expert_model("technical")
    
    if anti_expert_path:
        controller.load_anti_expert_model("technical", anti_expert_path)
