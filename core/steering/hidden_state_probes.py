"""
Hidden-state probes for real-time monitoring and closed-loop control.

Probes are lightweight linear classifiers that monitor specific properties
of hidden states during generation, enabling adaptive steering based on
real-time feedback.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ProbeType(Enum):
    """Types of probes for different monitoring tasks."""
    NOVELTY = "novelty"  # How novel/creative is the current state?
    COHERENCE = "coherence"  # How coherent is the current state?
    SOURCE_SIMILARITY = "source_similarity"  # Similarity to source text
    FACTUALITY = "factuality"  # How factual vs. hallucinatory?
    SENTIMENT = "sentiment"  # Emotional tone
    COMPLEXITY = "complexity"  # Linguistic complexity
    CUSTOM = "custom"  # User-defined probe


@dataclass
class ProbeConfig:
    """Configuration for a hidden-state probe."""
    name: str
    probe_type: ProbeType
    layer: int
    threshold: float = 0.5
    weight: float = 1.0
    active: bool = True


class HiddenStateProbe(nn.Module):
    """
    Linear probe for monitoring hidden states.
    
    A probe is a simple linear classifier that maps hidden states to
    scalar values representing specific properties.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        probe_type: ProbeType = ProbeType.CUSTOM,
        device: str = "cuda"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.probe_type = probe_type
        self.device = device
        
        # Linear probe: hidden_dim -> 1
        self.linear = nn.Linear(hidden_dim, 1).to(device)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute probe output.
        
        Args:
            hidden_states: Shape (batch_size, seq_len, hidden_dim)
        
        Returns:
            Probe scores of shape (batch_size, seq_len)
        """
        # Apply linear transformation
        scores = self.linear(hidden_states)  # (batch, seq, 1)
        scores = scores.squeeze(-1)  # (batch, seq)
        
        # Apply sigmoid for bounded output [0, 1]
        scores = torch.sigmoid(scores)
        
        return scores
    
    def probe_sequence(self, hidden_states: torch.Tensor) -> float:
        """
        Probe entire sequence and return mean score.
        
        Args:
            hidden_states: Shape (batch_size, seq_len, hidden_dim)
        
        Returns:
            Mean probe score
        """
        scores = self.forward(hidden_states)
        return scores.mean().item()
    
    def probe_last_token(self, hidden_states: torch.Tensor) -> float:
        """
        Probe only the last token.
        
        Args:
            hidden_states: Shape (batch_size, seq_len, hidden_dim)
        
        Returns:
            Probe score for last token
        """
        scores = self.forward(hidden_states)
        return scores[:, -1].mean().item()


class ProbeController:
    """
    Controller for managing multiple probes and implementing closed-loop control.
    
    Features:
    - Multiple probes for different properties
    - Threshold-based triggering
    - Callback system for adaptive steering
    - Real-time monitoring
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.probes: Dict[str, Tuple[HiddenStateProbe, ProbeConfig]] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        self.current_values: Dict[str, float] = {}
    
    def add_probe(
        self,
        name: str,
        probe: HiddenStateProbe,
        probe_type: ProbeType,
        layer: int,
        threshold: float = 0.5,
        weight: float = 1.0
    ) -> None:
        """Add a probe with configuration."""
        config = ProbeConfig(
            name=name,
            probe_type=probe_type,
            layer=layer,
            threshold=threshold,
            weight=weight
        )
        self.probes[name] = (probe, config)
        self.callbacks[name] = []
        logger.info(f"Added probe '{name}' of type {probe_type.value}")
    
    def remove_probe(self, name: str) -> None:
        """Remove a probe."""
        if name in self.probes:
            del self.probes[name]
            del self.callbacks[name]
            if name in self.current_values:
                del self.current_values[name]
            logger.info(f"Removed probe '{name}'")
    
    def register_callback(
        self,
        probe_name: str,
        callback: Callable[[str, float], None]
    ) -> None:
        """
        Register callback for when probe crosses threshold.
        
        Args:
            probe_name: Name of probe to monitor
            callback: Function(probe_name, value) called when threshold crossed
        """
        if probe_name in self.callbacks:
            self.callbacks[probe_name].append(callback)
    
    def monitor_layer(
        self,
        hidden_states: torch.Tensor,
        layer: int,
        mode: str = "last_token"
    ) -> Dict[str, float]:
        """
        Monitor all probes for a given layer.
        
        Args:
            hidden_states: Shape (batch_size, seq_len, hidden_dim)
            layer: Current layer
            mode: "last_token" or "mean"
        
        Returns:
            Dictionary of probe_name -> value
        """
        results = {}
        
        for name, (probe, config) in self.probes.items():
            if config.layer != layer or not config.active:
                continue
            
            # Compute probe value
            if mode == "last_token":
                value = probe.probe_last_token(hidden_states)
            else:  # mean
                value = probe.probe_sequence(hidden_states)
            
            results[name] = value
            self.current_values[name] = value
            
            # Check threshold and trigger callbacks
            if value > config.threshold:
                for callback in self.callbacks.get(name, []):
                    try:
                        callback(name, value)
                    except Exception as e:
                        logger.error(f"Callback error for probe '{name}': {e}")
        
        return results
    
    def get_probe_value(self, name: str) -> Optional[float]:
        """Get current value of a probe."""
        return self.current_values.get(name)
    
    def get_all_values(self) -> Dict[str, float]:
        """Get all current probe values."""
        return self.current_values.copy()
    
    def enable_probe(self, name: str) -> None:
        """Enable a probe."""
        if name in self.probes:
            self.probes[name][1].active = True
    
    def disable_probe(self, name: str) -> None:
        """Disable a probe."""
        if name in self.probes:
            self.probes[name][1].active = False
    
    def get_status(self) -> Dict:
        """Get controller status."""
        return {
            "device": self.device,
            "total_probes": len(self.probes),
            "active_probes": sum(1 for _, (_, c) in self.probes.items() if c.active),
            "probes": {
                name: {
                    "type": config.probe_type.value,
                    "layer": config.layer,
                    "threshold": config.threshold,
                    "weight": config.weight,
                    "active": config.active,
                    "current_value": self.current_values.get(name)
                }
                for name, (_, config) in self.probes.items()
            }
        }


class ClosedLoopController:
    """
    Closed-loop controller that uses probes to adaptively steer generation.
    
    Implements feedback loops where probe values influence steering parameters
    in real-time.
    """
    
    def __init__(
        self,
        probe_controller: ProbeController,
        device: str = "cuda"
    ):
        self.probe_controller = probe_controller
        self.device = device
        self.rules: List[Dict] = []
    
    def add_rule(
        self,
        probe_name: str,
        condition: str,  # "above", "below", "between"
        threshold: float,
        action: Callable[[float], Dict],
        threshold2: Optional[float] = None  # For "between"
    ) -> None:
        """
        Add a steering rule based on probe value.
        
        Args:
            probe_name: Probe to monitor
            condition: "above", "below", or "between"
            threshold: Threshold value (or lower bound for "between")
            threshold2: Upper bound for "between"
            action: Function(value) -> dict of steering adjustments
        """
        rule = {
            "probe_name": probe_name,
            "condition": condition,
            "threshold": threshold,
            "threshold2": threshold2,
            "action": action
        }
        self.rules.append(rule)
        
        # Register callback
        def callback(name: str, value: float):
            self._evaluate_rules(name, value)
        
        self.probe_controller.register_callback(probe_name, callback)
        
        logger.info(f"Added rule for probe '{probe_name}'")
    
    def _evaluate_rules(self, probe_name: str, value: float) -> None:
        """Evaluate all rules for a probe."""
        for rule in self.rules:
            if rule["probe_name"] != probe_name:
                continue
            
            condition = rule["condition"]
            threshold = rule["threshold"]
            threshold2 = rule["threshold2"]
            action = rule["action"]
            
            # Check condition
            triggered = False
            if condition == "above" and value > threshold:
                triggered = True
            elif condition == "below" and value < threshold:
                triggered = True
            elif condition == "between" and threshold2 is not None:
                if threshold < value < threshold2:
                    triggered = True
            
            # Execute action if triggered
            if triggered:
                try:
                    adjustments = action(value)
                    logger.info(f"Rule triggered for '{probe_name}': {adjustments}")
                    # Apply adjustments (would be connected to steering controllers)
                except Exception as e:
                    logger.error(f"Rule action error: {e}")
    
    def create_novelty_rule(
        self,
        low_threshold: float = 0.3,
        high_threshold: float = 0.7
    ) -> None:
        """
        Create rule for adaptive novelty steering.
        
        If novelty is low, increase novelty steering.
        If novelty is high, decrease it to maintain coherence.
        """
        def action(value: float) -> Dict:
            if value < low_threshold:
                return {"novelty_weight": 1.5, "message": "Increasing novelty"}
            elif value > high_threshold:
                return {"novelty_weight": 0.5, "message": "Decreasing novelty"}
            else:
                return {"novelty_weight": 1.0, "message": "Maintaining novelty"}
        
        self.add_rule("novelty", "between", low_threshold, high_threshold, action)
    
    def create_coherence_rule(
        self,
        min_coherence: float = 0.6
    ) -> None:
        """
        Create rule for coherence maintenance.
        
        If coherence drops below threshold, increase coherence steering.
        """
        def action(value: float) -> Dict:
            if value < min_coherence:
                return {
                    "coherence_weight": 1.5,
                    "entropy_target": 3.0,
                    "message": "Boosting coherence"
                }
            else:
                return {"coherence_weight": 1.0, "message": "Coherence OK"}
        
        self.add_rule("coherence", "below", min_coherence, action=action)
    
    def create_source_similarity_rule(
        self,
        max_similarity: float = 0.4
    ) -> None:
        """
        Create rule to prevent copying from source.
        
        If source similarity is too high, increase divergence.
        """
        def action(value: float) -> Dict:
            if value > max_similarity:
                return {
                    "divergence_weight": 2.0,
                    "anti_copy_strength": 1.5,
                    "message": "Reducing source similarity"
                }
            else:
                return {"divergence_weight": 1.0, "message": "Source distance OK"}
        
        self.add_rule("source_similarity", "above", max_similarity, action=action)
    
    def get_status(self) -> Dict:
        """Get controller status."""
        return {
            "total_rules": len(self.rules),
            "rules": [
                {
                    "probe": rule["probe_name"],
                    "condition": rule["condition"],
                    "threshold": rule["threshold"]
                }
                for rule in self.rules
            ],
            "probe_status": self.probe_controller.get_status()
        }


def create_novelty_probe(
    hidden_dim: int,
    layer: int,
    device: str = "cuda"
) -> HiddenStateProbe:
    """Create a novelty probe."""
    return HiddenStateProbe(hidden_dim, ProbeType.NOVELTY, device)


def create_coherence_probe(
    hidden_dim: int,
    layer: int,
    device: str = "cuda"
) -> HiddenStateProbe:
    """Create a coherence probe."""
    return HiddenStateProbe(hidden_dim, ProbeType.COHERENCE, device)


def create_source_similarity_probe(
    hidden_dim: int,
    layer: int,
    device: str = "cuda"
) -> HiddenStateProbe:
    """Create a source similarity probe."""
    return HiddenStateProbe(hidden_dim, ProbeType.SOURCE_SIMILARITY, device)
