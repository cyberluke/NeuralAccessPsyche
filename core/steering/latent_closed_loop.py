"""
Latent closed-loop steering for NRAM v5.

Connects probe output to subsequent representation intervention.

Causal sequence:
  hidden state at step t
  → probe score at step t
  → controller update
  → new alpha for step t+1
  → changed hidden state/logits at step t+1

Implements:
- Per-request controller state
- Configurable target
- Bounded P or PID update
- Anti-windup
- Maximum delta per step
- Warm-up steps
- Reset on request completion or abort
- Complete step-aligned telemetry
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PIDState:
    """PID controller state with anti-windup."""
    integral: float = 0.0
    previous_error: float = 0.0
    previous_output: float = 0.0


@dataclass
class ClosedLoopChannel:
    """One closed-loop channel connecting a probe to an intervention."""
    channel_id: str
    probe_id: str
    intervention_id: str
    target: float
    kp: float = 0.5  # Proportional gain
    ki: float = 0.0  # Integral gain
    kd: float = 0.0  # Derivative gain
    alpha_min: float = 0.0
    alpha_max: float = 2.0
    max_delta_per_step: float = 0.5
    warmup_steps: int = 0
    pid_state: PIDState = field(default_factory=PIDState)
    current_alpha: float = 0.0
    last_probe_score: float = 0.0
    last_error: float = 0.0
    step_count: int = 0


@dataclass
class ClosedLoopConfig:
    """Configuration for latent closed-loop steering."""
    enabled: bool = False
    channels: List[ClosedLoopChannel] = field(default_factory=list)
    telemetry_level: str = "summary"


@dataclass
class ClosedLoopTelemetry:
    """Telemetry from a closed-loop update."""
    request_id: str
    step: int
    channel_id: str
    probe_score: float
    target: float
    error: float
    pid_output: float
    new_alpha: float
    previous_alpha: float
    delta_alpha: float
    integral: float
    derivative: float
    warmup_active: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "step": self.step,
            "channel_id": self.channel_id,
            "probe_score": self.probe_score,
            "target": self.target,
            "error": self.error,
            "pid_output": self.pid_output,
            "new_alpha": self.new_alpha,
            "previous_alpha": self.previous_alpha,
            "delta_alpha": self.delta_alpha,
            "integral": self.integral,
            "derivative": self.derivative,
            "warmup_active": self.warmup_active,
        }


class LatentClosedLoopController:
    """
    Latent closed-loop controller.
    
    Connects probe scores to intervention strength via PID control.
    Ensures causal step alignment: probe at t → alpha at t+1.
    """
    
    def __init__(self, config: ClosedLoopConfig):
        self.config = config
        self._request_id: Optional[str] = None
        self._step: int = 0
        self._telemetry: List[ClosedLoopTelemetry] = []
        self._pending_alpha_updates: Dict[str, float] = {}
    
    def begin_request(self, request_id: str) -> None:
        """Begin a new request; reset all state."""
        self._request_id = request_id
        self._step = 0
        self._telemetry.clear()
        self._pending_alpha_updates.clear()
        for channel in self.config.channels:
            channel.pid_state = PIDState()
            channel.current_alpha = 0.0
            channel.last_probe_score = 0.0
            channel.last_error = 0.0
            channel.step_count = 0
    
    def end_request(self) -> None:
        """End request; reset state."""
        self._request_id = None
        self._step = 0
        self._pending_alpha_updates.clear()
    
    def update_from_probes(
        self,
        probe_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Update controller based on probe scores at current step.
        
        Returns mapping of intervention_id → new_alpha for NEXT step.
        
        CRITICAL: The returned alphas are applied at step t+1, not step t.
        This ensures causal alignment.
        """
        if not self.config.enabled:
            return {}
        
        updates: Dict[str, float] = {}
        
        for channel in self.config.channels:
            if channel.probe_id not in probe_scores:
                continue
            
            probe_score = probe_scores[channel.probe_id]
            channel.last_probe_score = probe_score
            channel.step_count += 1
            
            # Check warmup
            warmup_active = channel.step_count <= channel.warmup_steps
            if warmup_active:
                # During warmup, keep alpha at initial value
                updates[channel.intervention_id] = channel.current_alpha
                continue
            
            # Compute error (positive error means probe is below target)
            error = channel.target - probe_score
            channel.last_error = error
            
            # PID update
            pid_state = channel.pid_state
            
            # Proportional
            p_term = channel.kp * error
            
            # Integral with anti-windup
            pid_state.integral += error
            # Clamp integral to prevent windup
            max_integral = (channel.alpha_max - channel.alpha_min) / (channel.ki + 1e-10)
            pid_state.integral = max(-max_integral, min(max_integral, pid_state.integral))
            i_term = channel.ki * pid_state.integral
            
            # Derivative
            derivative = error - pid_state.previous_error
            pid_state.previous_error = error
            d_term = channel.kd * derivative
            
            # PID output
            pid_output = p_term + i_term + d_term
            
            # Compute new alpha
            previous_alpha = channel.current_alpha
            new_alpha = previous_alpha + pid_output
            
            # Clamp delta per step
            delta = new_alpha - previous_alpha
            if abs(delta) > channel.max_delta_per_step:
                delta = max(-channel.max_delta_per_step, min(channel.max_delta_per_step, delta))
                new_alpha = previous_alpha + delta
            
            # Clamp to bounds
            new_alpha = max(channel.alpha_min, min(channel.alpha_max, new_alpha))
            
            channel.current_alpha = new_alpha
            pid_state.previous_output = new_alpha
            
            # Record telemetry
            telemetry = ClosedLoopTelemetry(
                request_id=self._request_id or "",
                step=self._step,
                channel_id=channel.channel_id,
                probe_score=probe_score,
                target=channel.target,
                error=error,
                pid_output=pid_output,
                new_alpha=new_alpha,
                previous_alpha=previous_alpha,
                delta_alpha=new_alpha - previous_alpha,
                integral=pid_state.integral,
                derivative=derivative,
                warmup_active=False,
            )
            self._telemetry.append(telemetry)
            
            # Store for next step
            updates[channel.intervention_id] = new_alpha
        
        self._pending_alpha_updates = updates
        return updates
    
    def get_alpha_for_step(self, intervention_id: str) -> float:
        """Get the alpha that should be used at the current step."""
        return self._pending_alpha_updates.get(intervention_id, 0.0)
    
    def advance_step(self) -> None:
        """Advance to next step."""
        self._step += 1
    
    def get_telemetry(self) -> List[Dict[str, Any]]:
        """Get all telemetry events."""
        return [t.to_dict() for t in self._telemetry]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "enabled": self.config.enabled,
            "num_channels": len(self.config.channels),
            "channels": {
                ch.channel_id: {
                    "probe_id": ch.probe_id,
                    "intervention_id": ch.intervention_id,
                    "target": ch.target,
                    "current_alpha": ch.current_alpha,
                    "last_probe_score": ch.last_probe_score,
                    "last_error": ch.last_error,
                    "step_count": ch.step_count,
                }
                for ch in self.config.channels
            },
            "current_step": self._step,
        }
