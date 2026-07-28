"""
Latent closed-loop steering controller for NRAM.

Implements temporally valid feedback loop:
    hidden state at step t → probe score at step t → controller update
    → intervention strength at step t+1

Per-request controller state:
- target score, current error
- proportional, integral, derivative terms (PID)
- previous score, unclamped action, clamped effective strength, saturation state

The controller runs inside the forward hook path. It reads probe scores
emitted by probe interventions and updates the strength of actadd/conceptor
interventions for the NEXT decode step.

Critical: This is NOT a logit-level controller. It operates on hidden-state
intervention strengths, maintaining per-request PID state that is isolated
across concurrent requests via the batch context manager.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PIDState:
    """Per-request PID controller state."""
    target: float = 0.5
    kp: float = 1.0
    ki: float = 0.1
    kd: float = 0.05
    
    integral: float = 0.0
    previous_error: float = 0.0
    previous_score: float = 0.0
    
    # Output state
    unclamped_action: float = 0.0
    clamped_strength: float = 0.0
    strength_min: float = 0.0
    strength_max: float = 2.0
    
    # Saturation tracking
    saturated_high: bool = False
    saturated_low: bool = False
    
    # History for telemetry
    score_history: List[float] = field(default_factory=list)
    action_history: List[float] = field(default_factory=list)
    step_count: int = 0
    
    def reset(self) -> None:
        """Reset all state."""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_score = 0.0
        self.unclamped_action = 0.0
        self.clamped_strength = 0.0
        self.saturated_high = False
        self.saturated_low = False
        self.score_history.clear()
        self.action_history.clear()
        self.step_count = 0


@dataclass
class LatentLoopConfig:
    """Configuration for latent closed-loop controller."""
    enabled: bool = False
    probe_id: str = ""  # Which probe provides scores
    target_actadd_id: str = ""  # Which actadd intervention to control
    target_score: float = 0.5
    kp: float = 1.0
    ki: float = 0.1
    kd: float = 0.05
    strength_min: float = 0.0
    strength_max: float = 2.0
    initial_strength: float = 0.5
    update_interval_steps: int = 1  # Update every N decode steps
    anti_windup: bool = True  # Clamp integral to prevent windup
    integral_limit: float = 10.0


class LatentClosedLoopController:
    """
    PID controller for latent (hidden-state) closed-loop steering.
    
    Reads probe scores from the request context and updates intervention
    strength for the next decode step. Maintains per-request state that
    is isolated via the batch context manager.
    
    Lifecycle:
    - Created per-request when latent_loop config is present
    - Updated after each probe invocation in the forward hook
    - Strength changes take effect on the NEXT forward pass
    - State is cleaned up when request completes
    """
    
    def __init__(self, config: LatentLoopConfig):
        self.config = config
        self.pid = PIDState(
            target=config.target_score,
            kp=config.kp,
            ki=config.ki,
            kd=config.kd,
            strength_min=config.strength_min,
            strength_max=config.strength_max,
            clamped_strength=config.initial_strength,
        )
        self._last_update_step: int = -1
        self._probe_scores: Dict[int, float] = {}  # step -> score
    
    @property
    def current_strength(self) -> float:
        """Get current effective strength."""
        return self.pid.clamped_strength
    
    def receive_probe_score(self, step: int, score: float) -> None:
        """
        Receive a probe score for a given step.
        
        Called by probe interventions inside the forward hook.
        The score will be consumed on the next update() call.
        
        Args:
            step: Decode step at which score was computed
            score: Probe score value
        """
        self._probe_scores[step] = score
        self.pid.score_history.append(score)
        self.pid.previous_score = score
        logger.debug(f"Latent loop received probe score {score:.4f} at step {step}")
    
    def update(self, current_step: int) -> float:
        """
        Update controller and compute new strength for next step.
        
        Implements PID control:
            error = target - score
            integral += error (with anti-windup)
            derivative = error - prev_error
            action = kp*error + ki*integral + kd*derivative
            strength = clamp(initial + action, min, max)
        
        Args:
            current_step: Current decode step
        
        Returns:
            Updated strength for next step
        """
        if not self.config.enabled:
            return self.config.initial_strength
        
        # Check update interval
        if current_step - self._last_update_step < self.config.update_interval_steps:
            return self.pid.clamped_strength
        
        # Get latest probe score
        if not self._probe_scores:
            return self.pid.clamped_strength
        
        latest_step = max(self._probe_scores.keys())
        score = self._probe_scores[latest_step]
        
        # Compute error
        error = self.config.target_score - score
        
        # Update integral with anti-windup
        self.pid.integral += error
        if self.config.anti_windup:
            self.pid.integral = max(
                -self.config.integral_limit,
                min(self.config.integral_limit, self.pid.integral)
            )
        
        # Compute derivative
        derivative = error - self.pid.previous_error
        self.pid.previous_error = error
        
        # Compute PID action
        action = (
            self.config.kp * error
            + self.config.ki * self.pid.integral
            + self.config.kd * derivative
        )
        
        # Compute new strength
        unclamped = self.config.initial_strength + action
        clamped = max(self.config.strength_min, min(self.config.strength_max, unclamped))
        
        # Track saturation
        self.pid.saturated_high = unclamped > self.config.strength_max
        self.pid.saturated_low = unclamped < self.config.strength_min
        self.pid.unclamped_action = unclamped
        self.pid.clamped_strength = clamped
        self.pid.action_history.append(clamped)
        self.pid.step_count += 1
        
        self._last_update_step = current_step
        
        logger.debug(
            f"Latent loop update: step={current_step}, score={score:.4f}, "
            f"error={error:.4f}, action={action:.4f}, strength={clamped:.4f}"
        )
        
        return clamped
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get controller state for telemetry."""
        return {
            "enabled": self.config.enabled,
            "target_score": self.config.target_score,
            "current_strength": self.pid.clamped_strength,
            "unclamped_action": self.pid.unclamped_action,
            "integral": self.pid.integral,
            "previous_error": self.pid.previous_error,
            "saturated_high": self.pid.saturated_high,
            "saturated_low": self.pid.saturated_low,
            "step_count": self.pid.step_count,
            "score_history_len": len(self.pid.score_history),
            "action_history_len": len(self.pid.action_history),
            "last_scores": self.pid.score_history[-5:] if self.pid.score_history else [],
            "last_actions": self.pid.action_history[-5:] if self.pid.action_history else [],
        }
    
    def emit_telemetry(self, request_id: str) -> Dict[str, Any]:
        """Emit closed-loop telemetry event."""
        from nram_sglang.hooks.telemetry import HookTelemetry
        
        state = self.get_state_snapshot()
        last_score = state["last_scores"][-1] if state["last_scores"] else None
        
        event = HookTelemetry.emit_closed_loop_update(
            request_id=request_id,
            loop_type="latent",
            step=state["step_count"],
            probe_score=last_score,
            target=self.config.target_score,
            error=state["previous_error"],
            alpha_before=self.config.initial_strength,
            alpha_after=state["current_strength"],
        )
        
        return event


class LatentLoopRegistry:
    """
    Per-request registry for latent closed-loop controllers.
    
    Each request can have at most one latent loop controller.
    The registry is accessed by the forward hook to update strengths
    and by probe interventions to feed scores.
    """
    
    def __init__(self):
        self._controllers: Dict[str, LatentClosedLoopController] = {}
    
    def register(
        self,
        request_id: str,
        config: LatentLoopConfig,
    ) -> LatentClosedLoopController:
        """Register a controller for a request."""
        controller = LatentClosedLoopController(config)
        self._controllers[request_id] = controller
        logger.info(f"Registered latent loop controller for request {request_id}")
        return controller
    
    def get(self, request_id: str) -> Optional[LatentClosedLoopController]:
        """Get controller for a request."""
        return self._controllers.get(request_id)
    
    def remove(self, request_id: str) -> None:
        """Remove controller for a request."""
        self._controllers.pop(request_id, None)
    
    def get_all_ids(self) -> List[str]:
        """Get all request IDs with active controllers."""
        return list(self._controllers.keys())
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all controllers."""
        return {
            "active_controllers": len(self._controllers),
            "controllers": {
                rid: ctrl.get_state_snapshot()
                for rid, ctrl in self._controllers.items()
            },
        }


# Global registry
latent_loop_registry = LatentLoopRegistry()
