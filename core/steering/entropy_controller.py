"""Entropy controller for NRAM v5 — PID-based temperature servo.

This module implements dynamic entropy control during generation:
1. EntropyController: PID controller targeting specific entropy levels
2. Phase-based entropy profiles: Different targets for different generation phases
3. Closed-loop adjustment: Real-time temperature scaling based on measured entropy

The entropy servo replaces static temperature multipliers with a feedback loop
that maintains target entropy per phase, enabling:
- Low entropy during factual extraction (precision)
- High entropy during divergent concept generation (creativity)
- Medium entropy during synthesis (balance)

This prevents "psychedelic = random" and instead creates "phase-appropriate exploration".
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional


class PIDController:
    """Classic PID controller for continuous adjustment.
    
    Used by EntropyController to adjust temperature scale based on
    entropy error (difference between target and measured entropy).
    
    Formula:
        output = Kp * error + Ki * integral(error) + Kd * derivative(error)
    """
    
    def __init__(
        self,
        kp: float = 0.5,
        ki: float = 0.1,
        kd: float = 0.05,
        integral_min: float = -10.0,
        integral_max: float = 10.0,
    ):
        """Initialize PID controller.
        
        Args:
            kp: Proportional gain (reaction to current error)
            ki: Integral gain (reaction to accumulated error)
            kd: Derivative gain (reaction to error rate of change)
            integral_min: Minimum integral windup limit
            integral_max: Maximum integral windup limit
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_min = integral_min
        self.integral_max = integral_max
        
        self.integral = 0.0
        self.prev_error = 0.0
    
    def update(self, error: float) -> float:
        """Compute PID output for given error.
        
        Args:
            error: Current error (target - measured)
        
        Returns:
            PID output (scale factor for temperature)
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error
        self.integral = max(self.integral_min, min(self.integral_max, self.integral))
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = error - self.prev_error
        self.prev_error = error
        d_term = self.kd * derivative
        
        return p_term + i_term + d_term
    
    def reset(self) -> None:
        """Reset PID state (integral and previous error)."""
        self.integral = 0.0
        self.prev_error = 0.0


class EntropyController:
    """PID controller for entropy targeting during generation.
    
    Dynamically adjusts temperature to maintain target entropy per phase.
    Replaces static temperature multipliers with closed-loop control.
    
    Example:
        controller = EntropyController(
            target_entropy=4.0,
            kp=0.5,
            ki=0.1,
            kd=0.05,
        )
        
        # During generation
        entropy = compute_entropy(logits)
        scale = controller.adjust(logits, batch_index=0)
        # logits are now scaled to move toward target entropy
    """
    
    def __init__(
        self,
        target_entropy: float = 4.0,
        kp: float = 0.5,
        ki: float = 0.1,
        kd: float = 0.05,
        scale_min: float = 0.6,
        scale_max: float = 1.8,
    ):
        """Initialize entropy controller.
        
        Args:
            target_entropy: Target entropy value (in nats)
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            scale_min: Minimum temperature scale factor
            scale_max: Maximum temperature scale factor
        """
        self.target = target_entropy
        self.scale_min = scale_min
        self.scale_max = scale_max
        
        self.pid = PIDController(kp=kp, ki=ki, kd=kd)
        
        # Telemetry
        self.last_entropy = 0.0
        self.last_scale = 1.0
    
    def compute_entropy(self, logits: Any, batch_index: int) -> float:
        """Compute entropy of logit distribution.
        
        Args:
            logits: Logit tensor of shape (batch_size, vocab_size)
            batch_index: Index in batch to compute entropy for
        
        Returns:
            Entropy value in nats
        """
        # Convert logits to probabilities
        # Use softmax for numerical stability
        max_logit = logits[batch_index].max()
        exp_logits = (logits[batch_index] - max_logit).exp()
        probs = exp_logits / exp_logits.sum()
        
        # Compute entropy: H = -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        entropy = -(probs * (probs + 1e-10).log()).sum().item()
        
        return entropy
    
    def adjust(self, logits: Any, batch_index: int) -> float:
        """Adjust logits to move toward target entropy.
        
        Computes current entropy, determines scale factor via PID,
        and applies scale to logits (logits / scale = temperature adjustment).
        
        Args:
            logits: Logit tensor of shape (batch_size, vocab_size)
            batch_index: Index in batch to adjust
        
        Returns:
            Applied scale factor (for telemetry)
        """
        # Compute current entropy
        entropy = self.compute_entropy(logits, batch_index)
        self.last_entropy = entropy
        
        # Compute error (positive = entropy too low, need to increase temperature)
        error = self.target - entropy
        
        # Get PID output (scale adjustment)
        pid_output = self.pid.update(error)
        
        # Convert PID output to scale factor
        # PID output is additive adjustment to base scale of 1.0
        scale = 1.0 + pid_output
        scale = max(self.scale_min, min(self.scale_max, scale))
        
        self.last_scale = scale
        
        # Apply scale to logits (divide by scale = adjust temperature)
        # Higher scale = lower temperature = lower entropy
        # Lower scale = higher temperature = higher entropy
        logits[batch_index] = logits[batch_index] / scale
        
        return scale
    
    def reset(self) -> None:
        """Reset controller state (PID integral and previous error)."""
        self.pid.reset()
        self.last_entropy = 0.0
        self.last_scale = 1.0


# Phase-based entropy profiles
# Each phase has a different target entropy for appropriate steering
ENTROPY_PHASE_PROFILES: Dict[str, float] = {
    "extraction": 2.5,      # Low entropy — factual precision
    "questioning": 4.0,     # Medium entropy — exploratory
    "divergence": 6.0,      # High entropy — creative leaps
    "synthesis": 4.5,       # Medium entropy — converging
    "formulation": 3.0,     # Low entropy — precise formulation
}


class PhaseAwareEntropyController:
    """Entropy controller with phase-based target switching.
    
    Automatically adjusts target entropy based on generation phase.
    Phases are determined by generation progress (0.0 to 1.0).
    
    Example:
        controller = PhaseAwareEntropyController(
            phase_profile=ENTROPY_PHASE_PROFILES,
            kp=0.5,
            ki=0.1,
            kd=0.05,
        )
        
        # During generation
        progress = generated_tokens / max_tokens
        scale = controller.adjust(logits, batch_index=0, progress=progress)
    """
    
    def __init__(
        self,
        phase_profile: Optional[Dict[str, float]] = None,
        kp: float = 0.5,
        ki: float = 0.1,
        kd: float = 0.05,
        scale_min: float = 0.6,
        scale_max: float = 1.8,
    ):
        """Initialize phase-aware entropy controller.
        
        Args:
            phase_profile: Mapping of phase name to target entropy
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            scale_min: Minimum temperature scale factor
            scale_max: Maximum temperature scale factor
        """
        self.phase_profile = phase_profile or ENTROPY_PHASE_PROFILES
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.scale_min = scale_min
        self.scale_max = scale_max
        
        # Current controller (recreated when phase changes)
        self.current_phase = None
        self.controller = None
        
        # Telemetry
        self.phase_history = []
    
    def _get_phase_for_progress(self, progress: float) -> str:
        """Determine phase based on generation progress.
        
        Args:
            progress: Generation progress (0.0 to 1.0)
        
        Returns:
            Phase name
        """
        if progress < 0.2:
            return "extraction"
        elif progress < 0.4:
            return "questioning"
        elif progress < 0.6:
            return "divergence"
        elif progress < 0.8:
            return "synthesis"
        else:
            return "formulation"
    
    def adjust(
        self,
        logits: Any,
        batch_index: int,
        progress: float,
    ) -> float:
        """Adjust logits based on current phase and entropy.
        
        Args:
            logits: Logit tensor of shape (batch_size, vocab_size)
            batch_index: Index in batch to adjust
            progress: Generation progress (0.0 to 1.0)
        
        Returns:
            Applied scale factor (for telemetry)
        """
        # Determine current phase
        phase = self._get_phase_for_progress(progress)
        
        # Recreate controller if phase changed
        if phase != self.current_phase:
            target_entropy = self.phase_profile.get(phase, 4.0)
            self.controller = EntropyController(
                target_entropy=target_entropy,
                kp=self.kp,
                ki=self.ki,
                kd=self.kd,
                scale_min=self.scale_min,
                scale_max=self.scale_max,
            )
            self.current_phase = phase
            self.phase_history.append({
                "phase": phase,
                "progress": progress,
                "target_entropy": target_entropy,
            })
        
        # Adjust using current controller
        return self.controller.adjust(logits, batch_index)
    
    def reset(self) -> None:
        """Reset controller state and phase history."""
        if self.controller:
            self.controller.reset()
        self.current_phase = None
        self.controller = None
        self.phase_history = []
