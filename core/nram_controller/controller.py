"""NRAM state controller — orchestrates persona compilation and steering."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from core.contracts.nram import CompiledTokenPolicy, NRAMState, SteeringPolicy
from core.persona.compiler import compile_policy, policy_hash
from core.persona.profiles import DEFAULT_PROFILE, PROFILES

logger = logging.getLogger(__name__)


class NRAMController:
    """Coordinates NRAM state, policy compilation, and steering decisions."""

    def __init__(self) -> None:
        self._active_profile = DEFAULT_PROFILE
        self._last_policy: Optional[SteeringPolicy] = None
        self._last_hash: Optional[str] = None

    def get_profile(self, name: Optional[str] = None) -> NRAMState:
        profile_name = name or self._active_profile
        return PROFILES.get(profile_name, PROFILES[DEFAULT_PROFILE])

    def compile(
        self,
        profile_name: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        max_tokens: int = 512,
    ) -> SteeringPolicy:
        """Compile persona state into a steering policy deterministically."""
        profile = self.get_profile(profile_name)

        if overrides:
            state_dict = profile.model_dump()
            for key, value in overrides.items():
                if key in state_dict and value is not None:
                    state_dict[key] = value
            profile = NRAMState(**state_dict)

        policy = compile_policy(profile, max_tokens=max_tokens)
        self._last_policy = policy
        self._last_hash = policy_hash(policy)

        logger.info(
            f"NRAMController: compiled policy for profile={profile_name or self._active_profile}, "
            f"hash={self._last_hash}, +bias={policy.positive_bias}, -bias={policy.negative_bias}"
        )
        return policy

    @property
    def last_policy_hash(self) -> Optional[str]:
        return self._last_hash

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "engine": "sglang",
            "active_profile": self._active_profile,
            "available_profiles": list(PROFILES.keys()),
            "controls": {
                "token_biasing": True,
                "token_masking": True,
                "repetition_penalty": True,
                "structured_planning": True,
                "dynamic_logits": True,
            },
            "verified": {
                "pre_sampling_logit_modification": False,  # Set True after forced-token test
                "streaming": True,
                "structured_output": True,
            },
        }
