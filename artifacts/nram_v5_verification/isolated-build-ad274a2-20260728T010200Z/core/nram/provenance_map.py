"""
Feature 3: Phenomenon -> Provenance closed-loop mapping.

Maps each NRAM phenomenon's SteeringLayer to a TokenOrigin provenance label,
so dashboards can show WHICH steering mechanism caused WHAT token classification.

This is APPLIED_POLICY mapping: it reports the steering layer that was active,
not a causal proof that the layer changed a specific sampled token.
"""
from __future__ import annotations

from typing import Dict

from core.nram.events import TokenOrigin
from core.nram.phenomena import PHENOMENA, SteeringLayer

# SteeringLayer -> TokenOrigin provenance
LAYER_TO_ORIGIN: Dict[SteeringLayer, TokenOrigin] = {
    SteeringLayer.LOGIT_BIAS: TokenOrigin.POSITIVE_LOGIT_BIAS,
    SteeringLayer.LOGIT_MASK: TokenOrigin.HARD_MASK,
    SteeringLayer.REPETITION: TokenOrigin.NEGATIVE_LOGIT_BIAS,
    SteeringLayer.PROMPT: TokenOrigin.APPLIED_POLICY,
    SteeringLayer.PLANNER: TokenOrigin.PLANNER_CONSTRAINT,
    SteeringLayer.FRAGMENT_INJECTION: TokenOrigin.FORCED_INJECTION,
    SteeringLayer.POSTPROCESS: TokenOrigin.POSTPROCESS,
}


def phenomenon_origin(phenomenon_id: str) -> TokenOrigin:
    """Return the TokenOrigin provenance for a phenomenon by id."""
    from core.nram.phenomena import PHENOMENA_BY_ID
    phen = PHENOMENA_BY_ID.get(phenomenon_id)
    if phen is None:
        return TokenOrigin.APPLIED_POLICY
    return LAYER_TO_ORIGIN.get(phen.steering_layer, TokenOrigin.APPLIED_POLICY)


def full_map() -> Dict[str, Dict[str, str]]:
    """Return the complete phenomenon -> {layer, origin} mapping for UI/API."""
    result: Dict[str, Dict[str, str]] = {}
    for phen in PHENOMENA:
        result[phen.id] = {
            "label_en": phen.label_en,
            "label_cs": phen.label_cs,
            "steering_layer": phen.steering_layer.value,
            "provenance_origin": LAYER_TO_ORIGIN.get(
                phen.steering_layer, TokenOrigin.APPLIED_POLICY
            ).value,
        }
    return result


def derive_origin_counts(
    phenomenon_counts: Dict[str, int],
    model_generated: int = 0,
) -> Dict[str, int]:
    """Derive TokenOrigin counts from fired phenomenon counts.

    Given how many times each phenomenon fired, attribute those tokens to the
    corresponding provenance origin. Remaining tokens are MODEL_GENERATED.
    """
    counts: Dict[str, int] = {o.value: 0 for o in TokenOrigin}
    for phen_id, n in phenomenon_counts.items():
        origin = phenomenon_origin(phen_id)
        counts[origin.value] = counts.get(origin.value, 0) + n
    counts[TokenOrigin.MODEL_GENERATED.value] = model_generated
    return counts
