"""
NRAM representation control modules.

These modules implement the mathematical operations for hidden-state interventions:
- Activation Addition: h' = h + alpha * v
- Multi-vector control: multiple vectors combined
- Conceptor steering: subspace projection via low-rank SVD
- Latent closed-loop: PID controller from probe scores to intervention strength
- Semantic closed-loop: chunk-level feedback orchestration
- Branch-and-tournament: multi-branch generation with scoring

All operations execute INSIDE the model forward pass via hooks or at
the orchestration layer for semantic-level control.
"""
from nram_sglang.representation.activation_addition import (
    ActivationAdditionRuntime,
    ActivationVectorArtifact,
)
from nram_sglang.representation.multi_vector import (
    MultiVectorController,
    CombinationMode,
    CombinationResult,
)
from nram_sglang.representation.conceptor import (
    ConceptorRuntime,
    ConceptorArtifact,
)
from nram_sglang.representation.latent_loop import (
    LatentClosedLoopController,
    LatentLoopConfig,
    LatentLoopRegistry,
    latent_loop_registry,
)
from nram_sglang.representation.semantic_loop import (
    SemanticClosedLoopController,
    SemanticLoopConfig,
    SemanticScore,
    SemanticAction,
)
from nram_sglang.representation.branch_tournament import (
    BranchAndTournamentGenerator,
    TournamentConfig,
    BranchResult,
    BranchStatus,
)

__all__ = [
    "ActivationAdditionRuntime",
    "ActivationVectorArtifact",
    "MultiVectorController",
    "CombinationMode",
    "CombinationResult",
    "ConceptorRuntime",
    "ConceptorArtifact",
    "LatentClosedLoopController",
    "LatentLoopConfig",
    "LatentLoopRegistry",
    "latent_loop_registry",
    "SemanticClosedLoopController",
    "SemanticLoopConfig",
    "SemanticScore",
    "SemanticAction",
    "BranchAndTournamentGenerator",
    "TournamentConfig",
    "BranchResult",
    "BranchStatus",
]
