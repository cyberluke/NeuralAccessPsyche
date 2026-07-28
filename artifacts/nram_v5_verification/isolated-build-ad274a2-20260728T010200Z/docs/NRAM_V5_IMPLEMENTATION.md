# NRAM v5 Implementation Guide

> **Status correction (2026-07-27):** This historical file describes design
> and CPU helper APIs. It is not live-runtime evidence. ActAdd, conceptors,
> hidden-state probes, latent closed loop, and DExperts are not wired to the
> pinned SGLang model. See `NRAM_V5_RUNTIME_WIRING.md` for truthful status.

## Overview

NRAM v5 is a multi-layer inference control system for Qwen models via SGLang. It provides sophisticated steering capabilities across multiple dimensions:

1. **Symbolic Control** - Token-level constraints and phrase masking
2. **Entropy Control** - Dynamic temperature adjustment via PID servo
3. **Concept Injection** - Phase-aware semantic steering
4. **Activation Addition** - Contrastive vector steering in hidden space
5. **Multi-Vector Control** - Advanced vector combination strategies
6. **Conceptor Steering** - Subspace-based concept blending
7. **Hidden-State Probes** - Real-time monitoring and closed-loop control
8. **DExperts** - Expert/anti-expert model combination

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Request                              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              Layer 1: Symbolic Control                       │
│  • TokenTrieConstraint (phrase masking)                     │
│  • SourceNgramBlocker (anti-copy)                           │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              Layer 2: Entropy Control                        │
│  • EntropyController (PID servo)                            │
│  • PhaseAwareEntropyController                              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              Layer 3: Concept Injection                      │
│  • ConceptCapsule (semantic units)                          │
│  • ConceptInjector (phase-aware)                            │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              Layer 4: Activation Steering                    │
│  • ActivationAdditionController                             │
│  • VectorCollector (contrastive pairs)                      │
│  • VectorRegistry (persistent vectors)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              Layer 5: Advanced Control                       │
│  • MultiVectorController                                    │
│  • ConceptorController (subspace blending)                  │
│  • HiddenStateProbes (closed-loop)                          │
│  • DExpertsController (expert models)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    SGLang Engine                             │
│              (Qwen model inference)                         │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Symbolic Control

#### TokenTrieConstraint
Phrase-aware masking using trie data structure. Prevents forbidden phrases and enforces required patterns.

```python
from core.steering.phrase_constraints import TokenTrieConstraint

constraint = TokenTrieConstraint(
    forbidden_phrases=["I cannot", "I'm sorry", "As an AI"],
    required_phrases=["The solution", "This approach"]
)
```

#### SourceNgramBlocker
Prevents copying from source documents by blocking n-gram sequences.

```python
from core.steering.phrase_constraints import SourceNgramBlocker

blocker = SourceNgramBlocker(
    source_text=original_document,
    n=8,  # Block 8-grams
    tokenizer=tokenizer
)
```

### 2. Entropy Control

#### EntropyController
PID-based temperature servo that maintains target entropy levels.

```python
from core.steering.entropy_controller import EntropyController

controller = EntropyController(
    target_entropy=4.0,
    kp=0.5, ki=0.1, kd=0.05,
    scale_min=0.6, scale_max=1.8
)

# During generation
scale = controller.adjust(logits, batch_index=0)
```

#### PhaseAwareEntropyController
Automatically adjusts target entropy based on generation phase.

```python
from core.steering.entropy_controller import PhaseAwareEntropyController

controller = PhaseAwareEntropyController(
    phase_profile={
        "extraction": 2.5,    # Low entropy - factual
        "questioning": 4.0,   # Medium entropy - exploratory
        "divergence": 6.0,    # High entropy - creative
        "synthesis": 4.5,     # Medium entropy - converging
        "formulation": 3.0    # Low entropy - precise
    }
)
```

### 3. Concept Injection

#### ConceptCapsule
Semantic units with multiple lexical forms and activation rules.

```python
from core.steering.concept_capsules import ConceptCapsule, ConceptCapsuleRegistry

capsule = ConceptCapsule(
    concept_id="innovation",
    en_tokens=["innovation", "breakthrough"],
    cs_tokens=["inovace", "průlom"],
    activation_phase="divergence",
    max_uses=3
)

registry = ConceptCapsuleRegistry()
registry.add_capsule(capsule)
```

#### ConceptInjector
Applies concept injection during decoding with phase-dependent strength.

```python
from core.steering.concept_capsules import ConceptInjector

injector = ConceptInjector(registry=registry, tokenizer=tokenizer)
injector.apply_soft_injection(
    logits=logits,
    batch_index=0,
    phase="divergence",
    progress=0.5,
    base_strength=0.5
)
```

### 4. Activation Steering

#### ActivationAdditionController
Manages contrastive activation vectors for steering.

```python
from core.steering.activation_addition import ActivationVector, ActivationAdditionController

controller = ActivationAdditionController(device="cuda")

vector = ActivationVector(
    name="novelty",
    vector=torch.randn(5120),  # Qwen hidden dim
    layer=20,
    scale=1.0,
    phase="divergence"
)

controller.add_vector(vector)
modified_hidden = controller.apply_to_hidden_states(hidden_states, layer=20)
```

#### VectorCollector
Collects activation vectors from contrastive prompt pairs.

```python
from core.steering.activation_addition import VectorCollector

collector = VectorCollector(model, tokenizer, device="cuda")
vector = collector.collect(
    positive_prompt="Creative innovative solution",
    negative_prompt="Boring routine approach",
    layer=20,
    aggregation="mean"
)
```

#### VectorRegistry
Persistent storage and retrieval of activation vectors.

```python
from core.steering.activation_vectors import VectorRegistry, create_novelty_vector

registry = VectorRegistry(vectors_dir="data/activation_vectors")
novelty_vec = create_novelty_vector(registry, layer=20, device="cuda")
```

### 5. Multi-Vector Control

#### MultiVectorController
Advanced combination of multiple activation vectors.

```python
from core.steering.multi_vector_controller import MultiVectorController, CombinationMode

controller = MultiVectorController(
    device="cuda",
    combination_mode=CombinationMode.PROJECTED  # Orthogonalize vectors
)

controller.add_vector(
    name="novelty",
    vector=novelty_vec,
    weight=1.2,
    active_phases=["divergence", "synthesis"],
    priority=1
)

controller.add_vector(
    name="concreteness",
    vector=concrete_vec,
    weight=1.0,
    active_phases=["synthesis", "formulation"],
    priority=2
)

combined = controller.combine_vectors(hidden_dim=5120, layer=20)
```

### 6. Conceptor Steering

#### ConceptorController
Subspace-based concept blending using conceptor matrices.

```python
from core.steering.conceptor_steering import ConceptorController

controller = ConceptorController(device="cuda")

# Create conceptor from multiple vectors
controller.create_conceptor_from_vectors(
    name="creativity",
    vectors=[novelty_vec, imagination_vec],
    aperture=2.0,  # Softness of projection
    layer=20,
    weight=1.0
)

# Blend conceptors using AND/OR operations
blended = controller.blend_conceptors(layer=20, blend_mode="or")
```

#### LowRankSubspaceSteering
Efficient low-rank subspace projections.

```python
from core.steering.conceptor_steering import LowRankSubspaceSteering

steering = LowRankSubspaceSteering(rank=64, device="cuda")
steering.add_subspace(
    name="creative_space",
    basis_vectors=basis,  # Orthonormal basis
    layer=20,
    weight=1.0
)

projected = steering.project_to_subspace(hidden_states, "creative_space")
```

### 7. Hidden-State Probes

#### ProbeController
Real-time monitoring of hidden states for closed-loop control.

```python
from core.steering.hidden_state_probes import ProbeController, create_novelty_probe

probe_controller = ProbeController(device="cuda")

novelty_probe = create_novelty_probe(hidden_dim=5120, layer=20)
probe_controller.add_probe(
    name="novelty",
    probe=novelty_probe,
    layer=20,
    threshold=0.7
)

# Monitor during generation
values = probe_controller.monitor_layer(hidden_states, layer=20)
novelty_score = values["novelty"]
```

#### ClosedLoopController
Adaptive steering based on probe feedback.

```python
from core.steering.hidden_state_probes import ClosedLoopController

closed_loop = ClosedLoopController(probe_controller, device="cuda")

# Create rules for adaptive control
closed_loop.create_novelty_rule(low_threshold=0.3, high_threshold=0.7)
closed_loop.create_coherence_rule(min_coherence=0.6)
closed_loop.create_source_similarity_rule(max_similarity=0.4)
```

### 8. DExperts

#### DExpertsController
Combines expert and anti-expert models at inference time.

```python
from core.steering.dexperts import DExpertsController, ExpertType

controller = DExpertsController(
    base_model=base_model,
    base_tokenizer=tokenizer,
    device="cuda"
)

# Register creativity expert
controller.register_expert(
    name="creativity",
    expert_type=ExpertType.CREATIVITY,
    model_path="path/to/creativity/expert",
    alpha=1.0,  # Expert weight
    beta=0.5,   # Anti-expert weight
    active_phases=["divergence"]
)

# Load models
controller.load_expert_model("creativity")
controller.load_anti_expert_model("creativity", "path/to/boring/anti-expert")

# Get combined logits
combined_logits = controller.get_combined_logits(
    input_ids=input_ids,
    attention_mask=attention_mask
)
```

## Integration with SGLang

All components integrate with SGLang through the `NRAMLogitProcessor`:

```python
from core.steering.nram_logit_processor import NRAMLogitProcessor

processor = NRAMLogitProcessor(
    phrase_constraint=constraint,
    entropy_controller=entropy_ctrl,
    concept_injector=concept_inj,
    activation_controller=act_ctrl,
    multi_vector_controller=multi_ctrl,
    conceptor_controller=conceptor_ctrl,
    probe_controller=probe_ctrl,
    dexperts_controller=dexperts_ctrl
)
```

The processor is serialized and sent to SGLang for each request:

```python
payload["custom_logit_processor"] = processor.serialize()
payload["custom_params"] = {
    "positive_token_ids": [...],
    "negative_token_ids": [...],
    # ... other parameters
}
```

## Usage Example

Complete example combining multiple layers:

```python
from core.steering.nram_v5_integration import NRAMV5Integration

# Initialize NRAM v5
nram = NRAMV5Integration(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    device="cuda"
)

# Configure symbolic control
nram.add_forbidden_phrases(["I cannot", "As an AI"])
nram.enable_source_blocking(source_text=document, n=8)

# Configure entropy control
nram.configure_entropy_control(
    phase_profile={
        "extraction": 2.5,
        "divergence": 6.0,
        "synthesis": 4.5
    }
)

# Configure concept injection
nram.add_concept(
    concept_id="innovation",
    tokens=["innovation", "breakthrough", "novel"],
    activation_phase="divergence",
    max_uses=3
)

# Load activation vectors
nram.load_activation_vectors("data/activation_vectors")

# Configure multi-vector control
nram.configure_multi_vector(
    vectors=["novelty", "concreteness"],
    weights=[1.2, 1.0],
    combination_mode="projected"
)

# Configure closed-loop control
nram.enable_closed_loop_control(
    probes=["novelty", "coherence"],
    rules=["novelty_maintenance", "coherence_floor"]
)

# Generate with NRAM v5
response = nram.generate(
    prompt="Describe a revolutionary product concept",
    max_tokens=512,
    temperature=0.7,
    phase_sequence=["extraction", "divergence", "synthesis", "formulation"]
)

print(response["text"])
```

## Performance Considerations

### Memory Usage
- **Activation Vectors**: ~20KB per vector (5120 dims × 4 bytes)
- **Conceptor Matrices**: ~100MB per matrix (5120×5120)
- **Expert Models**: ~14GB per model (Qwen2.5-7B)
- **Probes**: ~20KB per probe (linear layer)

### Computational Overhead
- **Symbolic Control**: <1ms per token (trie lookup)
- **Entropy Control**: ~2ms per token (PID computation)
- **Concept Injection**: <1ms per token (vector addition)
- **Activation Addition**: ~5ms per token (vector broadcast)
- **Multi-Vector**: ~10ms per token (combination)
- **Conceptor**: ~50ms per token (matrix multiplication)
- **DExperts**: ~100ms per token (multiple forward passes)

### Optimization Tips
1. **Layer Selection**: Use middle layers (15-25) for activation steering
2. **Vector Pruning**: Remove vectors with low impact scores
3. **Conceptor Rank**: Use low-rank approximation for large models
4. **Expert Caching**: Keep frequently-used experts loaded
5. **Probe Sampling**: Monitor probes every N tokens, not every token

## Testing

Run the comprehensive test suite:

```bash
# Test all components
pytest tests/integration/test_nram_v5_ab_testing.py -v
pytest tests/integration/test_activation_addition.py -v

# Test specific components
pytest tests/integration/test_nram_v5_ab_testing.py::TestTokenTrie -v
pytest tests/integration/test_activation_addition.py::TestVectorCollector -v
```

## Troubleshooting

### Common Issues

1. **Vectors not loading**: Check that vector files exist in `data/activation_vectors/`
2. **Expert models OOM**: Reduce number of loaded experts or use CPU offloading
3. **Entropy oscillation**: Reduce PID gains (kp, ki, kd)
4. **Conceptor instability**: Increase aperture parameter (>2.0)
5. **Probe drift**: Recalibrate probes with fresh data

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("nram.v5")
logger.setLevel(logging.DEBUG)
```

## Future Enhancements

1. **GPU-Optimized Kernels**: Custom CUDA kernels for conceptor operations
2. **Adaptive Vector Selection**: ML-based vector selection per context
3. **Distributed Experts**: Multi-GPU expert model serving
4. **Online Probe Training**: Continuously update probes during generation
5. **Compositional Concepts**: Combine concepts using algebraic operations

## References

- [Activation Addition](https://research-information.bris.ac.uk/en/publications/activation-addition-steering-language-models-without-optimization/)
- [Conceptor Theory](https://www.researchgate.net/publication/265771452_Conceptors)
- [DExperts](https://arxiv.org/abs/2105.03023)
- [SGLang Documentation](https://sgl-project.github.io/)

## License

This implementation is part of the NeuralAccessPsyche project.
