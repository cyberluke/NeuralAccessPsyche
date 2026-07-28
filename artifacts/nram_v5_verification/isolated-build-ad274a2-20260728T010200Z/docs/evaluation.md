# Evaluation

NeuralAccessPsyche steering is validated by an **A/B evaluation harness** that compares the NRAM-steered alias against the unsteered baseline on identical prompts. The goal is to prove the persona changes style *without* degrading substance — and to report the result honestly.

## The A/B harness (`evaluation/`)

The harness runs each prompt through **two** configurations and compares the outputs:

| Arm | Model alias | NRAM |
|-----|-------------|:---:|
| **Baseline** | `qwen3-14b-awq-baseline` | off; virtual route over the loaded Qwen checkpoint |
| **Steered** | `nram-qwen3-14b-awq` | on for supported `nram` controls |

Both arms use the **same prompt set, same temperature, and same seed** so differences are attributable to steering, not sampling noise. Outputs are scored by **deterministic heuristics** (no LLM-as-judge required for the core metrics), so scores are reproducible.

## Deterministic heuristics

Metrics are computed with rule-based, repeatable measurements:

- **Corporate jargon density** — count of a fixed jargon lexicon (e.g. *synergy, leveraging, solutioning, ecosystem, paradigm, best-in-class*) per unit length.
- **Persona fidelity** — presence/coverage of the intended rhetorical properties (uncomfortable truth, rejected assumption, human need, product revelation, sensory metaphor, final turn) as defined by the structured plan.
- **Relevance** — lexical/topical overlap between the prompt and the answer (the response must still address the actual request).
- **Coherence** — sentence-level consistency / grammaticality signals and absence of self-contradiction.
- **Factual error rate** — count of verifiably wrong claims.
- **Repeated-phrase rate** — incidence of duplicated n-grams / loops.

Because these are deterministic, re-running the harness on the same outputs yields the same scores.

## Success criteria

The steered arm is considered a success only if **all** of the following hold relative to baseline:

| Metric | Requirement |
|--------|-------------|
| Corporate jargon | **≥ 25% reduction** |
| Persona fidelity | **≥ 20% improvement** |
| Relevance loss | **≤ 5%** |
| Coherence loss | **≤ 5%** |
| Factual errors | **no increase** |
| Repeated phrases | **no significant rise** |

The first two show the steering works; the last four show it does **not** trade away substance to get there. A large jargon reduction that collapses coherence or relevance is a **failure**, not a win.

## Honesty policy

- **Report failure honestly.** If any criterion is not met, the result is reported as a failure. We do not soften, reframe, or quietly drop failing metrics.
- **No post-hoc metric tuning.** Thresholds and heuristics are fixed **before** the run. We do not adjust definitions, weights, or thresholds after seeing results to manufacture a pass.
- **Report the numbers.** Every run records the per-metric deltas for both arms, the prompt set, temperature, and seed, so results are reproducible and auditable.
- **Negative results are kept.** A failed run is documented, not deleted — it informs the next iteration.

## Running the harness

```bash
# Run the A/B evaluation over the prompt set
python -m evaluation.run

# Compare two saved outputs directly
python -m evaluation.score --baseline out/baseline.jsonl --steered out/steered.jsonl
```

The harness writes a results summary (per-metric deltas + pass/fail against the criteria above) that should be committed alongside any steering change.
