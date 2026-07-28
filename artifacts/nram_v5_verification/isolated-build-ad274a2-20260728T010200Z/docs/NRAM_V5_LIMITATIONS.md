# NRAM v5 Limitations

## Proven in this bounded pass

- Stable dill class serialization and zero-argument construction in both containers.
- Official server-side output history via pinned SGLang `Req.__init__`.
- Public API to real Qwen3-AWQ/RTX sampling for forced token and structural masks.
- Processor-originated direct telemetry with per-request scheduler state.
- One-step live entropy-target movement and request reset.
- Local vocabulary-logit capsule, soft-window, and sparse-vector effects.

## Implemented but not broadly scientifically validated

- Phrase/source controls beyond the documented bilingual and 8-gram fixtures.
- Entropy PID over long generations, broad prompt suites, and target tuning.
- Natural-language impact of capsule/soft/vector controls.
- Persona prompt policy and orchestration quality.
- Streamlit UI-to-runtime propagation has no automated browser acceptance suite.

## Design-only helpers, not live features

Activation addition, conceptors, hidden-state probes, DExperts helper classes,
and hallucination/evidence helpers remain importable CPU code. They are not
attached to the loaded SGLang model and must not support runtime claims.

## Not implemented

Real DExperts distributions; ActAdd; ReFT; soft prompts; model-layer conceptors;
trained probes; latent closed loop; semantic novelty loop; scientifically
evaluated evidence guard; divergent branch-and-tournament; attention-head
gating; KV-cache firewall; GPU-native semantic control.

Requests for these advanced runtime mechanisms fail with HTTP 400 instead of
falling back to prompts or static vocabulary bias.

## Scientific limitations

- No labeled human preference or evidence-guard dataset.
- No representation artifacts, module-hook telemetry, or representation ablation.
- No complete multi-seed canonical prompt suite, confidence intervals, or corrected significance tests.
- The working tree began dirty; the final diff incorporates identified pre-existing NRAM work.
- CPU helper tests prove formulas only, not live inference wiring.

Verdict: `PARTIALLY FUNCTIONAL`.

