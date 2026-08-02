---
status: canonical
version: 1.0
source: active mission request, 2026-08-02
scope: NRAM Streamlit runtime and research cockpit
---

# NRAM Streamlit Runtime and Research Cockpit

This document records the canonical mission scope for the cockpit vertical
slice. The active mission request is authoritative for the complete normative
requirements, including the ten cockpit features, streaming observability
contract, four verbosity modes, token chroma rules, isolation requirements,
acceptance gates, and the explicit prohibition on resuming DExperts training,
H100 work, or causal ablation.

## Preserved constraints

- DExperts remains disabled by default behind `NRAM_DEXPERTS_ENABLED=false`.
- The existing Streamlit, API, control-plane, SGLang, and NRAM architecture
  must be extended rather than replaced with a disconnected demo.
- Ordinary OpenAI-compatible streaming remains unchanged for clients that do
  not opt into telemetry.
- Telemetry is bounded, request-scoped, versioned, and must never transfer
  full vocabulary logits, probability vectors, or hidden states.
- Missing metrics are represented as unavailable (`null` plus a reason), not
  fabricated zeros.
- Runtime claims require focused tests and one real RTX/SGLang smoke test when
  that runtime is available.

## Required vertical slice

The implementation must prove normal generation, real control wiring,
capability-driven method selection, DExperts feature-flag behavior, minimal
telemetry suppression, research streaming evidence, token-order preservation,
request isolation, bounded payloads, and graceful degradation when telemetry
or runtime capabilities are unavailable.

The complete user-provided mission text remains the normative source for this
specification; this repository record preserves its scope and non-negotiable
constraints without silently changing them.
