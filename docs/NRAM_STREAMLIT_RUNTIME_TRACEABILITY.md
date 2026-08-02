# NRAM Streamlit Runtime Cockpit Traceability

| ID | Requirement | Implementation | Verification | Status |
|---|---|---|---|---|
| OBS-01 | Versioned opt-in streaming envelope | [`NRAMStreamEvent`](../..//core/contracts/observability.py:26), [`_sse_observability()`](../..//api/routes.py:629) | [`test_research_events_are_ordered_and_request_scoped()`](../..//tests/contract/test_stream_observability.py:32) | Implemented |
| OBS-02 | Minimal mode has no telemetry events | [`_sse_observability()`](../../api/routes.py:638) | [`test_minimal_mode_preserves_normal_openai_stream_without_telemetry()`](../../tests/contract/test_stream_observability.py:20) | Implemented |
| OBS-03 | Bounded reduced token evidence | [`TokenEvidence`](../../core/contracts/observability.py:42), stream payload reason | contract test and source inspection | Partial: upstream streaming IDs unavailable |
| CAP-01 | Capability-driven method/control deck | [`main()`](../../streamlit_cockpit.py:113), `/v1/nram/capabilities` | Streamlit smoke when runtime available | Implemented |
| CAP-02 | DExperts absent when disabled | [`dexperts_enabled()`](../../api/routes.py:33), capability filtering | [`test_dexperts_feature_flag.py`](../../tests/test_dexperts_feature_flag.py) | Implemented |
| UI-01 | Four verbosity modes are request payload values | [`request_body()`](../../streamlit_cockpit.py:79) | request contract inspection | Implemented |
| UI-02 | Incremental token rendering and fallback | [`render_chroma()`](../../streamlit_cockpit.py:62), stream loop | Streamlit smoke when runtime available | Partial: token IDs unavailable upstream |
| ISO-01 | Request-local event ordering and IDs | [`NRAMStreamEvent`](../../core/contracts/observability.py:26) | contract test | Implemented |
| SAFE-01 | No full logits or hidden states cross boundary | [`TokenEvidence`](../../core/contracts/observability.py:42) | payload contract test | Implemented |
| EXP-01 | Export bounded sanitized replay | Raw Trace tab in [`main()`](../../streamlit_cockpit.py:113) | manual UI smoke | Implemented |
