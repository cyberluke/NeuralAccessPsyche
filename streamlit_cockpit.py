"""NRAM research cockpit backed by the existing OpenAI-compatible API.

Run with ``streamlit run streamlit_cockpit.py``.  The browser receives normal
OpenAI chunks plus opt-in ``event: nram`` envelopes; no model is called by the
UI directly.
"""
from __future__ import annotations

import html
import json
import os
import time
from typing import Any, Dict, Iterable, List

import httpx
import streamlit as st


API_BASE = os.environ.get("NRAM_API_BASE", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("NRAM_API_KEY", "dev-nram-key")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
LEVELS = ("minimal", "standard", "research", "trace")
COLORS = {
    "ARGMAX_FLIPPED_BY_STEERING": "#c084fc",
    "TOKEN_FAVORED_BY_STEERING": "#4ade80",
    "BASE_SELECTION_PRESERVED": "#60a5fa",
    "STOCHASTIC_OR_UNCLASSIFIED": "#94a3b8",
}


@st.cache_resource
def client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(180.0, connect=10.0), limits=httpx.Limits(max_connections=10))


@st.cache_data(ttl=10, show_spinner=False)
def get_json(path: str) -> Dict[str, Any]:
    try:
        response = client().get(f"{API_BASE}{path}", headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # UI remains usable when the backend is down.
        return {"_error": str(exc)}


def init_state() -> None:
    st.session_state.setdefault("verbosity", "minimal")
    st.session_state.setdefault("events", [])
    st.session_state.setdefault("tokens", [])
    st.session_state.setdefault("selected_token", None)
    st.session_state.setdefault("last_request", None)


def parse_sse(response: httpx.Response) -> Iterable[tuple[str, Any]]:
    event_name = "message"
    for line in response.iter_lines():
        if line.startswith("event: "):
            event_name = line[7:].strip()
        elif line.startswith("data: "):
            raw = line[6:]
            if raw == "[DONE]":
                yield event_name, raw
                event_name = "message"
                continue
            try:
                yield event_name, json.loads(raw)
            except json.JSONDecodeError:
                yield "warning", {"message": "malformed_event"}


def render_chroma(tokens: List[Dict[str, Any]], target: Any) -> None:
    fragments = []
    for index, token in enumerate(tokens):
        text = html.escape(str(token.get("token_text", "")))
        classification = token.get("classification", "STOCHASTIC_OR_UNCLASSIFIED")
        color = COLORS.get(classification, COLORS["STOCHASTIC_OR_UNCLASSIFIED"])
        warning = token.get("warning")
        outline = " outline: 1px solid #fb923c;" if warning else ""
        fragments.append(
            f'<span title="token {index} · {html.escape(classification)}" '
            f'style="color:{color}; padding:1px 2px; border-radius:3px;{outline}">{text}</span>'
        )
    target.markdown("<div class='chroma'>" + "".join(fragments) + "</div>", unsafe_allow_html=True)


def request_body(prompt: str, model: str, profile: str, level: str, strength: float,
                 temperature: float, max_tokens: int, deterministic: bool, supported: Dict[str, Any]) -> Dict[str, Any]:
    options: Dict[str, Any] = {
        "enabled": True,
        "profile": profile,
        "method": profile,
        "intensity": strength,
        "observability_level": level,
    }
    if level != "minimal":
        options["include_telemetry"] = True
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0 if deterministic else temperature,
        "seed": 271 if deterministic else None,
        "stream": True,
        "nram": options,
    }


def main() -> None:
    st.set_page_config(page_title="NRAM Research Cockpit", page_icon="NRAM", layout="wide")
    init_state()
    st.markdown("""<style>
    .stApp { background:#0b1020; color:#e5e7eb; }
    .chroma { background:#111827; border:1px solid #334155; border-radius:8px; padding:1.2rem; min-height:120px; line-height:1.8; white-space:pre-wrap; }
    .badge { display:inline-block; padding:.2rem .55rem; margin-right:.3rem; border:1px solid #475569; border-radius:999px; font-size:.78rem; }
    </style>""", unsafe_allow_html=True)

    capabilities = get_json("/nram/capabilities")
    profiles = get_json("/nram/profiles")
    if capabilities.get("_error"):
        st.error("NRAM API unavailable: " + capabilities["_error"])
        return
    controls = capabilities.get("controls", {})
    wired = {key: value for key, value in controls.items() if value is True or (isinstance(value, dict) and value.get("runtime_wired"))}
    models = capabilities.get("loaded_model_ids") or ["nram-qwen3-14b-awq"]
    profile_data = profiles.get("profiles", profiles.get("data", {}))
    advertised_profiles = capabilities.get("virtual_routes", {}).get("persona_profiles", {})
    profile_names = list(advertised_profiles.values()) or (list(profile_data) if isinstance(profile_data, dict) else [])
    profile_names = profile_names or ["normal", "microdose", "threshold", "psychedelic", "peak", "dissociative"]

    st.title("NRAM Research Cockpit")
    st.caption(f"Backend: {capabilities.get('engine', 'unavailable')} · Model: {models[0]} · telemetry is request-scoped")
    st.markdown("<span class='badge'>Capability contract loaded</span>" +
                ("<span class='badge'>DExperts disabled</span>" if "dexperts" not in controls else ""), unsafe_allow_html=True)

    with st.sidebar:
        st.header("Control Deck")
        model = st.selectbox("Model", models, key="cockpit_model")
        profile = st.selectbox("NRAM method/profile", profile_names, key="cockpit_profile")
        strength = st.slider("Steering strength", 0.0, 1.0, 0.5, 0.05, key="cockpit_strength")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05, key="cockpit_temperature")
        max_tokens = st.number_input("Max tokens", 1, 8192, 256, key="cockpit_max_tokens")
        deterministic = st.toggle("Deterministic seed", value=True, key="cockpit_deterministic")
        level = st.selectbox("Verbosity", LEVELS, key="verbosity")
        st.caption("Only controls reported as runtime-wired are sent.")
        if st.button("Refresh capabilities"):
            get_json.clear()
            st.rerun()

    prompt = st.text_area("Prompt", "Explain how NRAM steering can be evaluated without overstating causality.", height=120)
    generate = st.button("Generate", type="primary")
    if generate:
        body = request_body(prompt, model, profile, level, strength, temperature, int(max_tokens), deterministic, wired)
        st.session_state.events = []
        st.session_state.tokens = []
        st.session_state.selected_token = None
        st.session_state.last_request = body
        response_placeholder = st.empty()
        status_placeholder = st.empty()
        content = ""
        started = time.perf_counter()
        try:
            with client().stream("POST", f"{API_BASE}/chat/completions", json=body, headers=HEADERS) as response:
                response.raise_for_status()
                for event_name, data in parse_sse(response):
                    if event_name == "nram" and isinstance(data, dict):
                        st.session_state.events.append(data)
                        if data.get("event_type") == "token":
                            token = data.get("payload", {})
                            st.session_state.tokens.append(token)
                            content += token.get("token_text", "")
                            render_chroma(st.session_state.tokens, response_placeholder)
                    elif event_name == "message" and isinstance(data, dict):
                        for choice in data.get("choices", []):
                            content += (choice.get("delta") or {}).get("content", "")
                        if level == "minimal":
                            response_placeholder.markdown(content)
                    status_placeholder.caption(f"Streaming · {len(st.session_state.tokens)} evidence events · {time.perf_counter() - started:.1f}s")
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
        st.session_state["final_response"] = content

    tabs = st.tabs(["Live", "Tokens", "Steering", "Candidates", "Performance", "Events", "Raw Trace"])
    with tabs[0]:
        if st.session_state.get("final_response") and not st.session_state.tokens:
            st.markdown(st.session_state.final_response)
        st.caption("Legend: purple argmax flip · green favored · blue preserved · gray unavailable · orange outline warning")
    with tabs[1]:
        if st.session_state.tokens:
            labels = [f"{i}: {t.get('token_text', '')!r}" for i, t in enumerate(st.session_state.tokens)]
            selected = st.selectbox("Token inspector", range(len(labels)), format_func=lambda i: labels[i])
            st.json(st.session_state.tokens[selected])
        else:
            st.info("No token evidence available. The ordinary response remains usable.")
    with tabs[2]:
        st.json({"method": st.session_state.get("cockpit_profile"), "strength": st.session_state.get("cockpit_strength"), "supported": list(wired)})
    with tabs[3]:
        if "branch_tournament" in wired:
            st.info("Candidate panel is enabled by the capability contract; no branch events were emitted for this request.")
        else:
            st.info("Single-path execution; candidate arena is disabled by the backend contract.")
    with tabs[4]:
        deltas = [t.get("selected_token_delta_logp") for t in st.session_state.tokens if t.get("selected_token_delta_logp") is not None]
        st.metric("Evidence tokens", len(st.session_state.tokens))
        st.metric("Mean delta logp", f"{sum(deltas)/len(deltas):.4f}" if deltas else "N/A")
    with tabs[5]:
        for event in st.session_state.events[-100:]:
            st.caption(f"#{event.get('sequence_number')} · {event.get('event_type')} · {event.get('monotonic_timestamp_ms', 0):.1f} ms")
    with tabs[6]:
        if st.session_state.events:
            export = json.dumps({"schema_version": "nram.replay.v1", "request": st.session_state.last_request, "response": st.session_state.get("final_response", ""), "events": st.session_state.events[-100:]}, indent=2, ensure_ascii=False)
            st.download_button("Download sanitized JSON replay", export, "nram-replay.json", "application/json")
            st.code(export[:12000], language="json")
        else:
            st.info("No bounded trace to export.")


if __name__ == "__main__":
    main()
