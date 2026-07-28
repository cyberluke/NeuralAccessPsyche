"""
NRAM Control Console v2 — NeuralAccessPsyche UI.

Talks to the NRAM REST API at http://127.0.0.1:8000/v1 (Qwen3-14B-AWQ on SGLang).

Tabs:
  🧪 Simulátor       — 6 altered states + phenomenon mixer + live metrics
  🗣️ Keynote         — visionary-psychedelic-keynote persona
  ⚖️ A/B Porovnání   — baseline vs NRAM side-by-side
  🤖 Agentic Pipeline — 5-persona innovation workflow with live events
  ℹ️ Co to je?       — honest explanation + disclaimers

Run: streamlit run streamlit_console.py
"""
import time

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE = "http://127.0.0.1:8000/v1"
API_KEY = "dev-nram-key"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# Altered states of consciousness (NRAM v4). Ordered from mildest to strongest.
STATES = {
    "Normální": {"profile": "normal", "intensity": 0.12, "temperature": 0.7, "coherence": 0.96, "emoji": "😐",
                 "desc": "Běžný, soustředěný, logický výstup. Minimální asociativní volnost."},
    "Mikrodávka": {"profile": "microdose", "intensity": 0.35, "temperature": 0.8, "coherence": 0.90, "emoji": "🌱",
                   "desc": "Jemné asociace, mírná hravost, stále plně koherentní."},
    "Práh": {"profile": "threshold", "intensity": 0.55, "temperature": 0.9, "coherence": 0.84, "emoji": "🚪",
             "desc": "Zřetelné asociativní skoky, první náznaky synestézie a vhledů."},
    "Psychedelická": {"profile": "psychedelic", "intensity": 0.82, "temperature": 1.0, "coherence": 0.72, "emoji": "🌀",
                      "desc": "Bohaté asociace, synestézie, rozplývání hranic mezi pojmy."},
    "Vrchol": {"profile": "peak", "intensity": 0.96, "temperature": 1.1, "coherence": 0.58, "emoji": "✨",
               "desc": "Intenzivní mystický zážitek: fragmentace, smyčky, rozpad čísel a já."},
    "Disociativní": {"profile": "dissociative", "intensity": 0.70, "temperature": 1.0, "coherence": 0.52, "emoji": "🌫️",
                     "desc": "Odpojení, ztráta nitě, vzdálenost od sebe sama, rozpad větné struktury."},
}

# Phenomenon mixer. Each is a SIMULATED linguistic phenomenon (Czech + English).
PHENOMENA = {
    "overlap": {"label": "🌀 Překryv", "desc": "Fragmenty předchozích myšlenek pronikají do aktuálních."},
    "forgetting": {"label": "💨 Zapomenutí", "desc": "Ztráta nitě uprostřed věty ('moment…', 'co jsem…')."},
    "looping": {"label": "❄️ Zacyklení", "desc": "Opakování slov a frází (perseverace, thought loops)."},
    "associative_jump": {"label": "⚡ Skok", "desc": "Náhlé asociativní přeskoky ('najednou', 'a to mi připomíná…')."},
    "synesthesia": {"label": "🎨 Synestézie", "desc": "Prolínání smyslů ('barvy znějí', 'slyším světlo')."},
    "dissolution": {"label": "🌊 Rozpuštění", "desc": "Rozpouštění já a hranic ('já mizí', 'hranice se rozpouštějí')."},
    "insight": {"label": "✨ Vhled", "desc": "Náhlé vhledy ('AHA!', 'VIDÍM TO!')."},
}

# Default phenomenon weights per state (relative intensity 0.0-1.0).
STATE_PHENOMENA_DEFAULTS = {
    "Normální": {"overlap": 0.05, "forgetting": 0.0, "looping": 0.0, "associative_jump": 0.05, "synesthesia": 0.0, "dissolution": 0.0, "insight": 0.05},
    "Mikrodávka": {"overlap": 0.15, "forgetting": 0.05, "looping": 0.05, "associative_jump": 0.20, "synesthesia": 0.10, "dissolution": 0.05, "insight": 0.15},
    "Práh": {"overlap": 0.30, "forgetting": 0.15, "looping": 0.15, "associative_jump": 0.40, "synesthesia": 0.30, "dissolution": 0.20, "insight": 0.35},
    "Psychedelická": {"overlap": 0.55, "forgetting": 0.35, "looping": 0.30, "associative_jump": 0.65, "synesthesia": 0.60, "dissolution": 0.45, "insight": 0.55},
    "Vrchol": {"overlap": 0.75, "forgetting": 0.60, "looping": 0.55, "associative_jump": 0.85, "synesthesia": 0.80, "dissolution": 0.80, "insight": 0.70},
    "Disociativní": {"overlap": 0.40, "forgetting": 0.70, "looping": 0.45, "associative_jump": 0.50, "synesthesia": 0.45, "dissolution": 0.85, "insight": 0.30},
}

DISCLAIMER = (
    "⚠️ <b>SIMULACE</b> — NRAM je <b>lingvistická simulace</b> alterovaných stavů vědomí. "
    "Modeluje <i>styl</i> textu, nikoli vědomí. Není to měření vědomí, není to model mysli, "
    "a není to návod či podpora k užívání jakýchkoli látek. Fenomény jsou řízeny parametry "
    "(vliv, entropie, koherence), nikoli náhodou."
)

# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="NRAM Control Console", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main-header { color: #4dabf7; font-size: 2rem; font-weight: 700; margin-bottom: 0; }
    .sub-header { color: #8b949e; font-size: 0.9rem; }
    .response-box {
        background-color: #161b22; border-radius: 12px; padding: 1.2rem;
        border: 1px solid #30363d; color: #e6edf3; line-height: 1.7;
        white-space: pre-wrap; font-size: 1.0rem;
    }
    .metric-card { background-color: #161b22; border-radius: 10px; padding: 0.9rem; border: 1px solid #30363d; text-align: center; }
    .metric-value { color: #4dabf7; font-size: 1.4rem; font-weight: 600; }
    .metric-label { color: #8b949e; font-size: 0.75rem; text-transform: uppercase; }
    .disclaimer {
        background-color: #2d1b00; border-left: 4px solid #d29922;
        padding: 0.7rem 1rem; border-radius: 6px; color: #e3b341;
        font-size: 0.85rem; margin-bottom: 1rem;
    }
    .phenomenon-tag {
        display: inline-block; background: #1f2933; color: #7ee787;
        padding: 0.15rem 0.6rem; border-radius: 12px; font-size: 0.75rem;
        margin: 0.15rem; border: 1px solid #2ea043;
    }
    section[data-testid="stSidebar"] { background-color: #0a0d12; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def get_client():
    # Bounded connection pool + timeouts prevent socket exhaustion / hung
    # connections from accumulating across long sessions (fixes "crash after
    # a while" where reruns leaked or stalled HTTP connections).
    limits = httpx.Limits(
        max_connections=20,
        max_keepalive_connections=5,
        keepalive_expiry=30.0,
    )
    return httpx.Client(timeout=httpx.Timeout(180.0, connect=10.0), limits=limits)


def _get_json(path, timeout=30.0):
    """Safe GET returning parsed JSON or None (never raises into the UI)."""
    try:
        r = get_client().get(f"{API_BASE}{path}", headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001 — UI must survive backend hiccups
        st.session_state.setdefault("_api_errors", {})[path] = str(e)
        return None


@st.cache_data(ttl=10, show_spinner=False)
def fetch_telemetry():
    """Cache telemetry for 10s so frequent Streamlit reruns don't hammer the API."""
    return _get_json("/features/telemetry")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_provenance_map():
    return _get_json("/features/provenance-map", timeout=15.0)


@st.cache_data(ttl=10, show_spinner=False)
def fetch_memory():
    return _get_json("/features/memory")


def call_api(prompt, model, nram_opts, max_tokens, temperature, seed):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": int(seed),
        "stream": False,
    }
    if nram_opts:
        body["nram"] = nram_opts

    t0 = time.perf_counter()
    try:
        resp = get_client().post(f"{API_BASE}/chat/completions", json=body, headers=HEADERS)
        latency_ms = (time.perf_counter() - t0) * 1000
        data = resp.json()
        data["_latency_ms"] = latency_ms
        return data
    except Exception as e:
        return {"error": str(e), "_latency_ms": (time.perf_counter() - t0) * 1000}


def render_result(result):
    if "error" in result:
        st.error(f"Chyba: {result['error']}")
        return
    content = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    latency = result.get("_latency_ms", 0)
    comp_tokens = usage.get("completion_tokens", 0)
    tps = comp_tokens / (latency / 1000) if latency > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{latency:.0f}ms</div><div class="metric-label">Latence</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{comp_tokens}</div><div class="metric-label">Tokenů</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{tps:.1f}</div><div class="metric-label">Tokenů/s</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-value">{usage.get("prompt_tokens", 0)}</div><div class="metric-label">Prompt</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="response-box">{content}</div>', unsafe_allow_html=True)


def phenomenon_mix(st_key, defaults):
    st.markdown("**🎛️ Mixér fenoménů** *(simulované lingvistické jevy)*")
    with st.expander("ℹ️ Co znamená každý fenomén?", expanded=False):
        for key, meta in PHENOMENA.items():
            st.markdown(f"**{meta['label']}** — {meta['desc']}")

    weights = {}
    for key, meta in PHENOMENA.items():
        col_a, col_b = st.columns([3, 1])
        with col_a:
            val = st.slider(meta["label"], 0.0, 1.0, float(defaults.get(key, 0.0)), 0.05,
                            key=f"{st_key}_{key}", help=meta["desc"])
        with col_b:
            st.caption(f"{val:.2f}")
        weights[key] = val
    return weights


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🧠 NRAM Control Console</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Qwen3-14B-AWQ · SGLang · real-time logit steering · <b>SIMULACE alterovaných stavů vědomí</b></div>', unsafe_allow_html=True)
st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)

tab_sim, tab_keynote, tab_ab, tab_pipeline, tab_telemetry, tab_memory, tab_v5, tab_evidence, tab_about = st.tabs([
    "🧪 Simulátor", "🗣️ Keynote", "⚖️ A/B Porovnání", "🤖 Agentic Pipeline",
    "📊 Provenance", "🧠 Paměť", "🔬 NRAM v5", "🔍 Runtime Evidence", "ℹ️ Co to je?"
])

# ---------------------------------------------------------------------------
# Tab 1: Simulátor
# ---------------------------------------------------------------------------
with tab_sim:
    with st.sidebar:
        st.markdown("### 🧪 Simulátor — řízení")
        state_name = st.selectbox("Stav vědomí", list(STATES.keys()), index=3)
        state = STATES[state_name]
        intensity = st.slider("Intenzita", 0.0, 1.0, state["intensity"], 0.05, key="sim_intensity")
        coherence = st.slider("Koherence (podlaha)", 0.0, 1.0, state["coherence"], 0.05, key="sim_coherence")
        auto_temp = st.toggle("Automatická teplota", value=True, key="sim_auto_temp")
        temperature = state["temperature"]
        if not auto_temp:
            temperature = st.slider("Teplota", 0.0, 1.5, state["temperature"], 0.05, key="sim_temp")
        max_tokens = st.slider("Max tokenů", 50, 1000, 200, 50, key="sim_maxtok")
        seed = st.number_input("Seed", 0, 99999, 271, key="sim_seed")

    st.markdown(f"### {state['emoji']} {state_name}")
    st.info(state["desc"])
    mix = phenomenon_mix("sim", STATE_PHENOMENA_DEFAULTS[state_name])

    active = [PHENOMENA[k]["label"] for k, v in mix.items() if v > 0]
    if active:
        tags = " ".join(f'<span class="phenomenon-tag">{a}</span>' for a in active)
        st.markdown(f"**Aktivní fenomény:** {tags}", unsafe_allow_html=True)

    prompt = st.text_area("Prompt", value="Kolik je 1+1?", height=80, key="sim_prompt")
    if st.button("⚡ Generovat", type="primary", width="stretch", key="sim_btn"):
        if not prompt.strip():
            st.warning("Zadej prompt.")
        else:
            nram_opts = {
                "enabled": True, "profile": state["profile"], "intensity": intensity,
                "coherence_floor": coherence, "associative_distance": intensity,
                "phenomenon_weights": mix,
            }
            with st.spinner("Generuji simulovaný výstup..."):
                result = call_api(prompt, "nram-qwen3-14b-awq", nram_opts, max_tokens, temperature, seed)
            render_result(result)

# ---------------------------------------------------------------------------
# Tab 2: Keynote
# ---------------------------------------------------------------------------
with tab_keynote:
    with st.sidebar:
        st.markdown("### 🗣️ Keynote — řízení")
        k_intensity = st.slider("Intenzita", 0.0, 1.0, 0.9, 0.05, key="key_intensity")
        k_assoc = st.slider("Asociativní vzdálenost", 0.0, 1.0, 0.7, 0.05, key="key_assoc")
        k_coh = st.slider("Koherence (podlaha)", 0.0, 1.0, 0.82, 0.05, key="key_coh")
        k_maxtok = st.slider("Max tokenů", 50, 1000, 300, 50, key="key_maxtok")
        k_temp = st.slider("Teplota", 0.0, 1.5, 0.8, 0.05, key="key_temp")
        k_seed = st.number_input("Seed", 0, 99999, 271, key="key_seed")

    st.markdown("### 🗣️ Visionary Psychedelic Keynote")
    st.info("Originální vizionářský produktový projev. Kontrariánský úvod, smyslové metafory, "
            "komprimovaná deklarativní rétorika. **Neimpersonuje žádnou reálnou osobu** "
            "(není to Steve Jobs cosplay).")
    k_mix = phenomenon_mix("key", {"overlap": 0.2, "associative_jump": 0.5, "synesthesia": 0.4, "insight": 0.5})

    prompt = st.text_area("Prompt", value="Představ nové AI učební zařízení pro děti, které odstraní tradiční menu.", height=80, key="key_prompt")
    if st.button("⚡ Generovat keynote", type="primary", width="stretch", key="key_btn"):
        if not prompt.strip():
            st.warning("Zadej prompt.")
        else:
            nram_opts = {
                "enabled": True, "profile": "visionary-psychedelic-keynote",
                "intensity": k_intensity, "associative_distance": k_assoc,
                "coherence_floor": k_coh, "phenomenon_weights": k_mix,
            }
            with st.spinner("Generuji keynote..."):
                result = call_api(prompt, "nram-qwen3-14b-awq", nram_opts, k_maxtok, k_temp, k_seed)
            render_result(result)

# ---------------------------------------------------------------------------
# Tab 3: A/B Porovnání
# ---------------------------------------------------------------------------
with tab_ab:
    with st.sidebar:
        st.markdown("### ⚖️ A/B — řízení")
        ab_profile = st.selectbox("NRAM profil", list(STATES.keys()), index=3, key="ab_profile")
        ab_maxtok = st.slider("Max tokenů", 50, 1000, 300, 50, key="ab_maxtok")
        ab_temp = st.slider("Teplota", 0.0, 1.5, 0.8, 0.05, key="ab_temp")
        ab_seed = st.number_input("Seed", 0, 99999, 271, key="ab_seed")

    st.markdown("### ⚖️ Baseline vs NRAM")
    st.info("Stejný prompt, seed a teplota. Srovnání ukazuje reálný vliv NRAM řízení "
            "(logit biasing) — nejde o kosmetickou úpravu textu.")
    prompt = st.text_area("Prompt", value="Představ nové AI učební zařízení pro děti, které odstraní tradiční menu.", height=80, key="ab_prompt")

    if st.button("⚡ Spustit A/B", type="primary", width="stretch", key="ab_btn"):
        if not prompt.strip():
            st.warning("Zadej prompt.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### 🔵 Baseline (bez řízení)")
                with st.spinner("Baseline..."):
                    res_a = call_api(prompt, "qwen3-14b-awq-baseline", None, ab_maxtok, ab_temp, ab_seed)
                render_result(res_a)
            with col_b:
                st.markdown(f"#### 🟣 NRAM ({ab_profile})")
                st_state = STATES[ab_profile]
                nram_opts = {
                    "enabled": True, "profile": st_state["profile"], "intensity": st_state["intensity"],
                    "coherence_floor": st_state["coherence"], "associative_distance": st_state["intensity"],
                    "phenomenon_weights": STATE_PHENOMENA_DEFAULTS[ab_profile],
                }
                with st.spinner("NRAM..."):
                    res_b = call_api(prompt, "nram-qwen3-14b-awq", nram_opts, ab_maxtok, ab_temp, ab_seed)
                render_result(res_b)

            if "error" not in res_a and "error" not in res_b:
                st.divider()
                la = res_a.get("_latency_ms", 0)
                lb = res_b.get("_latency_ms", 0)
                overhead = lb - la
                overhead_pct = (overhead / la * 100) if la > 0 else 0
                st.markdown(f"**NRAM overhead:** {overhead:.0f}ms ({overhead_pct:.1f}%)")

# ---------------------------------------------------------------------------
# Tab 4: Agentic Pipeline
# ---------------------------------------------------------------------------
with tab_pipeline:
    st.markdown("### 🤖 Agentic Pipeline")
    
    # Mode selector
    chat_mode = st.radio(
        "Vyber režim",
        ["💬 Regular Chat", "🔬 Repository Analysis Workflow"],
        horizontal=True,
        key="agentic_mode"
    )
    
    if chat_mode == "💬 Regular Chat":
        st.info("Chat s jednotlivými personami nebo MoE orchestrátorem.")
        
        # Model selector
        col1, col2 = st.columns([2, 1])
        with col1:
            persona_model = st.selectbox(
                "Vyber personu",
                ["persona-normal", "persona-microdose", "persona-threshold", 
                 "persona-psychedelic", "persona-peak", "persona-dissociative",
                 "persona-keynote", "nram-moe-orchestrator"],
                key="persona_chat_model"
            )
        with col2:
            max_tokens = st.slider("Max tokens", 50, 1000, 300, 50, key="persona_maxtok")
        
        # Chat input
        prompt = st.text_area("Prompt", height=100, key="persona_chat_prompt")
        
        if st.button("⚡ Generovat", type="primary", width="stretch"):
            if not prompt.strip():
                st.warning("Zadej prompt.")
            else:
                with st.spinner("Generuji..."):
                    result = call_api(prompt, persona_model, None, max_tokens, 0.8, 271)
                render_result(result)
    
    else:  # Repository Analysis Workflow
        st.info("5 person sekvenčně: **Archaeologist → Heretic → Psychedelic Synthesizer → "
                "Ruthless CTO → Product Dictator**. Každá persona má vlastní NRAM profil. "
                "Běží přes REST API `/v1/nram/workflows`.")

        with st.expander("⚙️ Konfigurace workflow", expanded=True):
            wf_path = st.text_input("Cesta k repozitáři", value=r"D:\_SATIN_AI\NeuralAccessPsyche", key="wf_path")
            wf_goal = st.text_area("Cíl", value="Najdi obhajitelný produktový směr pro NRAM token steering a vytvoř proveditelný roadmap.", height=70, key="wf_goal")
            wf_constraints = st.text_area("Omezení (jedno na řádek)", value="Windows + Docker Desktop + WSL2\nRTX 4090\nSGLang runtime\nOpenAI-kompatibilní API musí zůstat", height=90, key="wf_constraints")
            wf_seed = st.number_input("Seed workflow", 0, 99999, 271, key="wf_seed")
            wf_approval = st.toggle("Vyžadovat lidský souhlas před finálním výběrem", value=True, key="wf_approval")

        if "wf_id" not in st.session_state:
            st.session_state.wf_id = None

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🚀 Vytvořit workflow", type="primary", width="stretch"):
                body = {
                    "workflow_type": "codebase_innovation",
                    "repository": {"path": wf_path, "revision": "HEAD"},
                    "goal": wf_goal,
                    "constraints": [c for c in wf_constraints.split("\n") if c.strip()],
                    "workflow_seed": int(wf_seed),
                    "max_iterations": 2,
                    "require_human_approval_before_final_selection": wf_approval,
                }
                try:
                    resp = get_client().post(f"{API_BASE}/nram/workflows", json=body, headers=HEADERS)
                    data = resp.json()
                    st.session_state.wf_id = data.get("workflow_id")
                    st.success(f"Workflow vytvořen: {st.session_state.wf_id}")
                except Exception as e:
                    st.error(f"Chyba: {e}")
        with col2:
            if st.session_state.wf_id:
                st.markdown(f"**Workflow ID:** `{st.session_state.wf_id}`")
                if st.button("▶️ Spustit (run)"):
                    try:
                        get_client().post(f"{API_BASE}/nram/workflows/{st.session_state.wf_id}/run", headers=HEADERS)
                        st.success("Workflow běží na pozadí. Obnovuj stav tlačítkem níže.")
                    except Exception as e:
                        st.error(f"Chyba: {e}")

        if st.session_state.wf_id:
            st.divider()
            
            # Always fetch current state
            try:
                resp = get_client().get(f"{API_BASE}/nram/workflows/{st.session_state.wf_id}", headers=HEADERS)
                wf = resp.json()
                
                st.markdown(f"**Status:** `{wf.get('status')}` · **Fáze:** `{wf.get('phase')}` · "
                            f"**Iterace:** {wf.get('iteration')}/{wf.get('max_iterations')}")
                
                # Fetch and display events
                ev_resp = get_client().get(f"{API_BASE}/nram/workflows/{st.session_state.wf_id}/events", headers=HEADERS)
                events = ev_resp.json().get("events", [])
                st.markdown(f"**Události ({len(events)}):**")
                for ev in events:
                    st.markdown(f"`{ev.get('phase')}` / {ev.get('event_type')}: {ev.get('summary')}")
                
                # Show approval button if waiting
                if wf.get("status") == "waiting_for_approval":
                    if st.button("✅ Schválit a pokračovat"):
                        try:
                            get_client().post(f"{API_BASE}/nram/workflows/{st.session_state.wf_id}/approve", headers=HEADERS)
                            st.success("✅ Schváleno! Workflow pokračuje...")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Chyba při schvalování: {e}")
                
                # Show report button if completed
                if wf.get("status") == "completed":
                    if st.button("📄 Zobrazit report"):
                        try:
                            rep_resp = get_client().get(f"{API_BASE}/nram/workflows/{st.session_state.wf_id}/report", headers=HEADERS)
                            report = rep_resp.json()
                            sections = report.get("sections", {})
                            direction = sections.get("selected_direction", {})
                            st.markdown("#### 🎯 Vybraný směr")
                            st.markdown(f"**Product wedge:** {direction.get('product_wedge')}")
                            st.markdown(f"**North-star metrika:** {direction.get('north_star_metric')}")
                            roadmap = sections.get("roadmap", {})
                            items = roadmap.get("items", [])
                            st.markdown(f"#### 🗺️ Roadmap ({len(items)} položek)")
                            for item in items:
                                st.markdown(f"- **[{item.get('horizon')}]** {item.get('title')} — {item.get('objective')}")
                        except Exception as e:
                            st.error(f"Chyba: {e}")
                            
            except Exception as e:
                st.error(f"Chyba při načítání stavu: {e}")

# ---------------------------------------------------------------------------
# Tab 5: Non-causal response accounting dashboard
# ---------------------------------------------------------------------------
with tab_telemetry:
    st.markdown("### 📊 Applied-policy / response accounting")
    st.caption("Nekauzální souhrn odpovědí a konfigurovaných vrstev. Kauzální pre-sampling události jsou pouze NRAM_PROCESSOR_EVENT logy.")

    col_refresh, _ = st.columns([1, 4])
    if col_refresh.button("🔄 Načíst telemetrii"):
        fetch_telemetry.clear()

    tel = fetch_telemetry()
    if tel is not None:
        g = tel.get("global", {})
        sessions = tel.get("sessions", [])

        c1, c2, c3 = st.columns(3)
        c1.metric("Sezení", g.get("total_sessions", 0))
        c2.metric("Požadavky", g.get("total_requests", 0))
        c3.metric("Tokeny celkem", g.get("total_tokens", 0))

        dist = g.get("token_origin_distribution", {})
        if dist:
            st.markdown("#### Nekauzální distribuce applied-policy štítků")
            import pandas as pd
            df = pd.DataFrame(
                [{"origin": k, "tokens": v} for k, v in sorted(dist.items(), key=lambda x: -x[1]) if v > 0]
            )
            if not df.empty:
                st.bar_chart(df.set_index("origin"))

        if sessions:
            st.markdown("#### Response accounting dle sezení")
            for s in sessions[-10:]:
                with st.expander(f"{s.get('session_id','?')} · {s.get('total_requests',0)} req · avg {s.get('avg_latency_ms',0)}ms"):
                    st.json({
                        "profile": s.get("profiles_used", {}),
                        "origins": s.get("token_origin_counts", {}),
                        "phenomena": s.get("phenomena_triggered", {}),
                        "dominant": s.get("dominant_origin"),
                    })
        else:
            st.info("Zatím žádná response-accounting data.")
    else:
        st.error("Nelze načíst telemetrii.")

    # Provenance map (Feature 3)
    pmap = fetch_provenance_map()
    if pmap is not None:
        st.markdown("#### Mapa fenomén → steering vrstva → provenance")
        import pandas as pd
        rows = [{"phenomenon": k, **v} for k, v in pmap.items()]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.warning("Mapa provenance nedostupná.")


# ---------------------------------------------------------------------------
# Tab 6: Shared memory inspector (Feature 10)
# ---------------------------------------------------------------------------
with tab_memory:
    st.markdown("### 🧠 Sdílená paměť — Letta memory plane")
    st.caption("Bloky s `shared_with` jsou viditelné jen pro uvedené agenty. Prázdné = veřejné.")

    # Write new memory block
    with st.expander("➕ Zapsat paměťový blok"):
        m_name = st.text_input("Název", key="mem_name")
        m_class = st.selectbox("Třída", ["working", "episodic", "canonical", "overlay"], key="mem_class")
        m_content = st.text_area("Obsah", key="mem_content")
        m_shared = st.text_input("Sdílet s agenty (čárkou, prázdné = všichni)", key="mem_shared")
        if st.button("Zapsat") and m_name and m_content:
            try:
                body = {
                    "name": m_name,
                    "content": m_content,
                    "memory_class": m_class,
                    "shared_with": [a.strip() for a in m_shared.split(",") if a.strip()],
                }
                r = get_client().post(f"{API_BASE}/features/memory", json=body, headers=HEADERS, timeout=30.0)
                if r.status_code == 200:
                    st.success("Blok zapsán.")
                    fetch_memory.clear()  # refresh cached block list
                else:
                    st.error(f"Chyba: {r.text}")
            except Exception as e:
                st.error(f"Chyba zápisu: {e}")

    if st.button("🔄 Načíst bloky"):
        fetch_memory.clear()

    mem = fetch_memory()
    if mem is not None:
        blocks = mem.get("blocks", [])
        st.markdown(f"**{mem.get('count', 0)} bloků**")
        if blocks:
            import pandas as pd
            df = pd.DataFrame([{
                "id": b["id"][:12],
                "name": b["name"],
                "class": b["memory_class"],
                "shared_with": ",".join(b.get("shared_with", [])) or "(všichni)",
                "tokens": b.get("token_estimate", 0),
                "read_only": "🔒" if b.get("read_only") else "",
            } for b in blocks])
            st.dataframe(df, width="stretch", hide_index=True)

            st.markdown("#### Test viditelnosti pro agenta")
            agent = st.text_input("Jméno agenta", "persona-analyst", key="mem_agent")
            if st.button("Zkompilovat kontext"):
                ctx = _get_json(f"/features/memory/context/{agent}")
                if ctx is not None:
                    st.json({
                        "visible_blocks": ctx.get("block_ids_used", []),
                        "total_tokens": ctx.get("total_tokens", 0),
                        "messages": ctx.get("messages", []),
                    })
                else:
                    st.error("Nepodařilo se zkompilovat kontext.")
        else:
            st.info("Zatím žádné paměťové bloky.")
    else:
        st.error("Nelze načíst paměť.")


# ---------------------------------------------------------------------------
# Tab 7: NRAM v5 Advanced Controls
# ---------------------------------------------------------------------------
with tab_v5:
    st.markdown("### 🔬 NRAM v5 — Advanced Steering Controls")
    st.info(
        "Experimental research controls for representation-level steering. "
        "These mechanisms operate on hidden states, not just logits."
    )
    
    with st.sidebar:
        st.markdown("### 🔬 NRAM v5 — řízení")
        v5_profile = st.selectbox("Stav", list(STATES.keys()), index=3, key="v5_profile")
        v5_maxtok = st.slider("Max tokenů", 50, 1000, 300, 50, key="v5_maxtok")
        v5_temp = st.slider("Teplota", 0.0, 1.5, 0.8, 0.05, key="v5_temp")
        v5_seed = st.number_input("Seed", 0, 99999, 271, key="v5_seed")
    
    # Representation Control
    st.markdown("#### 🎯 Representation Control (Activation Addition)")
    col_rep1, col_rep2 = st.columns(2)
    with col_rep1:
        rep_enabled = st.toggle("Enable representation control", value=False, key="v5_rep_enabled")
        rep_vector_id = st.text_input("Vector ID", value="novelty_vs_paraphrase_layer20", key="v5_rep_vector")
        rep_alpha = st.slider("Alpha (strength)", -2.0, 2.0, 1.0, 0.1, key="v5_rep_alpha")
    with col_rep2:
        rep_layer = st.selectbox("Target layer", ["auto", "layer_10", "layer_20", "layer_30"], key="v5_rep_layer")
        rep_scope = st.selectbox("Token scope", ["both", "prefill", "decode"], key="v5_rep_scope")
    
    # Conceptor Steering
    st.markdown("#### 🔷 Conceptor Steering")
    col_con1, col_con2 = st.columns(2)
    with col_con1:
        conceptor_enabled = st.toggle("Enable conceptor", value=False, key="v5_con_enabled")
        conceptor_id = st.text_input("Conceptor ID", value="creativity_conceptor", key="v5_con_id")
        conceptor_alpha = st.slider("Conceptor alpha", -2.0, 2.0, 1.0, 0.1, key="v5_con_alpha")
    with col_con2:
        conceptor_aperture = st.slider("Aperture", 0.1, 10.0, 2.0, 0.1, key="v5_con_aperture")
        conceptor_mode = st.selectbox("Mode", ["positive", "negative", "and", "or"], key="v5_con_mode")
    
    # Closed-Loop Control
    st.markdown("#### 🔄 Latent Closed-Loop Control")
    col_loop1, col_loop2 = st.columns(2)
    with col_loop1:
        loop_enabled = st.toggle("Enable closed loop", value=False, key="v5_loop_enabled")
        loop_probe = st.text_input("Probe ID", value="novelty_probe", key="v5_loop_probe")
        loop_target = st.slider("Target value", 0.0, 1.0, 0.7, 0.05, key="v5_loop_target")
    with col_loop2:
        loop_kp = st.slider("Kp (proportional)", 0.0, 2.0, 0.5, 0.1, key="v5_loop_kp")
        loop_ki = st.slider("Ki (integral)", 0.0, 1.0, 0.0, 0.05, key="v5_loop_ki")
        loop_kd = st.slider("Kd (derivative)", 0.0, 1.0, 0.0, 0.05, key="v5_loop_kd")
    
    # Semantic Closed Loop
    st.markdown("#### 📝 Semantic Closed Loop")
    col_sem1, col_sem2 = st.columns(2)
    with col_sem1:
        sem_enabled = st.toggle("Enable semantic loop", value=False, key="v5_sem_enabled")
        sem_chunk_size = st.slider("Chunk size (tokens)", 8, 64, 24, 4, key="v5_sem_chunk")
    with col_sem2:
        sem_source = st.text_area("Source text (optional)", height=80, key="v5_sem_source")
        sem_target_sim = st.slider("Target source similarity", 0.0, 1.0, 0.3, 0.05, key="v5_sem_sim")
    
    # Branch Tournament
    st.markdown("#### 🌳 Branch-and-Tournament Search")
    col_br1, col_br2 = st.columns(2)
    with col_br1:
        branch_enabled = st.toggle("Enable branch tournament", value=False, key="v5_branch_enabled")
        branch_count = st.slider("Number of branches", 2, 8, 4, 1, key="v5_branch_count")
    with col_br2:
        branch_length = st.slider("Branch length (tokens)", 16, 128, 32, 8, key="v5_branch_length")
        branch_single = st.toggle("Single trajectory mode", value=False, key="v5_branch_single")
    
    # DExperts
    st.markdown("#### 🎓 DExperts (Democratized Experts)")
    col_dex1, col_dex2 = st.columns(2)
    with col_dex1:
        dexperts_enabled = st.toggle("Enable DExperts", value=False, key="v5_dex_enabled")
        dexperts_expert = st.text_input("Expert name", value="creativity", key="v5_dex_expert")
        dexperts_alpha = st.slider("Alpha (expert weight)", 0.0, 3.0, 1.0, 0.1, key="v5_dex_alpha")
    with col_dex2:
        dexperts_anti = st.text_input("Anti-expert name (optional)", value="", key="v5_dex_anti")
        dexperts_beta = st.slider("Beta (anti-expert weight)", 0.0, 3.0, 0.5, 0.1, key="v5_dex_beta")
    
    # Telemetry
    st.markdown("#### 📊 Telemetry Level")
    telemetry_level = st.selectbox("Telemetry", ["none", "summary", "detailed"], index=1, key="v5_telemetry")
    
    # Generate button
    prompt = st.text_area(
        "Prompt",
        value="Explain quantum entanglement using a novel metaphor.",
        height=80,
        key="v5_prompt"
    )
    
    if st.button("⚡ Generate with NRAM v5", type="primary", width="stretch", key="v5_btn"):
        if not prompt.strip():
            st.warning("Enter a prompt.")
        else:
            # Build NRAM options
            nram_opts = {
                "enabled": True,
                "profile": STATES[v5_profile]["profile"],
                "intensity": STATES[v5_profile]["intensity"],
                "coherence_floor": STATES[v5_profile]["coherence"],
                "phenomenon_weights": STATE_PHENOMENA_DEFAULTS[v5_profile],
                "telemetry": {"level": telemetry_level},
            }
            
            # Add representation control
            if rep_enabled:
                nram_opts["representation"] = {
                    "enabled": True,
                    "vector_id": rep_vector_id,
                    "alpha": rep_alpha,
                    "layer": rep_layer,
                    "scope": rep_scope,
                }
            
            # Add conceptor steering (NRAM v5)
            if conceptor_enabled:
                nram_opts["conceptor_steering"] = True
                nram_opts["conceptor_id"] = conceptor_id
                nram_opts["conceptor_alpha"] = conceptor_alpha
                nram_opts["conceptor_aperture"] = conceptor_aperture
            
            # Add latent closed loop (NRAM v5)
            if loop_enabled:
                nram_opts["latent_closed_loop"] = True
                nram_opts["latent_loop_target_probe"] = loop_probe
                nram_opts["latent_loop_target_value"] = loop_target
                nram_opts["latent_loop_kp"] = loop_kp
            
            # Add semantic closed loop (NRAM v5)
            if sem_enabled:
                nram_opts["semantic_closed_loop"] = True
                nram_opts["semantic_loop_block_size"] = sem_chunk_size
                nram_opts["semantic_loop_target_score"] = sem_target_sim
            
            # Add branch tournament (NRAM v5)
            if branch_enabled:
                nram_opts["branch_tournament"] = True
                nram_opts["branch_tournament_branch_count"] = branch_count
                nram_opts["branch_tournament_max_tokens"] = branch_length
            
            # Add DExperts (NRAM v5)
            if dexperts_enabled:
                nram_opts["dexperts"] = True
                nram_opts["dexperts_expert"] = dexperts_expert
                nram_opts["dexperts_alpha"] = dexperts_alpha
                nram_opts["dexperts_beta"] = dexperts_beta
            
            with st.spinner("Generating with NRAM v5..."):
                result = call_api(
                    prompt,
                    "nram-qwen3-14b-awq",
                    nram_opts,
                    v5_maxtok,
                    v5_temp,
                    v5_seed
                )
            
            render_result(result)
            
            # Show telemetry if available
            if "nram_telemetry" in result:
                st.markdown("#### 📊 Telemetry")
                with st.expander("View telemetry data"):
                    st.json(result["nram_telemetry"])


# ---------------------------------------------------------------------------
# Tab 8: Runtime Evidence
# ---------------------------------------------------------------------------
with tab_evidence:
    st.markdown("### 🔍 Runtime Evidence — Capability States")
    st.markdown(
        "This tab displays **evidence-based capability states** from the `/nram/capabilities` endpoint. "
        "Each mechanism reports its verification status based on actual test results, not just imports."
    )
    
    @st.cache_data(ttl=30, show_spinner=False)
    def fetch_capabilities():
        return _get_json("/nram/capabilities")
    
    caps = fetch_capabilities()
    
    if caps is None:
        st.error("⚠️ Cannot reach API. Is the server running at http://127.0.0.1:8000?")
    else:
        # Engine info
        st.markdown(f"**Engine:** `{caps.get('engine', 'unknown')}` | "
                   f"**Base Model:** `{caps.get('actual_base_model', 'unknown')}`")
        st.markdown(f"**SGLang Enabled:** {caps.get('sglang_enabled', False)}")
        
        st.divider()
        
        # Controls with evidence states
        controls = caps.get("controls", {})
        
        # Filter to NRAM v5 mechanisms
        v5_mechanisms = [
            "activation_addition",
            "multi_vector_representation",
            "conceptor_steering",
            "hidden_state_probes",
            "latent_closed_loop",
            "semantic_closed_loop",
            "branch_tournament",
            "dexperts",
            "batch_context",
        ]
        
        st.markdown("#### NRAM v5 Mechanism States")
        
        for mech_name in v5_mechanisms:
            mech = controls.get(mech_name, {})
            if not mech or not isinstance(mech, dict):
                continue
            
            state = mech.get("state", "UNKNOWN")
            runtime_wired = mech.get("runtime_wired", False)
            mechanism = mech.get("mechanism", "unknown")
            
            # State badge
            if state == "CAUSALLY_PROVEN":
                state_badge = "🟢 CAUSALLY_PROVEN"
                state_color = "green"
            elif state == "EXECUTING":
                state_badge = "🟡 EXECUTING"
                state_color = "orange"
            elif state == "CONFIGURED":
                state_badge = "🔵 CONFIGURED"
                state_color = "blue"
            else:
                state_badge = "⚪ UNKNOWN"
                state_color = "gray"
            
            with st.expander(f"**{mech_name.replace('_', ' ').title()}** — {state_badge}"):
                st.markdown(f"**State:** `{state}`")
                st.markdown(f"**Runtime Wired:** {'✅ Yes' if runtime_wired else '❌ No'}")
                st.markdown(f"**Mechanism:** `{mechanism}`")
                
                if "last_proof_artifact" in mech:
                    st.markdown(f"**Last Proof Artifact:** `{mech['last_proof_artifact']}`")
                
                if "proof_tests" in mech:
                    st.markdown("**Proof Tests:**")
                    for test in mech["proof_tests"]:
                        st.markdown(f"- `{test}`")
                
                if "note" in mech:
                    st.info(mech["note"])
        
        st.divider()
        
        # Verified capabilities
        verified = caps.get("verified", {})
        st.markdown("#### Verified Capabilities")
        
        verified_cols = st.columns(3)
        for idx, (key, value) in enumerate(verified.items()):
            col = verified_cols[idx % 3]
            with col:
                if value:
                    st.success(f"✅ {key.replace('_', ' ').title()}")
                else:
                    st.warning(f"⚠️ {key.replace('_', ' ').title()}")
        
        st.divider()
        
        # Raw JSON
        with st.expander("📄 Raw Capabilities JSON"):
            st.json(caps)

# ---------------------------------------------------------------------------
# Tab 9: Co to je?
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown("### ℹ️ Co je NRAM?")
    st.markdown(f"""
{DISCLAIMER}

**🔬 Co to je?**

NRAM (Neural Random Access Memory) je experimentální systém, který moduluje výstup
LLM v reálném čase pomocí **logit-processoru** (mění pravděpodobnosti tokenů *před*
samplingem) a **system promptu** odpovídajícího danému stavu. Není to jen přidávání
náhodných slov — skutečně mění způsob generování.

**🌀 Jaké fenomény simuluje?**

| Fenomén | Popis |
|---|---|
| 🌀 Překryv (overlap) | Fragmenty předchozích myšlenek pronikají do aktuálních |
| 💨 Zapomenutí (forgetting) | Ztráta nitě uprostřed věty ("moment…", "co jsem…") |
| ❄️ Zacyklení (looping) | Opakování slov a frází (perseverace, thought loops) |
| ⚡ Skok (associative jump) | Náhlé asociativní přeskoky ("najednou", "a to mi připomíná…") |
| 🎨 Synestézie (synesthesia) | Prolínání smyslů ("barvy znějí", "slyším světlo") |
| 🌊 Rozpuštění (dissolution) | Rozpouštění já a hranic ("já mizí", "hranice se rozpouštějí") |
| ✨ Vhled (insight) | Náhlé vhledy ("AHA!", "VIDÍM TO!") |

**🎯 Co to (ne)dokazuje?**

- **Strukturální podobnost** s trip reporty (fragmentace, smyčky, směs vhledů a zmatení) — ANO
- **Fenomény řízené stavem** (vliv, entropie, koherence), ne náhodou — ANO
- **Měření vědomí** — **NE** (model není vědomý, jen simuluje styl)
- **Podpora užívání látek** — **NE** (explicitně odmítáme)

**⚙️ Technický stack**

- **Model:** Qwen3-14B-AWQ (4-bit, Marlin kernel)
- **Inference:** SGLang s custom logit processorem
- **Řízení:** OpenAI-kompatibilní API + NRAM REST API
- **Ověření:** A/B evaluace (baseline vs NRAM, stejný seed)
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    '<div class="sub-header" style="text-align:center;">'
    'NeuralAccessPsyche · NRAM metriky jsou <b>simulační a řídicí veličiny</b>, nikoli měření vědomí · '
    'Systém nepodporuje ani nenavádí k užívání jakýchkoli látek'
    '</div>',
    unsafe_allow_html=True,
)
