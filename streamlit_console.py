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
    return httpx.Client(timeout=180.0)


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

tab_sim, tab_keynote, tab_ab, tab_pipeline, tab_telemetry, tab_memory, tab_about = st.tabs([
    "🧪 Simulátor", "🗣️ Keynote", "⚖️ A/B Porovnání", "🤖 Agentic Pipeline",
    "📊 Provenance", "🧠 Paměť", "ℹ️ Co to je?"
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
    if st.button("⚡ Generovat", type="primary", use_container_width=True, key="sim_btn"):
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
    if st.button("⚡ Generovat keynote", type="primary", use_container_width=True, key="key_btn"):
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

    if st.button("⚡ Spustit A/B", type="primary", use_container_width=True, key="ab_btn"):
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
    st.markdown("### 🤖 Agentic Innovation Pipeline")
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
        if st.button("🚀 Vytvořit workflow", type="primary", use_container_width=True):
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
        if st.button("🔄 Obnovit stav"):
            try:
                resp = get_client().get(f"{API_BASE}/nram/workflows/{st.session_state.wf_id}", headers=HEADERS)
                wf = resp.json()
                st.markdown(f"**Status:** `{wf.get('status')}` · **Fáze:** `{wf.get('phase')}` · "
                            f"**Iterace:** {wf.get('iteration')}/{wf.get('max_iterations')}")
                ev_resp = get_client().get(f"{API_BASE}/nram/workflows/{st.session_state.wf_id}/events", headers=HEADERS)
                events = ev_resp.json().get("events", [])
                st.markdown(f"**Události ({len(events)}):**")
                for ev in events:
                    st.markdown(f"`{ev.get('phase')}` / {ev.get('event_type')}: {ev.get('summary')}")
                if wf.get("status") == "waiting_for_approval":
                    if st.button("✅ Schválit a pokračovat"):
                        try:
                            get_client().post(f"{API_BASE}/nram/workflows/{st.session_state.wf_id}/approve", headers=HEADERS)
                            st.success("Schváleno. Obnov stav.")
                        except Exception as e:
                            st.error(f"Chyba: {e}")
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
# Tab 5: Provenance dashboard (Feature 9)
# ---------------------------------------------------------------------------
with tab_telemetry:
    st.markdown("### 📊 Token Provenance — jaké řízení způsobilo jaké tokeny")
    st.caption("Applied-policy klasifikace: ukazuje aktivní steering vrstvu, ne kauzální důkaz.")

    col_refresh, _ = st.columns([1, 4])
    if col_refresh.button("🔄 Načíst telemetrii"):
        st.session_state["_tel_refresh"] = True

    try:
        tel = get_client().get(f"{API_BASE}/features/telemetry", headers=HEADERS, timeout=30.0).json()
        g = tel.get("global", {})
        sessions = tel.get("sessions", [])

        c1, c2, c3 = st.columns(3)
        c1.metric("Sezení", g.get("total_sessions", 0))
        c2.metric("Požadavky", g.get("total_requests", 0))
        c3.metric("Tokeny celkem", g.get("total_tokens", 0))

        dist = g.get("token_origin_distribution", {})
        if dist:
            st.markdown("#### Globální distribuce původu tokenů")
            import pandas as pd
            df = pd.DataFrame(
                [{"origin": k, "tokens": v} for k, v in sorted(dist.items(), key=lambda x: -x[1]) if v > 0]
            )
            if not df.empty:
                st.bar_chart(df.set_index("origin"))

        if sessions:
            st.markdown("#### Telemetrie dle sezení")
            for s in sessions[-10:]:
                with st.expander(f"{s.get('session_id','?')} · {s.get('total_requests',0)} req · avg {s.get('avg_latency_ms',0)}ms"):
                    st.json({
                        "profile": s.get("profiles_used", {}),
                        "origins": s.get("token_origin_counts", {}),
                        "phenomena": s.get("phenomena_triggered", {}),
                        "dominant": s.get("dominant_origin"),
                    })
        else:
            st.info("Zatím žádná telemetrie. Spusť generování v Simulátoru.")
    except Exception as e:
        st.error(f"Nelze načíst telemetrii: {e}")

    # Provenance map (Feature 3)
    try:
        pmap = get_client().get(f"{API_BASE}/features/provenance-map", headers=HEADERS, timeout=30.0).json()
        st.markdown("#### Mapa fenomén → steering vrstva → provenance")
        import pandas as pd
        rows = [{"phenomenon": k, **v} for k, v in pmap.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Mapa provenance nedostupná: {e}")


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
                else:
                    st.error(f"Chyba: {r.text}")
            except Exception as e:
                st.error(f"Chyba zápisu: {e}")

    if st.button("🔄 Načíst bloky"):
        st.session_state["_mem_refresh"] = True

    try:
        mem = get_client().get(f"{API_BASE}/features/memory", headers=HEADERS, timeout=30.0).json()
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
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("#### Test viditelnosti pro agenta")
            agent = st.text_input("Jméno agenta", "persona-analyst", key="mem_agent")
            if st.button("Zkompilovat kontext"):
                ctx = get_client().get(
                    f"{API_BASE}/features/memory/context/{agent}", headers=HEADERS, timeout=30.0
                ).json()
                st.json({
                    "visible_blocks": ctx.get("block_ids_used", []),
                    "total_tokens": ctx.get("total_tokens", 0),
                    "messages": ctx.get("messages", []),
                })
        else:
            st.info("Zatím žádné paměťové bloky.")
    except Exception as e:
        st.error(f"Nelze načíst paměť: {e}")


# ---------------------------------------------------------------------------
# Tab 7: Co to je?
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
