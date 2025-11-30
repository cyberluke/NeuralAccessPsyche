import streamlit as st
from annotated_text import annotated_text
import numpy as np
import random
import os
from typing import List, Dict, Tuple, Any
from core.guidance_handler import GuidanceHandler, GUIDANCE_AVAILABLE

st.set_page_config(
    page_title="NRAM Token Stream v4",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background-color: #0f1419;
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        color: #4dabf7;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #6c757d;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    
    .stTextArea textarea {
        background-color: #1a1f25;
        border: 1px solid #2d3748;
        color: #e2e8f0;
        border-radius: 8px;
    }
    
    .token-container {
        background-color: #1a1f25;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid #2d3748;
    }
    
    .stats-card {
        background-color: #1a1f25;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #2d3748;
        margin-bottom: 1rem;
    }
    
    .stat-label {
        color: #6c757d;
        font-size: 0.8rem;
    }
    
    .stat-value {
        color: #4dabf7;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .sidebar .stSlider > div > div {
        background-color: #2d3748;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #12171d;
        border-right: 1px solid #2d3748;
    }
    
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1.5rem;
    }
    
    .stButton > button[kind="primary"] {
        background-color: #22c55e;
        color: white;
    }
    
    div[data-testid="stHorizontalBlock"] {
        gap: 0.5rem;
    }
    
    .phenomena-item {
        display: flex;
        align-items: center;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
    
    .phenomena-color {
        width: 12px;
        height: 12px;
        border-radius: 3px;
        margin-right: 8px;
        display: inline-block;
    }
    
    .consciousness-state {
        text-align: center;
        padding: 0.5rem;
        background: linear-gradient(135deg, #1a1f25 0%, #2d3748 100%);
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

TOKEN_CATEGORIES = {
    "claude_ai": {"color": "#a78bfa", "label": "Claude AI", "emoji": "🤖"},
    "overlap": {"color": "#f472b6", "label": "Překryv", "emoji": "🔗"},
    "forgotten": {"color": "#fbbf24", "label": "Zapomenuté", "emoji": "💭"},
    "loop": {"color": "#f87171", "label": "Zacyklení", "emoji": "🔄"},
    "jump": {"color": "#a855f7", "label": "Skok", "emoji": "⚡"},
    "synesthesia": {"color": "#22d3ee", "label": "Synestéze", "emoji": "🎨"},
    "dissolution": {"color": "#fb923c", "label": "Rozpuštění", "emoji": "💫"},
    "fragmentation": {"color": "#ef4444", "label": "Fragmentace", "emoji": "💔"},
    "echo": {"color": "#84cc16", "label": "Ozvěna", "emoji": "🔊"},
    "tangent": {"color": "#14b8a6", "label": "Odbočka", "emoji": "↪"},
    "insight": {"color": "#eab308", "label": "Vhled", "emoji": "✨"},
}

CONSCIOUSNESS_STATES = {
    "Normální": {"intensity_range": (0.0, 0.2), "color": "#6b7280"},
    "Mikrodávka": {"intensity_range": (0.2, 0.4), "color": "#22c55e"},
    "Psychedelická": {"intensity_range": (0.4, 0.6), "color": "#f97316"},
    "Prahová": {"intensity_range": (0.6, 0.75), "color": "#ec4899"},
    "Vrchol": {"intensity_range": (0.75, 0.9), "color": "#a855f7"},
    "Disociativní": {"intensity_range": (0.9, 1.0), "color": "#06b6d4"},
}

@st.cache_resource
def get_guidance_handler():
    """Get or create the GuidanceHandler instance"""
    return GuidanceHandler()

def initialize_session_state():
    if "nram_state" not in st.session_state:
        st.session_state.nram_state = {
            "consciousness": 0.5,
            "entropy": 0.215,
            "coherence": 0.942,
            "momentum": 0.058,
        }
    if "token_stream" not in st.session_state:
        st.session_state.token_stream = []
    if "original_text" not in st.session_state:
        st.session_state.original_text = ""
    if "statistics" not in st.session_state:
        st.session_state.statistics = {
            "api_calls": 0,
            "total_tokens": 0,
            "claude_tokens": 0,
            "phenomena_count": 0,
        }
    if "phenomena_counts" not in st.session_state:
        st.session_state.phenomena_counts = {cat: 0 for cat in TOKEN_CATEGORIES}
    if "show_original" not in st.session_state:
        st.session_state.show_original = False
    if "neural_insight" not in st.session_state:
        st.session_state.neural_insight = None
    if "guidance_used" not in st.session_state:
        st.session_state.guidance_used = False

def tokenize_with_phenomena(text: str, intensity: float, state: str) -> List[Dict]:
    words = text.split()
    tokens = []
    phenomena_counts = {cat: 0 for cat in TOKEN_CATEGORIES}
    
    base_probability = intensity * 0.5
    
    state_multipliers = {
        "Normální": 0.1,
        "Mikrodávka": 0.3,
        "Psychedelická": 0.5,
        "Prahová": 0.7,
        "Vrchol": 0.85,
        "Disociativní": 0.95,
    }
    
    multiplier = state_multipliers.get(state, 0.3)
    
    for i, word in enumerate(words):
        if random.random() < base_probability * multiplier:
            category = random.choice(list(TOKEN_CATEGORIES.keys()))
            phenomena_counts[category] += 1
            tokens.append({
                "text": word,
                "category": category,
                "is_phenomenon": True,
            })
        else:
            tokens.append({
                "text": word,
                "category": None,
                "is_phenomenon": False,
            })
    
    return tokens, phenomena_counts

def generate_nram_response(prompt: str, intensity: float, temperature: float, consciousness_state: str) -> Dict:
    """Generate response using Microsoft Guidance with consciousness-aware processing"""
    try:
        handler = get_guidance_handler()
        
        result = handler.process_with_guidance_sync(
            query=prompt,
            consciousness_level=consciousness_state,
            temperature=temperature,
            max_tokens=500
        )
        
        return result
    except Exception as e:
        return {
            "main_response": f"Chyba při generování odpovědi: {str(e)}",
            "consciousness_level": consciousness_state,
            "neural_insight": None,
            "token_phenomena": [],
            "coherence_score": 0.0,
            "raw_tokens": [],
            "guidance_used": False
        }

def update_nram_state(intensity: float, temperature: float):
    st.session_state.nram_state["consciousness"] = intensity
    st.session_state.nram_state["entropy"] = 0.1 + (intensity * 0.4) + random.uniform(-0.05, 0.05)
    st.session_state.nram_state["coherence"] = 0.95 - (intensity * 0.3) + random.uniform(-0.02, 0.02)
    st.session_state.nram_state["momentum"] = abs(temperature - 0.5) * 0.2 + random.uniform(0, 0.05)

def render_token_stream(tokens: List[Dict]):
    annotated_elements = []
    
    for i, token in enumerate(tokens):
        if token["is_phenomenon"] and token["category"]:
            cat_info = TOKEN_CATEGORIES[token["category"]]
            annotated_elements.append(
                (token["text"], cat_info["label"], cat_info["color"])
            )
            annotated_elements.append(" ")
        else:
            annotated_elements.append(token["text"] + " ")
    
    annotated_text(*annotated_elements)

def get_current_state_name(intensity: float) -> str:
    for state_name, state_info in CONSCIOUSNESS_STATES.items():
        min_val, max_val = state_info["intensity_range"]
        if min_val <= intensity < max_val:
            return state_name
    return "Disociativní"

def main():
    initialize_session_state()
    
    with st.sidebar:
        st.markdown("### 🧠 NRAM v4")
        st.markdown('<p style="color: #6c757d; font-size: 0.8rem;">Neuronová Paměť s Modulací Vědomí</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### Stav vědomí")
        
        selected_state = st.radio(
            "Vyberte stav",
            list(CONSCIOUSNESS_STATES.keys()),
            horizontal=True,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("#### Intenzita NRAM")
        
        state_range = CONSCIOUSNESS_STATES[selected_state]["intensity_range"]
        default_intensity = (state_range[0] + state_range[1]) / 2
        
        intensity = st.slider(
            "Intenzita",
            min_value=0.0,
            max_value=1.0,
            value=default_intensity,
            step=0.01,
            format="%.0f%%",
            label_visibility="collapsed"
        )
        st.progress(intensity)
        
        st.markdown("---")
        st.markdown("#### Teplota")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            temperature = st.slider(
                "Teplota",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.05,
                label_visibility="collapsed"
            )
        with col2:
            auto_temp = st.checkbox("Auto", value=False)
        
        if auto_temp:
            temperature = intensity * 1.5 + 0.3
        
        st.markdown("---")
        st.markdown("#### Rychlost animace")
        animation_speed = st.slider(
            "Rychlost",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            format="%.1fx",
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("#### Stav NRAM")
        
        state_color = CONSCIOUSNESS_STATES[selected_state]["color"]
        st.markdown(f"""
        <div class="consciousness-state">
            <span style="color: {state_color}; font-weight: 600;">{selected_state}</span>
        </div>
        """, unsafe_allow_html=True)
        
        update_nram_state(intensity, temperature)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Vědomí", f"{st.session_state.nram_state['consciousness']:.3f}")
            st.metric("Koherence", f"{st.session_state.nram_state['coherence']:.3f}")
        with col2:
            st.metric("Entropie", f"{st.session_state.nram_state['entropy']:.3f}")
            st.metric("Momentum", f"{st.session_state.nram_state['momentum']:.3f}")
        
        st.markdown("---")
        st.markdown("#### Fenomény")
        
        for cat_key, cat_info in TOKEN_CATEGORIES.items():
            count = st.session_state.phenomena_counts.get(cat_key, 0)
            total = sum(st.session_state.phenomena_counts.values()) or 1
            percentage = (count / total) * 100 if total > 0 else 0
            
            st.markdown(f"""
            <div class="phenomena-item">
                <span class="phenomena-color" style="background-color: {cat_info['color']};"></span>
                <span style="color: #a0aec0;">{cat_info['emoji']} {cat_info['label']}</span>
                <span style="color: #4dabf7; margin-left: auto;">{percentage:.0f}%</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### Statistiky")
        
        st.markdown(f"""
        <div class="stats-card">
            <div><span class="stat-label">API volání</span><br><span class="stat-value">{st.session_state.statistics['api_calls']}</span></div>
        </div>
        <div class="stats-card">
            <div><span class="stat-label">Celkem tokenů</span><br><span class="stat-value">{st.session_state.statistics['total_tokens']}</span></div>
        </div>
        <div class="stats-card">
            <div><span class="stat-label">Z Claude AI</span><br><span class="stat-value">{st.session_state.statistics['claude_tokens']}</span></div>
        </div>
        <div class="stats-card">
            <div><span class="stat-label">Fenomény</span><br><span class="stat-value">{st.session_state.statistics['phenomena_count']}</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🧠 NRAM Token Stream v4</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Skutečné Claude API • Token-level modulace • Simulace změněných stavů vědomí</p>', unsafe_allow_html=True)
    
    user_input = st.text_area(
        "Zadejte text pro analýzu",
        placeholder="Ahoj, kdo jsi?",
        height=100,
        label_visibility="collapsed"
    )
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
    
    with col1:
        generate_btn = st.button("✨ Generovat", type="primary", use_container_width=True)
    with col2:
        reset_btn = st.button("↺ Reset", use_container_width=True)
    with col3:
        show_original_btn = st.button("👁 Ukázat originál", use_container_width=True)
    
    if generate_btn and user_input:
        with st.spinner("Generuji odpověď s Microsoft Guidance..."):
            result = generate_nram_response(user_input, intensity, temperature, selected_state)
            
            response_text = result.get("main_response", "")
            st.session_state.original_text = response_text
            st.session_state.neural_insight = result.get("neural_insight")
            st.session_state.guidance_used = result.get("guidance_used", False)
            
            if result.get("raw_tokens"):
                tokens = []
                phenomena_counts = {cat: 0 for cat in TOKEN_CATEGORIES}
                phenomenon_mapping = {
                    "coherent": "claude_ai",
                    "overlap": "overlap",
                    "forgotten": "forgotten",
                    "looping": "loop",
                    "jumping": "jump",
                    "synesthetic": "synesthesia",
                    "dissolution": "dissolution",
                    "fragmentation": "fragmentation",
                    "echo": "echo",
                    "tangent": "tangent",
                    "insight": "insight"
                }
                
                for raw_token in result["raw_tokens"]:
                    phenomenon = raw_token.get("phenomenon", "coherent")
                    category = phenomenon_mapping.get(phenomenon, "claude_ai")
                    is_phenomenon = phenomenon != "coherent"
                    
                    if is_phenomenon:
                        phenomena_counts[category] = phenomena_counts.get(category, 0) + 1
                    
                    tokens.append({
                        "text": raw_token.get("text", ""),
                        "category": category if is_phenomenon else None,
                        "is_phenomenon": is_phenomenon,
                    })
                
                st.session_state.token_stream = tokens
                st.session_state.phenomena_counts = phenomena_counts
            else:
                current_state = get_current_state_name(intensity)
                tokens, phenomena_counts = tokenize_with_phenomena(response_text, intensity, current_state)
                st.session_state.token_stream = tokens
                st.session_state.phenomena_counts = phenomena_counts
            
            if result.get("coherence_score"):
                st.session_state.nram_state["coherence"] = result["coherence_score"]
            
            st.session_state.statistics["api_calls"] += 1
            st.session_state.statistics["total_tokens"] += len(st.session_state.token_stream)
            st.session_state.statistics["claude_tokens"] += len([t for t in st.session_state.token_stream if t["is_phenomenon"]])
            st.session_state.statistics["phenomena_count"] = sum(st.session_state.phenomena_counts.values())
    
    if reset_btn:
        st.session_state.token_stream = []
        st.session_state.original_text = ""
        st.session_state.phenomena_counts = {cat: 0 for cat in TOKEN_CATEGORIES}
        st.session_state.show_original = False
        st.rerun()
    
    if show_original_btn:
        st.session_state.show_original = not st.session_state.show_original
    
    if st.session_state.neural_insight:
        guidance_badge = "🎯 MS Guidance" if st.session_state.guidance_used else "⚡ OpenAI"
        st.info(f"{guidance_badge} | **Neurální vhled:** {st.session_state.neural_insight}")
    
    st.markdown('<div class="token-container">', unsafe_allow_html=True)
    st.markdown("#### 🧠 NRAM-modulovaný výstup:")
    
    if st.session_state.show_original and st.session_state.original_text:
        st.markdown("**Originální text:**")
        st.write(st.session_state.original_text)
    elif st.session_state.token_stream:
        render_token_stream(st.session_state.token_stream)
    else:
        st.markdown('<p style="color: #6c757d;">Zadejte text a klikněte na "Generovat" pro zobrazení token streamu.</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    legend_cols = st.columns(len(TOKEN_CATEGORIES))
    for i, (cat_key, cat_info) in enumerate(TOKEN_CATEGORIES.items()):
        with legend_cols[i % len(legend_cols)]:
            st.markdown(f"""
            <div style="display: flex; align-items: center; font-size: 0.75rem;">
                <span style="width: 10px; height: 10px; background-color: {cat_info['color']}; border-radius: 50%; margin-right: 5px;"></span>
                <span style="color: #a0aec0;">{cat_info['label']}</span>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
