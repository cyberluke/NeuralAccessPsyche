# Overview

This is a FastAPI-based application that provides an OpenAI-compatible API with Neural Random Access Memory (NRAM) simulation capabilities. The system offers real-time visualization of neural network states, token stream processing with consciousness level awareness, and advanced API metrics tracking. It combines LLM processing with simulated neural memory states to create an enhanced AI interaction layer.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Backend Framework
**Problem:** Need for a high-performance API server with WebSocket support and async capabilities.

**Solution:** FastAPI with Uvicorn server.

**Rationale:** FastAPI provides native async/await support, automatic OpenAPI documentation, built-on Pydantic validation, and excellent WebSocket handling for real-time state broadcasting.

## Neural Random Access Memory (NRAM) Core
**Problem:** Simulate neural memory states that evolve during LLM interactions.

**Solution:** NumPy-based memory simulation with pattern recognition matrices.

**Key Components:**
- Memory state vectors (configurable size, default 1024)
- Pattern recognition matrix for tracking neural patterns
- Consciousness level mapping (baseline → aware → enlightened → transcendent)
- Real-time state broadcasting via WebSocket

**Rationale:** NumPy provides efficient numerical operations for simulating memory states, and the consciousness level system maps temperature parameters to different awareness states.

## LLM Integration Architecture
**Problem:** Process chat completions while maintaining NRAM state coherence.

**Solution:** Multi-layered handler architecture:
- `LLMHandler`: Primary interface for chat completions
- `GuidanceHandler`: Microsoft Guidance-powered structured generation with consciousness-aware processing
- `NRAMConfigSuggester`: AI-powered configuration optimization using GPT-4o

**Design Pattern:** Handler pattern with dependency injection, allowing each layer to focus on specific responsibilities (routing → LLM processing → guidance → NRAM state management).

## Microsoft Guidance Integration
**Problem:** Need structured LLM output control with token-level phenomena detection.

**Solution:** `core/guidance_handler.py` integrates Microsoft Guidance library for:
- **Structured Generation**: Uses `gen()` for controlled text generation with regex constraints
- **Variable Capture**: Named captures for main_response, neural_insight, primary_phenomenon, coherence
- **Select Constraints**: Uses `select()` for constrained phenomenon type selection
- **Consciousness Mapping**: Maps Czech UI states to internal consciousness levels (baseline, aware, enlightened, transcendent, psychedelic, dissociative)

**Fallback Behavior**: When Guidance library is unavailable, automatically falls back to standard OpenAI API with simulated phenomena detection.

**Key Features**:
- Consciousness-aware system prompts for each awareness level
- Token-level phenomena weighting based on consciousness state
- Neural insight generation with consciousness-appropriate messages
- Pydantic schema for structured response validation (NRAMResponse)

## Authentication & Rate Limiting
**Problem:** Secure API access while preventing abuse.

**Solution:** Custom middleware stack:
- `TokenAuthMiddleware`: Bearer token validation on `/v1` routes
- `RateLimitMiddleware`: Time-window based rate limiting (60 requests/minute default)

**Exemptions:** Visualization and explorer routes bypass authentication for easy access.

**Pros:** Flexible middleware allows route-specific security policies.

**Cons:** Current token validation is simplified (production would need database-backed user management).

## Real-Time Visualization System
**Problem:** Monitor NRAM states and API performance in real-time.

**Solution:** Dual WebSocket system:
- `/ws/nram-state`: Broadcasts NRAM memory state updates
- WebSocket-based metrics streaming for API call visualization

**Frontend Stack:**
- D3.js for interactive visualizations
- Plotly for time-series graphing
- Custom CSS with dark theme optimized for neural network visualization

**Rationale:** WebSockets enable push-based updates without polling, reducing latency and server load.

## API Logging & Metrics
**Problem:** Track OpenAI API usage, performance, and costs.

**Solution:** Decorator-based logging system with centralized metrics:
- `@log_api_call` decorator for automatic call tracking
- `APIMetrics` class aggregating success rates, token usage, and latency
- File-based logging to `logs/openai_api.log`
- Real-time metrics broadcasting to connected WebSocket clients

**Rationale:** Decorator pattern keeps logging concerns separate from business logic while providing comprehensive observability.

## Static File Management
**Problem:** Serve visualization assets efficiently.

**Solution:** FastAPI StaticFiles mount at `/static` serving CSS, JavaScript, and other assets.

**Template Rendering:** Jinja2Templates for server-side HTML rendering of visualization pages.

## Streamlit Token Stream UI (NRAM v4)
**Problem:** Provide a user-friendly interface for exploring token streams with NLP visualization.

**Solution:** `streamlit_token_stream.py` creates a comprehensive Streamlit app with:
- **Annotated Text Visualization**: Token-level highlighting with category labels
- **Consciousness State Selector**: 6 states (Normální, Mikrodávka, Psychedelická, Prahová, Vrchol, Disociativní)
- **NRAM Controls**: Intensity slider, temperature control, animation speed
- **Real-time Metrics**: Consciousness level, entropy, coherence, momentum
- **Phenomena Tracking**: 11 token categories with color-coded visualization
- **Statistics Panel**: API calls, token counts, phenomena analysis

**Token Categories:**
- Claude AI, Překryv, Zapomenuté, Zacyklení, Skok, Synestéze
- Rozpuštění, Fragmentace, Ozvěna, Odbočka, Vhled

**Workflow:** Runs on port 5000 via `streamlit run streamlit_token_stream.py`

**Rationale:** Streamlit provides rapid prototyping for data-centric UIs with native support for annotated text visualization, complementing the main FastAPI application.

## Configuration Management
**Problem:** Optimize NRAM parameters based on performance.

**Solution:** AI-powered configuration suggester using GPT-4o to analyze performance history and recommend parameter adjustments.

**Key Features:**
- Rolling performance history (last 100 metrics)
- Pattern intensity and state stability analysis
- Consciousness level distribution tracking
- AI-generated configuration recommendations

# External Dependencies

## OpenAI API
**Purpose:** Primary LLM provider for chat completions.

**Model:** GPT-4o (fixed, requires explicit user request to change).

**Authentication:** Environment variable `OPENAI_API_KEY`.

**Integration Points:**
- `core/llm_handler.py`: Direct completion requests
- `core/guidance_handler.py`: Consciousness-enhanced processing
- `core/config_suggester.py`: Performance analysis and recommendations

## JavaScript Libraries
**D3.js v7:** Interactive data visualizations (network graphs, heatmaps, memory grids).

**Plotly:** Time-series plotting for NRAM state evolution.

**Purpose:** Real-time visualization of neural states and API metrics.

## Python Dependencies
**FastAPI:** Web framework and API routing.

**Uvicorn:** ASGI server for running FastAPI applications.

**Pydantic:** Request/response validation and serialization.

**NumPy:** Numerical operations for NRAM simulation.

**Streamlit:** Alternative UI for token stream exploration (separate app).

**OpenAI Python SDK v4.77.0:** Official client for OpenAI API interactions.

**Jinja2:** Template rendering for HTML visualization pages.

## Frontend Assets
**Google Fonts (Inter):** Typography for Streamlit interface.

**Custom CSS:** Dark-themed styling optimized for neural network visualization.

## Logging Infrastructure
**Python logging module:** Structured logging with file handlers.

**Log Directory:** `logs/` for persistent API call tracking.

**Format:** Timestamped entries with log levels for debugging and monitoring.