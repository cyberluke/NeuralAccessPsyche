from fastapi import APIRouter, HTTPException, Depends, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from core.llm_handler import LLMHandler, InferenceError
from core.nram import NRAM
from core.config_suggester import NRAMConfigSuggester
from core.engines.registry import (
    get_sglang_engine,
    should_route_to_sglang,
    sglang_enabled,
)
from core.engines.sglang_engine import SGLangEngineError
from core.contracts.openai import ChatCompletionRequest as EngineRequest
from core.tools.searxng_client import SearXNGClient
from utils.validators import validate_request
from utils.auth import get_current_user
from utils.api_logger import api_metrics
import logging
import json
import asyncio
import time
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")
llm_handler = LLMHandler()
nram = NRAM()
templates = Jinja2Templates(directory="templates")
config_suggester = NRAMConfigSuggester()
searxng_client = SearXNGClient()

# Supported model aliases
MODEL_ALIASES = {
    "nram-gpt-oss-20b": "nram-gpt-oss-20b",
    "gpt-oss-20b-baseline": "gpt-oss-20b-baseline",
    "deepseek-r1-qwen-7b-baseline": "deepseek-r1-qwen-7b-baseline",
    "nram-deepseek-r1-qwen-7b": "nram-deepseek-r1-qwen-7b",
    # Qwen3-14B-AWQ personas (selectable in agentic coding)
    "qwen3-14b-awq-baseline": "qwen3-14b-awq-baseline",
    "nram-qwen3-14b-awq": "nram-qwen3-14b-awq",
    # Persona models (route to NRAM profiles)
    "persona-normal": "persona-normal",
    "persona-microdose": "persona-microdose",
    "persona-threshold": "persona-threshold",
    "persona-psychedelic": "persona-psychedelic",
    "persona-peak": "persona-peak",
    "persona-dissociative": "persona-dissociative",
    "persona-keynote": "persona-keynote",
    # MoE orchestrator (queries multiple personas, synthesizes)
    "nram-moe-orchestrator": "nram-moe-orchestrator",
}

# Persona model -> NRAM profile mapping
_PERSONA_MODEL_MAP = {
    "persona-normal": "normal",
    "persona-microdose": "microdose",
    "persona-threshold": "threshold",
    "persona-psychedelic": "psychedelic",
    "persona-peak": "peak",
    "persona-dissociative": "dissociative",
    "persona-keynote": "visionary-psychedelic-keynote",
}

# MoE orchestrator personas (which personas to query)
_MOE_DEFAULT_PERSONAS = [
    "normal", "threshold", "psychedelic", "visionary-psychedelic-keynote",
]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 100
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    # NRAM extension — optional, validated server-side only
    nram: Optional[Dict[str, Any]] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


def _openai_error(status_code: int, message: str, err_type: str, code: str) -> JSONResponse:
    """Return an OpenAI-shaped error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": err_type,
                "param": None,
                "code": code,
            }
        },
    )


def _is_persona_model(model: str) -> bool:
    """Check if model is a persona model that routes to NRAM profiles."""
    return model in _PERSONA_MODEL_MAP


def _is_moe_model(model: str) -> bool:
    """Check if model is the MoE orchestrator."""
    return model == "nram-moe-orchestrator"


async def _route_persona(request: ChatCompletionRequest, model: str) -> Dict[str, Any]:
    """Route a persona-model request to SGLang with NRAM opts injected."""
    engine = get_sglang_engine()
    if engine is None:
        raise InferenceError("SGLang engine not available", code="engine_unavailable")

    profile = _PERSONA_MODEL_MAP[model]
    
    # State-dependent intensity: each state gets a different intensity level
    # This creates qualitatively different outputs, not just quantitative differences
    intensity_map = {
        "normal": 0.1,           # Minimal steering, structured analysis
        "microdose": 0.35,       # Subtle pattern recognition
        "threshold": 0.58,       # Bold associative leaps
        "psychedelic": 0.84,     # Extraordinary synthesis
        "peak": 1.0,             # Maximum revolutionary insights
        "dissociative": 0.70,    # Radical deconstruction
    }
    intensity = intensity_map.get(profile, 0.5)
    
    # Inject NRAM options from the persona profile
    nram_opts = {
        "enabled": True,
        "profile": profile,
        "intensity": intensity,
    }
    if request.nram:
        nram_opts.update(request.nram)
    request.nram = nram_opts
    
    # CRITICAL: Switch model to NRAM-enabled alias so engine activates logit processor
    public_model = request.model
    request.model = "nram-qwen3-14b-awq"

    engine_request = _to_engine_request(request, request.messages)
    response = await engine.complete(engine_request)
    
    result = response.model_dump()
    result["model"] = public_model  # Return original model name to client
    return result


async def _route_moe(request: ChatCompletionRequest) -> Dict[str, Any]:
    """MoE orchestrator: query personas in parallel, synthesize final answer."""
    engine = get_sglang_engine()
    if engine is None:
        raise InferenceError("SGLang engine not available", code="engine_unavailable")

    user_content = ""
    for msg in reversed(request.messages):
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
            break

    # Query each persona in parallel (allow substantial insights for strategic analysis)
    async def query_persona(profile: str):
        sub_request = _to_engine_request(request, request.messages)
        # CRITICAL: Switch model to NRAM-enabled alias so engine activates logit processor
        sub_request.model = "nram-qwen3-14b-awq"
        sub_request.nram = {"enabled": True, "profile": profile, "intensity": 0.9}
        sub_request.max_tokens = 1024
        try:
            result = await engine.complete(sub_request)
            # Keep FULL content for rich synthesis — no truncation
            return (profile, result.choices[0].message.content)
        except Exception as e:
            return (profile, f"[error: {e}]")

    # Run all persona queries concurrently
    tasks = [query_persona(p) for p in _MOE_DEFAULT_PERSONAS]
    persona_results = await asyncio.gather(*tasks)

    # Build synthesis prompt — respect user's max_tokens for strategic analysis
    persona_summaries = []
    for profile, output in persona_results:
        # Keep FULL content for rich synthesis — no truncation
        persona_summaries.append(f"[{profile} perspective]:\n{output}")

    # Determine synthesis length based on user's request
    requested_tokens = request.max_tokens or 2048
    synthesis_max_tokens = max(1024, min(requested_tokens, 4096))  # Increased minimum from 512

    synthesis_input = (
        f"Question: {user_content[:500]}\n\n"
        "Perspectives from multiple analytical lenses:\n" + "\n".join(persona_summaries) +
        f"\n\nSynthesize these perspectives into a comprehensive, strategic response. "
        f"Integrate the best insights into a coherent analysis. Be thorough, substantive, and detailed. "
        f"Provide concrete examples, data points, and actionable recommendations."
    )

    # Synthesis uses baseline model (no NRAM steering) for coherence
    synth_messages = [
        {"role": "system", "content": "You are a strategic synthesizer. Integrate multiple analytical perspectives into a comprehensive, well-structured response. Be thorough, substantive, insightful, and detailed. Provide concrete examples and actionable recommendations."},
        {"role": "user", "content": synthesis_input},
    ]
    synth_request = _to_engine_request(request, synth_messages)
    synth_request.model = "qwen3-14b-awq-baseline"
    synth_request.nram = None
    synth_request.max_tokens = synthesis_max_tokens
    synth_request.temperature = 0.5

    synth_response = await engine.complete(synth_request)
    result = synth_response.model_dump()
    result["model"] = "nram-moe-orchestrator"
    return result

@router.get("/visualize", response_class=HTMLResponse)
async def visualize_nram(request: Request):
    """Serve the NRAM visualization page"""
    return templates.TemplateResponse(
        "visualization.html",
        {"request": request}
    )

@router.websocket("/ws/nram-state")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time NRAM state updates"""
    await nram.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        nram.disconnect(websocket)

@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Handle chat completion requests — OpenAI-compatible, no fake 200 on failure."""
    try:
        validate_request(request)

        # Reject client-supplied processor injection attempts
        nram_opts = request.nram or {}
        _FORBIDDEN_FIELDS = {
            "custom_logit_processor", "serialized_processor",
            "processor_class", "python_code",
        }
        if _FORBIDDEN_FIELDS & set(nram_opts.keys()):
            return _openai_error(
                400,
                "Forbidden field in nram options: processor injection is not allowed.",
                "invalid_request_error",
                "forbidden_field",
            )

        # Route persona models (e.g., persona-peak -> NRAM peak profile)
        if _is_persona_model(request.model):
            return await _route_persona(request, request.model)

        # Route MoE orchestrator model
        if _is_moe_model(request.model):
            return await _route_moe(request)

        # Process through NRAM (state tracking only — messages returned unchanged)
        messages = nram.process_messages(request.messages)

        # Broadcast updated state for visualization
        await nram.broadcast_state()

        # Route to SGLang engine when feature flag is active for this model
        if should_route_to_sglang(request.model):
            return await _handle_sglang(request, messages)

        # Legacy path below
        # FIX defect 6: handle streaming requests
        if request.stream:
            return await _stream_completion(request, messages)

        # FIX defect 5: pass requested model alias
        response = await llm_handler.generate_response(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            model=request.model,
        )

        return response

    except SGLangEngineError as e:
        logger.error(f"SGLang engine error: {e.message}")
        return _openai_error(e.status_code, e.message, "inference_error", "upstream_inference_failed")

    except InferenceError as e:
        logger.error(f"Inference error: {e.message}")
        # FIX defect 8: return OpenAI-shaped error, not a fake 200
        return _openai_error(502, e.message, "inference_error", e.code)

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}", exc_info=True)
        return _openai_error(500, str(e), "inference_error", "upstream_inference_failed")


def _to_engine_request(request: ChatCompletionRequest, messages: list) -> EngineRequest:
    """Convert the route request model into the engine contract request."""
    return EngineRequest(
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stream=request.stream,
        stop=request.stop,
        seed=request.seed,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        response_format=request.response_format,
        tools=request.tools,
        tool_choice=request.tool_choice,
        nram=request.nram,
    )


async def _handle_sglang(request: ChatCompletionRequest, messages: list):
    """Route a request through the SGLang engine with tool call support."""
    engine = get_sglang_engine()
    if engine is None:
        return _openai_error(
            503,
            "SGLang engine is enabled but not available.",
            "inference_error",
            "engine_unavailable",
        )

    # Add SearXNG tools if no tools specified
    tools = request.tools
    if tools is None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "searxng_search",
                    "description": "Search the web for current information, market research, company analysis, or technology trends",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            },
                            "search_type": {
                                "type": "string",
                                "enum": ["general", "market_research", "company_analysis", "technology_trends"],
                                "description": "Type of search: general web search, market research, company analysis, or technology trends"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    engine_request = _to_engine_request(request, messages)
    engine_request.tools = tools

    if request.stream:
        generator = engine.stream(engine_request)
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    response = await engine.complete(engine_request)
    response_dict = response.model_dump()
    
    # Check if model wants to call a tool
    if response_dict.get("choices") and response_dict["choices"][0].get("message", {}).get("tool_calls"):
        tool_calls = response_dict["choices"][0]["message"]["tool_calls"]
        
        # Execute tool calls
        tool_results = []
        for tool_call in tool_calls:
            if tool_call["function"]["name"] == "searxng_search":
                try:
                    args = json.loads(tool_call["function"]["arguments"])
                    query = args.get("query", "")
                    search_type = args.get("search_type", "general")
                    
                    if search_type == "market_research":
                        result = await searxng_client.market_research(query)
                    elif search_type == "company_analysis":
                        result = await searxng_client.company_analysis(query)
                    elif search_type == "technology_trends":
                        result = await searxng_client.technology_trends(query)
                    else:
                        result = await searxng_client.search(query)
                    
                    tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "content": json.dumps(result, ensure_ascii=False)
                    })
                except Exception as e:
                    logger.error(f"Tool call error: {e}")
                    tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "content": json.dumps({"error": str(e)})
                    })
        
        # Add tool results to messages and get final response
        messages_with_tools = messages + [response_dict["choices"][0]["message"]] + tool_results
        engine_request.messages = messages_with_tools
        engine_request.tools = None  # Don't allow nested tool calls
        
        final_response = await engine.complete(engine_request)
        return final_response.model_dump()
    
    return response_dict

@router.get("/models")
async def list_models(current_user: dict = Depends(get_current_user)):
    """List available models — OpenAI-compatible /v1/models."""
    return {
        "object": "list",
        "data": [
            {
                "id": alias,
                "object": "model",
                "created": 0,
                "owned_by": "neuralaccesspsyche",
                "permission": [],
            }
            for alias in MODEL_ALIASES
        ],
    }


@router.get("/nram/capabilities")
async def nram_capabilities(current_user: dict = Depends(get_current_user)):
    """Report the active engine and which steering controls are verified."""
    engine_active = sglang_enabled()
    return {
        "engine": "sglang" if engine_active else "legacy",
        "sglang_enabled": engine_active,
        "served_aliases": list(MODEL_ALIASES.keys()),
        "controls": {
            "prompt_steering": True,
            "structured_output_planning": True,
            "hard_token_masking": True,
            "soft_logit_biasing": True,
            "dynamic_repetition_penalty": True,
            "activation_steering": False,
        },
        "verified": {
            "pre_sampling_logit_modification": False,
            "forced_token_proof": False,
            "streaming": True,
        },
    }


@router.get("/nram/token-policy/{profile}")
async def nram_token_policy(
    profile: str,
    current_user: dict = Depends(get_current_user),
):
    """Diagnostics: show token IDs, decoded text, skipped reasons, and bias values.

    Requires authentication. Returns the compiled policy for a persona profile.
    """
    from core.persona.profiles import PROFILES
    from core.persona.compiler import compile_policy
    from core.steering.tokenizer_bias import TokenBiasCompiler

    state = PROFILES.get(profile)
    if state is None:
        return _openai_error(404, f"Unknown profile: {profile}", "invalid_request_error", "unknown_profile")

    policy = compile_policy(state)

    engine = get_sglang_engine()
    tokenizer = getattr(engine, "_tokenizer", None) if engine else None

    if tokenizer is None:
        return {
            "profile": profile,
            "policy": policy.model_dump(),
            "tokens": None,
            "note": "No tokenizer loaded (set NRAM_TOKENIZER_PATH). Showing lexemes only.",
        }

    compiler = TokenBiasCompiler(tokenizer)
    compiled, diagnostics = compiler.compile_with_diagnostics(policy)
    return {
        "profile": profile,
        "policy": policy.model_dump(),
        "compiled": compiled.model_dump(),
        "diagnostics": diagnostics,
    }

@router.get("/visualize/api", response_class=HTMLResponse)
async def visualize_api(request: Request):
    """Serve the API visualization dashboard"""
    return templates.TemplateResponse(
        "api_visualization.html",
        {"request": request}
    )

@router.websocket("/ws/api-viz")
async def api_visualization_websocket(websocket: WebSocket):
    """WebSocket endpoint for API visualization"""
    await api_metrics.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        api_metrics.disconnect(websocket)

@router.get("/nram/explorer", response_class=HTMLResponse)
async def nram_explorer(request: Request):
    """Serve the NRAM architecture explorer page"""
    try:
        return templates.TemplateResponse(
            "nram_explorer.html",
            {"request": request}
        )
    except Exception as e:
        logger.error(f"Error serving NRAM explorer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/nram-explorer")
async def nram_explorer_websocket(websocket: WebSocket):
    """WebSocket endpoint for NRAM architecture exploration"""
    await websocket.accept()
    try:
        while True:
            # Check for client messages (e.g., reset command)
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("action") == "reset":
                    nram.reset_state()
            except json.JSONDecodeError:
                pass

            # Get current NRAM state
            state = nram.get_explorer_state()

            # Send state update
            await websocket.send_json(state)

            # Brief delay to prevent overwhelming the client
            await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"WebSocket error in NRAM explorer: {str(e)}")
    finally:
        try:
            await websocket.close()
        except:
            pass

@router.get("/nram/config/analyze")
async def analyze_nram_config(current_user: dict = Depends(get_current_user)):
    """Get NRAM configuration analysis and suggestions"""
    try:
        # Get current state from NRAM
        state = nram.get_explorer_state()

        # Analyze performance and get suggestions
        analysis = await config_suggester.analyze_performance(state)

        return analysis
    except Exception as e:
        logger.error(f"Error analyzing NRAM configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/nram/config/update")
async def update_nram_config(
    config: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Update NRAM configuration based on suggestions"""
    try:
        # Update config suggester
        new_config = config_suggester.update_config(config)

        # Update NRAM with new configuration
        nram.update_configuration(new_config)

        return {"message": "Configuration updated successfully", "config": new_config}
    except Exception as e:
        logger.error(f"Error updating NRAM configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))