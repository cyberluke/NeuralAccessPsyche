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

# Supported model aliases
MODEL_ALIASES = {
    "nram-gpt-oss-20b": "nram-gpt-oss-20b",
    "gpt-oss-20b-baseline": "gpt-oss-20b-baseline",
    "deepseek-r1-qwen-7b-baseline": "deepseek-r1-qwen-7b-baseline",
    "nram-deepseek-r1-qwen-7b": "nram-deepseek-r1-qwen-7b",
}


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
    """Route a request through the SGLang engine (streaming or non-streaming)."""
    engine = get_sglang_engine()
    if engine is None:
        return _openai_error(
            503,
            "SGLang engine is enabled but not available.",
            "inference_error",
            "engine_unavailable",
        )

    engine_request = _to_engine_request(request, messages)

    if request.stream:
        generator = engine.stream(engine_request)
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    response = await engine.complete(engine_request)
    return response.model_dump()


async def _stream_completion(request: ChatCompletionRequest, messages: list) -> StreamingResponse:
    """Placeholder SSE streaming — full implementation in Phase 12."""
    # For now, run non-streaming and emit the result as a single SSE chunk
    response = await llm_handler.generate_response(
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        model=request.model,
    )

    chunk_id = response["id"]
    created = response["created"]
    model = response["model"]
    content = response["choices"][0]["message"]["content"]

    async def event_generator():
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        final_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

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