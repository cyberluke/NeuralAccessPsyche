from fastapi import APIRouter, HTTPException, Depends, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from core.llm_handler import LLMHandler
from core.nram import NRAM
from core.config_suggester import NRAMConfigSuggester
from utils.validators import validate_request
from utils.auth import get_current_user
from utils.api_logger import api_metrics
import logging
import json
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")
llm_handler = LLMHandler()
nram = NRAM()
templates = Jinja2Templates(directory="templates")
config_suggester = NRAMConfigSuggester()

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 100
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

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

@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Handle chat completion requests"""
    try:
        validate_request(request)

        # Process through NRAM
        modified_messages = nram.process_messages(request.messages)

        # Broadcast updated state
        await nram.broadcast_state()

        # Get response from LLM
        response = await llm_handler.generate_response(
            messages=modified_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        return response
    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_models(current_user: dict = Depends(get_current_user)):
    """List available models"""
    return {
        "data": [
            {
                "id": "custom-nram-model",
                "object": "model",
                "owned_by": "organization",
                "permission": []
            }
        ]
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