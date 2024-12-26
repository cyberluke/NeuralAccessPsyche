from fastapi import APIRouter, HTTPException, Depends, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from core.llm_handler import LLMHandler
from core.nram import NRAM
from utils.validators import validate_request
from utils.auth import get_current_user
from utils.api_logger import api_metrics # Added import
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
llm_handler = LLMHandler()
nram = NRAM()
templates = Jinja2Templates(directory="templates")

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

@router.get("/visualize/api", response_class=HTMLResponse) #Added route
async def visualize_api(request: Request):
    """Serve the API visualization dashboard"""
    return templates.TemplateResponse(
        "api_visualization.html",
        {"request": request}
    )

@router.websocket("/ws/api-viz") #Added route
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