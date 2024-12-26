from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from core.llm_handler import LLMHandler
from core.nram import NRAM
from utils.validators import validate_request
from utils.auth import get_current_user

router = APIRouter()
llm_handler = LLMHandler()
nram = NRAM()

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

@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        validate_request(request)
        
        # Process through NRAM
        modified_messages = nram.process_messages(request.messages)
        
        # Get response from LLM
        response = await llm_handler.generate_response(
            messages=modified_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_models(current_user: dict = Depends(get_current_user)):
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
