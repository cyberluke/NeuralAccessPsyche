from fastapi import HTTPException
from typing import Any

def validate_request(request: Any) -> bool:
    """Validate incoming request"""
    if not hasattr(request, "messages") or not request.messages:
        raise HTTPException(
            status_code=400,
            detail="Messages are required"
        )
        
    if not all(
        isinstance(msg, dict) and "role" in msg and "content" in msg
        for msg in request.messages
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid message format"
        )
        
    if hasattr(request, "temperature") and (
        request.temperature < 0 or request.temperature > 2
    ):
        raise HTTPException(
            status_code=400,
            detail="Temperature must be between 0 and 2"
        )
        
    return True
