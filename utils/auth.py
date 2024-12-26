from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from typing import Dict

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def verify_token(token: str) -> bool:
    """Verify API token"""
    # In production, this should check against a secure database
    return token.startswith("Bearer ") and len(token) > 10

def get_current_user(api_key: str = Security(api_key_header)) -> Dict:
    """Get current user from token"""
    if not api_key or not verify_token(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )
    # In production, this should fetch actual user data
    return {"id": "default_user"}
