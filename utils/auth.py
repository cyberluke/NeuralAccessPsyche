from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from typing import Dict
import hmac
import os

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def verify_token(token: str) -> bool:
    """Verify the configured local bearer token using constant-time comparison."""
    expected = os.environ.get("NRAM_API_KEY", "dev-nram-key")
    if not expected or not token or not token.startswith("Bearer "):
        return False
    supplied = token[len("Bearer "):]
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))

def get_current_user(api_key: str = Security(api_key_header)) -> Dict:
    """Get current user from token"""
    if not api_key or not verify_token(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )
    # In production, this should fetch actual user data
    return {"id": "default_user"}
