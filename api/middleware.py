from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from utils.auth import verify_token
from utils.rate_limiter import RateLimiter
import logging

logger = logging.getLogger(__name__)

class TokenAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            if request.url.path.startswith("/v1"):
                token = request.headers.get("Authorization")
                logger.info(f"Authenticating request to {request.url.path}")
                if not token or not verify_token(token):
                    logger.warning(f"Authentication failed for request to {request.url.path}")
                    return JSONResponse(
                        status_code=401,
                        content={"error": "Invalid authentication token"}
                    )
                logger.info("Authentication successful")
            return await call_next(request)
        except Exception as e:
            logger.error(f"Middleware error: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": f"Middleware error: {str(e)}"}
            )

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.rate_limiter = RateLimiter()
        logger.info("Rate limit middleware initialized")

    async def dispatch(self, request: Request, call_next):
        try:
            if request.url.path.startswith("/v1"):
                token = request.headers.get("Authorization")
                logger.debug(f"Checking rate limit for token: {token[:8]}...")
                if token and not self.rate_limiter.check_rate_limit(token):
                    logger.warning(f"Rate limit exceeded for token: {token[:8]}...")
                    return JSONResponse(
                        status_code=429,
                        content={"error": "Rate limit exceeded"}
                    )
            return await call_next(request)
        except Exception as e:
            logger.error(f"Rate limit error: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": f"Rate limit error: {str(e)}"}
            )