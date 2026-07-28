from fastapi import Request
from fastapi.responses import JSONResponse
from utils.auth import verify_token
from utils.rate_limiter import RateLimiter
import logging

logger = logging.getLogger(__name__)

class TokenAuthMiddleware:
    """Pure ASGI auth middleware that preserves downstream disconnect events."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive=receive)
        try:
            # Skip authentication for explorer and visualization routes
            if any(request.url.path.startswith(prefix) for prefix in [
                "/v1/nram/explorer",
                "/v1/visualize",
                "/v1/ws/nram-explorer",
                "/v1/ws/api-viz"
            ]):
                await self.app(scope, receive, send)
                return

            if request.url.path.startswith("/v1"):
                token = request.headers.get("Authorization")
                logger.info(f"Authenticating request to {request.url.path}")
                if not token or not verify_token(token):
                    logger.warning(f"Authentication failed for request to {request.url.path}")
                    response = JSONResponse(
                        status_code=401,
                        content={"error": "Invalid authentication token"}
                    )
                    await response(scope, receive, send)
                    return
                logger.info("Authentication successful")
            await self.app(scope, receive, send)
        except Exception as e:
            logger.error(f"Middleware error: {str(e)}", exc_info=True)
            response = JSONResponse(
                status_code=500,
                content={"error": f"Middleware error: {str(e)}"}
            )
            await response(scope, receive, send)

class RateLimitMiddleware:
    """Pure ASGI rate limiter; BaseHTTPMiddleware obscures socket disconnects."""

    def __init__(self, app):
        self.app = app
        self.rate_limiter = RateLimiter()
        logger.info("Rate limit middleware initialized")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive=receive)
        try:
            # Skip rate limiting for explorer and visualization routes
            if any(request.url.path.startswith(prefix) for prefix in [
                "/v1/nram/explorer",
                "/v1/visualize",
                "/v1/ws/nram-explorer",
                "/v1/ws/api-viz"
            ]):
                await self.app(scope, receive, send)
                return

            if request.url.path.startswith("/v1"):
                token = request.headers.get("Authorization")

                # Handle missing token
                if not token:
                    logger.debug("No token provided for rate limiting")
                    await self.app(scope, receive, send)
                    return

                # Safe token logging
                token_prefix = token[:8] if token else "none"
                logger.debug(f"Checking rate limit for token: {token_prefix}...")

                if not self.rate_limiter.check_rate_limit(token):
                    logger.warning(f"Rate limit exceeded for token: {token_prefix}...")
                    response = JSONResponse(
                        status_code=429,
                        content={"error": "Rate limit exceeded"}
                    )
                    await response(scope, receive, send)
                    return
            await self.app(scope, receive, send)
        except Exception as e:
            logger.error(f"Rate limit error: {str(e)}", exc_info=True)
            response = JSONResponse(
                status_code=500,
                content={"error": f"Rate limit error: {str(e)}"}
            )
            await response(scope, receive, send)
