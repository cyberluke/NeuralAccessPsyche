from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from utils.auth import verify_token
from utils.rate_limiter import RateLimiter

class TokenAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            if request.url.path.startswith("/v1"):
                token = request.headers.get("Authorization")
                if not token or not verify_token(token):
                    return JSONResponse(
                        status_code=401,
                        content={"error": "Invalid authentication token"}
                    )
            return await call_next(request)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Middleware error: {str(e)}"}
            )

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.rate_limiter = RateLimiter()

    async def dispatch(self, request: Request, call_next):
        try:
            if request.url.path.startswith("/v1"):
                token = request.headers.get("Authorization")
                if token and not self.rate_limiter.check_rate_limit(token):
                    return JSONResponse(
                        status_code=429,
                        content={"error": "Rate limit exceeded"}
                    )
            return await call_next(request)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Rate limit error: {str(e)}"}
            )