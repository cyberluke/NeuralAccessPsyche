import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from api.middleware import TokenAuthMiddleware, RateLimitMiddleware
from api.routes import router
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Custom LLM API with NRAM",
    description="OpenAI-compatible API with Neural Random Access Memory simulation",
    version="1.0.0"
)

# CORS middleware with WebSocket support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(TokenAuthMiddleware)
app.add_middleware(RateLimitMiddleware)

# Mount templates directory
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP Exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"General Exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": f"Internal server error: {str(exc)}"}
    )

# Routes
app.include_router(router)

# Root endpoint
@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Custom LLM API with NRAM simulation. Use /v1/chat/completions for chat completion."}

if __name__ == "__main__":
    logger.info("Starting application server")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )