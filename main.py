import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.middleware import TokenAuthMiddleware, RateLimitMiddleware
from api.routes import router

app = FastAPI(
    title="Custom LLM API with NRAM",
    description="OpenAI-compatible API with Neural Random Access Memory simulation",
    version="1.0.0"
)

# CORS middleware
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

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": f"Internal server error: {str(exc)}"}
    )

# Routes
app.include_router(router, prefix="/v1")

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Custom LLM API with NRAM simulation. Use /v1/chat/completions for chat completion."}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )