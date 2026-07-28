import logging
import json
import time
from typing import Dict, Any, Optional, Set
from functools import wraps
import traceback
import os
from fastapi import WebSocket

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
logger = logging.getLogger("openai.api")
logger.setLevel(logging.DEBUG)

# Add file handler for API calls
api_handler = logging.FileHandler("logs/openai_api.log")
api_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger.addHandler(api_handler)

class APIMetrics:
    def __init__(self):
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_tokens = 0
        self.total_latency = 0.0
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        """Handle new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        # Send current metrics upon connection
        await self.broadcast_metrics()

    def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        self.active_connections.remove(websocket)

    async def record_call(self, success: bool, tokens: int, latency: float, call_details: Dict):
        """Record API call metrics and broadcast updates"""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        self.total_tokens += tokens
        self.total_latency += latency

        # Broadcast updates
        await self.broadcast_call(call_details)
        await self.broadcast_metrics()

    async def broadcast_metrics(self):
        """Broadcast current metrics to all connected clients"""
        metrics = {
            "type": "metrics",
            "metrics": {
                "total_calls": self.total_calls,
                "successful_calls": self.successful_calls,
                "failed_calls": self.failed_calls,
                "success_rate": f"{self.success_rate:.2%}",
                "total_tokens": self.total_tokens,
                "average_latency": f"{self.average_latency:.2f}s"
            }
        }

        for connection in self.active_connections.copy():
            try:
                await connection.send_json(metrics)
            except Exception as e:
                logger.error(f"Error broadcasting metrics: {str(e)}")
                self.active_connections.remove(connection)

    async def broadcast_call(self, call_details: Dict):
        """Broadcast API call details to all connected clients"""
        message = {
            "type": "call",
            "call": call_details
        }

        for connection in self.active_connections.copy():
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting call: {str(e)}")
                self.active_connections.remove(connection)

    @property
    def success_rate(self) -> float:
        return self.successful_calls / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def average_latency(self) -> float:
        return self.total_latency / self.total_calls if self.total_calls > 0 else 0.0

# Global metrics tracker
api_metrics = APIMetrics()

def log_api_call(func):
    """Decorator to log OpenAI API calls with detailed metrics"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        call_id = int(time.time() * 1000)

        # Log request
        logger.info(f"API Call {call_id} - Starting request to OpenAI API")
        logger.debug(f"API Call {call_id} - Request parameters: {json.dumps(kwargs, default=str)}")

        try:
            # Execute API call
            response = await func(*args, **kwargs)

            # Calculate metrics
            latency = time.time() - start_time
            tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0

            # Log success
            logger.info(
                f"API Call {call_id} - Success - Latency: {latency:.2f}s, "
                f"Tokens: {tokens}, Model: {kwargs.get('model', 'unknown')}"
            )

            # Record and broadcast metrics
            await api_metrics.record_call(
                True,
                tokens,
                latency,
                {
                    "id": call_id,
                    "status": "success",
                    "latency": f"{latency:.2f}s",
                    "tokens": tokens,
                    "model": kwargs.get("model", "unknown")
                }
            )

            return response

        except Exception as e:
            # Calculate error metrics
            latency = time.time() - start_time

            # Get detailed error information
            error_type = type(e).__name__
            error_trace = traceback.format_exc()

            # Log error with context
            logger.error(
                f"API Call {call_id} - Failed - {error_type}: {str(e)}\n"
                f"Context: {json.dumps(kwargs, default=str)}\n"
                f"Traceback: {error_trace}"
            )

            # Record and broadcast failed metrics
            await api_metrics.record_call(
                False,
                0,
                latency,
                {
                    "id": call_id,
                    "status": "error",
                    "error": str(e),
                    "latency": f"{latency:.2f}s"
                }
            )

            # Re-raise the exception
            raise

    return wrapper

def get_api_metrics() -> Dict[str, Any]:
    """Get current API metrics"""
    return {
        "total_calls": api_metrics.total_calls,
        "successful_calls": api_metrics.successful_calls,
        "failed_calls": api_metrics.failed_calls,
        "success_rate": f"{api_metrics.success_rate:.2%}",
        "total_tokens": api_metrics.total_tokens,
        "average_latency": f"{api_metrics.average_latency:.2f}s"
    }

def log_api_metrics():
    """Log current API metrics"""
    metrics = get_api_metrics()
    logger.info(f"API Metrics: {json.dumps(metrics, indent=2)}")