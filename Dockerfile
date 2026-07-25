# ---------------------------------------------------------------------------
# NeuralAccessPsyche (NRAM) API service.
#
# A FastAPI app that wraps the SGLang inference backend and exposes an
# OpenAI-compatible API with NRAM steering. It talks to the `sglang`
# service over the docker compose network — it does NOT run inference
# itself and does NOT contain the model weights.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# Avoid Python writing .pyc files and force unbuffered stdout/stderr so
# logs stream to `docker compose logs` immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --- Install runtime dependencies ------------------------------------------
# The project declares its dependencies in pyproject.toml. We copy it first
# and install the package so the dependency layer is cached and re-used
# across builds. If the full pyproject dependency set (guidance, streamlit,
# transformers, plotly, ...) is undesirable for the API image, swap the
# `pip install .` line for the explicit runtime list in the comment below.
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install . || \
    pip install \
        "fastapi>=0.115.6" \
        "uvicorn[standard]>=0.34.0" \
        "httpx" \
        "pydantic>=2.10.4" \
        "numpy>=2.2.1" \
        "jinja2>=3.1.5" \
        "python-multipart>=0.0.20" \
        "dill" \
        "openai>=1.58.1" \
        "websockets>=14.1"

# --- Copy application source ------------------------------------------------
# Deliberately NOT copied: tests/, logs/, .git/, attached_assets/, and the
# GGUF model (the model lives on the host and is mounted into sglang, not
# baked into any image).
COPY main.py ./
COPY api/ ./api/
COPY core/ ./core/
COPY utils/ ./utils/
COPY templates/ ./templates/
COPY static/ ./static/

EXPOSE 8000

# Run the FastAPI app with uvicorn on all interfaces inside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
