# ---------------------------------------------------------------------------
# SGLang inference container for NeuralAccessPsyche (NRAM)
#
# This is a THIN WRAPPER around the official SGLang image. It does NOT
# download any model. The local GGUF model is mounted READ-ONLY at /models
# by docker compose (see compose.yaml).
#
# The image tag is intentionally an ARG so it can be pinned to a sha256
# digest at build time, e.g.:
#   docker build --build-arg SGLANG_IMAGE=lmsysorg/sglang@sha256:<digest> .
# ---------------------------------------------------------------------------
ARG SGLANG_IMAGE=lmsysorg/sglang:latest
FROM ${SGLANG_IMAGE}

# NRAM-specific helper scripts live here.
COPY entrypoint.sh /opt/nram/entrypoint.sh
COPY healthcheck.py /opt/nram/healthcheck.py

# Make the entrypoint executable.
RUN chmod +x /opt/nram/entrypoint.sh

# Healthcheck hits the SGLang /health endpoint on its internal port.
HEALTHCHECK --interval=15s --timeout=10s --retries=60 --start-period=300s \
    CMD python3 /opt/nram/healthcheck.py || exit 1

ENTRYPOINT ["/opt/nram/entrypoint.sh"]
