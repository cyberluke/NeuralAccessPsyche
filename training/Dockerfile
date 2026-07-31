ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:UNAVAILABLE
FROM ${BASE_IMAGE}
ARG SOURCE_COMMIT=UNAVAILABLE
LABEL org.opencontainers.image.source-revision=${SOURCE_COMMIT} org.opencontainers.image.title="NRAM Qwen3-14B LoRA training"
ENV PIP_NO_INDEX=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 python3-pip && rm -rf /var/lib/apt/lists/*
COPY requirements-training.lock /opt/nram/requirements-training.lock
RUN python3 -m pip install --no-cache-dir -r /opt/nram/requirements-training.lock
COPY nram_training /opt/nram/nram_training
WORKDIR /opt/nram
ENTRYPOINT ["python3", "-m", "nram_training"]
