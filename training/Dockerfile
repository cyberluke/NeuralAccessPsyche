ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:3986465b3dd3b4d602c07061f2cff417e0bfb24810129408d4eb12e111015a6c
FROM ${BASE_IMAGE}
ARG SOURCE_COMMIT=UNAVAILABLE
LABEL org.opencontainers.image.source-revision=${SOURCE_COMMIT} org.opencontainers.image.title="NRAM Qwen3-14B LoRA training"
ENV PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3-pip && rm -rf /var/lib/apt/lists/* && python3.12 -m venv /opt/venv
COPY requirements-training.lock /opt/nram/requirements-training.lock
RUN pip install --no-cache-dir --index-url https://pypi.org/simple -r /opt/nram/requirements-training.lock
COPY nram_training /opt/nram/nram_training
WORKDIR /opt/nram
ENTRYPOINT ["python3", "-m", "nram_training"]
