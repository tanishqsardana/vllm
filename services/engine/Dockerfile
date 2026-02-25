FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_SHA=dev
ARG BUILD_TIME=dev
ARG VLLM_VERSION=0.6.4.post1
ARG FASTAPI_VERSION=0.115.6
ARG UVICORN_VERSION=0.32.1
ARG HTTPX_VERSION=0.28.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HF_HOME=/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    VLLM_CACHE_ROOT=/cache/vllm \
    BUILD_SHA=${BUILD_SHA} \
    BUILD_TIME=${BUILD_TIME}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    curl \
    ca-certificates \
    openssh-server \
    tini \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && python -m pip install --upgrade \
    pip==24.3.1 \
    setuptools==75.6.0 \
    wheel==0.45.1

RUN python -m pip install \
    "vllm==${VLLM_VERSION}" \
    "fastapi==${FASTAPI_VERSION}" \
    "uvicorn==${UVICORN_VERSION}" \
    "httpx==${HTTPX_VERSION}"

RUN useradd --create-home --uid 10001 --shell /bin/bash appuser \
    && mkdir -p /app /cache \
    && chown -R appuser:appuser /app /cache

WORKDIR /app
COPY services/engine/entrypoint.sh /app/entrypoint.sh
COPY services/engine/bootstrap.sh /app/bootstrap.sh
COPY services/engine/sitecustomize.py /app/sitecustomize.py
RUN chmod +x /app/entrypoint.sh /app/bootstrap.sh \
    && mkdir -p /var/run/sshd

USER root
EXPOSE 22 8000

ENTRYPOINT ["/usr/bin/tini", "--", "/app/bootstrap.sh"]
