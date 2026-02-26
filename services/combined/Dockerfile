FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_SHA=dev
ARG BUILD_TIME=dev
ARG VLLM_VERSION=0.6.4.post1
ARG FASTAPI_VERSION=0.115.6
ARG UVICORN_VERSION=0.32.1
ARG HTTPX_VERSION=0.28.1
ARG TRANSFORMERS_VERSION=4.46.3
ARG HUGGINGFACE_HUB_VERSION=0.26.5
ARG TQDM_VERSION=4.66.5
ARG PROMETHEUS_CLIENT_VERSION=0.21.1
ARG PYNVML_VERSION=11.5.3

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
    curl \
    ca-certificates \
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
    "httpx==${HTTPX_VERSION}" \
    "transformers==${TRANSFORMERS_VERSION}" \
    "huggingface_hub==${HUGGINGFACE_HUB_VERSION}" \
    "tqdm==${TQDM_VERSION}" \
    "prometheus_client==${PROMETHEUS_CLIENT_VERSION}" \
    "pynvml==${PYNVML_VERSION}"

RUN useradd --create-home --uid 10001 --shell /bin/bash appuser \
    && mkdir -p /app /cache /data \
    && chown -R appuser:appuser /app /cache /data

WORKDIR /app
COPY services/combined/entrypoint.sh /app/entrypoint.sh
COPY services/combined/gateway /app/gateway
RUN chmod +x /app/entrypoint.sh

USER appuser
EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
