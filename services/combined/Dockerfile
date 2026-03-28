FROM nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1

ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_SHA=dev
ARG BUILD_TIME=dev
ARG FASTAPI_VERSION=0.115.6
ARG UVICORN_VERSION=0.32.1
ARG HTTPX_VERSION=0.28.1
ARG PROMETHEUS_CLIENT_VERSION=0.21.1
ARG PYNVML_VERSION=11.5.3
ARG PYYAML_VERSION=6.0.3
ARG PYJWT_VERSION=2.10.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HF_HOME=/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    VLLM_CACHE_ROOT=/cache/vllm \
    BUILD_SHA=${BUILD_SHA} \
    BUILD_TIME=${BUILD_TIME} \
    DYN_FILE_KV=/data/dynamo_kv \
    DYNAMO_FRONTEND_PORT=8001 \
    DYNAMO_WORKER_METRICS_PORT=8081

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    "fastapi==${FASTAPI_VERSION}" \
    "uvicorn==${UVICORN_VERSION}" \
    "httpx==${HTTPX_VERSION}" \
    "prometheus_client==${PROMETHEUS_CLIENT_VERSION}" \
    "pynvml==${PYNVML_VERSION}" \
    "pyyaml==${PYYAML_VERSION}" \
    "PyJWT==${PYJWT_VERSION}"

RUN useradd --create-home --uid 10001 --shell /bin/bash appuser \
    && mkdir -p /app /cache /data \
    && chown -R appuser:appuser /app /cache /data

WORKDIR /app
COPY services/combined/entrypoint.sh /app/entrypoint.sh
COPY services/combined/gateway /app/gateway
COPY config /app/config
COPY profiles /app/profiles
RUN chmod +x /app/entrypoint.sh

USER appuser
EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
