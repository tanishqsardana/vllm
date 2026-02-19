# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Minimal deps (include build-essential + python headers so Triton/torch.compile can JIT)
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-dev git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast resolver) then vLLM + deps via uv. If uv ever fails on a platform,
# swap `uv pip install ...` with `pip3 install ...` as a fallback.
RUN pip3 install --no-cache-dir --upgrade pip uv \
    && uv pip install --system --no-cache-dir --extra-index-url https://wheels.vllm.ai/nightly vllm "huggingface_hub[cli]" \
    && apt-get update && apt-get install -y --no-install-recommends git-lfs python3-dev && git lfs install

# Cache location inside container; can be mounted as a volume for reuse
ENV HF_HOME=/data/hf-cache
RUN mkdir -p $HF_HOME

# Disable torch.compile/inductor and FlashInfer to avoid NVCC/CUDA toolkit needs on T4; force eager path.
ENV TORCHINDUCTOR_DISABLE=1
ENV TORCH_COMPILE_DISABLE=1
ENV FLASHINFER_DISABLE=1

ARG MODEL_ID="Qwen/Qwen3-0.6B"
ARG DOWNLOAD_WEIGHTS=0
ENV MODEL_ID=${MODEL_ID}

# Optional: bake model weights at build time (set DOWNLOAD_WEIGHTS=1). This works only if the
# model is public; for gated/private models provide a token via BuildKit secret.
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ "${DOWNLOAD_WEIGHTS}" = "1" ]; then \
      if [ -f /run/secrets/HF_TOKEN ]; then export HUGGINGFACE_TOKEN=$(cat /run/secrets/HF_TOKEN); fi; \
      huggingface-cli download "$MODEL_ID" --local-dir "/models/$MODEL_ID" --local-dir-use-symlinks False; \
    fi

# Expose OpenAI-compatible HTTP server port
EXPOSE 8000

# Start vLLM OpenAI server with the smallest Qwen3 instruct model
# If weights were baked in, we point to /models; otherwise vLLM will pull to /data/hf-cache at first run.
# Decide at runtime: use baked weights if present, otherwise pull from HF.
ENTRYPOINT ["/bin/bash", "-lc", "if [ -d /models/${MODEL_ID} ]; then MODEL_REF=/models/${MODEL_ID}; else MODEL_REF=${MODEL_ID}; fi; export CC=${CC:-gcc}; export CXX=${CXX:-g++}; exec vllm serve ${MODEL_REF} --served-model-name ${MODEL_ID} --host 0.0.0.0 --port 8000 --api-key changeme --download-dir /data/hf-cache --max-model-len 8192 --enforce-eager --attention-backend TRITON_ATTN"]
