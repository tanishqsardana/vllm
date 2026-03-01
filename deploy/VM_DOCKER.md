# VM Docker Run

Example single-container run on a VM:

```bash
docker run --rm -it \
  --gpus all \
  -p 8000:8000 \
  -v /opt/vllm/cache:/cache \
  -v /opt/vllm/data:/data \
  -e MODEL_ID=Qwen/Qwen2.5-7B-Instruct \
  -e ADMIN_AUTH_MODE=static_token \
  -e ADMIN_TOKEN=replace-with-strong-token \
  -e DB_PATH=/data/controlplane.db \
  -e GPU_HOURLY_RATE=1.8 \
  -e BUILD_SHA=$(git rev-parse --short HEAD) \
  -e BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
  <your-image>:phase5
```

Then open:

- `http://<vm-ip>:8000/ui`

## Optional OIDC Mode

Switch auth mode and add OIDC env:

```bash
-e ADMIN_AUTH_MODE=oidc \
-e JWKS_URL=https://issuer.example.com/.well-known/jwks.json \
-e OIDC_ISSUER=https://issuer.example.com/ \
-e OIDC_AUDIENCE=control-plane-admin \
-e ADMIN_GROUP=platform-admins
```
