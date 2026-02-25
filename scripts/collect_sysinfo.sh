#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_id>"
  exit 1
fi

RUN_ID="$1"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="${REPO_ROOT}/results/${RUN_ID}"
SYSINFO_PATH="${RUN_DIR}/sysinfo.json"
CONTAINER_NAME="${CONTAINER_NAME:-engine}"

mkdir -p "${RUN_DIR}"

python3 - "$SYSINFO_PATH" "$CONTAINER_NAME" <<'PY'
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sysinfo_path = Path(sys.argv[1])
container_name_hint = sys.argv[2]


def run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def parse_ram_mb() -> int:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return 0
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    kb = int(parts[1])
                    return kb // 1024
                except ValueError:
                    return 0
    return 0


def parse_cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return platform.processor() or "unknown"
    for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.lower().startswith("model name"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return platform.processor() or "unknown"


def parse_os_name() -> str:
    os_release = Path("/etc/os-release")
    if not os_release.exists():
        return platform.system()
    values = {}
    for line in os_release.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k] = v.strip().strip('"')
    return values.get("PRETTY_NAME", values.get("NAME", platform.system()))


def parse_gpu_topology():
    query = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )

    gpus = []
    driver_version = ""
    total_vram_mb = 0

    for line in query.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name = parts[0]
        mem_match = re.search(r"(\d+)", parts[1])
        memory_mb = int(mem_match.group(1)) if mem_match else 0
        driver_version = parts[2] or driver_version

        gpus.append(
            {
                "name": name,
                "memory_mb": memory_mb,
            }
        )
        total_vram_mb += memory_mb

    full_smi = run(["nvidia-smi"])
    cuda_version = ""
    match = re.search(r"CUDA Version:\s*([0-9.]+)", full_smi)
    if match:
        cuda_version = match.group(1)

    return {
        "gpu_count": len(gpus),
        "gpus": gpus,
        "total_vram_mb": total_vram_mb,
        "driver_version": driver_version,
        "cuda_version": cuda_version,
    }


def parse_container_env(env_list):
    env_map = {}
    for item in env_list or []:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        env_map[k] = v
    return env_map


def parse_docker_info(name_hint):
    container_id = run(
        ["docker", "ps", "--filter", f"name=^{name_hint}$", "--format", "{{.ID}}"]
    )

    if not container_id:
        container_id = run(
            ["docker", "ps", "--filter", f"name={name_hint}", "--format", "{{.ID}}"]
        )

    if not container_id:
        return {
            "container_name": name_hint,
            "image_tag": "",
            "image_id": "",
            "tensor_parallel_size": os.getenv("TENSOR_PARALLEL")
            or os.getenv("TENSOR_PARALLEL_SIZE")
            or "",
        }

    inspect_raw = run(["docker", "inspect", container_id.splitlines()[0]])
    try:
        inspect = json.loads(inspect_raw)[0]
    except Exception:
        inspect = {}

    env_map = parse_container_env(((inspect.get("Config") or {}).get("Env") or []))

    return {
        "container_name": (inspect.get("Name", "").lstrip("/") or name_hint),
        "image_tag": ((inspect.get("Config") or {}).get("Image") or ""),
        "image_id": inspect.get("Image", ""),
        "tensor_parallel_size": env_map.get("TENSOR_PARALLEL")
        or env_map.get("TENSOR_PARALLEL_SIZE")
        or os.getenv("TENSOR_PARALLEL")
        or os.getenv("TENSOR_PARALLEL_SIZE")
        or "",
    }


gpu_topology = parse_gpu_topology()
docker_info = parse_docker_info(container_name_hint)

payload = {
    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "hostname": socket.gethostname(),
    "os": parse_os_name(),
    "kernel": platform.release(),
    "cpu": {
        "model": parse_cpu_model(),
        "cores": os.cpu_count() or 0,
    },
    "ram_total_mb": parse_ram_mb(),
    "gpu_topology": gpu_topology,
    "docker": {
        "container_name": docker_info["container_name"],
        "image_tag": docker_info["image_tag"],
        "image_id": docker_info["image_id"],
    },
    "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
    "tensor_parallel_size": docker_info["tensor_parallel_size"],
}

sysinfo_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(sysinfo_path)
PY
