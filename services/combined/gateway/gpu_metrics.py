from __future__ import annotations

import asyncio
import subprocess
from typing import Any

from .metrics import GatewayMetrics
from .utils import json_log


def _to_float(value: str) -> float:
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"n/a", "[not supported]"}:
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


class GPUMetricsPoller:
    def __init__(self, metrics: GatewayMetrics, poll_interval_seconds: float = 2.0) -> None:
        self.metrics = metrics
        self.poll_interval_seconds = max(0.5, poll_interval_seconds)
        self._task: asyncio.Task | None = None
        self._nvml = None
        self._nvml_initialized = False
        self._nvml_attempted = False
        self._backend_logged: set[str] = set()
        self._backend_unavailable_logged: set[str] = set()

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="gpu-metrics-poller")

    async def stop(self) -> None:
        if self._task is None:
            return

        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

        if self._nvml is not None and self._nvml_initialized:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False

    async def _run(self) -> None:
        while True:
            try:
                samples = self._collect_samples()
                self.metrics.set_gpu_metrics(samples)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                json_log("gpu_metrics_poll_error", message=str(exc))

            await asyncio.sleep(self.poll_interval_seconds)

    def _collect_samples(self) -> list[dict[str, Any]]:
        nvml_samples = self._collect_with_nvml()
        if nvml_samples is not None:
            return nvml_samples
        return self._collect_with_nvidia_smi()

    def _collect_with_nvml(self) -> list[dict[str, Any]] | None:
        if not self._nvml_attempted:
            self._nvml_attempted = True
            try:
                import pynvml  # type: ignore

                self._nvml = pynvml
                self._nvml.nvmlInit()
                self._nvml_initialized = True
                if "pynvml" not in self._backend_logged:
                    json_log("gpu_metrics_backend", backend="pynvml")
                    self._backend_logged.add("pynvml")
            except Exception as exc:
                self._nvml = None
                self._nvml_initialized = False
                self._log_backend_unavailable_once(backend="pynvml", message=str(exc))

        if self._nvml is None or not self._nvml_initialized:
            return None

        samples: list[dict[str, Any]] = []
        count = int(self._nvml.nvmlDeviceGetCount())
        for idx in range(count):
            handle = self._nvml.nvmlDeviceGetHandleByIndex(idx)
            util = self._nvml.nvmlDeviceGetUtilizationRates(handle)
            mem = self._nvml.nvmlDeviceGetMemoryInfo(handle)

            power_watts = 0.0
            temperature_celsius = 0.0
            try:
                power_watts = float(self._nvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
            except Exception:
                pass
            try:
                temperature_celsius = float(
                    self._nvml.nvmlDeviceGetTemperature(handle, self._nvml.NVML_TEMPERATURE_GPU)
                )
            except Exception:
                pass

            samples.append(
                {
                    "gpu_index": idx,
                    "utilization_percent": float(util.gpu),
                    "memory_used_bytes": float(mem.used),
                    "memory_total_bytes": float(mem.total),
                    "power_watts": power_watts,
                    "temperature_celsius": temperature_celsius,
                }
            )
        return samples

    def _collect_with_nvidia_smi(self) -> list[dict[str, Any]]:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            proc = subprocess.run(command, capture_output=True, text=True, check=False, timeout=3)
        except FileNotFoundError:
            self._log_backend_unavailable_once(backend="nvidia-smi", message="binary not found")
            return []
        except Exception as exc:
            self._log_backend_unavailable_once(backend="nvidia-smi", message=str(exc))
            return []

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            if stderr:
                self._log_backend_unavailable_once(backend="nvidia-smi", message=stderr)
            return []

        samples: list[dict[str, Any]] = []
        for idx, line in enumerate(proc.stdout.splitlines()):
            line = line.strip()
            if not line:
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 5:
                continue

            mem_used_mib = _to_float(parts[1])
            mem_total_mib = _to_float(parts[2])
            samples.append(
                {
                    "gpu_index": idx,
                    "utilization_percent": _to_float(parts[0]),
                    "memory_used_bytes": mem_used_mib * 1024.0 * 1024.0,
                    "memory_total_bytes": mem_total_mib * 1024.0 * 1024.0,
                    "power_watts": _to_float(parts[3]),
                    "temperature_celsius": _to_float(parts[4]),
                }
            )

        if samples and "nvidia-smi" not in self._backend_logged:
            json_log("gpu_metrics_backend", backend="nvidia-smi")
            self._backend_logged.add("nvidia-smi")
        return samples

    def _log_backend_unavailable_once(self, backend: str, message: str) -> None:
        if backend in self._backend_unavailable_logged:
            return
        json_log("gpu_metrics_backend_unavailable", backend=backend, message=message)
        self._backend_unavailable_logged.add(backend)
