from __future__ import annotations

import asyncio
import csv
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_float(value: str) -> float:
    cleaned = value.strip().replace("W", "")
    if cleaned in {"", "N/A", "[Not Supported]"}:
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


class GPUProbe:
    def __init__(self, output_csv: Path, interval_s: float = 1.0) -> None:
        self.output_csv = output_csv
        self.interval_s = interval_s
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._rows: List[Dict[str, Any]] = []

        self._writer = None
        self._file_handle = None

        self._backend = "none"
        self._nvml = None
        self._nvml_handles = []

        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            self._nvml = pynvml
            self._nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            self._backend = "pynvml"
            return
        except Exception:
            self._nvml = None
            self._nvml_handles = []

        if shutil.which("nvidia-smi"):
            self._backend = "nvidia-smi"
        else:
            self._backend = "none"

    async def start(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self._file_handle = self.output_csv.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file_handle,
            fieldnames=[
                "timestamp",
                "epoch_s",
                "gpu_index",
                "gpu_name",
                "gpu_util_percent",
                "mem_used_mb",
                "mem_total_mb",
                "power_w",
                "temperature_c",
            ],
        )
        self._writer.writeheader()
        self._file_handle.flush()

        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> Dict[str, Any]:
        self._stop_event.set()
        if self._task is not None:
            await self._task

        if self._file_handle is not None:
            self._file_handle.flush()
            self._file_handle.close()

        if self._backend == "pynvml" and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

        return self._summarize()

    def get_samples(self) -> List[Dict[str, Any]]:
        return list(self._rows)

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            timestamp = _utc_now_iso()
            epoch_s = time.time()

            rows = self._sample_once(timestamp, epoch_s)
            for row in rows:
                self._rows.append(row)
                if self._writer is not None:
                    self._writer.writerow(row)

            if self._file_handle is not None:
                self._file_handle.flush()

            await asyncio.sleep(self.interval_s)

    def _sample_once(self, timestamp: str, epoch_s: float) -> List[Dict[str, Any]]:
        if self._backend == "pynvml":
            return self._sample_pynvml(timestamp, epoch_s)
        if self._backend == "nvidia-smi":
            return self._sample_nvidia_smi(timestamp, epoch_s)
        return []

    def _sample_pynvml(self, timestamp: str, epoch_s: float) -> List[Dict[str, Any]]:
        assert self._nvml is not None
        rows: List[Dict[str, Any]] = []

        for gpu_index, handle in enumerate(self._nvml_handles):
            try:
                name = self._nvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="ignore")

                util = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                mem = self._nvml.nvmlDeviceGetMemoryInfo(handle)

                try:
                    power_w = self._nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except Exception:
                    power_w = 0.0

                try:
                    temp_c = float(
                        self._nvml.nvmlDeviceGetTemperature(
                            handle,
                            self._nvml.NVML_TEMPERATURE_GPU,
                        )
                    )
                except Exception:
                    temp_c = 0.0

                rows.append(
                    {
                        "timestamp": timestamp,
                        "epoch_s": epoch_s,
                        "gpu_index": gpu_index,
                        "gpu_name": str(name),
                        "gpu_util_percent": float(util.gpu),
                        "mem_used_mb": float(mem.used / (1024 * 1024)),
                        "mem_total_mb": float(mem.total / (1024 * 1024)),
                        "power_w": power_w,
                        "temperature_c": temp_c,
                    }
                )
            except Exception:
                continue

        return rows

    def _sample_nvidia_smi(self, timestamp: str, epoch_s: float) -> List[Dict[str, Any]]:
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]

        try:
            output = subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True)
        except Exception:
            return []

        rows: List[Dict[str, Any]] = []
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            rows.append(
                {
                    "timestamp": timestamp,
                    "epoch_s": epoch_s,
                    "gpu_index": int(_to_float(parts[0])),
                    "gpu_name": parts[1],
                    "gpu_util_percent": _to_float(parts[2]),
                    "mem_used_mb": _to_float(parts[3]),
                    "mem_total_mb": _to_float(parts[4]),
                    "power_w": _to_float(parts[5]),
                    "temperature_c": _to_float(parts[6]),
                }
            )

        return rows

    def _summarize(self) -> Dict[str, Any]:
        if not self._rows:
            return {
                "backend": self._backend,
                "sample_count": 0,
                "peak_mem_used_mb": 0.0,
                "avg_gpu_util": 0.0,
                "peak_power_w": 0.0,
            }

        total_mem_by_ts = defaultdict(float)
        total_power_by_ts = defaultdict(float)
        gpu_utils = []

        for row in self._rows:
            ts = float(row["epoch_s"])
            total_mem_by_ts[ts] += float(row["mem_used_mb"])
            total_power_by_ts[ts] += float(row["power_w"])
            gpu_utils.append(float(row["gpu_util_percent"]))

        peak_mem_used_mb = max(total_mem_by_ts.values()) if total_mem_by_ts else 0.0
        peak_power_w = max(total_power_by_ts.values()) if total_power_by_ts else 0.0
        avg_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0

        return {
            "backend": self._backend,
            "sample_count": len(self._rows),
            "peak_mem_used_mb": peak_mem_used_mb,
            "avg_gpu_util": avg_gpu_util,
            "peak_power_w": peak_power_w,
        }

    def total_memory_by_timestamp(self, start_epoch_s: float, end_epoch_s: float) -> List[Dict[str, float]]:
        bucket = defaultdict(float)
        for row in self._rows:
            ts = float(row.get("epoch_s", 0.0))
            if start_epoch_s <= ts <= end_epoch_s:
                bucket[ts] += float(row.get("mem_used_mb", 0.0))

        return [
            {"epoch_s": ts, "total_mem_used_mb": total}
            for ts, total in sorted(bucket.items(), key=lambda item: item[0])
        ]
