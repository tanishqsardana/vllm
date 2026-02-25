from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _fmt(value: Any, digits: int = 2) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_results_json(run_dir: Path, payload: Dict[str, Any]) -> Path:
    path = run_dir / "results.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _gpu_names(system_info: Dict[str, Any]) -> str:
    gpus = system_info.get("gpu_topology", {}).get("gpus", [])
    names = [gpu.get("name", "unknown") for gpu in gpus]
    if not names:
        return "unknown"
    return ", ".join(names)


def _build_concurrency_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| conc | mode | rps | tok/s | tok/s/gpu | p95 (ms) | p99 (ms) | err% |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| {conc} | {mode} | {rps} | {tok_s} | {tok_s_gpu} | {p95} | {p99} | {err} |".format(
                conc=row.get("conc", "-"),
                mode=row.get("mode", "-"),
                rps=_fmt(float(row.get("rps", 0.0))),
                tok_s=_fmt(float(row.get("completion_tokens_per_sec", 0.0))),
                tok_s_gpu=_fmt(float(row.get("tok_per_s_per_gpu", 0.0))),
                p95=_fmt(float(row.get("p95_ms", 0.0))),
                p99=_fmt(float(row.get("p99_ms", 0.0))),
                err=_fmt(float(row.get("error_rate", 0.0)) * 100.0),
            )
        )

    return "\n".join(lines)


def build_scorecard(payload: Dict[str, Any]) -> str:
    system_info = payload.get("system_info", {})
    model = payload.get("model", {})
    ladder_rows = payload.get("suites", {}).get("concurrency_ladder", {}).get("results", [])
    long_context_rows = payload.get("suites", {}).get("long_context", {}).get("results", [])
    soak = payload.get("suites", {}).get("soak", {})

    lines = []
    lines.append(f"# Scorecard: {payload.get('run_id', 'unknown')}")
    lines.append("")
    lines.append("## GPU")
    lines.append(f"- name(s): {_gpu_names(system_info)}")
    lines.append(f"- gpu_count: {payload.get('gpu_count', 0)}")
    lines.append(f"- total_vram_mb: {system_info.get('gpu_topology', {}).get('total_vram_mb', 0)}")
    lines.append(f"- tensor_parallel_size: {payload.get('tensor_parallel_size', 1)}")
    lines.append(f"- visible_gpus: {payload.get('visible_gpus', '')}")
    lines.append("")

    lines.append("## Model")
    lines.append(f"- model_id: {model.get('model_id', '')}")
    lines.append(f"- dtype: {model.get('dtype', '')}")
    lines.append(f"- max_model_len: {model.get('max_model_len', '')}")
    lines.append(f"- gpu_memory_utilization: {model.get('gpu_memory_utilization', '')}")
    lines.append("")

    lines.append("## Concurrency Ladder")
    lines.append(_build_concurrency_table(ladder_rows))
    lines.append("")

    lines.append("## Long Context")
    if long_context_rows:
        lines.append("| conc | success_rate | p95 (ms) |")
        lines.append("|---:|---:|---:|")
        for row in long_context_rows:
            lines.append(
                f"| {row.get('conc', '-')} | {_fmt(float(row.get('success_rate', 0.0)) * 100.0)}% | {_fmt(float(row.get('p95_ms', 0.0)))} |"
            )
    else:
        lines.append("No long-context results.")
    lines.append("")

    lines.append("## Soak")
    lines.append(f"- pass/fail: {'PASS' if soak.get('pass', False) else 'FAIL'}")
    lines.append(f"- latency_drift_percent: {_fmt(float(soak.get('latency_drift_percent', 0.0)))}")
    lines.append(f"- memory_drift_percent: {_fmt(float(soak.get('memory_drift_percent_vram', 0.0)))}")
    lines.append("")

    gpu_summary = payload.get("gpu_summary", {})
    lines.append("## GPU Telemetry Summary")
    lines.append(f"- peak_mem_used_mb: {_fmt(float(gpu_summary.get('peak_mem_used_mb', 0.0)))}")
    lines.append(f"- avg_gpu_util: {_fmt(float(gpu_summary.get('avg_gpu_util', 0.0)))}")
    lines.append(f"- peak_power_w: {_fmt(float(gpu_summary.get('peak_power_w', 0.0)))}")

    return "\n".join(lines) + "\n"


def write_scorecard(run_dir: Path, payload: Dict[str, Any]) -> Path:
    scorecard = build_scorecard(payload)
    path = run_dir / "scorecard.md"
    path.write_text(scorecard, encoding="utf-8")
    return path
