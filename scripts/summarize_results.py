#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "results_summary.csv"


def load_results_json() -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*/results.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return rows


def extract_conc20(ladder: List[Dict[str, Any]]) -> Dict[str, Any]:
    for row in ladder:
        if int(row.get("conc", 0)) == 20 and str(row.get("mode", "")) == "constant":
            return row
    return {}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records = load_results_json()

    summary_rows = []
    for result in records:
        system_info = result.get("system_info", {})
        gpus = system_info.get("gpu_topology", {}).get("gpus", [])
        gpu_name = gpus[0].get("name", "") if gpus else ""

        ladder_rows = result.get("suites", {}).get("concurrency_ladder", {}).get("results", [])
        conc20 = extract_conc20(ladder_rows)

        summary_rows.append(
            {
                "gpu_name": gpu_name,
                "gpu_count": result.get("gpu_count", 0),
                "model_preset": result.get("model_preset", ""),
                "image_tag": result.get("image_tag", ""),
                "conc_20_p95": conc20.get("p95_ms", 0.0),
                "conc_20_tok_per_s": conc20.get("completion_tokens_per_sec", 0.0),
                "tok_per_s_per_gpu": conc20.get("tok_per_s_per_gpu", 0.0),
                "peak_vram": result.get("gpu_summary", {}).get("peak_mem_used_mb", 0.0),
                "soak_pass": result.get("suites", {}).get("soak", {}).get("pass", False),
            }
        )

    fieldnames = [
        "gpu_name",
        "gpu_count",
        "model_preset",
        "image_tag",
        "conc_20_p95",
        "conc_20_tok_per_s",
        "tok_per_s_per_gpu",
        "peak_vram",
        "soak_pass",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
