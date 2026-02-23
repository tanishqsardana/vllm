#!/usr/bin/env python3
"""Throughput benchmark for OpenAI-compatible chat endpoints, with CSV output."""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
import urllib.error
import urllib.request
from queue import Empty, Queue


def _post_json(url: str, api_key: str, payload: dict, timeout_s: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        method="POST",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, api_key: str, timeout_s: int) -> dict:
    req = urllib.request.Request(
        url=url,
        method="GET",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _resolve_models(base_url: str, api_key: str, timeout_s: int) -> list[str]:
    data = _get_json(f"{base_url}/v1/models", api_key, timeout_s)
    models = data.get("data", [])
    if not models:
        raise RuntimeError("No models returned by /v1/models")
    model_ids = []
    for model in models:
        model_id = model.get("id")
        if model_id:
            model_ids.append(model_id)
    if not model_ids:
        raise RuntimeError("No model ids returned by /v1/models")
    return model_ids


def _parse_models(input_models: list[str]) -> list[str]:
    models: list[str] = []
    for item in input_models:
        for model in item.split(","):
            cleaned = model.strip()
            if cleaned:
                models.append(cleaned)
    return models


def _benchmark_model(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    requests: int,
    concurrency: int,
    max_tokens: int,
    timeout_seconds: int,
) -> dict:
    work_queue: Queue[int] = Queue()
    for i in range(requests):
        work_queue.put(i)

    stats = {
        "success": 0,
        "fail": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                work_queue.get_nowait()
            except Empty:
                return

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": False,
            }
            try:
                response = _post_json(
                    f"{base_url}/v1/chat/completions",
                    api_key,
                    payload,
                    timeout_seconds,
                )
                usage = response.get("usage", {})
                with lock:
                    stats["success"] += 1
                    stats["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
                    stats["completion_tokens"] += int(usage.get("completion_tokens", 0))
                    stats["total_tokens"] += int(usage.get("total_tokens", 0))
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
                with lock:
                    stats["fail"] += 1
            finally:
                work_queue.task_done()

    start = time.perf_counter()
    threads = []
    for _ in range(min(concurrency, requests)):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        elapsed = 1e-9

    return {
        "model": model,
        "base_url": base_url,
        "requests": requests,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "elapsed_seconds": round(elapsed, 3),
        "success_requests": stats["success"],
        "failed_requests": stats["fail"],
        "requests_per_second": round(stats["success"] / elapsed, 3),
        "completion_tokens_per_second": round(stats["completion_tokens"] / elapsed, 3),
        "total_tokens_per_second": round(stats["total_tokens"] / elapsed, 3),
        "prompt_tokens": stats["prompt_tokens"],
        "completion_tokens": stats["completion_tokens"],
        "total_tokens": stats["total_tokens"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="changeme")
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Model ids. Supports space-separated and comma-separated values.",
    )
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--prompt",
        default="Write a concise technical summary of GPU throughput tuning.",
    )
    parser.add_argument("--output-csv", default="throughput_results.csv")
    args = parser.parse_args()

    if args.requests < 1:
        raise ValueError("--requests must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")

    models = _parse_models(args.models)
    if not models:
        models = _resolve_models(args.base_url, args.api_key, args.timeout_seconds)

    results = []
    for model in models:
        result = _benchmark_model(
            base_url=args.base_url,
            api_key=args.api_key,
            model=model,
            prompt=args.prompt,
            requests=args.requests,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
        )
        results.append(result)
        print(json.dumps(result, indent=2))

    fieldnames = [
        "model",
        "base_url",
        "requests",
        "concurrency",
        "max_tokens",
        "elapsed_seconds",
        "success_requests",
        "failed_requests",
        "requests_per_second",
        "completion_tokens_per_second",
        "total_tokens_per_second",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ]
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    all_failed = all(row["success_requests"] == 0 for row in results)
    return 1 if all_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
