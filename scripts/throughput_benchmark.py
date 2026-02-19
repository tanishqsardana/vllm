#!/usr/bin/env python3
"""Simple throughput benchmark for an OpenAI-compatible chat endpoint."""

from __future__ import annotations

import argparse
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


def _resolve_model(base_url: str, api_key: str, timeout_s: int) -> str:
    data = _get_json(f"{base_url}/v1/models", api_key, timeout_s)
    models = data.get("data", [])
    if not models:
        raise RuntimeError("No models returned by /v1/models")
    model_id = models[0].get("id")
    if not model_id:
        raise RuntimeError("First model has no 'id' field")
    return model_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="changeme")
    parser.add_argument("--model", default="")
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--prompt",
        default="Write a concise technical summary of GPU throughput tuning.",
    )
    parser.add_argument("--output-json", default="throughput_result.json")
    args = parser.parse_args()

    if args.requests < 1:
        raise ValueError("--requests must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")

    model = args.model or _resolve_model(args.base_url, args.api_key, args.timeout_seconds)
    work_queue: Queue[int] = Queue()
    for i in range(args.requests):
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
                "messages": [{"role": "user", "content": args.prompt}],
                "max_tokens": args.max_tokens,
                "stream": False,
            }
            try:
                response = _post_json(
                    f"{args.base_url}/v1/chat/completions",
                    args.api_key,
                    payload,
                    args.timeout_seconds,
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
    for _ in range(min(args.concurrency, args.requests)):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start

    if elapsed <= 0:
        elapsed = 1e-9

    result = {
        "model": model,
        "base_url": args.base_url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
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

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print(json.dumps(result, indent=2))

    if stats["success"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
