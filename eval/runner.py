from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import re
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

CURRENT_DIR = Path(__file__).resolve().parent
import sys

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from client import OpenAICompatClient
from gpu_probe import GPUProbe
from metrics import (
    aggregate_request_metrics,
    latency_drift_percent,
    memory_drift_percent_vram,
    p95_by_prompt_bucket,
)
from report import write_results_json, write_scorecard


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "unknown"


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def detect_primary_gpu_name() -> str:
    command = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    try:
        output = subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True)
        first = output.splitlines()[0].strip()
        return first or "unknown-gpu"
    except Exception:
        return "unknown-gpu"


def default_run_id(model_preset: str, image_tag: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    gpu = slugify(detect_primary_gpu_name())
    parts = [ts, gpu, slugify(model_preset)]
    if image_tag:
        parts.append(f"img-{slugify(image_tag)}")
    return "_".join(parts)


def write_notes(run_dir: Path, note: Optional[str]) -> None:
    if not note:
        return
    (run_dir / "notes.txt").write_text(note.strip() + "\n", encoding="utf-8")


def write_request_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    seen = set()
    for record in records:
        for key in record.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def load_or_collect_sysinfo(repo_root: Path, run_id: str, run_dir: Path) -> Dict[str, Any]:
    sysinfo_path = run_dir / "sysinfo.json"
    if not sysinfo_path.exists():
        script = repo_root / "scripts" / "collect_sysinfo.sh"
        if script.exists():
            subprocess.run([str(script), run_id], cwd=repo_root, check=False)

    if sysinfo_path.exists():
        try:
            return json.loads(sysinfo_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def parse_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(str(value).strip())
        return parsed if parsed > 0 else None
    except Exception:
        return None


class MultiBucketSampler:
    def __init__(self, items: Sequence[Tuple[str, str]], rng: random.Random) -> None:
        self.items = list(items)
        self.rng = rng
        self.last_prompt: Optional[str] = None

    def sample(self) -> Tuple[str, str]:
        if not self.items:
            return "", "unknown"

        if len(self.items) == 1:
            prompt, bucket = self.items[0]
            self.last_prompt = prompt
            return prompt, bucket

        for _ in range(12):
            prompt, bucket = self.rng.choice(self.items)
            if prompt != self.last_prompt:
                self.last_prompt = prompt
                return prompt, bucket

        prompt, bucket = self.rng.choice(self.items)
        self.last_prompt = prompt
        return prompt, bucket


class MixedFairnessSampler:
    def __init__(
        self,
        short_prompts: Sequence[str],
        medium_prompts: Sequence[str],
        short_ratio: float,
        rng: random.Random,
    ) -> None:
        self.short_sampler = MultiBucketSampler([(p, "short") for p in short_prompts], rng)
        self.medium_sampler = MultiBucketSampler([(p, "medium") for p in medium_prompts], rng)
        self.short_ratio = short_ratio
        self.rng = rng
        self.last_prompt: Optional[str] = None

    def sample(self) -> Tuple[str, str]:
        for _ in range(12):
            if self.rng.random() < self.short_ratio:
                prompt, bucket = self.short_sampler.sample()
            else:
                prompt, bucket = self.medium_sampler.sample()
            if prompt != self.last_prompt:
                self.last_prompt = prompt
                return prompt, bucket

        if self.rng.random() < self.short_ratio:
            prompt, bucket = self.short_sampler.sample()
        else:
            prompt, bucket = self.medium_sampler.sample()
        self.last_prompt = prompt
        return prompt, bucket


def generate_long_context_prompt(max_model_len: int, fill_paragraph: str, seed: int) -> str:
    rng = random.Random(seed)
    target_tokens = max(256, int(max_model_len * 0.75))
    target_chars = target_tokens * 4

    sections = []
    char_count = 0
    section_idx = 1

    while char_count < target_chars:
        detail = rng.choice(
            [
                "Include assumptions and constraints.",
                "Capture edge cases and fallback paths.",
                "Note open questions and risks.",
                "Document expected output format.",
            ]
        )
        chunk = (
            f"Section {section_idx}: {fill_paragraph.strip()} "
            f"Operational note: {detail}"
        )
        sections.append(chunk)
        char_count += len(chunk)
        section_idx += 1

    body = "\n\n".join(sections)
    body = body[:target_chars]
    return (
        body
        + "\n\nUsing only the context above, produce a concise summary and list 5 implementation risks."
    )


async def run_warmup(
    client: OpenAICompatClient,
    samplers: Sequence[Any],
    warmup_requests: int,
    max_tokens: int,
    suite: str,
    conc: int,
    mode: str,
) -> None:
    if warmup_requests <= 0 or not samplers:
        return

    for i in range(warmup_requests):
        sampler = samplers[i % len(samplers)]
        prompt, bucket = sampler.sample()
        await client.chat_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            suite=suite,
            conc=conc,
            mode=mode,
            prompt_bucket=bucket,
        )


async def run_duration_test(
    *,
    client: OpenAICompatClient,
    suite_name: str,
    conc: int,
    mode: str,
    duration_s: int,
    warmup_requests: int,
    sampler_factory,
    max_tokens: int,
    abort_error_rate: float,
    burst_spike_every_s: int = 10,
    burst_spike_duration_s: int = 2,
    monitor_health: bool = False,
    health_check_interval_s: int = 30,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, float]]:
    base_samplers = [sampler_factory(worker_idx) for worker_idx in range(max(1, conc))]
    await run_warmup(
        client,
        samplers=base_samplers,
        warmup_requests=warmup_requests,
        max_tokens=max_tokens,
        suite=suite_name,
        conc=conc,
        mode=mode,
    )

    records: List[Dict[str, Any]] = []
    stop_event = asyncio.Event()
    state: Dict[str, Any] = {
        "aborted": False,
        "abort_reason": "",
        "total": 0,
        "errors": 0,
        "health_failed": False,
    }

    start_epoch = time.time()
    start_mono = time.monotonic()
    end_mono = start_mono + float(duration_s)

    async def worker(worker_idx: int, is_spike_worker: bool) -> None:
        sampler = sampler_factory(worker_idx)
        while not stop_event.is_set():
            now_mono = time.monotonic()
            if now_mono >= end_mono:
                break

            if mode == "bursty" and is_spike_worker:
                elapsed = now_mono - start_mono
                if (elapsed % burst_spike_every_s) >= burst_spike_duration_s:
                    await asyncio.sleep(0.05)
                    continue

            prompt, bucket = sampler.sample()
            rec = await client.chat_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                suite=suite_name,
                conc=conc,
                mode=mode,
                prompt_bucket=bucket,
                req_id=uuid.uuid4().hex[:8],
            )
            rec["worker_id"] = worker_idx
            records.append(rec)

            state["total"] += 1
            if int(rec.get("http_status", 0)) != 200:
                state["errors"] += 1

            if state["total"] >= max(20, conc):
                error_rate = state["errors"] / max(1, state["total"])
                if error_rate > abort_error_rate:
                    state["aborted"] = True
                    state["abort_reason"] = (
                        f"error_rate {error_rate:.4f} exceeded threshold {abort_error_rate:.4f}"
                    )
                    stop_event.set()
                    break

    async def health_monitor() -> None:
        while not stop_event.is_set():
            now_mono = time.monotonic()
            if now_mono >= end_mono:
                return

            status, _, _ = await client.get_status_json("/healthz")
            if status != 200:
                state["aborted"] = True
                state["health_failed"] = True
                state["abort_reason"] = f"healthz returned {status}"
                stop_event.set()
                return

            await asyncio.sleep(float(health_check_interval_s))

    tasks = [asyncio.create_task(worker(i, False)) for i in range(max(1, conc))]
    if mode == "bursty":
        tasks.extend(
            asyncio.create_task(worker(conc + i, True)) for i in range(max(1, conc))
        )
    if monitor_health:
        tasks.append(asyncio.create_task(health_monitor()))

    try:
        while not stop_event.is_set() and time.monotonic() < end_mono:
            await asyncio.sleep(0.2)
    finally:
        stop_event.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    end_epoch = time.time()
    duration_actual = max(0.001, end_epoch - start_epoch)
    summary = aggregate_request_metrics(records, duration_actual)
    summary.update(
        {
            "suite": suite_name,
            "conc": conc,
            "mode": mode,
            "duration_s": duration_actual,
            "aborted": bool(state["aborted"]),
            "abort_reason": state["abort_reason"],
            "health_failed": bool(state["health_failed"]),
        }
    )

    return summary, records, {"start_epoch_s": start_epoch, "end_epoch_s": end_epoch}


async def run_fixed_request_test(
    *,
    client: OpenAICompatClient,
    suite_name: str,
    request_count: int,
    conc: int,
    sampler_factory,
    max_tokens: int,
    warmup_requests: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, float]]:
    samplers = [sampler_factory(worker_idx) for worker_idx in range(max(1, conc))]
    await run_warmup(
        client,
        samplers=samplers,
        warmup_requests=warmup_requests,
        max_tokens=max_tokens,
        suite=suite_name,
        conc=conc,
        mode="constant",
    )

    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(request_count):
        queue.put_nowait(i)

    records: List[Dict[str, Any]] = []
    start_epoch = time.time()

    async def worker(worker_idx: int) -> None:
        sampler = samplers[worker_idx]
        while True:
            try:
                _ = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            prompt, bucket = sampler.sample()
            rec = await client.chat_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                suite=suite_name,
                conc=conc,
                mode="constant",
                prompt_bucket=bucket,
                req_id=uuid.uuid4().hex[:8],
            )
            rec["worker_id"] = worker_idx
            records.append(rec)
            queue.task_done()

    await asyncio.gather(*(worker(i) for i in range(max(1, conc))))
    end_epoch = time.time()

    duration_actual = max(0.001, end_epoch - start_epoch)
    summary = aggregate_request_metrics(records, duration_actual)
    summary.update(
        {
            "suite": suite_name,
            "conc": conc,
            "mode": "constant",
            "duration_s": duration_actual,
            "success_rate": summary["success_count"] / max(1, request_count),
        }
    )

    return summary, records, {"start_epoch_s": start_epoch, "end_epoch_s": end_epoch}


async def run_api_contract(
    client: OpenAICompatClient,
    short_prompts: Sequence[str],
    health_timeout_s: int,
    completion_timeout_s: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    livez_status, _, _ = await client.get_status_json("/livez")

    healthz_status = 0
    health_deadline = time.time() + health_timeout_s
    while time.time() < health_deadline:
        status, _, _ = await client.get_status_json("/healthz")
        healthz_status = status
        if status == 200:
            break
        await asyncio.sleep(2)

    prompt = short_prompts[0] if short_prompts else "Say hello in five words."

    completion_record = await asyncio.wait_for(
        client.chat_completion(
            prompt=prompt,
            max_tokens=128,
            suite="api_contract",
            conc=1,
            mode="constant",
            prompt_bucket="short",
            req_id=uuid.uuid4().hex[:8],
        ),
        timeout=max(10, completion_timeout_s),
    )

    passed = (
        livez_status == 200
        and healthz_status == 200
        and int(completion_record.get("http_status", 0)) == 200
        and bool(completion_record.get("response_snippet"))
    )

    return (
        {
            "pass": passed,
            "livez_status": livez_status,
            "healthz_status": healthz_status,
            "completion_status": completion_record.get("http_status", 0),
        },
        [completion_record],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 evaluation harness")
    parser.add_argument("--model-preset", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--config-dir", default="eval/config")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    repo_root = CURRENT_DIR.parent
    config_dir = (repo_root / args.config_dir).resolve()
    results_root = (repo_root / args.results_root).resolve()

    models_cfg = load_yaml(config_dir / "models.yaml")
    suites_cfg = load_yaml(config_dir / "suites.yaml")
    prompts_cfg = load_yaml(config_dir / "prompts.yaml")

    presets = models_cfg.get("model_presets", {})
    if args.model_preset not in presets:
        raise SystemExit(f"Unknown model preset: {args.model_preset}")

    model_cfg = presets[args.model_preset]
    image_tag = os.getenv("IMAGE_TAG", "")
    run_id = args.run_id.strip() or default_run_id(args.model_preset, image_tag)

    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_notes(run_dir, args.note.strip() or None)

    system_info = load_or_collect_sysinfo(repo_root, run_id, run_dir)

    model_id = os.getenv("MODEL_ID", model_cfg.get("model_id", ""))
    dtype = os.getenv("DTYPE", model_cfg.get("dtype", ""))
    max_model_len = int(os.getenv("MAX_MODEL_LEN", model_cfg.get("max_model_len", 8192)))
    gpu_memory_utilization = float(
        os.getenv("GPU_MEMORY_UTILIZATION", model_cfg.get("gpu_memory_utilization", 0.90))
    )
    tensor_parallel_size = (
        parse_positive_int(os.getenv("TENSOR_PARALLEL"))
        or parse_positive_int(os.getenv("TENSOR_PARALLEL_SIZE"))
        or parse_positive_int(system_info.get("tensor_parallel_size"))
        or parse_positive_int(model_cfg.get("tensor_parallel_size", 1))
    )

    gpu_count = int(system_info.get("gpu_topology", {}).get("gpu_count", 0) or 0)
    visible_gpus = system_info.get("cuda_visible_devices", os.getenv("CUDA_VISIBLE_DEVICES", ""))

    short_prompts = list(prompts_cfg.get("short", []))
    medium_prompts = list(prompts_cfg.get("medium", []))
    combined_items = [(p, "short") for p in short_prompts] + [(p, "medium") for p in medium_prompts]
    if not combined_items:
        combined_items = [("Say hello in five words.", "short")]

    long_cfg = prompts_cfg.get("long_context", {})
    fill_paragraph = long_cfg.get(
        "fill_paragraph",
        "Synthetic long-context content for stress testing.",
    )
    long_prompt = generate_long_context_prompt(max_model_len, fill_paragraph, args.seed)

    probe = GPUProbe(run_dir / "gpu_timeseries.csv", interval_s=1.0)
    await probe.start()

    results: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": utc_now_iso(),
        "base_url": args.base_url,
        "model_preset": args.model_preset,
        "image_tag": image_tag,
        "seed": args.seed,
        "gpu_count": gpu_count,
        "tensor_parallel_size": tensor_parallel_size,
        "visible_gpus": visible_gpus,
        "system_info": system_info,
        "model": {
            "model_id": model_id,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
        },
        "suites": {},
    }

    try:
        async with OpenAICompatClient(args.base_url, model_id=model_id) as client:
            api_cfg = suites_cfg.get("api_contract", {})
            api_summary, api_records = await run_api_contract(
                client,
                short_prompts,
                health_timeout_s=int(api_cfg.get("health_timeout_s", 600)),
                completion_timeout_s=int(api_cfg.get("completion_timeout_s", 120)),
            )
            results["suites"]["api_contract"] = api_summary
            write_request_csv(run_dir / "requests_api_contract_1_constant.csv", api_records)

            ladder_cfg = suites_cfg.get("concurrency_ladder", {})
            ladder_results: List[Dict[str, Any]] = []
            ladder_aborted = False

            conc_levels = list(ladder_cfg.get("conc_levels", [1, 5, 20, 50]))
            modes = list(ladder_cfg.get("modes", ["constant", "bursty"]))

            for conc in conc_levels:
                for mode in modes:
                    if ladder_aborted:
                        break

                    def ladder_sampler_factory(worker_idx: int):
                        rng = random.Random(args.seed + worker_idx * 997 + conc * 31)
                        return MultiBucketSampler(combined_items, rng)

                    summary, records, _ = await run_duration_test(
                        client=client,
                        suite_name="concurrency_ladder",
                        conc=int(conc),
                        mode=str(mode),
                        duration_s=int(ladder_cfg.get("duration_s", 120)),
                        warmup_requests=int(ladder_cfg.get("warmup_requests", 20)),
                        sampler_factory=ladder_sampler_factory,
                        max_tokens=128,
                        abort_error_rate=float(ladder_cfg.get("abort_error_rate", 0.05)),
                        burst_spike_every_s=int(ladder_cfg.get("burst_spike_every_s", 10)),
                        burst_spike_duration_s=int(ladder_cfg.get("burst_spike_duration_s", 2)),
                    )

                    tok_per_s = float(summary.get("completion_tokens_per_sec", 0.0))
                    summary["tok_per_s_per_gpu"] = tok_per_s / max(1, gpu_count)
                    ladder_results.append(summary)

                    csv_name = f"requests_concurrency_ladder_{conc}_{mode}.csv"
                    write_request_csv(run_dir / csv_name, records)

                    if bool(summary.get("aborted")):
                        ladder_aborted = True

            results["suites"]["concurrency_ladder"] = {
                "aborted": ladder_aborted,
                "results": ladder_results,
            }

            fairness_cfg = suites_cfg.get("mixed_prompt_fairness", {})

            def fairness_sampler_factory(worker_idx: int):
                rng = random.Random(args.seed + 50000 + worker_idx * 223)
                return MixedFairnessSampler(
                    short_prompts,
                    medium_prompts,
                    short_ratio=float(fairness_cfg.get("short_ratio", 0.70)),
                    rng=rng,
                )

            fairness_summary, fairness_records, _ = await run_duration_test(
                client=client,
                suite_name="mixed_prompt_fairness",
                conc=int(fairness_cfg.get("concurrency", 20)),
                mode="constant",
                duration_s=int(fairness_cfg.get("duration_s", 120)),
                warmup_requests=int(fairness_cfg.get("warmup_requests", 20)),
                sampler_factory=fairness_sampler_factory,
                max_tokens=128,
                abort_error_rate=1.0,
            )
            fairness_summary["p95_short_ms"] = p95_by_prompt_bucket(fairness_records, "short")
            fairness_summary["p95_medium_ms"] = p95_by_prompt_bucket(fairness_records, "medium")
            fairness_summary["tok_per_s_per_gpu"] = float(
                fairness_summary.get("completion_tokens_per_sec", 0.0)
            ) / max(1, gpu_count)

            results["suites"]["mixed_prompt_fairness"] = fairness_summary
            write_request_csv(
                run_dir
                / f"requests_mixed_prompt_fairness_{fairness_summary.get('conc', 20)}_constant.csv",
                fairness_records,
            )

            long_cfg_suite = suites_cfg.get("long_context", {})
            long_results: List[Dict[str, Any]] = []

            for conc in long_cfg_suite.get("conc_levels", [1, 5]):

                def long_sampler_factory(worker_idx: int):
                    rng = random.Random(args.seed + 90000 + worker_idx * 13 + int(conc) * 7)
                    return MultiBucketSampler([(long_prompt, "long_context")], rng)

                summary, records, _ = await run_fixed_request_test(
                    client=client,
                    suite_name="long_context",
                    request_count=int(long_cfg_suite.get("request_count", 20)),
                    conc=int(conc),
                    sampler_factory=long_sampler_factory,
                    max_tokens=64,
                    warmup_requests=0,
                )
                summary["tok_per_s_per_gpu"] = float(
                    summary.get("completion_tokens_per_sec", 0.0)
                ) / max(1, gpu_count)
                long_results.append(summary)
                write_request_csv(run_dir / f"requests_long_context_{conc}_constant.csv", records)

            results["suites"]["long_context"] = {"results": long_results}

            soak_cfg = suites_cfg.get("soak", {})

            def soak_sampler_factory(worker_idx: int):
                rng = random.Random(args.seed + 120000 + worker_idx * 149)
                return MultiBucketSampler(combined_items, rng)

            soak_summary, soak_records, soak_window = await run_duration_test(
                client=client,
                suite_name="soak",
                conc=int(soak_cfg.get("concurrency", 10)),
                mode="constant",
                duration_s=int(soak_cfg.get("duration_s", 1800)),
                warmup_requests=int(soak_cfg.get("warmup_requests", 50)),
                sampler_factory=soak_sampler_factory,
                max_tokens=128,
                abort_error_rate=float(soak_cfg.get("error_rate_fail_threshold", 0.001)),
                monitor_health=True,
                health_check_interval_s=int(soak_cfg.get("health_check_interval_s", 30)),
            )
            write_request_csv(
                run_dir / f"requests_soak_{soak_summary.get('conc', 10)}_constant.csv",
                soak_records,
            )

            total_vram_mb = float(system_info.get("gpu_topology", {}).get("total_vram_mb", 0.0))
            soak_mem_series = probe.total_memory_by_timestamp(
                start_epoch_s=float(soak_window["start_epoch_s"]),
                end_epoch_s=float(soak_window["end_epoch_s"]),
            )
            mem_drift = memory_drift_percent_vram(
                total_mem_by_ts=soak_mem_series,
                total_vram_mb=total_vram_mb,
                soak_duration_s=float(soak_summary.get("duration_s", 0.0)),
            )

            soak_summary["latency_drift_percent"] = latency_drift_percent(soak_records)
            soak_summary["memory_drift_percent_vram"] = mem_drift
            soak_summary["tok_per_s_per_gpu"] = float(
                soak_summary.get("completion_tokens_per_sec", 0.0)
            ) / max(1, gpu_count)

            fail_reasons: List[str] = []
            if bool(soak_summary.get("health_failed")):
                fail_reasons.append("engine_unhealthy")
            if float(soak_summary.get("error_rate", 0.0)) > float(
                soak_cfg.get("error_rate_fail_threshold", 0.001)
            ):
                fail_reasons.append("error_rate_threshold")
            if abs(float(mem_drift)) > float(
                soak_cfg.get("memory_drift_fail_threshold_percent_vram", 5.0)
            ):
                fail_reasons.append("memory_drift_threshold")

            soak_summary["pass"] = len(fail_reasons) == 0
            soak_summary["failure_reasons"] = fail_reasons
            results["suites"]["soak"] = soak_summary

    finally:
        results["gpu_summary"] = await probe.stop()

    results_json_path = write_results_json(run_dir, results)
    scorecard_path = write_scorecard(run_dir, results)

    print(f"Run ID: {run_id}")
    print(f"Results JSON: {results_json_path}")
    print(f"Scorecard: {scorecard_path}")


if __name__ == "__main__":
    asyncio.run(main())
