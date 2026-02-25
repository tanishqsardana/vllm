# Eval Harness (Phase 1)

This harness benchmarks the existing OpenAI-compatible vLLM endpoint and writes reproducible artifacts to:

- `results/<run_id>/results.json`
- `results/<run_id>/scorecard.md`
- `results/<run_id>/requests_*.csv`
- `results/<run_id>/gpu_timeseries.csv`
- `results/<run_id>/sysinfo.json`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r eval/requirements.txt
```

## Quick Start (single preset)

```bash
./scripts/eval_run.sh qwen25_3b http://localhost:8000
```

Optional environment variables:

- `IMAGE_TAG` (included in `run_id` and report)
- `CONTAINER_NAME` (default: `engine`, used by sysinfo collection)
- `SEED` (default: `42`)
- `NOTE` (stored in `results/<run_id>/notes.txt`)

## Matrix Run

```bash
./scripts/eval_matrix.sh http://localhost:8000
```

By default this pauses between presets (`PAUSE_BETWEEN=1`) so the operator can switch GPU machines.

## Summarize Across Machines

```bash
python3 scripts/summarize_results.py
```

Outputs `results/results_summary.csv` for cross-machine comparison.

## What Gets Tested

- API contract (`/livez`, `/healthz`, `/v1/chat/completions`)
- Concurrency ladder (`[1,5,20,50]`, constant + bursty)
- Mixed prompt fairness (70% short / 30% medium)
- Long-context behavior (~75% of max model length)
- Soak stability (default 30 minutes)

## Notes

- Prompt generation settings are fixed for comparability:
  - `temperature=0`, `top_p=1`, `presence_penalty=0`, `frequency_penalty=0`
  - non-streaming requests
  - system prompt: `You are a helpful assistant.`
- GPU telemetry works with `pynvml` when available, otherwise falls back to `nvidia-smi`.
