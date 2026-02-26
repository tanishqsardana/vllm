from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("gateway")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(ts: datetime) -> str:
    return ts.isoformat()


def latency_ms(start: datetime, end: datetime) -> int:
    return max(0, int((end - start).total_seconds() * 1000))


def json_log(event: str, **fields: Any) -> None:
    payload = {
        "ts": to_iso(utc_now()),
        "event": event,
        **fields,
    }
    logger.info(json.dumps(payload, separators=(",", ":"), default=str))


def percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    if len(values) == 1:
        return int(values[0])
    q = min(max(quantile, 0.0), 1.0)
    idx = int(math.ceil(q * len(values))) - 1
    idx = max(0, min(idx, len(values) - 1))
    return int(values[idx])


def extract_prompt_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []

    messages = payload.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            parts.extend(_extract_from_message(msg))

    prompt = payload.get("prompt")
    if prompt is not None:
        parts.extend(_extract_any_text(prompt))

    return "\n".join(part for part in parts if part)


def _extract_from_message(message: Any) -> list[str]:
    if isinstance(message, str):
        return [message]
    if not isinstance(message, dict):
        return _extract_any_text(message)

    out: list[str] = []
    content = message.get("content")
    out.extend(_extract_any_text(content))

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        out.extend(_extract_any_text(tool_calls))

    name = message.get("name")
    if isinstance(name, str):
        out.append(name)

    return out


def _extract_any_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_extract_any_text(item))
        return out
    if isinstance(value, dict):
        if "text" in value and isinstance(value["text"], str):
            return [value["text"]]
        return [json.dumps(value, separators=(",", ":"), sort_keys=True)]
    return [str(value)]


def extract_completion_text(response_json: dict[str, Any]) -> str:
    out: list[str] = []
    choices = response_json.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                out.extend(_extract_any_text(content))
            text = choice.get("text")
            if isinstance(text, str):
                out.append(text)
    return "\n".join(part for part in out if part)


class TokenEstimator:
    def __init__(self, model_id: str, hf_token: str | None = None, trust_remote_code: bool = False) -> None:
        self.model_id = model_id
        self.hf_token = hf_token
        self.trust_remote_code = trust_remote_code
        self._tokenizer = None
        self._attempted = False

    def approx_tokens(self, text: str) -> int:
        if not text:
            return 0

        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return max(1, math.ceil(len(text) / 4))

        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return max(1, math.ceil(len(text) / 4))

    def _get_tokenizer(self):
        if self._attempted:
            return self._tokenizer

        self._attempted = True
        try:
            from transformers import AutoTokenizer  # type: ignore

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=self.hf_token,
                trust_remote_code=self.trust_remote_code,
                use_fast=True,
            )
        except Exception as exc:  # pragma: no cover - environment-dependent
            json_log(
                "tokenizer_load_failed",
                model_id=self.model_id,
                message=str(exc),
            )
            self._tokenizer = None

        return self._tokenizer

