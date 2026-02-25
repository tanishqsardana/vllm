from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import httpx

SYSTEM_PROMPT = "You are a helpful assistant."


class TokenEstimator:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._tokenizer = None
        self._initialized = False

    def _init_tokenizer(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                local_files_only=True,
                trust_remote_code=False,
                use_fast=True,
            )
        except Exception:
            self._tokenizer = None

    def estimate(self, text: str) -> int:
        self._init_tokenizer()
        if self._tokenizer is not None:
            try:
                return int(len(self._tokenizer.encode(text, add_special_tokens=False)))
            except Exception:
                pass
        return max(1, len(text) // 4)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _extract_completion_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""

    message = (choices[0] or {}).get("message") or {}
    content = message.get("content", "")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    pieces.append(item["text"])
        return "\n".join(pieces)
    return str(content)


def _extract_completion_tokens(payload: Dict[str, Any], completion_text: str) -> int:
    usage = payload.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if isinstance(completion_tokens, int):
        return completion_tokens
    return max(1, len(completion_text) // 4)


class OpenAICompatClient:
    def __init__(self, base_url: str, model_id: str, timeout_s: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout_s = timeout_s
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=10.0))
        self.token_estimator = TokenEstimator(model_id)

    async def __aenter__(self) -> "OpenAICompatClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        await self.client.aclose()

    async def get_status_json(self, path: str) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
        try:
            response = await self.client.get(f"{self.base_url}{path}")
            payload: Optional[Dict[str, Any]] = None
            try:
                payload = response.json()
            except Exception:
                payload = None
            return response.status_code, payload, None
        except Exception as exc:
            return 0, None, exc.__class__.__name__

    async def chat_completion(
        self,
        *,
        prompt: str,
        max_tokens: int,
        suite: str,
        conc: int,
        mode: str,
        prompt_bucket: str,
        req_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        req_id = req_id or uuid.uuid4().hex[:8]
        start_monotonic = time.perf_counter()
        start_epoch = time.time()
        start_time = utc_now_iso()

        request_body = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{prompt}\n\n[req_id:{req_id}]"},
            ],
            "temperature": 0,
            "top_p": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "stream": False,
            "max_tokens": max_tokens,
        }

        status = 0
        error_type = ""
        completion_tokens = 0
        snippet = ""

        try:
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=request_body,
            )
            status = response.status_code
            if status == 200:
                payload = response.json()
                completion_text = _extract_completion_text(payload)
                completion_tokens = _extract_completion_tokens(payload, completion_text)
                snippet = completion_text.replace("\n", " ")[:220]
            else:
                error_type = f"http_{status}"
                snippet = response.text.replace("\n", " ")[:220]
        except Exception as exc:
            error_type = exc.__class__.__name__
            snippet = str(exc).replace("\n", " ")[:220]

        end_monotonic = time.perf_counter()
        end_epoch = time.time()
        end_time = utc_now_iso()

        latency_ms = (end_monotonic - start_monotonic) * 1000.0
        prompt_token_estimate = self.token_estimator.estimate(prompt)
        tokens_per_second = completion_tokens / max(1e-9, latency_ms / 1000.0)

        return {
            "start_time": start_time,
            "end_time": end_time,
            "start_epoch_s": start_epoch,
            "end_epoch_s": end_epoch,
            "latency_ms": latency_ms,
            "http_status": status,
            "error_type": error_type,
            "prompt_char_len": len(prompt),
            "prompt_token_estimate": prompt_token_estimate,
            "completion_tokens": completion_tokens,
            "tokens_per_second": tokens_per_second,
            "response_snippet": snippet,
            "prompt_bucket": prompt_bucket,
            "req_id": req_id,
            "suite": suite,
            "conc": conc,
            "mode": mode,
        }
