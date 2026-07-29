import json
import os
import time
from typing import Optional

import httpx

from modelbench.backends.base import BackendRunner, InferenceResult


class OpenAIBackend(BackendRunner):
    """Cloud candidate backend for OpenAI-compatible chat-completions APIs."""

    def __init__(self, base_url: str = "https://api.openai.com/v1"):
        self._base_url = base_url
        self._model_id: Optional[str] = None
        self._api_key: Optional[str] = None

    def start(self, model_id: str) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        self._api_key = api_key
        self._model_id = model_id

    def warm_up(self) -> None:
        for _ in range(5):
            try:
                self.infer("Hello", max_tokens=5)
            except Exception:
                pass  # warm-up failures are silent

    def infer(self, prompt: str, max_tokens: int) -> InferenceResult:
        payload = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}

        ttft_ms: Optional[float] = None
        start_time = time.perf_counter()
        output_tokens = 0
        received_any_content = False

        try:
            with httpx.stream(
                "POST", f"{self._base_url}/chat/completions", json=payload, headers=headers, timeout=120
            ) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"OpenAI inference failed: HTTP {resp.status_code}")

                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices") or []
                    content = choices[0].get("delta", {}).get("content") if choices else None
                    if content:
                        received_any_content = True
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - start_time) * 1000

                    usage = chunk.get("usage")
                    if usage:
                        output_tokens = usage.get("completion_tokens", 0)

        except httpx.HTTPError:
            raise RuntimeError("OpenAI inference failed: request error") from None

        if not received_any_content or ttft_ms is None:
            raise RuntimeError("OpenAI inference failed: no response tokens received")

        total_latency_ms = (time.perf_counter() - start_time) * 1000
        elapsed_s = total_latency_ms / 1000
        tokens_per_sec = output_tokens / elapsed_s if elapsed_s > 0 and output_tokens > 0 else 0.0

        return InferenceResult(
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            output_tokens=output_tokens,
            tokens_per_sec=tokens_per_sec,
        )

    def stop(self) -> None:
        pass  # no local process to tear down
