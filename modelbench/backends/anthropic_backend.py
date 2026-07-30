import json
import os
import time
from typing import Optional

import httpx

from modelbench.backends.base import BackendRunner, InferenceResult

_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicBackend(BackendRunner):
    """Cloud candidate backend for the Anthropic Messages API."""

    def __init__(self, base_url: str = "https://api.anthropic.com/v1"):
        self._base_url = base_url
        self._model_id: Optional[str] = None
        self._api_key: Optional[str] = None

    def start(self, model_id: str) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
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
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        }

        ttft_ms: Optional[float] = None
        start_time = time.perf_counter()
        output_tokens = 0
        received_any_content = False
        output_chunks: list[str] = []

        try:
            with httpx.stream(
                "POST", f"{self._base_url}/messages", json=payload, headers=headers, timeout=120
            ) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"Anthropic inference failed: HTTP {resp.status_code}")

                event_type: Optional[str] = None
                for line in resp.iter_lines():
                    if line.startswith("event: "):
                        event_type = line[len("event: "):]
                        continue
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[len("data: "):])
                    except json.JSONDecodeError:
                        continue
                    # Each SSE event carries exactly one data line in Anthropic's
                    # format; consume event_type here so a data line without a
                    # preceding "event:" line (malformed stream) can't be
                    # misattributed to whatever event type came before it.
                    consumed_event_type, event_type = event_type, None

                    if consumed_event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            received_any_content = True
                            output_chunks.append(delta["text"])
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - start_time) * 1000
                    elif consumed_event_type == "message_delta":
                        usage = data.get("usage") or {}
                        output_tokens = usage.get("output_tokens", 0)

        except httpx.HTTPError:
            raise RuntimeError("Anthropic inference failed: request error") from None

        if not received_any_content or ttft_ms is None:
            raise RuntimeError("Anthropic inference failed: no response tokens received")

        total_latency_ms = (time.perf_counter() - start_time) * 1000
        elapsed_s = total_latency_ms / 1000
        tokens_per_sec = output_tokens / elapsed_s if elapsed_s > 0 and output_tokens > 0 else 0.0

        return InferenceResult(
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            output_tokens=output_tokens,
            tokens_per_sec=tokens_per_sec,
            output_text="".join(output_chunks),
        )

    def stop(self) -> None:
        pass  # no local process to tear down
