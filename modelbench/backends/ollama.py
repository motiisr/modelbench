import json
import socket
import subprocess
import time
from typing import Optional

import httpx

from modelbench.backends.base import BackendRunner, InferenceResult


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _poll_vram_mb() -> int:
    """Poll nvidia-smi for current GPU memory usage in MB. Returns 0 on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return 0


class OllamaBackend(BackendRunner):
    def __init__(self, port: Optional[int] = None):
        self._port = port or _find_free_port()
        self._process: Optional[subprocess.Popen] = None
        self._model_id: Optional[str] = None
        self._base_url = f"http://localhost:{self._port}"

    def start(self, model_id: str) -> None:
        self._model_id = model_id
        self._process = subprocess.Popen(
            ["ollama", "serve"],
            env={**__import__("os").environ, "OLLAMA_HOST": f"0.0.0.0:{self._port}"},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_until_ready()

    def _wait_until_ready(self, timeout: int = 30) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = httpx.get(f"{self._base_url}/api/tags", timeout=2)
                if resp.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"Ollama did not start within {timeout}s on port {self._port}")

    def warm_up(self) -> None:
        for _ in range(5):
            try:
                self.infer("Hello", max_tokens=5)
            except Exception:
                pass  # warm-up failures are silent

    def infer(self, prompt: str, max_tokens: int) -> InferenceResult:
        payload = {
            "model": self._model_id,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.0,
                "top_p": 1.0,
                "num_predict": max_tokens,
            },
        }

        ttft_ms: Optional[float] = None
        start_time = time.perf_counter()
        output_tokens = 0
        eval_duration_ns = 0

        try:
            with httpx.stream("POST", f"{self._base_url}/api/generate", json=payload, timeout=120) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"Ollama inference failed: HTTP {resp.status_code}")

                for line in resp.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if ttft_ms is None and chunk.get("response"):
                        ttft_ms = (time.perf_counter() - start_time) * 1000

                    if chunk.get("done"):
                        output_tokens = chunk.get("eval_count", 0)
                        eval_duration_ns = chunk.get("eval_duration", 0)
                        break

        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama inference failed: {e}") from e

        if ttft_ms is None:
            raise RuntimeError("Ollama inference failed: no response tokens received")

        total_latency_ms = (time.perf_counter() - start_time) * 1000
        elapsed_s = eval_duration_ns / 1_000_000_000 if eval_duration_ns > 0 else total_latency_ms / 1000
        tokens_per_sec = output_tokens / elapsed_s if elapsed_s > 0 and output_tokens > 0 else 0.0

        return InferenceResult(
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            output_tokens=output_tokens,
            tokens_per_sec=tokens_per_sec,
        )

    def stop(self) -> None:
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        time.sleep(2)  # WSL2: allow GPU memory to clear
