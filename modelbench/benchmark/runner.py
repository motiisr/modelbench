import subprocess
import time

from modelbench.backends.base import BackendRunner
from modelbench.benchmark.metrics import BenchmarkMetrics, aggregate
from modelbench.benchmark.prompts import SUITE_V1


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


class BenchmarkRunner:
    def __init__(self, backend: BackendRunner):
        self._backend = backend

    def run(self, model_id: str) -> BenchmarkMetrics:
        load_start = time.perf_counter()
        self._backend.start(model_id)
        model_load_time_ms = (time.perf_counter() - load_start) * 1000

        vram_model_mb = _poll_vram_mb()

        try:
            self._backend.warm_up()

            results = []
            vram_peak_mb = vram_model_mb

            for prompt in SUITE_V1:
                result = self._backend.infer(prompt.text, max_tokens=prompt.max_tokens)
                results.append(result)
                current_vram = _poll_vram_mb()
                if current_vram > vram_peak_mb:
                    vram_peak_mb = current_vram

        except Exception:
            self._backend.stop()
            raise

        self._backend.stop()

        return aggregate(
            results=results,
            model_load_time_ms=model_load_time_ms,
            vram_peak_mb=vram_peak_mb,
            vram_model_mb=vram_model_mb,
        )
