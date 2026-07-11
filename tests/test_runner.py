import pytest
from unittest.mock import MagicMock, patch
from modelbench.backends.base import BackendRunner, InferenceResult
from modelbench.benchmark.runner import BenchmarkRunner
from modelbench.benchmark.metrics import BenchmarkMetrics


def _make_fake_result() -> InferenceResult:
    return InferenceResult(ttft_ms=150.0, total_latency_ms=4000.0, output_tokens=50, tokens_per_sec=45.0)


class FakeBackend(BackendRunner):
    def __init__(self):
        self.started = False
        self.stopped = False
        self.warmed_up = False
        self.infer_calls = []

    def start(self, model_id: str) -> None:
        self.started = True

    def warm_up(self) -> None:
        self.warmed_up = True

    def infer(self, prompt: str, max_tokens: int) -> InferenceResult:
        self.infer_calls.append((prompt, max_tokens))
        return _make_fake_result()

    def stop(self) -> None:
        self.stopped = True


def test_runner_calls_start():
    backend = FakeBackend()
    runner = BenchmarkRunner(backend=backend)
    with patch("modelbench.benchmark.runner._poll_vram_mb", return_value=3840):
        runner.run(model_id="llama3.2:3b")
    assert backend.started


def test_runner_calls_warm_up_before_benchmark():
    backend = FakeBackend()
    runner = BenchmarkRunner(backend=backend)
    with patch("modelbench.benchmark.runner._poll_vram_mb", return_value=3840):
        runner.run(model_id="llama3.2:3b")
    assert backend.warmed_up


def test_runner_calls_stop():
    backend = FakeBackend()
    runner = BenchmarkRunner(backend=backend)
    with patch("modelbench.benchmark.runner._poll_vram_mb", return_value=3840):
        runner.run(model_id="llama3.2:3b")
    assert backend.stopped


def test_runner_runs_all_suite_v1_prompts():
    backend = FakeBackend()
    runner = BenchmarkRunner(backend=backend)
    with patch("modelbench.benchmark.runner._poll_vram_mb", return_value=3840):
        runner.run(model_id="llama3.2:3b")
    # Suite v1 has 5 prompts
    assert len(backend.infer_calls) == 5


def test_runner_returns_benchmark_metrics():
    backend = FakeBackend()
    runner = BenchmarkRunner(backend=backend)
    with patch("modelbench.benchmark.runner._poll_vram_mb", return_value=3840):
        metrics = runner.run(model_id="llama3.2:3b")
    assert isinstance(metrics, BenchmarkMetrics)
    assert metrics.tokens_per_sec == pytest.approx(45.0, rel=0.01)


def test_runner_calls_stop_even_if_infer_raises():
    backend = FakeBackend()
    backend.infer = MagicMock(side_effect=RuntimeError("Inference failed"))
    runner = BenchmarkRunner(backend=backend)
    with patch("modelbench.benchmark.runner._poll_vram_mb", return_value=3840):
        with pytest.raises(RuntimeError, match="Inference failed"):
            runner.run(model_id="llama3.2:3b")
    assert backend.stopped
