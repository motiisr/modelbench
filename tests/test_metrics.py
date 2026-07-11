import pytest
from modelbench.benchmark.metrics import BenchmarkMetrics, aggregate
from modelbench.backends.base import InferenceResult


def _make_results(ttft_values: list[float], latency_values: list[float]) -> list[InferenceResult]:
    results = []
    for ttft, latency in zip(ttft_values, latency_values):
        output_tokens = 50
        elapsed_s = latency / 1000
        results.append(InferenceResult(
            ttft_ms=ttft,
            total_latency_ms=latency,
            output_tokens=output_tokens,
            tokens_per_sec=output_tokens / elapsed_s if elapsed_s > 0 else 0,
        ))
    return results


def test_aggregate_ttft_p50():
    # 5 results with known TTFT values
    results = _make_results(
        ttft_values=[100.0, 150.0, 200.0, 250.0, 300.0],
        latency_values=[500.0, 600.0, 700.0, 800.0, 900.0],
    )
    metrics = aggregate(results, model_load_time_ms=5000.0, vram_peak_mb=3840, vram_model_mb=3200)
    assert metrics.ttft_p50 == pytest.approx(200.0, rel=0.05)


def test_aggregate_ttft_p90():
    results = _make_results(
        ttft_values=[100.0, 150.0, 200.0, 250.0, 300.0],
        latency_values=[500.0, 600.0, 700.0, 800.0, 900.0],
    )
    metrics = aggregate(results, model_load_time_ms=5000.0, vram_peak_mb=3840, vram_model_mb=3200)
    assert metrics.ttft_p90 == pytest.approx(280.0, rel=0.05)


def test_aggregate_ttft_p99():
    results = _make_results(
        ttft_values=[100.0, 150.0, 200.0, 250.0, 300.0],
        latency_values=[500.0, 600.0, 700.0, 800.0, 900.0],
    )
    metrics = aggregate(results, model_load_time_ms=5000.0, vram_peak_mb=3840, vram_model_mb=3200)
    assert metrics.ttft_p99 == pytest.approx(296.0, rel=0.05)


def test_aggregate_mean_tokens_per_sec():
    results = _make_results(
        ttft_values=[100.0, 100.0],
        latency_values=[1000.0, 2000.0],
    )
    metrics = aggregate(results, model_load_time_ms=0, vram_peak_mb=0, vram_model_mb=0)
    # Result 1: 50 tokens / 1s = 50 tps; Result 2: 50 tokens / 2s = 25 tps; mean = 37.5
    assert metrics.tokens_per_sec == pytest.approx(37.5, rel=0.01)


def test_aggregate_model_load_time_stored():
    results = _make_results([100.0], [500.0])
    metrics = aggregate(results, model_load_time_ms=8200.0, vram_peak_mb=3840, vram_model_mb=3200)
    assert metrics.model_load_time_ms == 8200.0


def test_aggregate_vram_stored():
    results = _make_results([100.0], [500.0])
    metrics = aggregate(results, model_load_time_ms=0, vram_peak_mb=4096, vram_model_mb=3500)
    assert metrics.vram_peak_mb == 4096
    assert metrics.vram_model_mb == 3500


def test_aggregate_requires_at_least_one_result():
    with pytest.raises(ValueError, match="at least one"):
        aggregate([], model_load_time_ms=0, vram_peak_mb=0, vram_model_mb=0)
