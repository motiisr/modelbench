import statistics
from dataclasses import dataclass

from modelbench.backends.base import InferenceResult


@dataclass(frozen=True)
class BenchmarkMetrics:
    model_load_time_ms: float
    ttft_p50: float
    ttft_p90: float
    ttft_p99: float
    latency_p50: float
    latency_p90: float
    latency_p99: float
    tokens_per_sec: float
    vram_peak_mb: int
    vram_model_mb: int


def _percentile(values: list[float], p: float) -> float:
    """Compute percentile p (0-100) using linear interpolation."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    index = (p / 100) * (n - 1)
    lower = int(index)
    upper = lower + 1
    if upper >= n:
        return sorted_vals[-1]
    fraction = index - lower
    return sorted_vals[lower] + fraction * (sorted_vals[upper] - sorted_vals[lower])


def aggregate(
    results: list[InferenceResult],
    model_load_time_ms: float,
    vram_peak_mb: int,
    vram_model_mb: int,
) -> BenchmarkMetrics:
    if not results:
        raise ValueError("aggregate() requires at least one InferenceResult")

    ttft_values = [r.ttft_ms for r in results]
    latency_values = [r.total_latency_ms for r in results]
    tps_values = [r.tokens_per_sec for r in results]

    return BenchmarkMetrics(
        model_load_time_ms=model_load_time_ms,
        ttft_p50=_percentile(ttft_values, 50),
        ttft_p90=_percentile(ttft_values, 90),
        ttft_p99=_percentile(ttft_values, 99),
        latency_p50=_percentile(latency_values, 50),
        latency_p90=_percentile(latency_values, 90),
        latency_p99=_percentile(latency_values, 99),
        tokens_per_sec=statistics.mean(tps_values),
        vram_peak_mb=vram_peak_mb,
        vram_model_mb=vram_model_mb,
    )
