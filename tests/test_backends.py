from modelbench.backends.base import InferenceResult


def test_inference_result_fields():
    r = InferenceResult(
        ttft_ms=120.5,
        total_latency_ms=4200.0,
        output_tokens=45,
        tokens_per_sec=42.3,
    )
    assert r.ttft_ms == 120.5
    assert r.total_latency_ms == 4200.0
    assert r.output_tokens == 45
    assert r.tokens_per_sec == 42.3


def test_inference_result_is_immutable():
    r = InferenceResult(ttft_ms=100.0, total_latency_ms=500.0, output_tokens=10, tokens_per_sec=20.0)
    try:
        r.ttft_ms = 999.0
        assert False, "Should have raised"
    except Exception:
        pass
