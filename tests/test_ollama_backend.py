import pytest
import httpx
from pytest_httpx import HTTPXMock
from modelbench.backends.ollama import OllamaBackend
from modelbench.backends.base import InferenceResult
from tests.conftest import make_ollama_stream_response


def test_infer_returns_inference_result(httpx_mock: HTTPXMock):
    backend = OllamaBackend(port=11500)
    stream_body = make_ollama_stream_response(
        tokens=["Hello", " world"],
        eval_count=10,
        eval_duration_ns=500_000_000,
    )
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:11500/api/generate",
        content=stream_body,
        headers={"Content-Type": "application/x-ndjson"},
    )
    result = backend.infer("Hello", max_tokens=20)
    assert isinstance(result, InferenceResult)
    assert result.output_tokens == 10
    assert result.ttft_ms > 0
    assert result.total_latency_ms >= result.ttft_ms


def test_infer_tokens_per_sec_calculated_correctly(httpx_mock: HTTPXMock):
    backend = OllamaBackend(port=11500)
    # 50 tokens in 1 second = 50 tps
    stream_body = make_ollama_stream_response(
        tokens=["word"] * 5,
        eval_count=50,
        eval_duration_ns=1_000_000_000,
    )
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:11500/api/generate",
        content=stream_body,
    )
    result = backend.infer("Hello", max_tokens=50)
    assert result.tokens_per_sec == pytest.approx(50.0, rel=0.01)


def test_infer_raises_on_http_error(httpx_mock: HTTPXMock):
    backend = OllamaBackend(port=11500)
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:11500/api/generate",
        status_code=500,
        text="Internal Server Error",
    )
    with pytest.raises(RuntimeError, match="Ollama inference failed"):
        backend.infer("Hello", max_tokens=20)


def test_infer_raises_on_empty_response(httpx_mock: HTTPXMock):
    backend = OllamaBackend(port=11500)
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:11500/api/generate",
        content=b"",
    )
    with pytest.raises(RuntimeError, match="Ollama inference failed"):
        backend.infer("Hello", max_tokens=20)
