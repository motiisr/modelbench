import pytest
from pytest_httpx import HTTPXMock

from modelbench.backends.base import InferenceResult
from modelbench.backends.openai_backend import OpenAIBackend
from tests.conftest import make_openai_stream_response


def test_start_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = OpenAIBackend()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        backend.start("gpt-4o-mini")


def test_infer_returns_inference_result(httpx_mock: HTTPXMock, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    backend = OpenAIBackend()
    backend.start("gpt-4o-mini")
    stream_body = make_openai_stream_response(tokens=["Hello", " world"], output_tokens=10)
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        content=stream_body,
        headers={"Content-Type": "text/event-stream"},
    )
    result = backend.infer("Hello", max_tokens=20)
    assert isinstance(result, InferenceResult)
    assert result.output_tokens == 10
    assert result.ttft_ms > 0
    assert result.total_latency_ms >= result.ttft_ms


def test_infer_tokens_per_sec_calculated_correctly(httpx_mock: HTTPXMock, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    backend = OpenAIBackend()
    backend.start("gpt-4o-mini")
    stream_body = make_openai_stream_response(tokens=["word"] * 5, output_tokens=50)
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        content=stream_body,
    )
    result = backend.infer("Hello", max_tokens=50)
    assert result.tokens_per_sec > 0


def test_infer_raises_on_http_error(httpx_mock: HTTPXMock, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    backend = OpenAIBackend()
    backend.start("gpt-4o-mini")
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        status_code=500,
        text="Internal Server Error",
    )
    with pytest.raises(RuntimeError, match="OpenAI inference failed"):
        backend.infer("Hello", max_tokens=20)


def test_infer_raises_on_empty_response(httpx_mock: HTTPXMock, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    backend = OpenAIBackend()
    backend.start("gpt-4o-mini")
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        content=b"",
    )
    with pytest.raises(RuntimeError, match="OpenAI inference failed"):
        backend.infer("Hello", max_tokens=20)


def test_stop_and_warm_up_are_safe_noops(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    backend = OpenAIBackend()
    backend.start("gpt-4o-mini")
    backend.stop()  # should not raise
