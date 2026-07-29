import pytest
from pytest_httpx import HTTPXMock

from modelbench.backends.anthropic_backend import AnthropicBackend
from modelbench.backends.base import InferenceResult
from tests.conftest import make_anthropic_stream_response


def test_start_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    backend = AnthropicBackend()
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        backend.start("claude-haiku-3-5")


def test_infer_returns_inference_result(httpx_mock: HTTPXMock, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    backend = AnthropicBackend()
    backend.start("claude-haiku-3-5")
    stream_body = make_anthropic_stream_response(tokens=["Hello", " world"], output_tokens=10)
    httpx_mock.add_response(
        method="POST",
        url="https://api.anthropic.com/v1/messages",
        content=stream_body,
        headers={"Content-Type": "text/event-stream"},
    )
    result = backend.infer("Hello", max_tokens=20)
    assert isinstance(result, InferenceResult)
    assert result.output_tokens == 10
    assert result.ttft_ms > 0
    assert result.total_latency_ms >= result.ttft_ms


def test_infer_tokens_per_sec_calculated_correctly(httpx_mock: HTTPXMock, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    backend = AnthropicBackend()
    backend.start("claude-haiku-3-5")
    stream_body = make_anthropic_stream_response(tokens=["word"] * 5, output_tokens=50)
    httpx_mock.add_response(
        method="POST",
        url="https://api.anthropic.com/v1/messages",
        content=stream_body,
    )
    result = backend.infer("Hello", max_tokens=50)
    assert result.tokens_per_sec > 0


def test_infer_raises_on_http_error(httpx_mock: HTTPXMock, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    backend = AnthropicBackend()
    backend.start("claude-haiku-3-5")
    httpx_mock.add_response(
        method="POST",
        url="https://api.anthropic.com/v1/messages",
        status_code=500,
        text="Internal Server Error",
    )
    with pytest.raises(RuntimeError, match="Anthropic inference failed"):
        backend.infer("Hello", max_tokens=20)


def test_infer_raises_on_empty_response(httpx_mock: HTTPXMock, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    backend = AnthropicBackend()
    backend.start("claude-haiku-3-5")
    httpx_mock.add_response(
        method="POST",
        url="https://api.anthropic.com/v1/messages",
        content=b"",
    )
    with pytest.raises(RuntimeError, match="Anthropic inference failed"):
        backend.infer("Hello", max_tokens=20)


def test_stop_and_warm_up_are_safe_noops(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    backend = AnthropicBackend()
    backend.start("claude-haiku-3-5")
    backend.stop()  # should not raise
