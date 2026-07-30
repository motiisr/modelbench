import pytest

from modelbench.backends.anthropic_backend import AnthropicBackend
from modelbench.backends.ollama import OllamaBackend
from modelbench.backends.openai_backend import OpenAIBackend
from modelbench.replay.registry import resolve_candidate


def test_resolve_known_anthropic_alias():
    candidate, kind = resolve_candidate("haiku")
    assert kind == "anthropic"
    assert isinstance(candidate.backend, AnthropicBackend)
    assert candidate.name == "haiku"
    assert candidate.model_id == "claude-haiku-3-5"


def test_resolve_known_openai_alias():
    candidate, kind = resolve_candidate("gpt-4o-mini")
    assert kind == "openai"
    assert isinstance(candidate.backend, OpenAIBackend)
    assert candidate.model_id == "gpt-4o-mini"


def test_resolve_colon_qualified_name_as_ollama():
    candidate, kind = resolve_candidate("llama3.3:70b")
    assert kind == "ollama"
    assert isinstance(candidate.backend, OllamaBackend)
    assert candidate.model_id == "llama3.3:70b"


def test_resolve_unknown_bare_name_raises_with_helpful_message():
    with pytest.raises(ValueError, match="Unknown candidate"):
        resolve_candidate("not-a-real-model")
