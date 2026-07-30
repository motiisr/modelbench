from modelbench.backends.anthropic_backend import AnthropicBackend
from modelbench.backends.ollama import OllamaBackend
from modelbench.backends.openai_backend import OpenAIBackend
from modelbench.replay.runner import Candidate

# Known short aliases for cloud models, so `--candidates haiku,gpt-4o-mini`
# reads naturally on the command line. Anything not listed here is assumed
# to be an Ollama model id (identified by a ":" tag separator, e.g.
# "llama3.3:70b") — extend this table rather than adding string-matching
# heuristics for new cloud aliases.
_CLOUD_ALIASES: dict[str, tuple[str, str]] = {
    "haiku": ("anthropic", "claude-haiku-3-5"),
    "sonnet": ("anthropic", "claude-3-5-sonnet"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "gpt-4o": ("openai", "gpt-4o"),
}

_BACKEND_CLASSES = {
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
    "ollama": OllamaBackend,
}


def resolve_candidate(name: str) -> tuple[Candidate, str]:
    """Resolve a `--candidates` entry to a Candidate + backend kind ("anthropic"/"openai"/"ollama")."""
    if name in _CLOUD_ALIASES:
        kind, model_id = _CLOUD_ALIASES[name]
    elif ":" in name:
        kind, model_id = "ollama", name
    else:
        known = ", ".join(_CLOUD_ALIASES)
        raise ValueError(
            f"Unknown candidate {name!r}. Use a known alias ({known}) or an "
            "Ollama model id containing ':' (e.g. llama3.3:70b)."
        )
    backend = _BACKEND_CLASSES[kind]()
    return Candidate(name=name, backend=backend, model_id=model_id), kind
