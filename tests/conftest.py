import json


def make_ollama_stream_response(tokens: list[str], eval_count: int = 10, eval_duration_ns: int = 500_000_000) -> bytes:
    """Build a fake Ollama streaming response body."""
    lines = []
    for token in tokens:
        lines.append(json.dumps({
            "model": "llama3.2:3b",
            "created_at": "2026-07-11T14:00:00Z",
            "response": token,
            "done": False,
        }))
    # Final chunk
    lines.append(json.dumps({
        "model": "llama3.2:3b",
        "created_at": "2026-07-11T14:00:00Z",
        "response": "",
        "done": True,
        "eval_count": eval_count,
        "eval_duration": eval_duration_ns,
    }))
    return "\n".join(lines).encode()
