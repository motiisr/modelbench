import json


def make_openai_stream_response(tokens: list[str], output_tokens: int = 10) -> bytes:
    """Build a fake OpenAI chat-completions SSE stream body (stream_options.include_usage)."""
    lines = []
    for token in tokens:
        chunk = {"choices": [{"delta": {"content": token}, "index": 0}]}
        lines.append(f"data: {json.dumps(chunk)}")
    usage_chunk = {
        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        "usage": {"completion_tokens": output_tokens},
    }
    lines.append(f"data: {json.dumps(usage_chunk)}")
    lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode()


def make_anthropic_stream_response(tokens: list[str], output_tokens: int = 10) -> bytes:
    """Build a fake Anthropic messages SSE stream body."""
    events = [("message_start", {"message": {"usage": {"output_tokens": 0}}})]
    for token in tokens:
        events.append(("content_block_delta", {"delta": {"type": "text_delta", "text": token}}))
    events.append(("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": output_tokens}}))
    events.append(("message_stop", {}))
    lines = []
    for event_type, data in events:
        lines.append(f"event: {event_type}")
        lines.append(f"data: {json.dumps(data)}")
        lines.append("")
    return ("\n".join(lines) + "\n").encode()


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
