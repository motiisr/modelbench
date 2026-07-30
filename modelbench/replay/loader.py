import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Sample:
    prompt: str
    expected: Optional[str] = None
    task_type: Optional[str] = None


def load_samples(path: Path) -> list[Sample]:
    """Parse a replay sample JSONL file. Rejects malformed lines with a
    clear, line-numbered error rather than silently skipping them."""
    samples = []
    with open(path) as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON on line {line_no}: {e}") from e
            if "prompt" not in data:
                raise ValueError(f"Missing required 'prompt' field on line {line_no}")
            samples.append(
                Sample(
                    prompt=data["prompt"],
                    expected=data.get("expected"),
                    task_type=data.get("task_type"),
                )
            )
    if not samples:
        raise ValueError(f"No samples found in {path}")
    return samples
