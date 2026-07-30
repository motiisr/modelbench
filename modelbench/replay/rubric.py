import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from modelbench.backends.base import BackendRunner

_EXACT_METHOD = "exact"
_FUZZY_METHOD = "fuzzy"
_LLM_JUDGE_METHOD = "llm_judge"

# Judge scores are an estimate, never ground truth — confidence is always
# below 1.0 (reserved for deterministic exact/fuzzy scoring) so callers can
# tell verified-by-comparison apart from judged-by-a-model.
_LLM_JUDGE_CONFIDENCE = 0.6

_JUDGE_PROMPT_TEMPLATE = (
    "Rate how well the RESPONSE answers the PROMPT on a scale from 0 to 10, "
    "where 10 is a perfect answer. Reply with only the number, nothing else.\n\n"
    "PROMPT: {prompt}\n\nRESPONSE: {output}"
)
_RATING_RE = re.compile(r"-?\d+(\.\d+)?")


@dataclass(frozen=True)
class ScoreResult:
    score: float       # 0.0-1.0
    confidence: float  # 0.0-1.0; 1.0 = deterministic, <1.0 = judge-derived estimate
    method: str


def score_exact_or_fuzzy(output: str, expected: str) -> ScoreResult:
    """Deterministic scoring for structured outputs with a known-correct answer."""
    output_norm = output.strip()
    expected_norm = expected.strip()
    if output_norm == expected_norm:
        return ScoreResult(score=1.0, confidence=1.0, method=_EXACT_METHOD)
    ratio = SequenceMatcher(None, output_norm, expected_norm).ratio()
    return ScoreResult(score=ratio, confidence=1.0, method=_FUZZY_METHOD)


def score_llm_judge(output: str, prompt: str, judge_backend: BackendRunner) -> ScoreResult:
    """Quality scoring for free-text output with no known-correct answer,
    via a cheap model acting as judge. Always carries confidence < 1.0."""
    judge_prompt = _JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, output=output)
    result = judge_backend.infer(judge_prompt, max_tokens=10)

    match = _RATING_RE.search(result.output_text)
    if match is None:
        raise ValueError(f"Could not parse a numeric rating from judge output: {result.output_text!r}")

    raw_rating = float(match.group())
    clamped_rating = max(0.0, min(raw_rating, 10.0))
    return ScoreResult(score=clamped_rating / 10, confidence=_LLM_JUDGE_CONFIDENCE, method=_LLM_JUDGE_METHOD)
