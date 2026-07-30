import pytest

from modelbench.backends.base import BackendRunner, InferenceResult
from modelbench.replay.rubric import ScoreResult, score_exact_or_fuzzy, score_llm_judge


def test_score_exact_or_fuzzy_exact_match():
    result = score_exact_or_fuzzy("4", "4")
    assert result == ScoreResult(score=1.0, confidence=1.0, method="exact")


def test_score_exact_or_fuzzy_exact_match_ignores_surrounding_whitespace():
    result = score_exact_or_fuzzy("  4 \n", "4")
    assert result.score == 1.0
    assert result.method == "exact"


def test_score_exact_or_fuzzy_partial_match_is_fuzzy():
    result = score_exact_or_fuzzy("The answer is 4", "4")
    assert result.method == "fuzzy"
    assert 0.0 <= result.score < 1.0
    assert result.confidence == 1.0  # fuzzy match is deterministic


def test_score_exact_or_fuzzy_no_match_scores_low():
    result = score_exact_or_fuzzy("completely unrelated text", "4")
    assert result.score < 0.3


class _FakeJudgeBackend(BackendRunner):
    def __init__(self, judge_output_text: str):
        self._judge_output_text = judge_output_text
        self.infer_calls: list[tuple[str, int]] = []

    def start(self, model_id: str) -> None:
        pass

    def warm_up(self) -> None:
        pass

    def infer(self, prompt: str, max_tokens: int) -> InferenceResult:
        self.infer_calls.append((prompt, max_tokens))
        return InferenceResult(
            ttft_ms=10.0,
            total_latency_ms=20.0,
            output_tokens=1,
            tokens_per_sec=50.0,
            output_text=self._judge_output_text,
        )

    def stop(self) -> None:
        pass


def test_score_llm_judge_parses_numeric_rating():
    judge = _FakeJudgeBackend(judge_output_text="8")
    result = score_llm_judge("Paris is the capital of France.", "What is the capital of France?", judge)
    assert result.score == pytest.approx(0.8)
    assert result.method == "llm_judge"
    assert result.confidence < 1.0  # judge scores are never ground truth


def test_score_llm_judge_includes_prompt_and_output_in_judge_call():
    judge = _FakeJudgeBackend(judge_output_text="10")
    score_llm_judge("some output", "some prompt", judge)
    assert len(judge.infer_calls) == 1
    judge_prompt, _ = judge.infer_calls[0]
    assert "some output" in judge_prompt
    assert "some prompt" in judge_prompt


def test_score_llm_judge_raises_on_unparseable_rating():
    judge = _FakeJudgeBackend(judge_output_text="not a number")
    with pytest.raises(ValueError, match="judge"):
        score_llm_judge("output", "prompt", judge)


def test_score_llm_judge_clamps_out_of_range_rating():
    judge = _FakeJudgeBackend(judge_output_text="15")
    result = score_llm_judge("output", "prompt", judge)
    assert result.score == 1.0
