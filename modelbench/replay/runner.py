import statistics
from dataclasses import dataclass
from typing import Optional

from modelbench.backends.base import BackendRunner, InferenceResult
from modelbench.replay.loader import Sample
from modelbench.replay.rubric import ScoreResult, score_exact_or_fuzzy, score_llm_judge

_DEFAULT_MAX_TOKENS = 500


def _percentile(values: list[float], p: float) -> float:
    """Compute percentile p (0-100) using linear interpolation."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    index = (p / 100) * (n - 1)
    lower = int(index)
    upper = lower + 1
    if upper >= n:
        return sorted_vals[-1]
    fraction = index - lower
    return sorted_vals[lower] + fraction * (sorted_vals[upper] - sorted_vals[lower])


@dataclass(frozen=True)
class Candidate:
    name: str
    backend: BackendRunner
    model_id: str


@dataclass(frozen=True)
class CandidateReport:
    name: str
    mean_tokens_per_sec: float
    latency_p50_ms: float
    mean_score: float
    min_confidence: float
    score_method: str      # common method across all scored samples, or "mixed"
    scores: list[float]    # every individual score (samples x repeats) — the
                            # distribution, not just its mean; a single point
                            # value would hide run-to-run variance that later
                            # phases (dashboard, continuous re-verification)
                            # need to show as a confidence range.
    sample_count: int          # number of unique samples
    total_inference_count: int # sample_count x repeats — distinct from
                                # sample_count so callers can't mistake one
                                # for the other when computing cost.


@dataclass(frozen=True)
class ReplayReport:
    candidates: list[CandidateReport]


def _common_method(scores: list[ScoreResult]) -> str:
    methods = {s.method for s in scores}
    return methods.pop() if len(methods) == 1 else "mixed"


class ReplayRunner:
    """Runs a set of samples against a set of candidate backends, scoring
    each response, so a replay comparison table can be built from the
    result. Each sample is run `repeats` times per candidate to guard
    against treating a single noisy run as representative (LLM outputs and
    judge scores are non-deterministic)."""

    def __init__(
        self,
        repeats: int = 3,
        judge_backend: Optional[BackendRunner] = None,
        judge_model_id: Optional[str] = None,
    ):
        if judge_backend is not None and judge_model_id is None:
            raise ValueError("judge_model_id is required when judge_backend is provided")
        self._repeats = repeats
        self._judge_backend = judge_backend
        self._judge_model_id = judge_model_id

    def run(self, samples: list[Sample], candidates: list[Candidate]) -> ReplayReport:
        if not samples:
            raise ValueError("run() requires at least one sample; got an empty samples list")

        if self._judge_backend is not None:
            self._judge_backend.start(self._judge_model_id)
            self._judge_backend.warm_up()
        try:
            return ReplayReport(candidates=[self._run_candidate(c, samples) for c in candidates])
        finally:
            if self._judge_backend is not None:
                self._judge_backend.stop()

    def _run_candidate(self, candidate: Candidate, samples: list[Sample]) -> CandidateReport:
        candidate.backend.start(candidate.model_id)
        try:
            candidate.backend.warm_up()
            results: list[InferenceResult] = []
            scores: list[ScoreResult] = []
            for sample in samples:
                for _ in range(self._repeats):
                    result = candidate.backend.infer(sample.prompt, max_tokens=_DEFAULT_MAX_TOKENS)
                    results.append(result)
                    scores.append(self._score(sample, result))
        finally:
            candidate.backend.stop()

        return CandidateReport(
            name=candidate.name,
            mean_tokens_per_sec=statistics.mean(r.tokens_per_sec for r in results),
            latency_p50_ms=_percentile([r.total_latency_ms for r in results], 50),
            mean_score=statistics.mean(s.score for s in scores),
            min_confidence=min(s.confidence for s in scores),
            score_method=_common_method(scores),
            scores=[s.score for s in scores],
            sample_count=len(samples),
            total_inference_count=len(results),
        )

    def _score(self, sample: Sample, result: InferenceResult) -> ScoreResult:
        if sample.expected is not None:
            return score_exact_or_fuzzy(result.output_text, sample.expected)
        if self._judge_backend is None:
            raise ValueError(
                f"Sample {sample.prompt!r} has no 'expected' field and no judge_backend was "
                "configured on ReplayRunner — cannot score free-text output."
            )
        return score_llm_judge(result.output_text, sample.prompt, self._judge_backend)
