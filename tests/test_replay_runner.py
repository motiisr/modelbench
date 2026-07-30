import pytest

from modelbench.backends.base import BackendRunner, InferenceResult
from modelbench.replay.loader import Sample
from modelbench.replay.runner import Candidate, ReplayRunner


class FakeCandidateBackend(BackendRunner):
    """Returns a fixed output_text/tokens_per_sec/latency per infer() call,
    and records lifecycle calls for ordering assertions."""

    def __init__(self, output_text: str = "some output", tokens_per_sec: float = 10.0, latency_ms: float = 100.0):
        self.started_with: str | None = None
        self.stopped = False
        self.warmed_up = False
        self.infer_calls: list[tuple[str, int]] = []
        self._output_text = output_text
        self._tokens_per_sec = tokens_per_sec
        self._latency_ms = latency_ms

    def start(self, model_id: str) -> None:
        self.started_with = model_id

    def warm_up(self) -> None:
        self.warmed_up = True

    def infer(self, prompt: str, max_tokens: int) -> InferenceResult:
        self.infer_calls.append((prompt, max_tokens))
        return InferenceResult(
            ttft_ms=10.0,
            total_latency_ms=self._latency_ms,
            output_tokens=5,
            tokens_per_sec=self._tokens_per_sec,
            output_text=self._output_text,
        )

    def stop(self) -> None:
        self.stopped = True


class RaisingBackend(BackendRunner):
    def __init__(self):
        self.stopped = False

    def start(self, model_id: str) -> None:
        pass

    def warm_up(self) -> None:
        pass

    def infer(self, prompt: str, max_tokens: int) -> InferenceResult:
        raise RuntimeError("boom")

    def stop(self) -> None:
        self.stopped = True


class FakeJudgeBackend(BackendRunner):
    def __init__(self, rating: str = "8"):
        self._rating = rating
        self.started_with: str | None = None
        self.warmed_up = False
        self.stopped = False

    def start(self, model_id: str) -> None:
        self.started_with = model_id

    def warm_up(self) -> None:
        self.warmed_up = True

    def infer(self, prompt: str, max_tokens: int) -> InferenceResult:
        return InferenceResult(ttft_ms=1.0, total_latency_ms=2.0, output_tokens=1, tokens_per_sec=1.0, output_text=self._rating)

    def stop(self) -> None:
        self.stopped = True


def test_run_starts_and_stops_each_candidate():
    backend = FakeCandidateBackend()
    candidate = Candidate(name="my-model", backend=backend, model_id="my-model-id")
    runner = ReplayRunner(repeats=1)

    runner.run(samples=[Sample(prompt="hi", expected="hi")], candidates=[candidate])

    assert backend.started_with == "my-model-id"
    assert backend.warmed_up is True
    assert backend.stopped is True


def test_run_stops_backend_even_if_infer_raises():
    backend = RaisingBackend()
    candidate = Candidate(name="broken", backend=backend, model_id="broken-id")
    runner = ReplayRunner(repeats=1)

    with pytest.raises(RuntimeError, match="boom"):
        runner.run(samples=[Sample(prompt="hi", expected="hi")], candidates=[candidate])

    assert backend.stopped is True


def test_run_repeats_each_sample_configured_number_of_times():
    backend = FakeCandidateBackend()
    candidate = Candidate(name="my-model", backend=backend, model_id="my-model-id")
    runner = ReplayRunner(repeats=3)

    runner.run(samples=[Sample(prompt="a", expected="a"), Sample(prompt="b", expected="b")], candidates=[candidate])

    assert len(backend.infer_calls) == 6  # 2 samples x 3 repeats


def test_run_scores_samples_with_expected_using_exact_match():
    backend = FakeCandidateBackend(output_text="4")
    candidate = Candidate(name="my-model", backend=backend, model_id="my-model-id")
    runner = ReplayRunner(repeats=1)

    report = runner.run(samples=[Sample(prompt="2+2?", expected="4")], candidates=[candidate])

    assert report.candidates[0].mean_score == 1.0
    assert report.candidates[0].score_method == "exact"


def test_run_scores_samples_without_expected_using_judge():
    backend = FakeCandidateBackend(output_text="Paris")
    judge = FakeJudgeBackend(rating="9")
    candidate = Candidate(name="my-model", backend=backend, model_id="my-model-id")
    runner = ReplayRunner(repeats=1, judge_backend=judge, judge_model_id="judge-model-id")

    report = runner.run(samples=[Sample(prompt="capital of France?")], candidates=[candidate])

    assert report.candidates[0].mean_score == pytest.approx(0.9)
    assert report.candidates[0].score_method == "llm_judge"
    assert report.candidates[0].min_confidence < 1.0


def test_run_starts_warms_up_and_stops_judge_backend_once_for_the_whole_run():
    backend_a = FakeCandidateBackend(output_text="Paris")
    backend_b = FakeCandidateBackend(output_text="London")
    judge = FakeJudgeBackend(rating="9")
    candidates = [
        Candidate(name="model-a", backend=backend_a, model_id="a"),
        Candidate(name="model-b", backend=backend_b, model_id="b"),
    ]
    runner = ReplayRunner(repeats=1, judge_backend=judge, judge_model_id="judge-model-id")

    runner.run(samples=[Sample(prompt="capital?")], candidates=candidates)

    assert judge.started_with == "judge-model-id"
    assert judge.warmed_up is True
    assert judge.stopped is True


def test_judge_backend_without_judge_model_id_raises_at_construction():
    judge = FakeJudgeBackend()
    with pytest.raises(ValueError, match="judge_model_id"):
        ReplayRunner(repeats=1, judge_backend=judge)  # missing judge_model_id


def test_run_raises_clear_error_on_empty_samples():
    backend = FakeCandidateBackend()
    candidate = Candidate(name="my-model", backend=backend, model_id="my-model-id")
    runner = ReplayRunner(repeats=1)

    with pytest.raises(ValueError, match="samples"):
        runner.run(samples=[], candidates=[candidate])

    # Must fail before wasting a start()/warm_up() cycle on the candidate backend.
    assert backend.started_with is None


def test_candidate_report_preserves_individual_scores_not_just_the_mean():
    backend = FakeCandidateBackend(output_text="4")
    candidate = Candidate(name="my-model", backend=backend, model_id="my-model-id")
    runner = ReplayRunner(repeats=3)

    report = runner.run(samples=[Sample(prompt="2+2?", expected="4")], candidates=[candidate])

    assert report.candidates[0].scores == [1.0, 1.0, 1.0]


def test_candidate_report_distinguishes_sample_count_from_total_inference_count():
    backend = FakeCandidateBackend()
    candidate = Candidate(name="my-model", backend=backend, model_id="my-model-id")
    runner = ReplayRunner(repeats=3)

    report = runner.run(
        samples=[Sample(prompt="a", expected="a"), Sample(prompt="b", expected="b")],
        candidates=[candidate],
    )

    assert report.candidates[0].sample_count == 2
    assert report.candidates[0].total_inference_count == 6  # 2 samples x 3 repeats


def test_run_raises_if_sample_has_no_expected_and_no_judge_backend_configured():
    backend = FakeCandidateBackend()
    candidate = Candidate(name="my-model", backend=backend, model_id="my-model-id")
    runner = ReplayRunner(repeats=1)  # no judge_backend

    with pytest.raises(ValueError, match="judge_backend"):
        runner.run(samples=[Sample(prompt="capital of France?")], candidates=[candidate])


def test_run_report_contains_one_entry_per_candidate():
    backend_a = FakeCandidateBackend()
    backend_b = FakeCandidateBackend()
    candidates = [
        Candidate(name="model-a", backend=backend_a, model_id="a"),
        Candidate(name="model-b", backend=backend_b, model_id="b"),
    ]
    runner = ReplayRunner(repeats=1)

    report = runner.run(samples=[Sample(prompt="hi", expected="hi")], candidates=candidates)

    assert [c.name for c in report.candidates] == ["model-a", "model-b"]


def test_run_aggregates_mean_tokens_per_sec_and_latency_p50():
    backend = FakeCandidateBackend(tokens_per_sec=20.0, latency_ms=50.0)
    candidate = Candidate(name="my-model", backend=backend, model_id="my-model-id")
    runner = ReplayRunner(repeats=2)

    report = runner.run(samples=[Sample(prompt="hi", expected="hi")], candidates=[candidate])

    assert report.candidates[0].mean_tokens_per_sec == pytest.approx(20.0)
    assert report.candidates[0].latency_p50_ms == pytest.approx(50.0)
    assert report.candidates[0].sample_count == 1
