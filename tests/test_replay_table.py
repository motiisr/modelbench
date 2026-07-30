from modelbench.replay.runner import CandidateReport, ReplayReport
from modelbench.report.table import render_replay_table


def _make_report() -> ReplayReport:
    return ReplayReport(
        candidates=[
            CandidateReport(
                name="llama3.3:70b",
                mean_tokens_per_sec=45.2,
                latency_p50_ms=4200.0,
                mean_score=0.92,
                min_confidence=1.0,
                score_method="exact",
                scores=[1.0, 0.9, 0.86],
                sample_count=3,
                total_inference_count=9,
            ),
            CandidateReport(
                name="haiku",
                mean_tokens_per_sec=80.0,
                latency_p50_ms=900.0,
                mean_score=0.75,
                min_confidence=0.6,
                score_method="llm_judge",
                scores=[0.8, 0.7, 0.75],
                sample_count=3,
                total_inference_count=9,
            ),
        ]
    )


def _kinds() -> dict[str, str]:
    return {"llama3.3:70b": "ollama", "haiku": "anthropic"}


def test_render_replay_table_prints_output(capsys):
    render_replay_table(_make_report(), _kinds())
    captured = capsys.readouterr()
    assert len(captured.out) > 0


def test_render_replay_table_contains_candidate_names(capsys):
    render_replay_table(_make_report(), _kinds())
    captured = capsys.readouterr()
    assert "llama3.3:70b" in captured.out
    assert "haiku" in captured.out


def test_render_replay_table_shows_self_host_cost_for_local_candidate(capsys):
    render_replay_table(_make_report(), _kinds(), hardware_cost_per_hour=1.5)
    captured = capsys.readouterr()
    assert "1M tokens" in captured.out


def test_render_replay_table_does_not_fabricate_cost_for_cloud_candidate(capsys):
    render_replay_table(_make_report(), _kinds())
    captured = capsys.readouterr()
    assert "cloud" in captured.out.lower()


def test_render_replay_table_shows_quality_confidence_label(capsys):
    render_replay_table(_make_report(), _kinds())
    captured = capsys.readouterr()
    assert "60%" in captured.out or "0.6" in captured.out
