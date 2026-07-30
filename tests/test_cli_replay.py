from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from modelbench.cli.main import cli
from modelbench.replay.loader import Sample
from modelbench.replay.runner import CandidateReport, ReplayReport


def _make_report() -> ReplayReport:
    return ReplayReport(
        candidates=[
            CandidateReport(
                name="haiku",
                mean_tokens_per_sec=80.0,
                latency_p50_ms=900.0,
                mean_score=0.9,
                min_confidence=0.6,
                score_method="llm_judge",
                scores=[0.9, 0.9, 0.9],
                sample_count=1,
                total_inference_count=3,
            )
        ]
    )


def test_replay_command_exists():
    runner = CliRunner()
    result = runner.invoke(cli, ["replay", "--help"])
    assert result.exit_code == 0
    assert "candidates" in result.output.lower()


def test_replay_invokes_runner_with_loaded_samples_and_resolved_candidates(tmp_path):
    samples_file = tmp_path / "samples.jsonl"
    samples_file.write_text('{"prompt": "2+2?", "expected": "4"}\n')

    with patch("modelbench.cli.main.ReplayRunner") as MockRunner, \
         patch("modelbench.cli.main.render_replay_table") as mock_render:
        mock_instance = MagicMock()
        mock_instance.run.return_value = _make_report()
        MockRunner.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(cli, ["replay", str(samples_file), "--candidates", "haiku,llama3.3:70b"])

    assert result.exit_code == 0, result.output
    mock_instance.run.assert_called_once()
    called_samples = mock_instance.run.call_args.kwargs["samples"]
    assert called_samples == [Sample(prompt="2+2?", expected="4")]
    called_candidates = mock_instance.run.call_args.kwargs["candidates"]
    assert [c.name for c in called_candidates] == ["haiku", "llama3.3:70b"]
    mock_render.assert_called_once()


def test_replay_passes_repeats_option_to_runner(tmp_path):
    samples_file = tmp_path / "samples.jsonl"
    samples_file.write_text('{"prompt": "2+2?", "expected": "4"}\n')

    with patch("modelbench.cli.main.ReplayRunner") as MockRunner, \
         patch("modelbench.cli.main.render_replay_table"):
        mock_instance = MagicMock()
        mock_instance.run.return_value = _make_report()
        MockRunner.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(
            cli, ["replay", str(samples_file), "--candidates", "haiku", "--repeats", "5"]
        )

    assert result.exit_code == 0, result.output
    MockRunner.assert_called_once_with(repeats=5, judge_backend=None, judge_model_id=None)


def test_replay_wires_up_judge_backend_when_judge_option_given(tmp_path):
    samples_file = tmp_path / "samples.jsonl"
    samples_file.write_text('{"prompt": "capital of France?"}\n')  # no "expected" -> needs a judge

    with patch("modelbench.cli.main.ReplayRunner") as MockRunner, \
         patch("modelbench.cli.main.render_replay_table"):
        mock_instance = MagicMock()
        mock_instance.run.return_value = _make_report()
        MockRunner.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(
            cli, ["replay", str(samples_file), "--candidates", "llama3.3:70b", "--judge", "haiku"]
        )

    assert result.exit_code == 0, result.output
    _, kwargs = MockRunner.call_args
    assert kwargs["judge_model_id"] == "claude-haiku-3-5"
    assert kwargs["judge_backend"] is not None


def test_replay_rejects_unknown_candidate_with_clear_error(tmp_path):
    samples_file = tmp_path / "samples.jsonl"
    samples_file.write_text('{"prompt": "2+2?", "expected": "4"}\n')

    runner = CliRunner()
    result = runner.invoke(cli, ["replay", str(samples_file), "--candidates", "not-a-real-model"])

    assert result.exit_code != 0
    assert "Unknown candidate" in result.output


def test_replay_rejects_missing_samples_file():
    runner = CliRunner()
    result = runner.invoke(cli, ["replay", "/no/such/file.jsonl", "--candidates", "haiku"])
    assert result.exit_code != 0


def test_replay_json_flag_prints_valid_json_instead_of_table(tmp_path):
    import json as json_module

    samples_file = tmp_path / "samples.jsonl"
    samples_file.write_text('{"prompt": "2+2?", "expected": "4"}\n')

    with patch("modelbench.cli.main.ReplayRunner") as MockRunner, \
         patch("modelbench.cli.main.render_replay_table") as mock_render:
        mock_instance = MagicMock()
        mock_instance.run.return_value = _make_report()
        MockRunner.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(
            cli, ["replay", str(samples_file), "--candidates", "haiku", "--json"]
        )

    assert result.exit_code == 0, result.output
    mock_render.assert_not_called()  # --json replaces the human table, doesn't add to it
    parsed = json_module.loads(result.output)
    assert parsed["candidates"][0]["name"] == "haiku"
    assert parsed["candidates"][0]["mean_score"] == 0.9


def test_replay_without_json_flag_still_renders_table(tmp_path):
    samples_file = tmp_path / "samples.jsonl"
    samples_file.write_text('{"prompt": "2+2?", "expected": "4"}\n')

    with patch("modelbench.cli.main.ReplayRunner") as MockRunner, \
         patch("modelbench.cli.main.render_replay_table") as mock_render:
        mock_instance = MagicMock()
        mock_instance.run.return_value = _make_report()
        MockRunner.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(cli, ["replay", str(samples_file), "--candidates", "haiku"])

    assert result.exit_code == 0, result.output
    mock_render.assert_called_once()


def test_replay_catches_runner_failure_and_shows_clean_error_not_a_traceback(tmp_path):
    samples_file = tmp_path / "samples.jsonl"
    samples_file.write_text('{"prompt": "2+2?", "expected": "4"}\n')

    with patch("modelbench.cli.main.ReplayRunner") as MockRunner:
        mock_instance = MagicMock()
        mock_instance.run.side_effect = RuntimeError("backend connection failed")
        MockRunner.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(cli, ["replay", str(samples_file), "--candidates", "haiku"])

    assert result.exit_code != 0
    assert "backend connection failed" in result.output
    assert "Traceback" not in result.output
