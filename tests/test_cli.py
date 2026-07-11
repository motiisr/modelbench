import json
import pytest
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from modelbench.cli.main import cli
from modelbench.benchmark.metrics import BenchmarkMetrics
from modelbench.store import BenchmarkRecord


def _make_metrics() -> BenchmarkMetrics:
    return BenchmarkMetrics(
        model_load_time_ms=8200.0,
        ttft_p50=180.0, ttft_p90=210.0, ttft_p99=280.0,
        latency_p50=4200.0, latency_p90=5100.0, latency_p99=6800.0,
        tokens_per_sec=45.2,
        vram_peak_mb=3840, vram_model_mb=3200,
    )


def test_run_command_exists():
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "model" in result.output.lower()


def test_show_command_exists():
    runner = CliRunner()
    result = runner.invoke(cli, ["show", "--help"])
    assert result.exit_code == 0


def test_run_invokes_benchmark_runner(tmp_path):
    runner = CliRunner()
    mock_metrics = _make_metrics()

    with patch("modelbench.cli.main.BenchmarkRunner") as MockRunner, \
         patch("modelbench.cli.main.OllamaBackend") as MockBackend, \
         patch("modelbench.cli.main._default_store_path", return_value=tmp_path / "benchmarks.json"), \
         patch("modelbench.cli.main._get_ollama_version", return_value="0.3.6"), \
         patch("modelbench.cli.main._detect_hardware", return_value=("rtx4080", 16)):

        mock_runner_instance = MagicMock()
        mock_runner_instance.run.return_value = mock_metrics
        MockRunner.return_value = mock_runner_instance

        result = runner.invoke(cli, ["run", "llama3.2:3b"])

    assert result.exit_code == 0, result.output
    mock_runner_instance.run.assert_called_once_with("llama3.2:3b")


def test_run_saves_result(tmp_path):
    runner = CliRunner()
    mock_metrics = _make_metrics()
    store_path = tmp_path / "benchmarks.json"

    with patch("modelbench.cli.main.BenchmarkRunner") as MockRunner, \
         patch("modelbench.cli.main.OllamaBackend"), \
         patch("modelbench.cli.main._default_store_path", return_value=store_path), \
         patch("modelbench.cli.main._get_ollama_version", return_value="0.3.6"), \
         patch("modelbench.cli.main._detect_hardware", return_value=("rtx4080", 16)):

        mock_runner_instance = MagicMock()
        mock_runner_instance.run.return_value = mock_metrics
        MockRunner.return_value = mock_runner_instance

        runner.invoke(cli, ["run", "llama3.2:3b"])

    assert store_path.exists()
    data = json.loads(store_path.read_text())
    assert len(data["benchmarks"]) == 1
    assert data["benchmarks"][0]["model"] == "llama3.2:3b"


def test_show_with_no_results(tmp_path):
    runner = CliRunner()
    with patch("modelbench.cli.main._default_store_path", return_value=tmp_path / "benchmarks.json"):
        result = runner.invoke(cli, ["show"])
    assert result.exit_code == 0
    assert "no results" in result.output.lower()


def test_show_filters_by_model(tmp_path):
    from modelbench.store import save_result
    store_path = tmp_path / "benchmarks.json"
    save_result(
        store_path=store_path,
        model="llama3.2:3b",
        backend="ollama",
        backend_version="0.3.6",
        hardware="rtx4080",
        vram_gb=16,
        prompt_suite_version="v1",
        metrics=_make_metrics(),
    )
    save_result(
        store_path=store_path,
        model="mistral:7b",
        backend="ollama",
        backend_version="0.3.6",
        hardware="rtx4080",
        vram_gb=16,
        prompt_suite_version="v1",
        metrics=_make_metrics(),
    )
    runner_obj = CliRunner()
    with patch("modelbench.cli.main._default_store_path", return_value=store_path):
        result = runner_obj.invoke(cli, ["show", "llama3.2:3b"])
    assert "llama3.2:3b" in result.output
    assert "mistral:7b" not in result.output
