from modelbench.benchmark.metrics import BenchmarkMetrics
from modelbench.report.table import render_table
from modelbench.store import BenchmarkRecord


def _make_record() -> BenchmarkRecord:
    return BenchmarkRecord(
        id="llama3.2-3b-ollama-rtx4080-20260711",
        model="llama3.2:3b",
        backend="ollama",
        backend_version="0.3.6",
        hardware="rtx4080",
        vram_gb=16,
        prompt_suite_version="v1",
        timestamp="2026-07-11T14:00:00Z",
        results=BenchmarkMetrics(
            model_load_time_ms=8200.0,
            ttft_p50=180.0,
            ttft_p90=210.0,
            ttft_p99=280.0,
            latency_p50=4200.0,
            latency_p90=5100.0,
            latency_p99=6800.0,
            tokens_per_sec=45.2,
            vram_peak_mb=3840,
            vram_model_mb=3200,
        ),
    )


def test_render_table_returns_string(capsys):
    record = _make_record()
    render_table(record)
    captured = capsys.readouterr()
    assert len(captured.out) > 0


def test_render_table_contains_model_name(capsys):
    record = _make_record()
    render_table(record)
    captured = capsys.readouterr()
    assert "llama3.2:3b" in captured.out


def test_render_table_contains_ttft(capsys):
    record = _make_record()
    render_table(record)
    captured = capsys.readouterr()
    assert "180" in captured.out  # p50 TTFT


def test_render_table_contains_tokens_per_sec(capsys):
    record = _make_record()
    render_table(record)
    captured = capsys.readouterr()
    assert "45.2" in captured.out


def test_render_table_contains_vram(capsys):
    record = _make_record()
    render_table(record)
    captured = capsys.readouterr()
    assert "3840" in captured.out


def test_render_table_contains_load_time(capsys):
    record = _make_record()
    render_table(record)
    captured = capsys.readouterr()
    assert "8.2" in captured.out  # 8200ms → 8.2s
