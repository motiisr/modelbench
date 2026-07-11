import json
import pytest
from pathlib import Path
from modelbench.store import save_result, load_results, BenchmarkRecord
from modelbench.benchmark.metrics import BenchmarkMetrics


def _make_metrics() -> BenchmarkMetrics:
    return BenchmarkMetrics(
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
    )


def test_save_result_creates_file(tmp_path):
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
    assert store_path.exists()


def test_save_result_valid_json(tmp_path):
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
    data = json.loads(store_path.read_text())
    assert data["schema_version"] == "1"
    assert len(data["benchmarks"]) == 1


def test_save_result_appends_on_second_call(tmp_path):
    store_path = tmp_path / "benchmarks.json"
    for _ in range(2):
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
    data = json.loads(store_path.read_text())
    assert len(data["benchmarks"]) == 2


def test_load_results_returns_empty_list_if_no_file(tmp_path):
    store_path = tmp_path / "benchmarks.json"
    results = load_results(store_path)
    assert results == []


def test_load_results_returns_records(tmp_path):
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
    records = load_results(store_path)
    assert len(records) == 1
    assert records[0].model == "llama3.2:3b"
    assert records[0].backend == "ollama"
    assert records[0].results.tokens_per_sec == pytest.approx(45.2)


def test_record_id_is_unique_per_run(tmp_path):
    store_path = tmp_path / "benchmarks.json"
    for _ in range(2):
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
    data = json.loads(store_path.read_text())
    ids = [b["id"] for b in data["benchmarks"]]
    assert len(set(ids)) == 2
