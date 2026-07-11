import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from modelbench.benchmark.metrics import BenchmarkMetrics


def _get_cuda_version() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _make_id(model: str, backend: str, hardware: str, timestamp: str) -> str:
    date_part = timestamp[:10].replace("-", "")
    safe_model = model.replace(":", "-").replace("/", "-")
    return f"{safe_model}-{backend}-{hardware}-{date_part}"


def _ensure_unique_id(existing_ids: set[str], candidate: str) -> str:
    if candidate not in existing_ids:
        return candidate
    counter = 2
    while f"{candidate}-{counter}" in existing_ids:
        counter += 1
    return f"{candidate}-{counter}"


@dataclass(frozen=True)
class BenchmarkRecord:
    id: str
    model: str
    backend: str
    backend_version: str
    hardware: str
    vram_gb: int
    prompt_suite_version: str
    timestamp: str
    results: BenchmarkMetrics


def _record_to_dict(record: BenchmarkRecord) -> dict:
    return {
        "id": record.id,
        "model": record.model,
        "backend": record.backend,
        "backend_version": record.backend_version,
        "hardware": record.hardware,
        "vram_gb": record.vram_gb,
        "prompt_suite_version": record.prompt_suite_version,
        "timestamp": record.timestamp,
        "environment": {
            "os": platform.platform(),
            "driver_version": _get_cuda_version(),
        },
        "results": {
            "model_load_time_ms": record.results.model_load_time_ms,
            "ttft_p50": record.results.ttft_p50,
            "ttft_p90": record.results.ttft_p90,
            "ttft_p99": record.results.ttft_p99,
            "latency_p50": record.results.latency_p50,
            "latency_p90": record.results.latency_p90,
            "latency_p99": record.results.latency_p99,
            "tokens_per_sec": record.results.tokens_per_sec,
            "vram_peak_mb": record.results.vram_peak_mb,
            "vram_model_mb": record.results.vram_model_mb,
        },
    }


def _dict_to_record(d: dict) -> BenchmarkRecord:
    r = d["results"]
    return BenchmarkRecord(
        id=d["id"],
        model=d["model"],
        backend=d["backend"],
        backend_version=d["backend_version"],
        hardware=d["hardware"],
        vram_gb=d["vram_gb"],
        prompt_suite_version=d["prompt_suite_version"],
        timestamp=d["timestamp"],
        results=BenchmarkMetrics(
            model_load_time_ms=r["model_load_time_ms"],
            ttft_p50=r["ttft_p50"],
            ttft_p90=r["ttft_p90"],
            ttft_p99=r["ttft_p99"],
            latency_p50=r["latency_p50"],
            latency_p90=r["latency_p90"],
            latency_p99=r["latency_p99"],
            tokens_per_sec=r["tokens_per_sec"],
            vram_peak_mb=r["vram_peak_mb"],
            vram_model_mb=r["vram_model_mb"],
        ),
    )


def save_result(
    store_path: Path,
    model: str,
    backend: str,
    backend_version: str,
    hardware: str,
    vram_gb: int,
    prompt_suite_version: str,
    metrics: BenchmarkMetrics,
) -> BenchmarkRecord:
    store_path.parent.mkdir(parents=True, exist_ok=True)

    if store_path.exists():
        data = json.loads(store_path.read_text())
    else:
        data = {"schema_version": "1", "benchmarks": []}

    existing_ids = {b["id"] for b in data["benchmarks"]}
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    candidate_id = _make_id(model, backend, hardware, timestamp)
    unique_id = _ensure_unique_id(existing_ids, candidate_id)

    record = BenchmarkRecord(
        id=unique_id,
        model=model,
        backend=backend,
        backend_version=backend_version,
        hardware=hardware,
        vram_gb=vram_gb,
        prompt_suite_version=prompt_suite_version,
        timestamp=timestamp,
        results=metrics,
    )

    data["benchmarks"].append(_record_to_dict(record))
    store_path.write_text(json.dumps(data, indent=2))
    return record


def load_results(store_path: Path) -> list[BenchmarkRecord]:
    if not store_path.exists():
        return []
    data = json.loads(store_path.read_text())
    return [_dict_to_record(b) for b in data.get("benchmarks", [])]
