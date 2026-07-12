import subprocess
from pathlib import Path

import click
from rich.console import Console

from modelbench.backends.ollama import OllamaBackend
from modelbench.benchmark.runner import BenchmarkRunner
from modelbench.report.table import render_table
from modelbench.store import save_result, load_results

console = Console()


def _default_store_path() -> Path:
    return Path.home() / ".modelbench" / "benchmarks.json"


def _get_ollama_version() -> str:
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
        return result.stdout.strip().split()[-1] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _detect_hardware() -> tuple[str, int]:
    """Returns (hardware_label, vram_gb). Falls back to 'unknown' if nvidia-smi unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            name = parts[0].lower().replace(" ", "").replace("nvidia", "").replace("geforce", "")
            vram_mb = int(parts[1]) if len(parts) > 1 else 0
            return name, round(vram_mb / 1024)
    except Exception:
        pass
    return "unknown", 0


@click.group()
def cli():
    """ModelbBench — benchmark LLMs on your local hardware."""
    pass


@cli.command()
@click.argument("model")
@click.option("--backend", default="ollama", show_default=True, help="Backend to use.")
@click.option("--hardware", default=None, help="Hardware label (auto-detected if omitted).")
@click.option("--suite", default="v1", show_default=True, help="Prompt suite version.")
def run(model: str, backend: str, hardware: str, suite: str) -> None:
    """Run benchmark suite against MODEL."""
    if backend != "ollama":
        raise click.UsageError(f"Backend '{backend}' not supported in V1. Use 'ollama'.")

    hw_label, vram_gb = _detect_hardware()
    if hardware:
        hw_label = hardware

    console.print(f"\n[bold]ModelbBench[/bold] — benchmarking [cyan]{model}[/cyan] via {backend}...")
    console.print(f"  Hardware: [yellow]{hw_label}[/yellow] ({vram_gb}GB VRAM detected)")
    console.print(f"  Suite:    {suite}\n")

    ollama_backend = OllamaBackend()
    runner = BenchmarkRunner(backend=ollama_backend)

    try:
        metrics = runner.run(model)
    except RuntimeError as e:
        raise click.ClickException(str(e))

    record = save_result(
        store_path=_default_store_path(),
        model=model,
        backend=backend,
        backend_version=_get_ollama_version(),
        hardware=hw_label,
        vram_gb=vram_gb,
        prompt_suite_version=suite,
        metrics=metrics,
    )

    render_table(record)
    console.print(f"  Result saved → [dim]{_default_store_path()}[/dim]\n")


@cli.command()
@click.argument("model", required=False, default=None)
def show(model: str) -> None:
    """Show stored benchmark results. Optionally filter by MODEL."""
    records = load_results(_default_store_path())

    if model:
        records = [r for r in records if r.model == model]

    if not records:
        console.print("[dim]No results found.[/dim]" + (f" Run: modelbench run {model}" if model else ""))
        return

    for record in records:
        render_table(record)
