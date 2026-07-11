from rich.console import Console
from rich.table import Table
from rich import box

from modelbench.store import BenchmarkRecord

console = Console()


def render_table(record: BenchmarkRecord) -> None:
    r = record.results

    # Header
    console.print()
    console.print(
        f"[bold]ModelbBench[/bold] — [cyan]{record.model}[/cyan] on "
        f"[green]{record.backend}[/green] ({record.hardware.upper()})"
    )
    console.rule()
    console.print(f"  Model load time     [yellow]{r.model_load_time_ms / 1000:.1f}s[/yellow]")
    console.print(f"  Prompt suite        [dim]{record.prompt_suite_version} (5 prompts, temp=0.0)[/dim]")
    console.print(f"  Warm-up             [dim]5 requests (not counted)[/dim]")
    console.print()

    # Metrics table
    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
    table.add_column("Metric", style="dim", width=24)
    table.add_column("p50", justify="right")
    table.add_column("p90", justify="right")
    table.add_column("p99", justify="right")

    table.add_row(
        "Time to first token",
        f"{r.ttft_p50:.0f}ms",
        f"{r.ttft_p90:.0f}ms",
        f"{r.ttft_p99:.0f}ms",
    )
    table.add_row(
        "End-to-end latency",
        f"{r.latency_p50:.0f}ms",
        f"{r.latency_p90:.0f}ms",
        f"{r.latency_p99:.0f}ms",
    )
    table.add_row(
        "Tokens / sec",
        f"{r.tokens_per_sec:.1f}",
        "—",
        "—",
    )
    table.add_row(
        "VRAM (peak)",
        f"{r.vram_peak_mb} MB",
        "—",
        "—",
    )

    console.print(table)
    console.print()
