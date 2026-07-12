from rich.console import Console
from rich.table import Table
from rich import box

from modelbench.cost import compute_cost_comparison
from modelbench.store import BenchmarkRecord

console = Console()


def _format_cost(usd_per_1m: float) -> str:
    if usd_per_1m == 0.0:
        return "$0.00 / 1M tokens"
    if usd_per_1m < 0.01:
        return f"${usd_per_1m:.4f} / 1M tokens"
    return f"${usd_per_1m:.2f} / 1M tokens"


def _format_multiplier(self_cost: float, api_cost: float) -> str:
    if self_cost == 0.0:
        return "∞× cheaper"
    ratio = api_cost / self_cost
    return f"{ratio:.0f}× cheaper"


def render_table(record: BenchmarkRecord, hardware_cost_per_hour: float = 0.0) -> None:
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

    # Cost projection
    cc = compute_cost_comparison(
        tokens_per_sec=r.tokens_per_sec,
        hardware_cost_per_hour=hardware_cost_per_hour,
    )
    hw_label = f"${hardware_cost_per_hour:.2f}/hr" if hardware_cost_per_hour > 0 else "owned hw"
    console.print(f"  [bold]Self-hosting cost[/bold]    {_format_cost(cc.self_hosting_usd_per_1m)}  ({hw_label})")
    for name, api_price in cc.api_comparisons:
        mult = _format_multiplier(cc.self_hosting_usd_per_1m, api_price)
        console.print(f"  vs [dim]{name:<14}[/dim]  {_format_cost(api_price)}  → [green]{mult}[/green]")
    console.print()
