from dataclasses import dataclass

# USD per 1M output tokens for well-known APIs (as of mid-2025)
API_PRICES: dict[str, float] = {
    "GPT-4o mini": 0.60,
    "GPT-4o": 10.00,
}


@dataclass(frozen=True)
class CostComparison:
    self_hosting_usd_per_1m: float
    api_comparisons: list[tuple[str, float]]


def compute_cost_per_1m(tokens_per_sec: float, hardware_cost_per_hour: float = 0.0) -> float:
    """Cost in USD per 1M output tokens at a given throughput and hardware cost."""
    if tokens_per_sec <= 0:
        return 0.0
    tokens_per_hour = tokens_per_sec * 3600
    return (hardware_cost_per_hour / tokens_per_hour) * 1_000_000


def compute_cost_comparison(
    tokens_per_sec: float,
    hardware_cost_per_hour: float = 0.0,
    api_prices: dict[str, float] | None = None,
) -> CostComparison:
    """Compute self-hosting cost and compare against well-known API prices."""
    if api_prices is None:
        api_prices = API_PRICES
    self_cost = compute_cost_per_1m(tokens_per_sec, hardware_cost_per_hour)
    return CostComparison(
        self_hosting_usd_per_1m=self_cost,
        api_comparisons=[(name, price) for name, price in api_prices.items()],
    )
