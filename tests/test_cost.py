import pytest
from modelbench.cost import (
    compute_cost_per_1m,
    compute_cost_comparison,
    CostComparison,
    API_PRICES,
)


# --- compute_cost_per_1m ---

def test_cost_per_1m_owned_hardware_is_zero():
    # No hardware cost means $0 per token
    assert compute_cost_per_1m(tokens_per_sec=45.0, hardware_cost_per_hour=0.0) == 0.0


def test_cost_per_1m_rented_hardware():
    # $1/hr, 3600 tokens/hr (1 tok/s) → $1 / 3600 tokens * 1M = $277.78
    result = compute_cost_per_1m(tokens_per_sec=1.0, hardware_cost_per_hour=1.0)
    assert result == pytest.approx(277.78, rel=0.01)


def test_cost_per_1m_realistic():
    # 45 tok/s, $1/hr → 45 * 3600 = 162000 tok/hr → $1/162000 * 1M ≈ $6.17
    result = compute_cost_per_1m(tokens_per_sec=45.0, hardware_cost_per_hour=1.0)
    assert result == pytest.approx(6.17, rel=0.01)


def test_cost_per_1m_zero_tokens_per_sec():
    # Guard against division by zero
    assert compute_cost_per_1m(tokens_per_sec=0.0, hardware_cost_per_hour=1.0) == 0.0


# --- API_PRICES ---

def test_api_prices_contains_gpt4o_mini():
    assert "GPT-4o mini" in API_PRICES
    assert API_PRICES["GPT-4o mini"] > 0


def test_api_prices_contains_gpt4o():
    assert "GPT-4o" in API_PRICES
    assert API_PRICES["GPT-4o"] > API_PRICES["GPT-4o mini"]


# --- compute_cost_comparison ---

def test_cost_comparison_self_hosting_usd():
    cc = compute_cost_comparison(tokens_per_sec=45.0, hardware_cost_per_hour=0.0)
    assert cc.self_hosting_usd_per_1m == 0.0


def test_cost_comparison_includes_api_comparisons():
    cc = compute_cost_comparison(tokens_per_sec=45.0)
    assert len(cc.api_comparisons) >= 2


def test_cost_comparison_api_names_and_prices():
    cc = compute_cost_comparison(tokens_per_sec=45.0)
    names = [name for name, _ in cc.api_comparisons]
    assert "GPT-4o mini" in names
    assert "GPT-4o" in names


def test_cost_comparison_custom_api_prices():
    custom = {"FakeAPI": 5.00}
    cc = compute_cost_comparison(tokens_per_sec=10.0, api_prices=custom)
    assert cc.api_comparisons == [("FakeAPI", 5.00)]


def test_cost_comparison_rented_hardware():
    cc = compute_cost_comparison(tokens_per_sec=45.0, hardware_cost_per_hour=1.0)
    assert cc.self_hosting_usd_per_1m == pytest.approx(6.17, rel=0.01)


def test_cost_comparison_is_frozen():
    cc = compute_cost_comparison(tokens_per_sec=45.0)
    with pytest.raises((AttributeError, TypeError)):
        cc.self_hosting_usd_per_1m = 999  # type: ignore
