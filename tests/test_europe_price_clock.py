"""Synthetic, deterministic tests of Europe market clock and index builder."""
from datetime import date
import math

import numpy as np
import pandas as pd
import pytest

from pricing.europe_price_clock import (
    adjusted_quote, level_at, month_label, month_number, read_index,
)


def release(months=None):
    if months is None:
        months = [
            {"month": month_label(2026 * 12 + m - 1),
             "index": 100.0 * (1.01 ** (m - 1)), "n": 1400, "overlap": 36}
            for m in range(1, 9)
        ]
    return {"schema_version": 1, "status": "validated",
            "countries": {"DE": {"months": months}}}


def test_same_month_requires_no_index_and_no_fake_daily_adjustment():
    price, meta = adjusted_quote(10000, "2026-10-10", date(2026, 10, 29), "DE",
                                 index={"status": "unavailable"})
    assert price == 10000
    assert meta["status"] == "snapshot"


def test_absent_index_never_returns_a_false_date_adjustment():
    price, meta = adjusted_quote(10000, "2026-10-10", date(2026, 11, 1), "DE",
                                 index={"status": "unavailable"})
    assert price is None
    assert meta["status"] == "unavailable"


def test_fisher_observed_ratio():
    price, meta = adjusted_quote(10000, "2026-01-10", date(2026, 2, 3), "DE",
                                 index=release())
    assert price == pytest.approx(10100)
    assert meta["status"] == "observed"


def test_missing_month_uses_log_interpolation_not_linear_prices():
    r = release([
        {"month": "2026-01", "index": 100.0, "n": 1400},
        {"month": "2026-03", "index": 121.0, "n": 1400},
    ])
    point = level_at(r, "DE", "2026-02")
    assert point["method"] == "log_interpolation"
    assert point["level"] == pytest.approx(110.0)


def test_long_gap_is_unsupported():
    r = release([
        {"month": "2026-01", "index": 100.0, "n": 1400},
        {"month": "2026-08", "index": 121.0, "n": 1400},
    ])
    assert "error" in level_at(r, "DE", "2026-04")


def test_damped_holt_works_in_both_directions():
    r = release()
    future = level_at(r, "DE", "2026-11")
    assert future["method"] == "holt_forecast"
    assert future["level"] > 100.0 * 1.01 ** 7
    past = level_at(r, "DE", "2025-12")
    assert past["method"] == "holt_forecast"
    assert past["level"] < 100.0
    assert future["lower"] <= future["level"] <= future["upper"]


def test_holt_requires_six_adjacent_months_and_twelve_month_horizon():
    assert "error" in level_at(release(release()["countries"]["DE"]["months"][:5]),
                               "DE", "2026-06")
    assert "error" in level_at(release(), "DE", "2028-12")


def test_index_needs_real_country_support():
    price, meta = adjusted_quote(12000, "2026-01-10", "2026-03-02",
                                 "FR", index=release())
    assert price is None
    assert meta["status"] == "unavailable"


def test_builder_refuses_one_month_snapshot():
    from scripts.build_europe_hedonic_fisher import build
    data = pd.DataFrame([{
        "id": f"vehicle-{i}", "source_site": "portal", "seller_country": "DE",
        "make": "VW", "model": "Golf", "price_eur": 12000 + i,
        "mileage_km": 80000, "reg_year": 2020, "power_hp": 110,
        "first_seen_at": "2026-10-07", "fuel": "petrol",
        "transmission": "manual", "body_type": "hatchback",
    } for i in range(100)])
    result = build(data, min_month_rows=1, min_months=2)
    assert result["status"] == "unavailable"
    assert result["countries"] == {}
    assert result["coverage"] == [{"country": "DE", "month": "2026-10", "n": 100}]


def test_no_index_file_fails_closed(tmp_path):
    assert read_index(tmp_path / "missing.json")["status"] == "unavailable"


def test_invalid_month_rejected():
    with pytest.raises(ValueError):
        month_number("2026-13")
    with pytest.raises(ValueError):
        adjusted_quote(float("nan"), "2026-10-10", "2026-10-11", "DE")
