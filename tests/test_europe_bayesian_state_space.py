"""Explicitly separate years of official HICP data from one model pulse."""
import math
import pytest
from pricing.europe_bayesian_state_space import (
    conditional_integrated_effect, model_scenario_series)
from pricing.europe_price_clock import month_label, month_number


def eurostat_fixture():
    first = month_number("2020-01")
    last = month_number("2026-09")
    return {"schema_version": 1, "status": "validated",
            "countries": {"DE": {
                "first_month": "2020-01", "last_month": "2026-09",
                "months": [{"month": month_label(t),
                            "index": round(70 * (1.005 ** (t - first)), 4)}
                           for t in range(first, last+1)]}}}


def pilot_fixture(effect=0.075, sd=0.012):
    return {"status": "research_only", "country": "DE",
            "make": "volkswagen", "model": "golf",
            "early_month": "2026-09", "late_month": "2026-10",
            "relative_log_change": effect,
            "posterior_sd_log": sd,
            "applied_to_quote": False}


def test_one_month_conditional_posterior_matches_conjugate_math():
    sigma = .035
    measurement_sd = .02
    y = .1
    point = conditional_integrated_effect(
        "2026-09", "2026-10", "2026-10", y,
        measurement_sd, stationary_sd=sigma)
    v = sigma**2
    assert point["mean_log"] == pytest.approx(v/(v+measurement_sd**2)*y)
    assert point["var_log"] == pytest.approx(
        v*measurement_sd**2/(v+measurement_sd**2))
    assert point["months_from_anchor"] == 1


def test_long_official_history_does_not_create_past_model_observations():
    result = model_scenario_series(eurostat_fixture(), pilot_fixture(),
                                   lookback_months=60, forward_months=6)
    assert result["status"] == "research_only"
    assert result["observed_model_transitions"] == 1
    assert result["national_first_published_month"] == "2020-01"
    assert result["national_last_published_month"] == "2026-09"
    points = {x["month"]:x for x in result["points"]}
    assert len(points) == 67  # 60 historical + anchor + six future
    assert points["2026-09"]["national_index"] == 100
    assert points["2026-09"]["model_scenario_median"] == 100
    assert points["2026-09"]["model_scenario_lower"] == 100
    assert points["2026-09"]["model_scenario_upper"] == 100
    assert points["2025-01"]["national_data_method"] == "observed"
    assert points["2025-01"]["model_data_method"] == "assumption_driven_backcast"
    assert points["2025-01"]["model_observed"] is False
    assert points["2026-10"]["national_data_method"] == "holt_forecast"
    assert points["2026-10"]["model_data_method"] == "single_observed_pulse_month_posterior"
    assert points["2026-10"]["model_observed"] is False
    assert points["2026-10"]["model_scenario_median"] > points["2026-10"]["national_index"]
    assert points["2027-03"]["model_data_method"] == "assumption_driven_forward_scenario"
    assert result["applied_to_quote"] is False
    assert all(x["model_scenario_lower"]<=x["model_scenario_median"]<=x["model_scenario_upper"]
               for x in result["points"])
    # Posterior bands widen drastically away from the single observed pulse.
    def relative_width(row):
        return row["model_scenario_upper"]/row["model_scenario_lower"]
    assert relative_width(points["2021-09"])>relative_width(points["2026-10"])


def test_earlier_model_growth_is_backcast_with_correct_sign():
    params = dict(anchor="2026-09", observation="2026-10",
                  measured_effect=0.08, measurement_sd=0.015)
    before=conditional_integrated_effect(target="2026-08",**params)
    after=conditional_integrated_effect(target="2026-10",**params)
    assert before["mean_log"]<0
    assert after["mean_log"]>0


def test_no_relative_model_trend_leaks_into_production_without_evidence():
    assert model_scenario_series(eurostat_fixture(),None)["status"]=="unavailable"
    bad=pilot_fixture()
    bad["posterior_sd_log"]=float("nan")
    assert model_scenario_series(eurostat_fixture(),bad)["status"]=="unavailable"
    bad=pilot_fixture()
    bad["early_month"]="2026-07"
    assert model_scenario_series(eurostat_fixture(),bad)["status"]=="unavailable"
    assert model_scenario_series({"status":"unavailable"},pilot_fixture())["status"]=="unavailable"
    with pytest.raises(ValueError):
        model_scenario_series(eurostat_fixture(),pilot_fixture(),
                              lookback_months=73)


def test_posterior_conditional_sd_is_not_claimed_calibrated_car_interval():
    result=model_scenario_series(eurostat_fixture(),pilot_fixture())
    assert "not calibrated car-price intervals" in result["uncertainty_warning"]
    assert "NOT vehicle age" in result["depreciation_warning"]
    assert result["posterior_type"] == "two_stage_conditional_gaussian_ar1"


def test_zero_observed_pulse_means_national_median_but_not_zero_uncertainty():
    result=model_scenario_series(eurostat_fixture(),pilot_fixture(effect=0))
    for p in result["points"]:
        assert p["model_scenario_median"]==pytest.approx(p["national_index"],abs=.00001)
    first=result["points"][0]
    assert first["model_scenario_upper"]>first["model_scenario_lower"]
