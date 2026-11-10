"""Eurostat-anchored conditional Bayesian make/model residual scenarios.

National HICP levels have years of monthly observations. The make/model
listing dataset currently supplies a single uncertain Sep-Oct 2026 contrast.
An assumed stationary Gaussian AR(1) monthly relative log-growth state is
conditioned on this one model observation.

This is a two-stage empirical-Bayes APPROXIMATION, not joint Bayesian model
inference, a model-specific observed historical index, or depreciation.
Model scenarios never alter production prices.
"""
from __future__ import annotations
import math
from pricing.europe_eurostat import eurostat_level_at
from pricing.europe_price_clock import month_label, month_number

DEFAULT_PHI = 0.65
DEFAULT_STATIONARY_SD = 0.035
DEFAULT_EXTRA_MEASUREMENT_SD = 0.02
Z_95 = 1.959963984540054
MAX_BACKCAST_MONTHS = 72
MAX_FORECAST_MONTHS = 12


def conditional_integrated_effect(anchor, target, observation, measured_effect,
                                  measurement_sd, *, phi=DEFAULT_PHI,
                                  stationary_sd=DEFAULT_STATIONARY_SD):
    """Conditional Gaussian AR(1) integral on a monthly grid.

    x_t has covariance sigma^2 * phi^abs(t-s). A noisy measurement of x_obs
    updates all states, but distant ones remain determined by the prior.
    """
    anchor, target, observation = map(month_number, (anchor, target, observation))
    y, sd, phi, sigma = map(float, (
        measured_effect, measurement_sd, phi, stationary_sd))
    if (not all(math.isfinite(v) for v in (y, sd, phi, sigma))
            or sd <= 0 or sigma <= 0 or abs(phi) >= 1 or abs(y) > .4):
        raise ValueError("Invalid AR(1) model settings or measured pulse")
    if target == anchor:
        return {"mean_log": 0.0, "var_log": 0.0,
                "months_from_anchor": 0, "method": "anchor"}
    direction = 1 if target > anchor else -1
    months = (list(range(anchor + 1, target + 1)) if direction == 1
              else list(range(target + 1, anchor + 1)))
    n = len(months)
    if n > MAX_BACKCAST_MONTHS:
        raise ValueError("Model scenario horizon exceeds conservative limit")
    var = sigma * sigma
    cross = direction * var * sum(phi ** abs(k - observation) for k in months)
    # Exact variance of a sum of n stationary AR(1) states, O(n).
    sum_cov = n + 2.0 * sum((n - d) * phi ** d for d in range(1, n))
    unconditional_var = var * sum_cov
    denominator = var + sd * sd
    mean = (cross / denominator) * y
    posterior_var = max(0.0, unconditional_var - cross * cross / denominator)
    return {"mean_log": mean, "var_log": posterior_var,
            "months_from_anchor": target - anchor,
            "method": "conditional_gaussian_ar1"}


def model_scenario_series(eurostat_manifest, pilot, *, lookback_months=24,
                          forward_months=6, phi=DEFAULT_PHI,
                          stationary_sd=DEFAULT_STATIONARY_SD,
                          extra_measurement_sd=DEFAULT_EXTRA_MEASUREMENT_SD):
    """National observed history plus model-specific posterior scenarios.

    The selected pilot is a research-only lookup_pulse record. Never pass
    these inferred model-specific adjustment factors to production quotes.
    """
    if not isinstance(pilot, dict) or pilot.get("status") != "research_only":
        return {"status": "unavailable",
                "reason": "No qualifying make/model residual evidence."}
    if not 1 <= lookback_months <= MAX_BACKCAST_MONTHS:
        raise ValueError("Invalid historical display horizon")
    if not 0 <= forward_months <= MAX_FORECAST_MONTHS:
        raise ValueError("Invalid extrapolation horizon")
    if not (math.isfinite(float(extra_measurement_sd))
            and extra_measurement_sd >= 0):
        raise ValueError("Invalid measurement uncertainty")
    country = str(pilot.get("country", "")).upper()
    early = str(pilot.get("early_month", ""))
    late = str(pilot.get("late_month", ""))
    if month_number(late) - month_number(early) != 1:
        return {"status": "unavailable",
                "reason": "Requires a genuine consecutive-month listing comparison."}
    if (eurostat_manifest.get("status") != "validated"
            or country not in eurostat_manifest.get("countries", {})):
        return {"status": "unavailable",
                "reason": "A validated national Eurostat index is required."}
    try:
        y = float(pilot["relative_log_change"])
        pulse_sd = float(pilot["posterior_sd_log"])
        if (not math.isfinite(y) or not math.isfinite(pulse_sd)
                or not 0 < pulse_sd < .2 or abs(y) > .4):
            raise ValueError
    except (ValueError, TypeError, KeyError):
        return {"status": "unavailable",
                "reason": "Posterior make/model residual evidence is invalid."}
    anchor = month_number(early)
    baseline = eurostat_level_at(eurostat_manifest, country, early,
                                allow_holt=False)
    if "error" in baseline:
        return {"status": "unavailable",
                "reason": "No official national index in listing-reference month."}
    base_level = float(baseline["level"])
    # Explicit uncertainty floor for selection/summary effects omitted by
    # the two-stage empirical-Bayes pipeline; not learned from time history.
    effective_sd = math.hypot(pulse_sd, extra_measurement_sd)
    observation = month_number(late)
    points = []
    for t in range(anchor - lookback_months, anchor + forward_months + 1):
        month = month_label(t)
        national = eurostat_level_at(eurostat_manifest, country, month)
        if "error" in national:
            continue
        post = conditional_integrated_effect(
            early, month, late, y, effective_sd,
            phi=phi, stationary_sd=stationary_sd)
        mean = post["mean_log"]
        sd = math.sqrt(post["var_log"])
        national_index = 100.0 * float(national["level"]) / base_level
        if not math.isfinite(national_index) or not .1 < national_index < 10000:
            continue
        if t == anchor:
            kind = "normalized_anchor"
        elif t == observation:
            kind = "single_observed_pulse_month_posterior"
        elif t < observation:
            kind = "assumption_driven_backcast"
        else:
            kind = "assumption_driven_forward_scenario"
        points.append({
            "month": month,
            "national_index": round(national_index, 5),
            "model_scenario_median": round(national_index * math.exp(mean), 5),
            "model_scenario_lower": round(
                national_index * math.exp(mean - Z_95 * sd), 5),
            "model_scenario_upper": round(
                national_index * math.exp(mean + Z_95 * sd), 5),
            "posterior_relative_log": round(mean, 7),
            "posterior_relative_sd": round(sd, 7),
            "national_data_method": national["method"],
            "model_data_method": kind,
            "model_observed": False,
        })
    if not points:
        return {"status": "unavailable",
                "reason": "No overlapping national and model-scenario dates."}
    record = eurostat_manifest["countries"][country]
    return {
        "status": "research_only",
        "country": country,
        "make": pilot.get("make"), "model": pilot.get("model"),
        "national_source": "Eurostat HICP CP07112, I25, 2025=100",
        "national_first_published_month": record.get("first_month"),
        "national_last_published_month": record.get("last_month"),
        "anchor_month": early,
        "one_model_observation_month": late,
        "observed_model_transitions": 1,
        "observed_model_measurement_log": y,
        "observed_model_measurement_sd_effective": effective_sd,
        "prior_phi": float(phi),
        "prior_stationary_sd_monthly_log": float(stationary_sd),
        "posterior_type": "two_stage_conditional_gaussian_ar1",
        "uncertainty_warning": (
            "Bands condition on an assumed AR(1) prior and one approximate "
            "Bayesian-pulse estimate. They omit prior-parameter estimation, "
            "listing-selection and national-index forecast uncertainty. "
            "They are not calibrated car-price intervals."),
        "depreciation_warning": (
            "These are market-time scenarios for a constant-quality basket, "
            "NOT vehicle age- or mileage-specific depreciation."),
        "applied_to_quote": False,
        "points": points,
    }
