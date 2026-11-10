"""European asking-price market clock: observed hedonic Fisher + damped Holt scenarios.

The index is independently trained and reviewed. No index file means NO adjustment.
Prices are listing/ASK prices, not transactions. Monthly precision only.
"""
from __future__ import annotations

from datetime import date
import json
import math
from pathlib import Path

INDEX_PATH = Path(__file__).resolve().parents[1] / "models/europe-ask/hedonic_fisher.json"
MIN_OBSERVED_MONTHS_FOR_HOLT = 6
MAX_EXTRAPOLATION_MONTHS = 12
MAX_INTERPOLATION_GAP_MONTHS = 3


def month_number(value):
    """Integer year-month coordinate, without 30-day month approximations."""
    if isinstance(value, date):
        return value.year * 12 + value.month - 1
    s = str(value)
    year, month = int(s[:4]), int(s[5:7])
    if not (1950 <= year <= 2100 and 1 <= month <= 12):
        raise ValueError("Unsupported year/month")
    return year * 12 + month - 1


def month_label(number):
    year, index = divmod(int(number), 12)
    return f"{year:04d}-{index+1:02d}"


def read_index(path=INDEX_PATH):
    """An absent or invalid artifact is a disabled index, never a fictional flat one."""
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if obj.get("schema_version") != 1 or obj.get("status") not in ("validated", "unavailable"):
            raise ValueError("Invalid release status")
        if not isinstance(obj.get("countries"), dict):
            raise ValueError("No countries")
        return obj
    except (OSError, ValueError, TypeError, KeyError):
        return {"schema_version": 1, "status": "unavailable", "countries": {},
                "message": "No validated, multi-month hedonic Fisher index is published yet."}


def _observations(index, country):
    country_data = index.get("countries", {}).get(str(country).upper(), {})
    rows = country_data.get("months", []) if isinstance(country_data, dict) else []
    result = {}
    for row in rows:
        try:
            x = month_number(row["month"])
            y = float(row["index"])
            if math.isfinite(y) and y > 0 and int(row["n"]) >= 800:
                result[x] = {"level": y, "n": int(row["n"]),
                             "overlap": int(row.get("overlap", 0))}
        except (KeyError, ValueError, TypeError, OverflowError):
            continue
    return dict(sorted(result.items()))


def _holt_fit(log_levels, alpha=0.4, beta=0.18, phi=0.8):
    """Damped Holt on log index. Never interpret a fitted trend as observed data."""
    if len(log_levels) < MIN_OBSERVED_MONTHS_FOR_HOLT:
        return None
    level = float(log_levels[0])
    trend = float(log_levels[1] - log_levels[0])
    residuals = []
    for observed in log_levels[1:]:
        forecast = level + phi * trend
        residuals.append(float(observed - forecast))
        new_level = alpha * float(observed) + (1 - alpha) * forecast
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        level = new_level
    if not all(map(math.isfinite, [level, trend] + residuals)):
        return None
    # Diagnostic *model* uncertainty only. Never call this a vehicle-price interval.
    rms = math.sqrt(sum(x * x for x in residuals) / len(residuals))
    return level, trend, rms


def _holt_forecast(obs, target):
    keys = sorted(obs)
    if not keys or target in obs:
        return None
    before = target < keys[0]
    nearest = keys[0] if before else keys[-1]
    steps = abs(target - nearest)
    if steps == 0 or steps > MAX_EXTRAPOLATION_MONTHS:
        return None
    # Only the consecutive observed tail (or head for backward forecasting).
    # For future forecasts walk backwards from latest; for historical forecasts
    # walk forwards from earliest. Then reverse for fitting in forecast direction.
    ordered = list(reversed(keys)) if not before else keys
    contiguous = [ordered[0]]
    for k in ordered[1:]:
        if abs(k - contiguous[-1]) != 1:
            break
        contiguous.append(k)
    if len(contiguous) < MIN_OBSERVED_MONTHS_FOR_HOLT:
        return None
    recent = contiguous[:min(18, len(contiguous))]
    # The latest observed boundary is the last fitted point in either direction.
    series = list(reversed(recent))
    log_levels = [math.log(obs[m]["level"]) for m in series]
    fitted = _holt_fit(log_levels)
    if fitted is None:
        return None
    level, trend, rms = fitted
    phi = 0.8
    drift = trend * sum(phi ** k for k in range(1, steps + 1))
    # Guard against unreasonable extrapolation, not a fitted observation.
    drift = max(-0.35, min(0.35, drift))
    # Anchor the Holt trend to the actual boundary observation. The fitted
    # level is used to estimate the slope and forecast errors, not to introduce
    # an artificial jump at the sample boundary.
    forecast = math.exp(math.log(obs[nearest]["level"]) + drift)
    band = 1.282 * rms * math.sqrt(steps)
    return {"level": forecast, "method": "holt_forecast", "months_ahead": steps,
            "lower": forecast * math.exp(-band), "upper": forecast * math.exp(band),
            "rms": rms}


def level_at(index, country, chosen):
    """Monthly observed, log-interpolated, or bounded Holt extrapolated index level."""
    obs = _observations(index, country)
    key = month_number(chosen)
    if key in obs:
        return {"level": obs[key]["level"], "method": "observed", "n": obs[key]["n"]}
    if len(obs) < 2:
        return {"error": "Insufficient monthly listing observations for this country."}
    keys = list(obs)
    lower = [k for k in keys if k < key]
    upper = [k for k in keys if k > key]
    if lower and upper:
        l, r = lower[-1], upper[0]
        if r - l > MAX_INTERPOLATION_GAP_MONTHS:
            return {"error": "Historical gap exceeds the interpolation limit."}
        fraction = (key - l) / (r - l)
        value = math.exp((1 - fraction) * math.log(obs[l]["level"])
                         + fraction * math.log(obs[r]["level"]))
        return {"level": value, "method": "log_interpolation",
                "between": [month_label(l), month_label(r)]}
    forecast = _holt_forecast(obs, key)
    if forecast:
        return forecast
    return {"error": "A forecast needs at least six consecutive observed months "
                     "and is limited to twelve months outside the sample."}


def adjusted_quote(base_price, snapshot, valuation_date, country, index=None):
    """Returns date-adjusted ASK or None plus fully explicit provenance and status.

    Market index controls *quality-adjusted time movement only*. Age/mileage forecasts
    are separate and not silently imputed into this output.
    """
    base_month = month_label(month_number(snapshot))
    target_month = month_label(month_number(valuation_date))
    base = float(base_price)
    if not math.isfinite(base) or base <= 0:
        raise ValueError("Invalid base ASK price")
    if base_month == target_month:
        return base, {"status": "snapshot", "factor": 1.0,
                      "base_month": base_month, "target_month": target_month}
    index = read_index() if index is None else index
    if index.get("status") != "validated":
        return None, {"status": "unavailable", "reason": index.get("message") or
                      "No validated multi-month index is published.",
                      "base_month": base_month, "target_month": target_month}
    a = level_at(index, country, snapshot)
    b = level_at(index, country, valuation_date)
    if "error" in a or "error" in b:
        return None, {"status": "unavailable",
                      "reason": a.get("error") or b.get("error"),
                      "base_month": base_month, "target_month": target_month}
    factor = b["level"] / a["level"]
    if not (0.35 <= factor <= 2.5 and math.isfinite(factor)):
        return None, {"status": "unavailable", "reason": "Index adjustment outside safe bounds."}
    status = ("forecast" if "holt_forecast" in (a["method"], b["method"])
              else "interpolated" if "log_interpolation" in (a["method"], b["method"])
              else "observed")
    meta = {"status": status, "factor": factor, "base_month": base_month,
            "target_month": target_month, "base_index_method": a["method"],
            "target_index_method": b["method"], "country": str(country).upper()}
    # Trend uncertainty is not the vehicle model's confidence interval.
    if "lower" in b and "upper" in b and a["method"] == "observed":
        meta["trend_only_range"] = [base * b["lower"] / a["level"],
                                    base * b["upper"] / a["level"]]
    return base * factor, meta
