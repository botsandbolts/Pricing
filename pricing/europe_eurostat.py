"""Eurostat ECOICOP v2 used-car HICP, official monthly INDEX (I25, 2025=100).

HICP is a *consumer retail price-index proxy*, not Car Price Studio's
own hedonic/Fisher index or a direct index of classified ASK advertisements.
Offline, dated artifact only. Never fabricate a data point.
"""
from __future__ import annotations

from datetime import date
import json
import math
from pathlib import Path
from pricing.europe_price_clock import (
    _holt_forecast, month_label, month_number, MAX_INTERPOLATION_GAP_MONTHS,
    MAX_EXTRAPOLATION_MONTHS,
)

PATH = Path(__file__).resolve().parents[1] / "models" / "europe-ask" / "eurostat_hicp.json"
SOURCE_URL = "https://ec.europa.eu/eurostat/databrowser/view/prc_hicp_minr/default/table?lang=en"
SUPPORTED = frozenset(("DE", "FR", "NL", "BE", "AT", "ES", "IT", "PL"))
COUNTRY_NAMES = {"DE": "Germany", "FR": "France", "NL": "Netherlands",
                 "BE": "Belgium", "AT": "Austria", "ES": "Spain",
                 "IT": "Italy", "PL": "Poland"}


def read_eurostat(path=PATH):
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        if (manifest.get("status") != "validated" or
                manifest.get("schema_version") != 1 or
                manifest.get("dataset") != "prc_hicp_minr" or
                manifest.get("unit") != "I25" or
                manifest.get("coicop18") != "CP07112" or
                not isinstance(manifest.get("countries"), dict)):
            raise ValueError("Missing verified Eurostat used-car HICP contract")
        return manifest
    except (OSError, ValueError, TypeError):
        return {"status": "unavailable", "countries": {},
                "message": "Validated Eurostat used-car HICP data have not been packaged."}


def _observations(manifest, country):
    if str(country).upper() not in SUPPORTED:
        return {}
    record = manifest.get("countries", {}).get(str(country).upper(), {})
    months = record.get("months", []) if isinstance(record, dict) else []
    observed = {}
    for row in months:
        try:
            key = month_number(row["month"])
            value = float(row["index"])
            if math.isfinite(value) and value > 0:
                observed[key] = {"level": value}
        except (KeyError, ValueError, TypeError):
            continue
    return dict(sorted(observed.items()))


def eurostat_level_at(manifest, country, when, *, allow_holt=True):
    obs = _observations(manifest, country)
    key = month_number(when)
    if key in obs:
        return {"level": obs[key]["level"], "method": "observed"}
    if len(obs) < 12:
        return {"error": "Fewer than 12 actual monthly second-hand-car HICP observations."}
    keys = sorted(obs)
    lower = [x for x in keys if x < key]
    upper = [x for x in keys if x > key]
    if lower and upper:
        left, right = lower[-1], upper[0]
        if right - left > MAX_INTERPOLATION_GAP_MONTHS:
            return {"error": "The published statistical series has a long missing gap."}
        fraction = (key-left)/(right-left)
        return {"level": math.exp(math.log(obs[left]["level"])*(1-fraction)+
                                   math.log(obs[right]["level"])*fraction),
                "method": "log_interpolation",
                "between": [month_label(left), month_label(right)]}
    if not allow_holt:
        return {"error": "No official Eurostat monthly value is published for that date."}
    forecast = _holt_forecast(obs, key)
    if forecast:
        forecast["source_last_published_month"] = month_label(keys[-1])
        return forecast
    return {"error": f"Outside official data, and beyond the supported Holt window "
                     f"({MAX_EXTRAPOLATION_MONTHS} months)."}


def eurostat_adjusted_quote(base_price, snapshot, valuation_date, country,
                            manifest=None, *, allow_holt=True):
    base_month = month_label(month_number(snapshot))
    target_month = month_label(month_number(valuation_date))
    base = float(base_price)
    if not math.isfinite(base) or base <= 0:
        raise ValueError("Invalid base Euro ASK price")
    meta = {"status": "unavailable", "source": "Eurostat HICP · second-hand motor cars",
            "dataset": "prc_hicp_minr", "coicop18": "CP07112", "unit": "I25",
            "country": str(country).upper(), "base_month": base_month,
            "target_month": target_month,
            "limitation": "Official consumer second-hand-car index; proxy for "
                          "listing ASK movements, not a model-specific hedonic index."}
    if base_month == target_month:
        return base, {**meta, "status": "snapshot", "factor": 1.0}
    manifest = read_eurostat() if manifest is None else manifest
    if manifest.get("status") != "validated":
        return None, {**meta, "reason": manifest.get("message") or
                      "Eurostat data are not validated/packaged."}
    if meta["country"] not in SUPPORTED or meta["country"] not in manifest["countries"]:
        return None, {**meta, "reason": "Eurostat country index not available."}
    a = eurostat_level_at(manifest, country, snapshot, allow_holt=allow_holt)
    b = eurostat_level_at(manifest, country, valuation_date, allow_holt=allow_holt)
    if "error" in a or "error" in b:
        return None, {**meta, "reason": a.get("error") or b.get("error")}
    factor = b["level"] / a["level"]
    if not math.isfinite(factor) or not 0.35 <= factor <= 2.5:
        return None, {**meta, "reason": "Official-index movement exceeds sanity limits."}
    status = ("forecast" if "holt_forecast" in (a["method"], b["method"])
              else "interpolated" if "log_interpolation" in (a["method"], b["method"])
              else "observed")
    meta.update({"status": status, "factor": factor,
                 "base_index_method": a["method"],
                 "target_index_method": b["method"],
                 "source_last_published_month":
                    manifest["countries"][meta["country"]].get("last_month"),
                 "source_url": SOURCE_URL,
                 "data_updated": manifest.get("generated_at")})
    # The band covers only trend estimation, not a price prediction interval.
    if a["method"] == "observed" and b["method"] == "holt_forecast":
        meta["trend_only_range"] = [base*b["lower"]/a["level"],
                                    base*b["upper"]/a["level"]]
    return base*factor, meta
