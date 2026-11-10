"""Fetch official Eurostat second-hand motor-car monthly HICP (2025=100).

Uses Eurostat's JSON-stat 2.0 dissemination API, not scraped webpages.
Validates exact dataset / classification / units and preserves only observed data.
Run offline in GitHub Actions to keep app startup reliable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
from urllib.parse import urlencode

import requests

DATASET = "prc_hicp_minr"
CODE = "CP07112"  # ECOICOP 2018 classification: second-hand motor cars
UNIT = "I25"  # Index 2025 = 100, NOT growth rates
BASE_URL = ("https://ec.europa.eu/eurostat/api/dissemination/"
            "statistics/1.0/data/prc_hicp_minr")
COUNTRIES = ("DE", "FR", "NL", "BE", "AT", "ES", "IT", "PL")


def build_url(country):
    q = {"lang": "EN", "freq": "M", "unit": UNIT, "coicop18": CODE,
         "geo": country, "sinceTimePeriod": "2020-01"}
    return BASE_URL + "?" + urlencode(q)


def _category_index(obj):
    category = obj.get("category", {})
    idx = category.get("index", {})
    if isinstance(idx, dict):
        return {str(k): int(v) for k, v in idx.items()}
    if isinstance(idx, list):
        return {str(v): i for i, v in enumerate(idx)}
    raise ValueError("Malformed Eurostat JSON-stat category index")


def unpack_jsonstat(obj, *, expected_geo):
    """Decode flattened JSON-stat observations via full dimension coordinates."""
    dimensions = obj.get("id")
    sizes = obj.get("size")
    values = obj.get("value")
    if not isinstance(dimensions, list) or not isinstance(sizes, list) or len(dimensions) != len(sizes):
        raise ValueError("Not a JSON-stat dataset")
    if not isinstance(values, (dict, list)):
        raise ValueError("Eurostat omitted numeric observations")
    dimensions_idx = {}
    for dim in dimensions:
        item = obj.get("dimension", {}).get(dim)
        if not isinstance(item, dict):
            raise ValueError("Missing JSON-stat dimension " + dim)
        dimensions_idx[dim] = _category_index(item)
    # Exact semantic safeguards against upstream metadata changes.
    selectors = {"freq": "M", "unit": UNIT, "coicop18": CODE, "geo": expected_geo}
    for dim, value in selectors.items():
        if dim not in dimensions_idx or value not in dimensions_idx[dim]:
            raise ValueError(f"Eurostat response missing {dim}={value}")
    if "time" in dimensions_idx:
        time_key = "time"
    elif "TIME_PERIOD" in dimensions_idx:
        time_key = "TIME_PERIOD"
    else:
        raise ValueError("No monthly time dimension")
    selected = {dim: dimensions_idx[dim][code] for dim, code in selectors.items()}
    months = []
    for month, month_pos in dimensions_idx[time_key].items():
        if len(month) != 7 or month[4] != "-" or not ("01" <= month[5:] <= "12"):
            continue
        indexes = [month_pos if dim == time_key else selected[dim]
                   for dim in dimensions]
        pos = 0
        for ix, size in zip(indexes, sizes):
            pos = pos * int(size) + int(ix)
        raw = values.get(str(pos)) if isinstance(values, dict) else (
            values[pos] if pos < len(values) else None)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and 10 <= value <= 1000:
            months.append({"month": month, "index": round(value, 4)})
    months.sort(key=lambda row: row["month"])
    if len({x["month"] for x in months}) != len(months):
        raise ValueError("Duplicate Eurostat month")
    return months


def fetch_series(country, *, timeout=35, session=None):
    get = (session or requests).get
    url = build_url(country)
    response = get(url, headers={"Accept": "application/json",
                                 "User-Agent": "CarPriceStudioResearch/1.0 (public data)"},
                   timeout=timeout)
    response.raise_for_status()
    obj = response.json()
    months = unpack_jsonstat(obj, expected_geo=country)
    if len(months) < 12:
        raise ValueError(f"{country}: fewer than 12 published used-car index months")
    return {"months": months, "first_month": months[0]["month"],
            "last_month": months[-1]["month"], "source_query": url}


def build(*, min_countries=1):
    manifest = {"schema_version": 1, "status": "unavailable",
                "dataset": DATASET, "coicop18": CODE, "unit": UNIT,
                "reference": "Second-hand motor cars, national HICP 2025=100",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source_url": BASE_URL, "countries": {}, "errors": {}}
    for country in COUNTRIES:
        try:
            manifest["countries"][country] = fetch_series(country)
            row = manifest["countries"][country]
            print(country, len(row["months"]), row["first_month"], row["last_month"], flush=True)
        except (ValueError, requests.RequestException, KeyError, TypeError) as exc:
            manifest["errors"][country] = str(exc)[:240]
            print(country, "unavailable:", exc, flush=True)
    if len(manifest["countries"]) >= min_countries:
        manifest["status"] = "validated"
    manifest["message"] = (f"Eurostat second-hand motor-car monthly HICP: "
                           f"{len(manifest['countries'])}/{len(COUNTRIES)} usable countries.")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/europe-ask/eurostat_hicp.json")
    args = parser.parse_args()
    manifest = build()
    outfile = Path(args.output)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    if outfile.is_file():
        try:
            previous=json.loads(outfile.read_text(encoding="utf-8"))
            # Ignore the time of download when official observations are unchanged.
            # Avoid a meaningless new commit every scheduled run.
            if (previous.get("status")=="validated"
                    and previous.get("countries")==manifest["countries"]
                    and previous.get("errors")==manifest["errors"]):
                manifest["generated_at"]=previous.get("generated_at",manifest["generated_at"])
                print("Official series unchanged; preserving previous timestamp.",flush=True)
        except (OSError,ValueError,TypeError):
            pass
    outfile.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n",
                       encoding="utf-8")
    print(json.dumps({"status": manifest["status"],
                      "countries": {k: len(v["months"]) for k,v in manifest["countries"].items()},
                      "errors": manifest["errors"],
                      "output": str(outfile)}, sort_keys=True))
    if manifest["status"] != "validated":
        raise SystemExit("Eurostat release is unavailable; do not publish a flat/fake index.")


if __name__ == "__main__":
    main()
