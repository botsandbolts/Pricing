"""Derive November-2026 market-adjusted ASK prices for every European listing.

Adds 'current_depreciated_price_eur' to the 2M-row original Parquet. This is
a COUNTRY-LEVEL, monthly Eurostat second-hand-car HICP adjustment relative to
LAST_VERIFIED_AT. It is NOT vehicle-specific age/mileage depreciation, not the
Bayesian research model, and not a transacted resale value.

Original price_eur and all source columns are preserved byte-for-byte in
semantic value. Rows without reliable country/date/index evidence get NULL,
never an invented price. An adjacent status/ratio/source-month field documents
exactly how every result was computed.

Source license: KTHESAT/european-used-car-market-2m-listings,
CC BY-NC-SA 4.0 (derived data subject to source license).

Usage:
    python -m scripts.add_europe_nov2026_adjusted_price
    python -m scripts.add_europe_nov2026_adjusted_price --source local.parquet

Requires: pandas numpy pyarrow requests
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pricing.europe_eurostat import (
    SOURCE_URL, SUPPORTED, eurostat_adjusted_quote, read_eurostat,
)
from pricing.europe_price_clock import month_number

SOURCE = (
    "https://huggingface.co/datasets/KTHESAT/"
    "european-used-car-market-2m-listings/resolve/main/"
    "data/train-00000-of-00001.parquet"
)
TARGET_MONTH = "2026-11"
OUTPUT = Path("artifacts/europe-nov2026/europe_2m_nov2026_adjusted.parquet")
REPORT = Path("artifacts/europe-nov2026/adjustment_report.json")
PRICE_FIELD = "current_depreciated_price_eur"
FACTOR_FIELD = "current_depreciation_factor"
STATUS_FIELD = "current_depreciation_method"
FROM_FIELD = "current_depreciation_source_month"
REQUIRED = frozenset(("price_eur", "seller_country", "last_verified_at"))
LICENSE = "CC BY-NC-SA 4.0"
MAX_PRICE_EUR = 1_000_000
STALENESS_MONTHS_FOR_CURRENT_TRAINING = 2
AGE_MONTHS_FIELD = "months_since_last_verified"
FRESHNESS_FIELD = "eligible_as_current_price_training_label"
AGE_STATUS_FIELD = "vehicle_age_mileage_adjustment_status"


def _source_file(source, directory):
    path = Path(source)
    if path.is_file():
        return path
    u = urlparse(str(source))
    if u.scheme not in ("https",) or u.hostname != "huggingface.co":
        raise ValueError("Source must be a local Parquet path or Hugging Face HTTPS URL")
    import requests
    saved = Path(directory) / "source.parquet"
    with requests.get(str(source), stream=True, timeout=(25, 180)) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            raise ValueError("Dataset URL responded with HTML rather than Parquet.")
        with saved.open("wb") as out:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    out.write(chunk)
    if saved.stat().st_size < 1000:
        raise ValueError("Dataset download is unexpectedly small.")
    return saved


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _cached_adjustment(country, source_month, *, manifest, target, cache):
    key = (country, source_month, target)
    if key in cache:
        return cache[key]
    if country not in SUPPORTED or country not in manifest.get("countries", {}):
        result = (float("nan"), "unsupported_country")
    elif month_number(source_month) > month_number(target):
        result = (float("nan"), "source_after_target")
    else:
        factor, meta = eurostat_adjusted_quote(
            1.0, source_month + "-01", target + "-01", country,
            manifest=manifest, allow_holt=True,
        )
        if factor is None or not math.isfinite(factor):
            result = (float("nan"), "index_unavailable")
        else:
            result = (float(factor), str(meta["status"]))
    cache[key] = result
    return result


def adjust_record_batch(batch, *, manifest, target=TARGET_MONTH, cache=None):
    """Preserve input schema while appending nullable price + audit fields."""
    if cache is None:
        cache = {}
    if manifest.get("status") != "validated":
        raise ValueError("Unvalidated Eurostat manifest; refusing to adjust listings.")
    if not set(REQUIRED).issubset(batch.schema.names):
        raise ValueError("Missing source fields: " +
                         ", ".join(sorted(REQUIRED - set(batch.schema.names))))
    for new in (PRICE_FIELD, FACTOR_FIELD, STATUS_FIELD, FROM_FIELD,
                AGE_MONTHS_FIELD, FRESHNESS_FIELD, AGE_STATUS_FIELD):
        if new in batch.schema.names:
            raise ValueError("Output fields already exist in source; refuse double adjustment.")
    n = batch.num_rows
    country = (
        batch.column("seller_country").to_pandas().astype("string")
        .str.strip().str.upper()
    )
    verified = pd.to_datetime(
        batch.column("last_verified_at").to_pandas(),
        utc=True, errors="coerce", format="mixed",
    )
    source_months = verified.dt.strftime("%Y-%m")
    target_coordinate = month_number(target)
    month_coordinates = np.array([
        month_number(x) if isinstance(x, str) and len(x) >= 7
        else -1 for x in source_months
    ], dtype=np.int32)
    months_elapsed = np.where(
        month_coordinates < 0, -1, target_coordinate - month_coordinates
    ).astype(np.int32)
    original = pd.to_numeric(
        batch.column("price_eur").to_pandas(), errors="coerce"
    ).to_numpy(dtype="float64")
    adjusted = np.full(n, np.nan, dtype="float64")
    factors = np.full(n, np.nan, dtype="float64")
    methods = np.full(n, "invalid_date", dtype=object)
    valid_price = np.isfinite(original) & (original > 0) & (original <= MAX_PRICE_EUR)
    valid_country = country.notna() & country.str.len().gt(0)
    valid_date = verified.notna()
    methods[~valid_price] = "invalid_price"
    methods[valid_price & ~valid_country.to_numpy(dtype=bool)] = "missing_country"
    eligible = valid_price & valid_country.to_numpy(dtype=bool) & valid_date.to_numpy(dtype=bool)
    if bool(np.any(eligible)):
        source_key = (
            country.fillna("").astype(str) + "|" +
            source_months.fillna("").astype(str)
        )
        for key in source_key[eligible].unique():
            country_value, observed_month = key.split("|", 1)
            ratio, method = _cached_adjustment(
                country_value, observed_month,
                manifest=manifest, target=target, cache=cache)
            belongs = eligible & (source_key.to_numpy() == key)
            methods[belongs] = method
            if math.isfinite(ratio):
                factors[belongs] = ratio
                adjusted[belongs] = np.round(original[belongs] * ratio, 2)
    valid_adjusted = np.isfinite(adjusted) & (adjusted > 0)
    adjusted[~valid_adjusted] = np.nan
    methods[np.isfinite(factors) & ~valid_adjusted] = "invalid_adjusted_value"
    month_strings = source_months.where(source_months.notna(), None).tolist()
    result = batch.append_column(
        PRICE_FIELD, pa.array(adjusted, type=pa.float64(), mask=~valid_adjusted)
    )
    result = result.append_column(
        FACTOR_FIELD, pa.array(factors, type=pa.float64(), mask=~np.isfinite(factors))
    )
    result = result.append_column(STATUS_FIELD, pa.array(methods, type=pa.string()))
    result = result.append_column(FROM_FIELD, pa.array(month_strings, type=pa.string()))
    # A five-year-old advert is NOT a current-day supervised target, even
    # after country HICP normalization. No invented actual current odometer.
    current_eligible = (
        valid_adjusted &
        (months_elapsed >= 0) &
        (months_elapsed <= STALENESS_MONTHS_FOR_CURRENT_TRAINING)
    )
    result = result.append_column(
        AGE_MONTHS_FIELD, pa.array(
            months_elapsed.astype("float64"),
            type=pa.float64(), mask=months_elapsed < 0
        )
    )
    result = result.append_column(
        FRESHNESS_FIELD, pa.array(current_eligible, type=pa.bool_())
    )
    result = result.append_column(
        AGE_STATUS_FIELD, pa.array(
            np.where(
                ~valid_adjusted, "missing_market_time_adjustment",
                np.where(
                    current_eligible, "recent_listing_no_physical_projection",
                    "stale_listing_age_mileage_not_observed"
                )
            ).tolist(), type=pa.string()
        )
    )
    return result


def enrich(source, output, report, *, manifest=None, target=TARGET_MONTH,
           batch_size=75_000):
    if month_number(target) != month_number(TARGET_MONTH):
        # A date target can vary only when explicitly implementing and
        # verifying separate output names/contracts; this artifact is Nov '26.
        raise ValueError("This version is fixed to November 2026.")
    manifest = read_eurostat() if manifest is None else manifest
    if manifest.get("status") != "validated":
        raise ValueError("Official second-hand-car HICP artifact unavailable.")
    output, report = Path(output), Path(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    if Path(source).is_file() and output.resolve() == Path(source).resolve():
        raise ValueError("Never overwrite the source dataset.")
    counts = Counter()
    country_counts = Counter()
    source_month_counts = Counter()
    status_by_country = Counter()
    freshness_counts = Counter()
    age_status_counts = Counter()
    ratio_cache = {}
    rows = 0
    with tempfile.TemporaryDirectory(prefix="eu-nov26-") as scratch:
        local = _source_file(source, scratch)
        if output.resolve() == local.resolve():
            raise ValueError("Never overwrite the source dataset.")
        reader = pq.ParquetFile(local)
        if not REQUIRED.issubset(reader.schema_arrow.names):
            raise ValueError("Original Parquet missing required price/date/country fields.")
        with pq.ParquetWriter(
            str(output), reader.schema_arrow.append(
                pa.field(PRICE_FIELD, pa.float64()),
                pa.field(FACTOR_FIELD, pa.float64()),
                pa.field(STATUS_FIELD, pa.string()),
                pa.field(FROM_FIELD, pa.string()),
                pa.field(AGE_MONTHS_FIELD, pa.float64()),
                pa.field(FRESHNESS_FIELD, pa.bool_()),
                pa.field(AGE_STATUS_FIELD, pa.string()),
            ).with_metadata({
                **(reader.schema_arrow.metadata or {}),
                b"derived_data_license": LICENSE.encode(),
                b"original_dataset": (
                    b"KTHESAT/european-used-car-market-2m-listings"),
                b"target_month": target.encode(),
                b"date_source": b"last_verified_at",
                b"adjustment_method": (
                    b"Eurostat HICP CP07112 I25 country-level; Holt forecast "
                    b"where month not officially published"),
                b"warning": (
                    b"market-time proxy, NOT vehicle age/mileage depreciation "
                    b"or model-specific price prediction"),
                b"training_eligibility": (
                    b"Recent verified listings within 2 months are candidates "
                    b"for contemporaneous training only, not automatic labels; "
                    b"no reconstructed current odometer or physical ageing"),
            }), compression="zstd",
            compression_level=3,
        ) as writer:
            for batch in reader.iter_batches(batch_size=batch_size):
                out = adjust_record_batch(batch, manifest=manifest,
                                          target=target, cache=ratio_cache)
                # Ensure stable schema and field metadata in every row group.
                writer.write_table(pa.Table.from_batches([out]))
                rows += out.num_rows
                status = out.column(STATUS_FIELD).to_pylist()
                country = out.column("seller_country").to_pylist()
                months = out.column(FROM_FIELD).to_pylist()
                freshness_counts.update(
                    "recent_candidate" if x else "not_current_candidate"
                    for x in out.column(FRESHNESS_FIELD).to_pylist()
                )
                age_status_counts.update(
                    out.column(AGE_STATUS_FIELD).to_pylist()
                )
                counts.update(status)
                country_counts.update(str(x or "").upper() for x in country)
                source_month_counts.update(str(x) for x in months if x is not None)
                status_by_country.update(
                    f"{str(c or '').upper()}:{s}" for c, s in zip(country, status)
                )
        result = {
            "schema_version": 1,
            "dataset": "KTHESAT/european-used-car-market-2m-listings",
            "license": LICENSE, "source_uri": str(source),
            "source_sha256": _sha256(local),
            "derived_file_sha256": _sha256(output),
            "derived_file_name": output.name,
            "target_month": target,
            "source_date_column": "last_verified_at",
            "source_price_column": "price_eur",
            "new_price_column": PRICE_FIELD,
            "new_factor_column": FACTOR_FIELD,
            "new_method_column": STATUS_FIELD,
            "age_since_verification_column": AGE_MONTHS_FIELD,
            "physical_revaluation_status_column": AGE_STATUS_FIELD,
            "rows": rows,
            "adjusted_rows": int(sum(
                counts.get(k, 0) for k in
                ("observed", "forecast", "interpolated", "snapshot"))),
            "status_counts": dict(counts.most_common()),
            "freshness_counts": dict(freshness_counts.most_common()),
            "physical_projection_counts": dict(age_status_counts.most_common()),
            "freshness_threshold_months": STALENESS_MONTHS_FOR_CURRENT_TRAINING,
            "current_training_eligibility_field": FRESHNESS_FIELD,
            "country_counts": dict(country_counts.most_common()),
            "source_month_counts": dict(source_month_counts.most_common()),
            "status_by_country": dict(status_by_country.most_common()),
            "eurostat_latest_published": {
                c: record.get("last_month")
                for c, record in manifest.get("countries", {}).items()
            },
            "eurostat_dataset": "prc_hicp_minr",
            "eurostat_category": "CP07112",
            "eurostat_unit": "I25",
            "eurostat_source_url": SOURCE_URL,
            "calculation": (
                "price_eur * HICP(country, 2026-11) / "
                "HICP(country, year-month(last_verified_at)). "
                "Missing future HICP is Holt forecast, clearly labelled."),
            "limitations": (
                "National used-car consumer-price proxy only. Not actual "
                "seller price changes, model-specific Bayesian scenarios, "
                "vehicle age/mileage depreciation or completed sales. "
                "Old ads are excluded from current-label eligibility even "
                "when their index-adjusted ASK is available. "
                "last_verified_at is not original ad publication date. "
                "Missing/unsupported rows retain NULL derived prices."),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        report.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        # Fail rather than publish a truncated or silently unadjusted file.
        if rows != reader.metadata.num_rows:
            raise AssertionError("Row count mismatch between source and derivative")
        if result["adjusted_rows"] == 0:
            raise AssertionError("No rows qualified for date normalization")
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--batch-size", type=int, default=75_000)
    args = parser.parse_args()
    if args.batch_size < 100 or args.batch_size > 250_000:
        parser.error("--batch-size must be between 100 and 250000")
    stats = enrich(
        args.source, args.output, args.report, batch_size=args.batch_size
    )
    print(json.dumps({
        "source": stats["dataset"],
        "rows": stats["rows"],
        "adjusted_rows": stats["adjusted_rows"],
        "freshness_counts": stats["freshness_counts"],
        "status_counts": stats["status_counts"],
        "output": args.output,
        "report": args.report,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
