"""Full Parquet Nov-2026 price-date adjustment without losing old ASK labels."""
import json
import math
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.add_europe_nov2026_adjusted_price import (
    AGE_MONTHS_FIELD, AGE_STATUS_FIELD, FACTOR_FIELD, FRESHNESS_FIELD,
    FROM_FIELD, PRICE_FIELD, STATUS_FIELD, adjust_record_batch, enrich,
)
from pricing.europe_price_clock import month_label, month_number


def fixture_manifest():
    first=month_number("2020-01")
    last=month_number("2026-09")
    def country(factor):
        rows=[{"month":month_label(n),
               "index":round(80*factor*(1.0025**(n-first)),4)}
              for n in range(first,last+1)]
        return {"first_month":"2020-01","last_month":"2026-09",
                "months":rows}
    return {"status":"validated","schema_version":1,
            "countries":{"DE":country(1.0),"FR":country(1.2)}}


def fixture_batch():
    return pa.record_batch({
        "id":pa.array([1,2,3,4,5,6,7],type=pa.int64()),
        "price_eur":pa.array([10000.,12000.,8000.,9000.,-2.,15000.,7000.]),
        "seller_country":["DE","DE","FR","CH","DE","DE","DE"],
        "reg_year":pa.array([2021,2025,2017,2018,2019,2020,2024],type=pa.int64()),
        "mileage_km":pa.array([45000.,10000.,73000.,60000.,20000.,90000.,20000.]),
        "last_verified_at":[
            "2026-10-15T10:02:00Z", "2020-10-12", "2026-10-07",
            "2026-10-14", "2026-10-01", None, "2026-12-02"],
    })


def test_adjust_preserves_original_ask_and_distinguishes_stale_from_recent():
    batch=fixture_batch()
    out=adjust_record_batch(batch,manifest=fixture_manifest())
    assert out.num_rows==batch.num_rows
    assert set(batch.schema.names).issubset(set(out.schema.names))
    assert out.column("price_eur").to_pylist()==batch.column("price_eur").to_pylist()
    vals=out.column(PRICE_FIELD).to_pylist()
    factors=out.column(FACTOR_FIELD).to_pylist()
    status=out.column(STATUS_FIELD).to_pylist()
    recency=out.column(FRESHNESS_FIELD).to_pylist()
    age=out.column(AGE_MONTHS_FIELD).to_pylist()
    month=out.column(FROM_FIELD).to_pylist()
    physical=out.column(AGE_STATUS_FIELD).to_pylist()
    assert status[0]=="forecast" and vals[0]>0
    assert vals[0]==pytest.approx(round(10000*factors[0],2))
    assert recency[0] is True and age[0]==1
    # Registration in 2021 DOES NOT mean ad was published in 2021.
    assert month[0]=="2026-10"
    assert recency[1] is False and age[1]==73
    assert physical[1]=="stale_listing_age_mileage_not_observed"
    assert status[1]=="index_unavailable"  # outside 12-month Holt window?
    assert status[2]=="forecast" and recency[2] is True
    assert status[3]=="unsupported_country" and vals[3] is None
    assert status[4]=="invalid_price" and vals[4] is None
    assert status[5]=="invalid_date" and vals[5] is None
    assert status[6]=="source_after_target" and vals[6] is None
    assert not any(recency[i] for i in [1,3,4,5,6])


def test_full_parquet_streaming_preserves_records_and_writes_report(tmp_path):
    source=tmp_path/"source.parquet"
    dest=tmp_path/"current.parquet"
    report=tmp_path/"report.json"
    table=pa.Table.from_batches([fixture_batch()])
    pq.write_table(table,source,row_group_size=3)
    summary=enrich(source,dest,report,manifest=fixture_manifest(),
                   batch_size=2)
    output=pq.read_table(dest)
    assert output.num_rows==table.num_rows==summary["rows"]==7
    assert summary["adjusted_rows"]>=2
    assert summary["freshness_counts"]["recent_candidate"]==2
    assert output.column("id").to_pylist()==table.column("id").to_pylist()
    assert output.column("price_eur").to_pylist()==table.column("price_eur").to_pylist()
    meta=output.schema.metadata
    assert meta[b"derived_data_license"]==b"CC BY-NC-SA 4.0"
    assert meta[b"target_month"]==b"2026-11"
    assert meta[b"date_source"]==b"last_verified_at"
    obj=json.loads(report.read_text())
    assert obj["rows"]==7
    assert obj["derived_file_sha256"]
    assert obj["freshness_threshold_months"]==2
    assert "HICP" in obj["calculation"]


def test_refuses_double_adjustment_and_target_or_source_overwrite(tmp_path):
    batch=adjust_record_batch(fixture_batch(),manifest=fixture_manifest())
    with pytest.raises(ValueError,match="already exist"):
        adjust_record_batch(batch,manifest=fixture_manifest())
    path=tmp_path/"input.parquet"
    pq.write_table(pa.Table.from_batches([fixture_batch()]),path)
    with pytest.raises(ValueError,match="Never overwrite"):
        enrich(path,path,tmp_path/"r.json",manifest=fixture_manifest())
    with pytest.raises(ValueError,match="fixed to November"):
        enrich(path,tmp_path/"other.parquet",tmp_path/"r.json",
               manifest=fixture_manifest(),target="2026-12")


def test_invalid_manifest_and_missing_source_columns_fail_closed():
    with pytest.raises(ValueError,match="Unvalidated"):
        adjust_record_batch(fixture_batch(),manifest={"status":"unavailable"})
    with pytest.raises(ValueError,match="Missing source"):
        adjust_record_batch(pa.record_batch({"price_eur":[200.]}),
                            manifest=fixture_manifest())
