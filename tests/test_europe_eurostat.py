"""Eurostat JSON-stat safety, price clock, country policy and SGD-Huber smoke tests."""
from datetime import date
import math

import numpy as np
import pandas as pd
import pytest

from pricing.europe_eurostat import (
    eurostat_level_at, eurostat_adjusted_quote, read_eurostat,
)
from scripts.fetch_europe_eurostat_hicp import (
    unpack_jsonstat, build_url,
)


def published():
    return {
        "schema_version": 1, "status": "validated",
        "dataset": "prc_hicp_minr", "unit": "I25", "coicop18": "CP07112",
        "countries": {"DE": {"last_month": "2026-09",
                 "months": [{"month": f"{y:04d}-{m:02d}",
                             "index": 100.0 * 1.003 ** ((y-2025)*12 + m-1)}
                            for y in (2025, 2026)
                            for m in (range(1,13) if y==2025 else range(1,10))]}}}


def test_jsonstat_flattened_decoding_exact_category():
    obj = {"id": ["freq","unit","coicop18","geo","time"],
           "size": [1,1,1,1,2],
           "dimension": {
              "freq": {"category": {"index": {"M": 0}}},
              "unit": {"category": {"index": {"I25": 0}}},
              "coicop18": {"category": {"index": {"CP07112": 0}}},
              "geo": {"category": {"index": {"DE": 0}}},
              "time": {"category": {"index": {"2026-09": 0, "2026-10": 1}}},
           }, "value": {"0": 104.17, "1": 104.42}}
    assert unpack_jsonstat(obj,expected_geo="DE")==[
        {"month":"2026-09","index":104.17},{"month":"2026-10","index":104.42}]
    with pytest.raises(ValueError):
        unpack_jsonstat(obj,expected_geo="FR")


def test_api_query_uses_2026_econicop_and_reference_period():
    url=build_url("DE")
    assert "prc_hicp_minr?" in url
    assert "coicop18=CP07112" in url
    assert "unit=I25" in url


def test_official_ratio_and_proxy_disclosure():
    data=published()
    amount,info=eurostat_adjusted_quote(15000,"2025-01-09","2025-02-12",
                                        "DE",manifest=data)
    assert amount==pytest.approx(15000*1.003)
    assert info["status"]=="observed"
    assert info["source"].startswith("Eurostat HICP")
    assert "proxy" in info["limitation"]


def test_missing_current_month_uses_holt_and_labels_forecast():
    data=published()
    amount,info=eurostat_adjusted_quote(15000,"2026-09-15",date(2026,11,10),
                                        "DE",manifest=data)
    assert math.isfinite(amount) and amount>0
    assert info["status"]=="forecast"
    assert info["source_last_published_month"]=="2026-09"
    assert info["target_index_method"]=="holt_forecast"


def test_not_substitute_hicp_from_another_country():
    data=published()
    amount,info=eurostat_adjusted_quote(15000,"2025-01-01","2025-02-01",
                                        "FR",manifest=data)
    assert amount is None
    assert info["status"]=="unavailable"
    amount,info=eurostat_adjusted_quote(15000,"2025-01-01","2025-02-01",
                                        "CH",manifest=data)
    assert amount is None


def test_forecast_beyond_twelve_months_is_refused():
    result=eurostat_level_at(published(),"DE","2028-12")
    assert "error" in result


def test_no_implicit_fabrication_when_official_data_missing(tmp_path):
    assert read_eurostat(tmp_path/"missing.json")["status"]=="unavailable"
    amount, info=eurostat_adjusted_quote(12000,"2026-10-10","2026-11-10","DE",
                                        manifest={"status":"unavailable"})
    assert amount is None


def test_sgd_huber_pair_on_synthetic_common_vehicle_baskets():
    from scripts.build_europe_hedonic_fisher import pair_index
    rs=np.random.default_rng(17)
    common=[]
    for month, drift in (("2026-08",0.),("2026-09",0.025)):
        for i in range(1100):
            group=i%25
            common.append({
                "id":f"{month}-{i}", "month":month,
                "vehicle_group":f"Brand{group%5} · Model{group}",
                "source_site":"portal",
                "fuel":"petrol" if group%2 else "diesel",
                "transmission":"manual" if group%2 else "auto",
                "body_type":"sedan",
                "age":4.0+(group%6),
                "log_mileage":math.log1p(65000+group*2500),
                "power_hp":115+group,
                "price_eur":float(math.exp(9.3+drift-0.04*(group%6)+rs.normal(0,0.05)))
            })
    df=pd.DataFrame(common)
    r=pair_index(df[df.month=="2026-08"],df[df.month=="2026-09"],
                 min_overlap=800,sample_per_month=1100)
    assert r is not None
    assert 0.95 < r["fisher"] < 1.12
    assert r["overlap"] >= 10
