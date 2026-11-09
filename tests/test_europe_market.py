"""European release safety: schema parity, exact winner, separate market routing."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest

from pricing.europe_model import (CATEGORICAL,FEATURES,FEATURE_TYPES,NUMERIC,
    EuropeanModelUnavailable,encode_vehicle,estimate,available,load_bundle)

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/"models"/"europe-ask"


def fixture_schema():
    return {
        "feature_names":FEATURES,
        "feature_types":FEATURE_TYPES,
        "snapshot_date":"2026-06-01",
        "category_vocabulary":{k:{"__missing__":0,"volkswagen":1,
                     "golf":2,"de":3,"diesel":4,"manual":5}
                  for k in CATEGORICAL},
    }


def vehicle():
    return {
      "seller_country":"DE","make":"VW_that_is_not_known",
      "model":"Golf","reg_year":2020,"reg_month":6,
      "mileage_km":85000.,"power_hp":110.,
      "fuel":"diesel","transmission":"manual",
      "first_seen_at":"2026-06-01",
    }


def test_exact_feature_order_and_unknown_category_fallback():
    schema=fixture_schema()
    row=encode_vehicle(vehicle(),schema)
    assert row.shape==(1,len(FEATURES))
    assert row.dtype==np.float32
    assert row[0,0]==2020
    assert row[0,1]==85000
    assert row[0,len(NUMERIC)+CATEGORICAL.index("make")]!=row[0,len(NUMERIC)+CATEGORICAL.index("make")]
    assert row[0,len(NUMERIC)+CATEGORICAL.index("model")]==2
    assert row[0,len(NUMERIC)+CATEGORICAL.index("seller_country")]==3


def test_invalid_vehicle_has_no_estimate():
    for changes in ({"reg_year":2030},{"mileage_km":-100},
                    {"make":""},{"model":""},{"power_hp":9999}):
        bad={**vehicle(),**changes}
        with pytest.raises(ValueError):
            encode_vehicle(bad,fixture_schema())


def test_core_repo_us_routes_remain_unchanged():
    app=(ROOT/"app.py").read_text()
    assert "render_market_gateway()" in app
    assert "if _workspace=='🚗 Price my car':" in app
    assert "pricing.neural_experimental_ui" in app
    assert "render_europe" in app
    assert "Euro" not in app.split("if _workspace=='🚗 Price my car':")[1][:500]


def test_model_deploy_requires_packaged_schema_and_audit(tmp_path):
    with pytest.raises(EuropeanModelUnavailable):
        load_bundle(tmp_path)


def test_rank1_real_weights_when_packaged():
    if not available():
        pytest.skip("Real model is not packaged yet; must pass packaging workflow")
    model,schema,comp=load_bundle()
    assert comp["winning_model"]["origin"]=="eu-ask-candidate-ray-8"
    assert comp["winning_model"]["family"]=="aft_normal"
    assert comp["winning_model"]["iterations"]==600
    assert comp["winning_model"]["test_once"]["n"]==201085
    assert comp["winning_model"]["test_once"]["cvar95_absolute_eur"]<23000
    options=schema["category_vocabulary"]
    v=vehicle()
    for key,example in [("seller_country","de"),("make","volkswagen"),("model","golf")]:
        v[key]=example if example in options[key] else next(iter(options[key]))
    value=estimate(v,bundle=(model,schema,comp))
    assert np.isfinite(value) and 100<=value<=1_000_000
    assert (MODEL/"release_audit.json").is_file()
    audit=json.loads((MODEL/"release_audit.json").read_text())
    assert audit["verified"] is True
    assert audit["production_inference_parity_rows"]>=20
