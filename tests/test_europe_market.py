"""European ResMLP release regression tests: exact schema and safe inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pricing import europe_model
from pricing.europe_model import (CATEGORICAL, FEATURES, FEATURE_TYPES, NUMERIC,
    EuropeanModelUnavailable, encode_vehicle, estimate, available, load_bundle,
    batch_estimate, explain)

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/"models"/"europe-resmlp"


def fixture_schema():
    return {
        "feature_names":FEATURES,"feature_types":FEATURE_TYPES,
        "snapshot_date":"2026-10-10",
        "category_vocabulary":{k:{"__missing__":0,"volkswagen":1,
                   "golf":2,"de":3,"diesel":4,"manual":5} for k in CATEGORICAL},
    }


def vehicle():
    return {
        "seller_country":"de","make":"volkswagen","model":"golf",
        "reg_year":2020,"reg_month":6,"mileage_km":85000.,
        "power_hp":110.,"fuel":"diesel","transmission":"manual",
        "first_seen_at":"2026-10-10",
    }


def test_exact_feature_order_and_unknown_categories():
    values=fixture_schema()
    row=encode_vehicle(vehicle(),values)
    assert row.shape==(1,len(FEATURES))
    assert row.dtype==np.float32
    assert row[0,0]==2020 and row[0,1]==85000
    assert row[0,len(NUMERIC)+CATEGORICAL.index("make")]==1
    unseen={**vehicle(),"make":"unseen-brand"}
    missing=encode_vehicle(unseen,values)
    assert np.isnan(missing[0,len(NUMERIC)+CATEGORICAL.index("make")])


def test_invalid_vehicle_has_no_estimate():
    for changes in ({"reg_year":2030},{"mileage_km":-100},
                    {"make":""},{"model":""},{"power_hp":9999}):
        with pytest.raises(ValueError):
            encode_vehicle({**vehicle(),**changes},fixture_schema())


def test_core_us_routes_are_unchanged():
    app=(ROOT/"app.py").read_text()
    assert "render_market_gateway()" in app
    assert "if _workspace=='🚗 Price my car':" in app
    assert "pricing.neural_experimental_ui" in app
    assert "render_europe" in app
    assert "Euro" not in app.split("if _workspace=='🚗 Price my car':")[1][:500]


def test_neural_provenance_fails_closed_without_weights(tmp_path):
    with pytest.raises(EuropeanModelUnavailable):
        load_bundle(tmp_path)
    assert estimate.__module__=="pricing.europe_resmlp"


def test_verified_release_with_real_weights():
    if not available():
        pytest.skip("Production weights are downloaded in release workflow")
    model,schema,comp=load_bundle()
    top=comp["overall_champion"]
    assert top["origin"]=="eu-neural-wave2-5"
    assert top["architecture"]=="resmlp"
    assert top["loss"]=="tail_mae_eur"
    assert comp["test_rows"]==201085
    assert top["test_once"]["cvar95_absolute_eur"]<22000
    assert top["test_once"]["mae_eur"]<3200
    assert schema["train_only_vocabulary"] is True
    assert (MODEL/"release_manifest.json").is_file()
    options=schema["category_vocabulary"]
    car=vehicle()
    for key,example in [("seller_country","de"),("make","volkswagen"),
                        ("model","golf")]:
        car[key]=example if example in options[key] else next(iter(options[key]))
    one=estimate(car,bundle=(model,schema,comp))
    two=batch_estimate([car,car],bundle=(model,schema,comp))
    assert 100<=one<=1000000 and np.isfinite(one)
    assert np.allclose(two,[one,one],rtol=1e-6)
    scenarios=explain(car,bundle=(model,schema,comp))
    assert scenarios and all(np.isfinite(change) for _,change in scenarios)
    assert len(scenarios)>=2
