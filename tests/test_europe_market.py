"""European overnight ResMLP release regression tests: exact schema and safe inference."""
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
MODEL=ROOT/"models"/"europe-resmlp-overnight"


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
    assert estimate.__module__=="pricing.europe_resmlp_overnight"


def test_verified_release_with_real_weights():
    if not available():
        pytest.skip("Production weights are downloaded in release workflow")
    model,schema,comp=load_bundle()
    top=comp["overall_champion"]
    assert top["origin"]=="eu-overnight-20261111-wave1-0"
    assert top["architecture"]=="resmlp"
    assert top["loss"]=="weighted_tail_huber"
    assert top["neural_epoch"]==76
    assert comp["compared_models"]==30
    assert comp["all_trial_count"]==971
    assert comp["test_rows"]==192046
    assert top["test_once"]["cvar95_absolute_eur"]<22000
    assert top["test_once"]["mae_eur"]<3200
    assert schema["train_only_vocabulary"] is True
    assert set(schema["category_vocabulary"]["seller_country"])==set("de fr nl be at es it pl".split())
    assert schema["automatic_eurostat_adjustment"] is False
    assert schema["source_training_target"]=="original_price_eur"
    assert schema["source_run_id"]==42892216866
    assert (MODEL/"release_manifest.json").is_file()
    options=schema["category_vocabulary"]
    car=vehicle()
    for key,example in [("seller_country","de"),("make","volkswagen"),
                        ("model","golf")]:
        car[key]=example if example in options[key] else next(iter(options[key]))
    one=estimate(car,bundle=(model,schema,comp))
    two=batch_estimate([car,car],bundle=(model,schema,comp))
    assert 100<=one<=1000000 and np.isfinite(one)
    # GEMM kernels differ slightly by batch size under float32. Cent-level tolerance.
    assert np.allclose(two,[one,one],rtol=0,atol=0.08)
    scenarios=explain(car,bundle=(model,schema,comp))
    assert scenarios and all(np.isfinite(change) for _,change in scenarios)
    assert len(scenarios)>=2


def test_europe_calculator_never_reapplies_eurostat():
    source=(ROOT/"pricing/europe_market.py").read_text()
    form=source.split("def _edit_vehicle(schema):",1)[1].split("def _receipt(bundle):",1)[0]
    assert "eurostat_adjusted_quote(" not in form
    assert "adjusted_quote(" not in form
    assert '"status":"built_in_recent_market"' in form
    assert 'ResMLP Weighted Tail-Huber · October 2026 evidence' in source
    assert 'HOFM3 Tail-MAE · October 2026 evidence' not in form
    assert 'COUNTRY_OPTIONS' in form
    assert "eu_country_v2" in form
    assert "st.date_input(" not in form
    from pricing.europe_market import COUNTRIES,COUNTRY_OPTIONS
    assert len(COUNTRY_OPTIONS)==8
    assert set(COUNTRY_OPTIONS)==set("de fr nl be at es it pl".split())
    assert "CH" not in COUNTRIES
