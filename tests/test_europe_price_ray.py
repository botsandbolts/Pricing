"""Fast local/synthetic safety tests for European ask-price holdouts and AFT."""
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.europe_price_ray import (
    CATEGORICAL, FEATURES, INPUT_COLUMNS, NUMERIC,
    default_config, ids_sha, metrics, prepare, smoke, split_ids,
    validated_config, aft_huber_objective,
)


def _fake_rows(n=3000):
    rows=[]
    for i in range(n):
        rows.append({
            "id":str(i),"source_site":"site_a" if i % 2 else "site_b",
            "seller_country":"DE" if i%3 else "NL",
            "make":("BMW","VW","Toyota")[i%3],
            "model":("320d","Golf","Corolla")[i%3],
            "model_group":"compact","version":"Trim "+str(i%21),
            "price_eur":2500.+(i*37)%51000,
            "mileage_km":10000+i*37,"first_registration":"2021-03-15",
            "reg_year":2021,"power_hp":90+i%35,
            "seller_city":"City-"+str(i%17),"color":"blue",
            "fuel":"diesel","transmission":"automatic",
            "first_seen_at":"2026-10-07","body_type":"wagon",
            "drive_type":"front","doors":5,"seats":5,
            "images_count":8,"accident_free":1,
        })
    rows[1]["price_eur"]=12500.
    rows[2].update({k:v for k,v in rows[1].items() if k not in ("id","source_site","price_eur")})
    rows[2]["price_eur"]=17500.
    return pd.DataFrame(rows)


def test_fingerprint_split_keeps_identical_spec_different_prices_together():
    f=_fake_rows(3000)
    phases=split_ids(f)
    assert phases[1]==phases[2]
    assert set(np.unique(phases))=={"train","validation","test"}
    assert len(FEATURES)==len(NUMERIC)+len(CATEGORICAL)
    assert "price_eur" not in FEATURES
    assert ids_sha(np.array([3,2,1])) == ids_sha(np.array([1,3,2]))


def test_test_set_sealed_and_no_thinning(tmp_path):
    f=_fake_rows(3000)
    p=tmp_path/"source.parquet"
    pq.write_table(pa.Table.from_pandas(f,preserve_index=False),p)
    report=prepare(p,tmp_path/"split")
    assert report["counts"]["train"]+report["counts"]["validation"]+report["counts"]["test"]==3000
    assert report["deduplication_applied"] is False
    assert report["test_sealed_until_final_aggregation"]
    for phase in ("train","validation","test"):
        with np.load(tmp_path/"split"/(phase+".npz"),allow_pickle=False) as data:
            assert len(data["y"])==report["counts"][phase]
            assert data["x"].shape[1]==len(FEATURES)
    assert json.loads((tmp_path/"split"/"manifest.json").read_text())==report


def test_safety_bounds_and_aft_huber_gradients():
    c=default_config("aft_huber")
    validated_config(c)
    import xgboost as xgb
    dm=xgb.DMatrix(np.zeros((4,2),dtype=np.float32),
                   label=np.array([8.,9.,10.,11.],dtype=np.float32))
    gradient,hessian=aft_huber_objective(c)(
        np.array([8.,9.5,8.5,11.],dtype=np.float32),dm)
    assert np.isfinite(gradient).all() and np.isfinite(hessian).all()
    assert np.all(hessian>0)
    c["sigma"]=0
    try:
        validated_config(c)
    except ValueError:
        pass
    else:
        raise AssertionError("Illegal AFT scale accepted")


def test_cvar_is_worst_five_percent():
    observed=np.full(100,1000.)
    estimate=np.full(100,1000.)
    estimate[-5:]=2000.
    result=metrics(observed,estimate)
    assert result["cvar95_absolute_eur"]==1000
    assert result["mae_eur"]==50


def test_xgboost_objective_smoke():
    smoke()
