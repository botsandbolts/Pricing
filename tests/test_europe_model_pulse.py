"""Make/model relative-date evidence must never masquerade as a real price index."""
import math
import numpy as np
import pandas as pd
import pytest

from scripts.build_europe_hedonic_fisher import clean
from scripts.build_europe_model_pulse import (
    build, fit_relative_pulse, _robust_se, _trimmed_mean,
)
from pricing.europe_model_pulse import read_pulse, lookup_pulse
from pricing.europe_bayes_pulse import shrink_model_differences


def artificial_dates(n=360):
    rng=np.random.default_rng(1234)
    rows=[]
    groups=(("volkswagen","golf"),("bmw","320d"),("toyota","corolla"))
    for month_num,month in enumerate(("2026-09","2026-10")):
        for g,(make,model) in enumerate(groups):
            for i in range(n):
                registration=2015+(i%9)
                mileage=25000+(i%12)*8000
                hp=100+(i%8)*20
                trend=(0.07 if g==0 and month_num else 0)
                # Price differences reflect age, km and car model, plus mild noise.
                log_price=(9.2 + 0.30*g+0.012*(registration-2018)
                           -0.000003*mileage+trend+rng.normal(0,0.07))
                rows.append({
                    "id":f"{month}-{g}-{i}","source_site":"dealer",
                    "seller_country":"DE","make":make,"model":model,
                    "price_eur":float(math.exp(log_price)),
                    "mileage_km":mileage,"reg_year":registration,"power_hp":hp,
                    "first_seen_at":month+"-12","fuel":"petrol" if i%2 else "diesel",
                    "transmission":"manual" if i%3 else "automatic",
                    "body_type":"sedan",
                })
    return pd.DataFrame(rows)


def test_research_requires_multiple_genuine_months_and_never_marks_validated():
    d=artificial_dates(360)
    only_one=d[d.first_seen_at.str.startswith("2026-10")]
    report=build(only_one)
    assert report["status"]=="research_only"
    assert report["research_pairs"]==[]
    assert report["minimum_observed_months_to_enable_forecasting"]>=6


def test_adjacent_month_guardrails():
    d=clean(artificial_dates(360))
    with pytest.raises(ValueError):
        fit_relative_pulse(d,early="2026-07",late="2026-10")
    too_sparse=d.groupby(["vehicle_group","month"]).head(10).copy()
    assert fit_relative_pulse(too_sparse) is None


def test_sgd_huber_research_generates_shrunk_vehicle_specific_effects():
    r=build(artificial_dates(420))
    assert r["status"]=="research_only"
    assert len(r["research_pairs"])==1
    pair=r["research_pairs"][0]
    assert pair["early_month"]=="2026-09"
    assert pair["late_month"]=="2026-10"
    assert pair["model_count"]>=3
    assert pair["n_training"]>=1500
    assert all(0<=row["shrink_weight"]<=1 for row in pair["models"])
    assert all(0<=row["posterior_probability_positive"]<=1 for row in pair["models"])
    assert all("brand_peer_count" in row for row in pair["models"])
    # The first vehicle group has a real +7% difference, unlike other models.
    golf=next(x for x in pair["models"] if x["model"]=="golf")
    assert golf["relative_pct"]>0
    assert golf["interval_low_pct"]<=golf["relative_pct"]<=golf["interval_high_pct"]


def test_research_only_reader_and_lookup_no_production_application(tmp_path):
    data={"schema_version":1,"status":"research_only","research_pairs":[{
        "country":"DE","early_month":"2026-09","late_month":"2026-10",
        "models":[{"make":"volkswagen","model":"golf","relative_pct":2.5,
                   "interval_low_pct":-1.0,"interval_high_pct":5.5,
                   "n_early":340,"n_late":350}]
    }]}
    import json
    file=tmp_path/"pulse.json"
    file.write_text(json.dumps(data))
    observed=read_pulse(file)
    found=lookup_pulse(observed,"de","Volkswagen","Golf")
    assert found["relative_pct"]==2.5
    assert found["applied_to_quote"] is False
    assert found["status"]=="research_only"
    assert lookup_pulse(observed,"FR","Volkswagen","Golf") is None
    assert lookup_pulse(observed,"DE","Volkswagen","Polo") is None
    assert read_pulse(tmp_path/"missing.json")["status"]=="unavailable"


def test_robust_trimmed_mean_resists_large_price_outliers():
    sample=[0.03]*100+[25,-30]
    assert abs(_trimmed_mean(sample)-0.03)<0.001
    assert _robust_se(sample)>0


def test_two_level_bayesian_posterior_shrinks_sparse_models_and_uses_make_peers():
    items=[
        {"make":"volkswagen","model":"golf","raw":0.08,"se":0.012},
        {"make":"volkswagen","model":"polo","raw":0.03,"se":0.02},
        {"make":"volkswagen","model":"passat","raw":0.01,"se":0.025},
        {"make":"bmw","model":"320","raw":-0.05,"se":0.015},
        {"make":"bmw","model":"520","raw":-0.04,"se":0.018},
        {"make":"toyota","model":"corolla","raw":0.25,"se":0.25},
    ]
    r=shrink_model_differences(items)
    lookup={row["model"]:row for row in r}
    assert len(r)==6
    assert lookup["golf"]["brand_peer_count"]==2
    assert lookup["corolla"]["brand_peer_count"]==0
    assert lookup["corolla"]["posterior_log_change"]<0.1
    assert lookup["golf"]["posterior_log_change"]>0.03
    assert lookup["320"]["posterior_log_change"]<0
    for row in r:
        assert row["interval_low_log"]<=row["posterior_log_change"]<=row["interval_high_log"]
        assert 0<=row["posterior_probability_positive"]<=1
    # Known observations cannot be changed by an unrelated make's posterior
    r2=shrink_model_differences(items[:-1])
    look2={row["model"]:row for row in r2}
    assert look2["golf"]["posterior_log_change"]==pytest.approx(
        lookup["golf"]["posterior_log_change"])


def test_two_level_bayesian_posterior_rejects_invalid_noise_and_prior_scales():
    assert shrink_model_differences([{"make":"vw","model":"golf","raw":0.1,"se":0}])==[]
    with pytest.raises(ValueError):
        shrink_model_differences([{"make":"vw","raw":0.1,"se":0.02}],tau_make=-1)
