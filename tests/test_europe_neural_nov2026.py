import pandas as pd
import pytest
from scripts.europe_neural_nov2026_prepare import (
    COUNTRY_MAP, normalize_country, make_recent_cohort, MAX_AGE_MONTHS,
)
from scripts.europe_neural_nov2026_ray import (
    FAMILIES, LOSSES, base_config, safe_config,
)


def records():
    return pd.DataFrame({
        "seller_country":["Germany","DEU","de","Österreich","AUT","CH",
                          "France","FR","Spain","NL"],
        "make":["VW"]*10,"model":["Golf"]*10,"reg_year":[2020]*10,
        "mileage_km":[55000]*10,"price_eur":[12000.]*10,
        "first_seen_at":["2024-02-01"]*10,
        "last_verified_at":["2026-10-10","2026-11-03","2021-09-10",
                            "2026-10-10","2026-10-10","2026-10-10",
                            "2026-10-10","2026-12-10","2026-10-10","2026-10-10"],
        "eligible_as_current_price_training_label":[True]*10,
        "current_depreciated_price_eur":[1_000_000.]*10,
        "hypothetical_nov2026_mileage_km":[70000]*10,
    })


def test_country_iso2_aliases_merge_duplicates_without_unknown_guessing():
    raw=pd.Series(["DE","DEU","Germany","Deutschland","AT","Austria",
                   "Österreich","FR","france","NL","nld","CH",None,"nonsense"])
    got=normalize_country(raw).tolist()
    assert got[:4]==["DE"]*4
    assert got[4:7]==["AT"]*3
    assert got[7:9]==["FR"]*2
    assert got[9:11]==["NL"]*2
    assert pd.isna(got[11]) and pd.isna(got[12]) and pd.isna(got[13])


def test_historical_outdated_and_future_rows_are_filtered():
    sample=records()
    assert MAX_AGE_MONTHS==2
    frame=sample.copy()
    # Make synthetic fixture exceed the minimum 1000 good rows
    frame=pd.concat([frame]*210,ignore_index=True)
    recent,meta=make_recent_cohort(frame)
    assert meta["source_rows"]==2100
    assert meta["included_rows"]==7*210
    assert meta["excluded_rows"]==3*210
    assert set(recent.seller_country)=={"de","at","fr","es","nl"}
    assert set(recent.first_seen_at)=={"2026-10-10","2026-11-03"}
    assert (recent.price_eur==12000).all()
    assert "current_depreciated_price_eur" not in recent.columns
    assert "hypothetical_nov2026_mileage_km" not in recent.columns
    assert meta["no_derived_hicp_or_mileage_labels"] is True


def test_five_architectures_thirty_loss_hyperparameter_compatibility():
    assert set(FAMILIES)=={"resmlp","dcnv2","deepfm","hofm3","fttransformer"}
    assert len(LOSSES)==30
    for family in FAMILIES:
        cfg=base_config(family)
        assert safe_config(cfg)["family"]==family
    assert "tail_mae_eur" in LOSSES
    assert "aft_huber" in LOSSES
    assert "quantile50" in LOSSES
