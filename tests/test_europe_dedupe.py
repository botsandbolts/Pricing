"""Synthetic safety regression tests; no external data required."""
import csv
import json

import pandas as pd
import pytest

from pricing.europe_dedupe import main, norm, classify


def listing(**overrides):
    row = {
        "id":"one","source_site":"market_a","seller_country":"DE",
        "make":"Volkswagen","model":"Golf","version":"Golf 2.0 TDI R-Line",
        "price_eur":18800,"mileage_km":65000,
        "first_registration":"2021-04-17","reg_year":2021,"power_hp":150,
        "seller_city":"Hamburg","color":"Blau","fuel":"Diesel",
        "transmission":"Automatic","first_seen_at":"2026-10-07"
    }
    row.update(overrides)
    return row


def run(tmp_path, rows, *args):
    input_path=tmp_path/"input.csv"
    output=tmp_path/"audit"
    pd.DataFrame(rows).to_csv(input_path,index=False)
    main(["--input",str(input_path),"--output",str(output),*args])
    with (output/"audit.json").open() as f:
        report=json.load(f)
    with (output/"row_mapping.csv").open() as f:
        mapping=list(csv.DictReader(f))
    return report,mapping,output


def test_normalization():
    assert norm("  Citroën C4  ")=="citroen c4"
    assert norm("Mercedes-Benz")=="mercedes benz"


def test_strong_cross_post_and_repeat_id_merge_but_lookalikes_survive(tmp_path):
    rows=[
      listing(),
      listing(id="two",source_site="market_b",mileage_km=65005,price_eur=17990),
      listing(id="one",source_site="market_a",price_eur=18100),
      listing(id="three",source_site="market_c",first_registration="2021-04-18",
              mileage_km=65009,price_eur=18000),
      listing(id="four",source_site="market_d",mileage_km=65800,price_eur=17700),
      listing(id="five",source_site="market_e",version="Golf 1.5 TSI Style",fuel="Petrol",
              power_hp=130,price_eur=16800),
    ]
    report,mapping,output=run(tmp_path,rows)
    assert report["input_rows"]==6
    assert report["automatic_rows_removed"]==2
    assert report["canonical_rows"]==4
    assert {mapping[i]["canonical_row_number"] for i in (0,1,2)}=={"0"}
    assert {mapping[i]["canonical_row_number"] for i in (3,4,5)}=={"3","4","5"}
    assert report["review_pairs"]>=1
    assert (output/"canonical_listings.csv").is_file()
    assert len(pd.read_csv(output/"canonical_listings.csv"))==4


def test_never_merge_semantics_without_identity():
    a=dict(zip(("rid","site","extid","country","make","model","version",
                "label_id","reg_date","reg_year","km","power","city","color",
                "fuel","gear","price"),
               (0,"a","a","DE","bmw","320d","320d m sport",
                1,"2021-02-01",2021,65000,190,"hamburg","black",
                "diesel","automatic",20000)))
    b={**a,"rid":1,"site":"b","extid":"b","reg_date":"2021-04-01","km":65300}
    status,score,_=classify(a,b,semantic=0.999)
    assert status != "auto"


def test_source_id_not_enough_when_vehicle_identity_conflicts(tmp_path):
    rows=[listing(),listing(source_site="market_a",id="one",make="BMW",model="320d")]
    report,mapping,_=run(tmp_path,rows)
    assert report["automatic_rows_removed"]==0
    assert mapping[0]["canonical_row_number"]!=mapping[1]["canonical_row_number"]


def test_schema_failure_is_explicit(tmp_path):
    path=tmp_path/"bad.csv"
    pd.DataFrame([{"id":"x"}]).to_csv(path,index=False)
    with pytest.raises(ValueError,match="Missing expected dataset fields"):
        main(["--input",str(path),"--output",str(tmp_path/"out")])


def test_training_thinning_prefers_more_informative_record_without_claiming_duplicate(tmp_path):
    original=listing()
    sparse=listing(id="new-id",source_site="other-market",mileage_km=65500,
                   power_hp=None,first_registration=None,color=None,price_eur=14200)
    report,_,out=run(tmp_path,[original,sparse],"--thin-review-candidates")
    assert report["automatic_rows_removed"]==0
    assert report["review_pairs"]==1
    assert report["training_thinning"]["additional_rows_excluded"]==1
    assert report["training_thinning"]["verified_duplicates"] is False
    assert len(pd.read_csv(out/"canonical_listings.csv"))==2
    assert len(pd.read_csv(out/"training_thinned_listings.csv"))==1
    with (out/"training_thinning_decisions.csv").open(newline="") as f:
        decisions=list(csv.DictReader(f))
    assert decisions[0]["excluded_canonical_row"]=="1"
    assert decisions[0]["preferred_canonical_row"]=="0"
    assert (out/"training_thinning_mapping.csv").exists()


def test_high_price_does_not_make_a_listing_more_informative():
    from pricing.europe_prune import information_score
    x={"reg_date":"2020-01-01","km":50000,"power":150,"city":"berlin",
       "version":"1.5 TSI Sport","fuel":"petrol","gear":"automatic",
       "color":"red","extid":"a","price":3000}
    assert information_score(x)==information_score({**x,"price":900000})


def test_training_thinning_retains_different_engine_versions(tmp_path):
    a=listing()
    b=listing(id="two",source_site="market_b",version="Golf 1.5 TSI Style",
              fuel="Petrol",power_hp=130,mileage_km=65010)
    report,_,out=run(tmp_path,[a,b],"--thin-review-candidates")
    assert report["training_thinning"]["additional_rows_excluded"]==0
    assert len(pd.read_csv(out/"training_thinned_listings.csv"))==2
