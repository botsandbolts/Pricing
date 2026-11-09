"""Make-level sharding, global indices and hard-rule aggregation without network."""
import csv
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pricing.europe_dedupe import main as dedupe
from pricing.europe_make_shards import (
    GLOBAL_ID, SEGMENT, balance_makes, canonical_make, canonical_year, segment,
    collect, make_global_mapping, partition,
)


def listing(i, make, **kw):
    data = {
        "id":str(i), "source_site":"site_a","seller_country":"DE",
        "make":make,"model":"Model-A","version":"Model-A R Line 150",
        "price_eur":20000+i*7, "mileage_km":50000+i*100,
        "first_registration":"2021-05-07", "reg_year":2021, "power_hp":150,
        "seller_city":"Hamburg","color":"red","fuel":"diesel",
        "transmission":"automatic","first_seen_at":"2026-10-07",
    }
    data.update(kw)
    return data


def test_balance_never_splits_make_and_is_stable():
    weights={"a":100,"b":72,"c":44,"d":16,"e":9}
    one,loads=balance_makes(weights,3)
    assert len(one)==len(weights)
    assert sum(loads)==sum(weights.values())
    assert (one,loads)==balance_makes(weights,3)
    assert canonical_make(" VW ")==canonical_make("Volkswagen")
    assert canonical_make("Mercedes")=="mercedes benz"


def test_partition_link_recombine_global_rows(tmp_path):
    # Interleave makes so naive concatenation would destroy original indices.
    makes=["VW","BMW","Toyota","Volkswagen","BMW","Toyota",
           "BMW","Mercedes","Mercedes-Benz","VW","Toyota","BMW"]
    rows=[listing(i,make) for i,make in enumerate(makes)]
    rows[1]["id"]="same"
    rows[4].update(id="same",mileage_km=rows[1]["mileage_km"])
    rows[9].update(source_site="site_b",mileage_km=rows[0]["mileage_km"]+4,
                   version=rows[0]["version"],price_eur=18500)
    source=tmp_path/"input.parquet"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows),preserve_index=False),source)
    shards=tmp_path/"input-shards"
    report=partition(source,shards,3,batch_size=4)
    assert report["source_rows"]==len(rows)
    assert sum(report["shard_rows"])==len(rows)
    assert report["unique_makes"]==4
    assert report["segment_to_shard"][segment("Volkswagen","2021")]==report["segment_to_shard"][segment("VW","2021")]
    recovered=[]
    root=tmp_path/"reports"
    for i in range(3):
        f=shards/("shard-%02d.parquet"%i)
        values=pq.read_table(f)
        ids=values.column(GLOBAL_ID).to_pylist()
        recovered.extend(ids)
        for make in values.column("make").to_pylist():
            assert report["segment_to_shard"][segment(make,"2021")]==i
        out=root/("eu-make-audit-%02d"%i)
        dedupe(["--input",str(f),"--output",str(out),
                "--no-canonical-export","--thin-review-candidates"])
        make_global_mapping(f,out)
    assert sorted(recovered)==list(range(len(rows)))
    merged=collect(shards/"manifest.json",root,tmp_path/"combined")
    assert merged["source_rows"]==len(rows)
    assert merged["parallel_shards"]==3
    assert merged["automatic_identity_rows_removed"]>=1
    with (tmp_path/"combined"/"global_training_mapping.csv").open(newline="") as f:
        entries=list(csv.DictReader(f))
    assert len(entries)==len(rows)
    assert sorted(int(row["global_row"]) for row in entries)==list(range(len(rows)))
    assert all(0 <= int(r["global_training_rep"]) < len(rows) for r in entries)
    assert (tmp_path/"combined"/"combined_audit.json").is_file()


def test_imbalanced_brand_is_never_divided(tmp_path):
    path=tmp_path/"big.parquet"
    rows=[listing(i,"BMW") for i in range(25)]+[listing(25+i,"Audi") for i in range(3)]
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows),preserve_index=False),path)
    summary=partition(path,tmp_path/"chunks",4,batch_size=7)
    assert 25 in summary["shard_rows"]
    assert sum(summary["shard_rows"])==28


def test_make_year_blocking_does_not_compare_different_production_years(tmp_path):
    # Registration can be equal although actual production/model years differ.
    # Same marketplace and listing-ID, alike model, same year-of-registration:
    # no merge is allowed across the explicit year-block boundary.
    x=listing(0,"VW",id="same",production_year=2020,reg_year=2022)
    y=listing(1,"Volkswagen",id="same",production_year=2021,reg_year=2022,
              mileage_km=x["mileage_km"])
    z=listing(2,"VW",id="different",production_year=None,reg_year=2018)
    data=tmp_path/"years.parquet"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame([x,y,z]),preserve_index=False),data)
    chunks=tmp_path/"chunks"
    report=partition(data,chunks,2,batch_size=2)
    assert report["unique_makes"]==1
    assert report["unique_make_year_segments"]==3
    assert report["year_sources"]["production_year"]==2
    assert report["year_sources"]["reg_year"]==1
    assert canonical_year("2022-11-03")=="2022"
    assert canonical_year("nan")=="unknown"
    for i in range(2):
        f=chunks/("shard-%02d.parquet"%i)
        table=pq.read_table(f)
        for mk, yr, key in zip(table.column("make").to_pylist(),
                               table.column("production_year").to_pylist(),
                               table.column(SEGMENT).to_pylist()):
            expected=str(yr) if yr is not None else "2018"
            assert key==segment(mk,canonical_year(expected))
        out=tmp_path/("audit-%02d"%i)
        dedupe(["--input",str(f),"--output",str(out),
                "--thin-review-candidates","--no-canonical-export"])
        summary=json.loads((out/"audit.json").read_text())
        assert summary["automatic_rows_removed"]==0
        assert summary["review_pairs"]==0


def test_same_make_different_years_can_use_distinct_shards(tmp_path):
    rows=[listing(i,"BMW",reg_year=2020,production_year=2020) for i in range(15)]
    rows+=[listing(15+i,"BMW",reg_year=2022,production_year=2022) for i in range(15)]
    p=tmp_path/"input.parquet"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows),preserve_index=False),p)
    manifest=partition(p,tmp_path/"shards",2,batch_size=7)
    assert manifest["unique_makes"]==1
    assert manifest["unique_make_year_segments"]==2
    assert manifest["segment_to_shard"][segment("BMW","2020")] != manifest["segment_to_shard"][segment("BMW","2022")]
    assert sorted(manifest["shard_rows"])==[15,15]
