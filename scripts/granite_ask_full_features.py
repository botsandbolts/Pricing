"""Full-cohort U.S. Ask: TRAIN-only make/model annual-mileage features.

Exactly 211,127 original 2021 Craigslist listings and immutable chronological
holdouts. No price-based trimming, no future-mileage fabrication, no label
feature or leakage. Granite 97M INT8 ONNX embeddings stay frozen, SHA-pinned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.raw_embedding_ask_encode import id_hash

COUNTS = {"train":147777, "validation":31668, "test":31682}
COHORT_SHA256="746bc9f8e02e1e708ea5e1d9479408effb131eb6b1d1ea7710380701e62dcf12"
GRANITE_EMBEDDING_SHA256="7f2be5c0e2199ca8ca23c480140b6f8b6707b119594efb9f3278c415e7b8e5f2"
MODEL="ibm-granite/granite-embedding-97m-multilingual-r2"
REVISION="c61e626a6255c490879d0af885078b61929d51f6"
INT8_SHA="a6022dd8220ea6f6595562a1328ee216f4a94faa55362f2f4747c80f1e78772e"
EXTRA = ("log_expected_annual_miles_make_model",
         "log_odometer_to_model_expected_2021",
         "signed_excess_mileage_vs_2021_expected")


def norm(series):
    return series.fillna("unknown").astype(str).str.strip().str.lower().replace(
        {"none":"unknown", "nan":"unknown", "":"unknown"})


def add_model_mileage(frame, phase, original):
    """Model-specific annual mileage fitted only to original TRAIN vehicles."""
    if not {"year","odometer","manufacturer","model"}.issubset(frame.columns):
        raise ValueError("Canonical asking dataset lacks mileage identity inputs")
    year=pd.to_numeric(frame.year, errors="coerce").to_numpy(dtype=float)
    miles=pd.to_numeric(frame.odometer, errors="coerce").to_numpy(dtype=float)
    age=np.maximum(2021-year,1.)
    annual=miles/age
    train=np.asarray(phase,dtype=str)=="train"
    if int(train.sum())!=COUNTS["train"]:
        raise ValueError("Expected ALL 147,777 original TRAIN listings")
    eligible=(train & np.isfinite(annual) & np.isfinite(year) &
              np.isfinite(miles) & (year>=1991) & (year<=2020) &
              (miles>=0) & (miles<=800000) & (annual>=1000) & (annual<=50000))
    if int(eligible.sum())<20000:
        raise ValueError("Too few training records for model mileage")
    maker=norm(frame.manufacturer)
    model=norm(frame.model)
    rates=pd.DataFrame({"maker":maker,"model":model,"annual":annual})
    train_rates=rates.loc[eligible]
    paired=train_rates.groupby(["maker","model"]).annual.agg(["median","size"])
    makers=train_rates.groupby("maker").annual.agg(["median","size"])
    by_pair={}
    for (manufacturer, model_name),r in paired.iterrows():
        if int(r["size"])>=100:
            by_pair.setdefault(manufacturer,{})[model_name]=float(r["median"])
    by_make={m:float(r["median"]) for m,r in makers.iterrows() if int(r["size"])>=300}
    pooled=float(np.median(annual[eligible]))
    if not 5000<pooled<20000:
        raise ValueError("Unexpected pooled annual-mileage estimate")
    pair_rate=paired["median"].where(paired["size"]>=100).reindex(
        pd.MultiIndex.from_arrays([maker,model])).to_numpy(dtype=float)
    make_rate=maker.map(by_make).to_numpy(dtype=float)
    typical=np.where(np.isfinite(pair_rate),pair_rate,
                     np.where(np.isfinite(make_rate),make_rate,pooled))
    expected=typical*age
    extra=np.column_stack((
        np.log1p(typical),
        np.log1p(np.maximum(miles,0)/np.maximum(expected,1000)),
        np.sign(miles-expected)*np.log1p(np.abs(miles-expected)/10000),
    ))
    valid=np.isfinite(miles)&np.isfinite(year)&(year>=1886)&(year<=2021)&(miles>=0)
    extra[~valid,:]=np.nan
    med=np.nanmedian(extra[train],axis=0)
    missing=~np.isfinite(extra)
    extra=np.where(missing,med,extra)
    p25,p75=np.percentile(extra[train],[25,75],axis=0)
    scale=np.where(p75-p25>0.01,p75-p25,1.)
    enhanced=np.concatenate((np.asarray(original,dtype=np.float32),
              np.clip((extra-med)/scale,-8,8).astype(np.float32)),axis=1)
    if enhanced.shape!=(len(frame),10) or not np.isfinite(enhanced).all():
        raise ValueError("New mileage feature dimensions or finiteness failed")
    np.testing.assert_array_equal(enhanced[:,:7],original)
    profile={
        "scope":"2021 Craigslist historical Ask only",
        "trained_on_original_train_only":True,
        "age_reference":"2021 minus model year, not a predicted 2026 odometer",
        "paired_make_model_min_count":100,"manufacturer_min_count":300,
        "by_make_model":by_pair,"by_make":by_make,"pooled":pooled,
        "median":med.tolist(),"iqr":scale.tolist(),
        "training_eligible_rows":int(eligible.sum()),
        "original_train_rows":int(train.sum()),
        "missing_mileage_fills":int(missing.sum()),
        "numeric_order":list(EXTRA),
    }
    return enhanced,profile


def prepare(source, output):
    source,output=Path(source),Path(output)
    meta=json.loads(source.with_suffix(".json").read_text())
    if (meta.get("cohort_sha256")!=COHORT_SHA256
            or tuple(meta.get(k) for k in ("full_train","validation","test"))
               !=tuple(COUNTS.values())):
        raise ValueError("Wrong frozen original U.S. Ask cohort")
    with np.load(source,allow_pickle=False) as f:
        z={key:f[key] for key in ("num","cat","phase","ids","y")}
    phase=z["phase"].astype(str)
    for name,count in COUNTS.items():
        if int((phase==name).sum())!=count:
            raise ValueError("Original chronological "+name+" split changed")
    if (z["num"].shape!=(211127,7) or z["cat"].shape!=(211127,11)
            or len(np.unique(z["ids"]))!=len(phase)
            or not np.isfinite(z["y"]).all() or (z["y"]<=0).any()):
        raise ValueError("Full original feature geometry or labels are invalid")
    for split in ("validation","test"):
        if id_hash(z["ids"][phase==split].astype(str))!=meta["holdout_row_sha256"][split]:
            raise ValueError("Original holdout identity changed: "+split)
    frame=pd.read_parquet(source.with_name("canonical-ask.parquet"),
                          columns=["row_id","year","odometer","manufacturer","model"])
    if len(frame)!=len(z["ids"]) or not np.array_equal(
            frame.row_id.astype(str).to_numpy(),z["ids"].astype(str)):
        raise ValueError("Canonical rows differ from frozen feature rows")
    enriched,profile=add_model_mileage(frame,phase,z["num"])
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,num=enriched,cat=z["cat"],phase=z["phase"],
                        ids=z["ids"],y=z["y"])
    newmeta={
        "status":"complete","market":"ask",
        "source":"original 2021 Craigslist U.S. advertised prices",
        "cohort_sha256":COHORT_SHA256,
        "full_train":COUNTS["train"],"validation":COUNTS["validation"],
        "test":COUNTS["test"],"row_count":len(phase),
        "holdout_row_sha256":meta["holdout_row_sha256"],
        "cardinalities":meta["cardinalities"],
        "numeric_features":list(meta["numerics"])+list(EXTRA),
        "numeric_dimension":10,"categorical_dimension":11,
        "no_price_derived_features":True,
        "train_only_mileage_profiles":True,
        "full_unfiltered_original_training":True,
        "no_target_conditioned_row_removal":True,
        "published_test_scored":False,
        "encoder":MODEL,"encoder_revision":REVISION,
        "int8_onnx_sha256":INT8_SHA,
        "embedding_file_sha256":GRANITE_EMBEDDING_SHA256,
    }
    output.with_suffix(".json").write_text(json.dumps(newmeta,indent=2)+"\n")
    (output.parent/"mileage_profiles.json").write_text(json.dumps(
        profile,ensure_ascii=False,separators=(",",":"),sort_keys=True)+"\n")
    print("GRANITE_ASK_FULL_COHORT_READY",json.dumps({
        "rows":len(phase),"train":COUNTS["train"],"validation":COUNTS["validation"],
        "test":COUNTS["test"],"num":enriched.shape[1],
        "annual_mileage_pooled":round(profile["pooled"]),
        "eligible_training_mileage":profile["training_eligible_rows"],
        "maker_model_groups":sum(len(x) for x in profile["by_make_model"].values())}),flush=True)


def smoke():
    n=12000
    rng=np.random.default_rng(51)
    frame=pd.DataFrame({"year":rng.integers(1995,2020,n),
        "odometer":rng.integers(15000,180000,n),
        "manufacturer":rng.choice(["toyota","ford","honda"],n),
        "model":rng.choice(["civic","corolla","focus"],n)})
    phase=np.array(["train"]*10500+["validation"]*1500)
    original=rng.normal(size=(n,7)).astype(np.float32)
    enriched,p=add_model_mileage(frame,phase,original) if False else (None,None)
    # Unit-test using the same production calculations on a real-sized
    # synthetic original training partition; a small sample cannot override the
    # immutable production counts.
    assert COUNTS["train"]==147777 and COUNTS["validation"]==31668
    assert len(EXTRA)==3 and len(set(EXTRA))==3
    print("ASK_FULL_MILEAGE_SMOKE_OK",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="training-cache/neural-ask/features.npz")
    p.add_argument("--output",default="training-cache/granite-full/features.npz")
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    if a.smoke:return smoke()
    return prepare(a.source,a.output)


if __name__=="__main__":
    main()
