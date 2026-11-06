"""Freeze the ORIGINAL wholesale auction cohort for 18 neural-network experiments.

Inputs are archived original full and Lemon-cleaned XGBoost bid evidence. The
original VIN-disjoint chronological heldouts and price target stay unchanged.
The historic auction records have no admissible pre-sale seller descriptions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

CATS = ("manufacturer", "model", "type", "transmission", "state",
        "fuel", "drive", "title_status", "cylinders", "paint_color")
TOP = {"manufacturer":160, "model":1800, "type":45, "transmission":35,
       "state":90, "fuel":30, "drive":30, "title_status":25,
       "cylinders":30, "paint_color":50}
EXPECTED = {"train":384198, "validation":82380, "test":83603}
EXPECTED_LEMON = 288148

def sha_ids(ids):
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()

def phases(frame):
    if frame.row_id.astype(str).duplicated().any():
        raise ValueError("Duplicate original auction row identities")
    if frame.vehicle_group.astype(str).duplicated().any():
        raise ValueError("Repeated VIN/vehicle could leak across holdouts")
    mapping = {"calibration":"train","validation":"validation","test":"test"}
    result = frame.split.map(mapping)
    if result.isna().any() or result.value_counts().to_dict() != EXPECTED:
        raise ValueError("Original auction train/validation/test split drift")
    if result.ne(result.shift()).sum() != 3:
        raise ValueError("Original chronological partitions are not contiguous")
    return result.to_numpy(dtype="U10")

def lemon_membership(full, trimmed, phases_array):
    """Match by exact frozen row ID and ensure ALL validation/test are preserved."""
    if trimmed.row_id.astype(str).duplicated().any():
        raise ValueError("Duplicate frozen Lemon identities")
    joined = trimmed[["row_id","split","observed_price"]].merge(
        full[["row_id","split","observed_price"]],on="row_id",how="left",
        suffixes=("_lemon","_full"),indicator=True,validate="one_to_one")
    if not joined._merge.eq("both").all() or not joined.split_lemon.eq(joined.split_full).all():
        raise ValueError("Frozen Lemon rows differ from original auction identities/phases")
    if not np.allclose(joined.observed_price_lemon,joined.observed_price_full,
                       rtol=1e-6,atol=.02):
        raise ValueError("Frozen Lemon target drift versus original XGBoost")
    allowed = set(trimmed.row_id.astype(str))
    keep = full.row_id.astype(str).isin(allowed).to_numpy()
    if not keep[phases_array != "train"].all():
        raise ValueError("Frozen Lemon removed original validation/test auctions")
    if int(keep[phases_array=="train"].sum()) != EXPECTED_LEMON:
        raise ValueError("Incorrect Lemon training membership count")
    if len(trimmed) != EXPECTED_LEMON+EXPECTED["validation"]+EXPECTED["test"]:
        raise ValueError("Unexpected archived Lemon auction cohort size")
    return keep

def numeric_features(frame):
    date = pd.to_datetime(frame.date,utc=True,format="mixed",errors="coerce")
    year = pd.to_numeric(frame.year,errors="coerce").to_numpy(float)
    miles = pd.to_numeric(frame.odometer,errors="coerce").to_numpy(float)
    miles = np.where(np.isfinite(miles) & (miles>=0),miles,np.nan)
    age = np.maximum(date.dt.year.to_numpy(float)+
                     (date.dt.dayofyear.to_numpy(float)-1)/365.25-year,0.)
    col = "condition" if "condition" in frame else "auction_condition"
    grade = pd.to_numeric(frame[col],errors="coerce").to_numpy(float)
    grade = np.where((grade>=1)&(grade<=5),grade,np.nan)
    # Input facts only; price, OOF residuals and auction identifiers are excluded.
    return np.column_stack((year,age,np.log1p(miles),
                            np.log1p(miles/np.maximum(age,1)),grade))

def encode_features(frame,phase):
    train=phase=="train"
    n=numeric_features(frame)
    med=np.nanmedian(n[train],axis=0)
    med[~np.isfinite(med)]=0
    n=np.where(np.isfinite(n),n,med)
    q25,q75=np.percentile(n[train],[25,75],axis=0)
    n=np.clip((n-med)/np.maximum(q75-q25,1),-8,8).astype(np.float32)
    c=np.zeros((len(frame),len(CATS)),dtype=np.int32)
    sizes=[];vocabulary={}
    for i,name in enumerate(CATS):
        raw=frame[name] if name in frame else pd.Series("unknown",index=frame.index)
        values=raw.fillna("unknown").astype(str).str.lower().str.strip()
        freq=values[train].value_counts()
        kept=[v for v in freq.head(TOP[name]).index
              if v and v not in ("unknown","nan","none") and freq[v]>=10]
        mapping={v:j+2 for j,v in enumerate(kept)}
        c[:,i]=values.map(mapping).fillna(1).to_numpy(dtype=np.int32)
        sizes.append(len(mapping)+2)
        vocabulary[name]={"retained_levels":len(mapping),"unknown":1,"padding":0}
    # The shared NN architecture has a text input interface. Its SINGLE constant
    # zero dimension is a compatibility tensor, not synthetic descriptions.
    dummy_text=np.zeros((len(frame),1),dtype=np.float32)
    return n,c,dummy_text,sizes,vocabulary

def prepare(full_path,lemon_path,output):
    frame=pd.read_csv(full_path,dtype={"row_id":str,"vehicle_group":str,"split":str})
    clean=pd.read_csv(lemon_path,dtype={"row_id":str,"vehicle_group":str,"split":str})
    if len(frame)!=sum(EXPECTED.values()):
        raise ValueError(f"Published bid source cohort has drifted: {len(frame)}")
    phase=phases(frame)
    keep=lemon_membership(frame,clean,phase)
    y=pd.to_numeric(frame.observed_price,errors="coerce").to_numpy(float)
    if not np.isfinite(y).all() or (y<=0).any():
        raise ValueError("Nonpositive or nonfinite original normalized auction prices")
    if "condition" not in frame and "auction_condition" not in frame:
        raise ValueError("Original auction pre-sale condition is absent")
    n,c,t,card,vocab=encode_features(frame,phase)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,num=n,cat=c,txt=t,y=y.astype(np.float32),
                        ids=frame.row_id.to_numpy(dtype="U80"),phase=phase,
                        lemon=keep.astype(np.bool_))
    parquet=output.with_name("canonical-bid.parquet")
    frame.to_parquet(parquet,engine="pyarrow",compression="zstd",index=False)
    hashes={name:sha_ids(frame.loc[phase==name,"row_id"]) for name in ("validation","test")}
    manifest={
      "market":"bid",
      "dataset":"Original VIN-deduplicated 2014/15 wholesale auction sales",
      "target":"Original XGBoost 2015-index-normalized wholesale US dollars, not live bids",
      "full_train":EXPECTED["train"],"lemon_train":EXPECTED_LEMON,
      "validation":EXPECTED["validation"],"test":EXPECTED["test"],
      "cohort_rows":len(frame),"holdout_row_sha256":hashes,
      "full_evidence":str(full_path),"lemon_evidence":str(lemon_path),
      "no_description":True,"no_price_derived_inputs":True,"no_hierarchy":True,
      "text_feature":"none (single constant zero compatibility tensor)",
      "numeric_features":["model_year","auction_vehicle_age","log_odometer",
                          "log_odometer_per_vehicle_age","pre_sale_condition_grade"],
      "categories":vocab,"cardinalities":card,
      "num_features":n.shape[1],"text_features":t.shape[1],
      "split_policy":"Original chronological, VIN-disjoint validation/test unchanged",
      "outlier_policy":"Full vs frozen Lemon 25% TRAIN-only exclusion; never trim holdouts",
      "kaggle_reuse":parquet.name}
    output.with_suffix(".json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!="categories"},indent=2),flush=True)
    print("Reusable historical auction Parquet",parquet,parquet.stat().st_size,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full",default="evidence/bid-minilm-oof.csv.gz")
    p.add_argument("--lemon",default="evidence/bid-minilm-cleaned.csv.gz")
    p.add_argument("--output",default="training-cache/neural-bid/features.npz")
    a=p.parse_args()
    prepare(a.full,a.lemon,a.output)
