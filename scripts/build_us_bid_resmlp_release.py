"""Build and validate the U.S. Bid ResMLP release from the archived winner.

Usage:
  python -m scripts.build_us_bid_resmlp_release --source .../features.npz \
      --winner .../winner-huber_usd --output .../us-bid-resmlp-20261111

Must run in a disposable GitHub Actions runner, NOT in the public app.
Produces a small frozen inference profile (no 550k-row parquet at serving).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from pricing.neural_bid_blend_live import CATS
from pricing.us_bid_resmlp_live import (
    COHORT_SHA, EXPECTED, PREPROCESS, RUN_ID, USBidResMLP)
from scripts.us_bid_ray_current_features import add_model_mileage

def sha(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()

def normalized(col):
    return (col.fillna("unknown").astype(str).str.lower()
            .str.strip().replace({"": "unknown", "nan": "unknown"}))

def profiles(frame, phase, audit):
    dates = pd.to_datetime(frame["date"],utc=True,errors="coerce",format="mixed")
    year = pd.to_numeric(frame["year"],errors="coerce").to_numpy(float)
    age = dates.dt.year.to_numpy(float) + (
        dates.dt.dayofyear.to_numpy(float)-1)/365.25 - year
    miles = pd.to_numeric(frame["odometer"],errors="coerce").to_numpy(float)
    annual = miles / np.maximum(age,.5)
    eligible = ((phase=="train") & np.isfinite(annual) & np.isfinite(age) &
                (age>=1) & (age<=30) & (annual>=1000) & (annual<=50000) &
                np.isfinite(miles) & (miles>=0) & (miles<=800000))
    make, model = normalized(frame["manufacturer"]),normalized(frame["model"])
    counts = pd.DataFrame({"make":make,"model":model,"annual":annual})
    train = counts.loc[eligible]
    by_pair = train.groupby(["make","model"])["annual"].agg(["median","size"])
    by_make = train.groupby("make")["annual"].agg(["median","size"])
    pairs = {}
    for (maker, model_name), row in by_pair.iterrows():
        if int(row["size"]) >= 100:
            pairs.setdefault(maker,{})[model_name] = float(row["median"])
    makes = {maker:float(row["median"]) for maker,row in by_make.iterrows()
             if int(row["size"]) >= 300}
    pooled = float(np.median(annual[eligible]))
    if int(eligible.sum()) != audit["training_records_for_expected_mileage"]:
        raise ValueError("Original TRAIN-only mileage membership changed")
    if abs(pooled-audit["overall_train_median_annual_miles"]) > .001:
        raise ValueError("Average mileage rate changed")
    return {
        "source_run_id": RUN_ID, "training_only": True,
        "extra_feature_count": 3,
        "input_basis": "2014/2015 observed auction mileage and age",
        "no_synthetic_current_odometer": True,
        "by_make_model": pairs, "by_make": makes, "pooled": pooled,
        "median": audit["extra_median_train_only"],
        "iqr": audit["extra_iqr_train_only"],
        "paired_model_groups": sum(len(models) for models in pairs.values()),
        "manufacturer_groups":len(makes),
        "training_eligible_rows":int(eligible.sum()),
    }

def build(source, winner, root, output):
    source,winner,root,output=map(Path,(source,winner,root,output))
    with np.load(source,allow_pickle=False) as archive:
        z = {k:archive[k] for k in ("num","cat","phase","ids","y")}
    meta=json.loads(source.with_suffix(".json").read_text())
    if (meta["cohort_sha256"]!=COHORT_SHA or
        meta["cardinalities"] != json.loads((root/PREPROCESS).read_text())["cardinalities"] or
        z["num"].shape!=(550181,8) or z["cat"].shape!=(550181,10)):
        raise ValueError("Original historical U.S. Bid training source changed")
    frame=pd.read_parquet(source.with_name("canonical-bid.parquet"))
    if (len(frame)!=len(z["ids"]) or not np.array_equal(
            frame["row_id"].astype(str).to_numpy(),z["ids"].astype(str))):
        raise ValueError("Original VIN-disjoint auction rows do not align")
    enriched,audit = add_model_mileage(frame,z["phase"].astype(str),
                                      z["num"][:,:5])
    if not np.allclose(enriched,z["num"],atol=1e-6,rtol=1e-6):
        delta=np.abs(enriched-z["num"])
        raise ValueError("Training numeric feature parity failed, max "+str(delta.max()))
    p=profiles(frame,z["phase"].astype(str),audit)
    output.mkdir(parents=True,exist_ok=True)
    (output/"mileage_profiles.json").write_text(
        json.dumps(p,ensure_ascii=False,separators=(",",":"),sort_keys=True)+"\n")
    result=json.loads((winner/"result.json").read_text())
    cfg=result["configuration"]
    if (result["family"]!="resmlp" or result["loss"]!="huber_usd" or
        result["best_epoch"]!=44 or result["source_sha256"]!=COHORT_SHA or
        result["training_full_non_lemon"] is not True or result["test_scored"] is not False):
        raise ValueError("Not the specified winner #1")
    for k,expected in EXPECTED.items():
        if not np.isclose(result["original_validation"][k],expected,atol=.002):
            raise ValueError("Winner validation metric changed: "+k)
    if int(result["original_validation"]["rows"])!=82380:
        raise ValueError("Wrong original validation cohort")
    shutil.copy2(winner/"weights.pt",output/"weights.pt")
    manifest={
        "status": "release_candidate_unverified", "market": "bid",
        "source_workflow_run":RUN_ID, "source_shard":4,
        "model":"ResMLP + Huber USD + train-only make/model annual mileage",
        "configuration":cfg,"cardinalities":meta["cardinalities"],
        "cohort_sha256":COHORT_SHA,
        "training_full_non_lemon":True,"test_scored":False,
        "original_validation":result["original_validation"],
        "training_inner_CVaR95_USD":result["inner_CVaR95_USD"],
        "best_epoch":result["best_epoch"],
        "checkpoint_sha256":sha(output/"weights.pt"),
        "preprocessing_sha256":sha(root/PREPROCESS),
        "mileage_profile_sha256":sha(output/"mileage_profiles.json"),
        "label": "2015-index-normalized 2014/15 wholesale sale prices",
        "not_live_2026_accuracy":True,
    }
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (output/"verification.json").write_text(
        json.dumps({"status":"pending"},indent=2)+"\n")
    engine=USBidResMLP(root,_parity_test=True) if False else None
    # Install into a throwaway root with the exact production loader.
    checkroot=output.parent/"parity-root"
    if checkroot.exists():
        shutil.rmtree(checkroot)
    target=checkroot/"models/us-bid-resmlp-20261111"
    target.mkdir(parents=True)
    for file in ("weights.pt","manifest.json","mileage_profiles.json","verification.json"):
        shutil.copy2(output/file,target/file)
    prep_path=checkroot/PREPROCESS
    prep_path.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/PREPROCESS,prep_path)
    model=USBidResMLP(checkroot,_parity_test=True)
    # Compare exact original archived validation predictions against raw historic
    # fields transformed by the NEW live inference pipeline, never test rows.
    actual=pd.read_csv(winner/"validation.csv.gz",dtype={"row_id":str})
    if len(actual)!=82380 or actual["row_id"].duplicated().any():
        raise ValueError("Archived original validation receipts invalid")
    valid=frame.loc[z["phase"].astype(str)=="validation"].copy()
    if set(valid.row_id.astype(str))!=set(actual.row_id):
        raise ValueError("Original validation row identities changed")
    chosen=actual.iloc[np.linspace(0,len(actual)-1,128,dtype=int)]
    raw=valid.set_index(valid.row_id.astype(str),drop=False)
    maximum=0.0
    for row in chosen.itertuples(index=False):
        original=raw.loc[str(row.row_id)]
        condition=("condition" if "condition" in raw.columns
                   else "auction_condition")
        fields={"year":int(original["year"]),
                "odometer":float(original["odometer"]),
                "auction_date":str(original["date"])[:10],
                "auction_grade":original[condition]}
        for name in CATS:
            fields[name]=original[name] if name in original else "unknown"
        prediction=model.predict(fields)["price"]
        deviation=abs(float(prediction)-float(row.predicted_price))
        maximum=max(maximum,deviation)
        if not np.isclose(prediction,float(row.predicted_price),atol=.35,rtol=3e-5):
            raise ValueError(f"Serving validation parity failed row={row.row_id}: "
                             f"deviation={deviation}, prediction={prediction}, "
                             f"archive={row.predicted_price}")
    verification={
        "status":"passed",
        "source_workflow_run": RUN_ID,
        "original_validation_rows":82380,
        "validation_prediction_parity_passed":True,
        "validation_sampled_rows":len(chosen),
        "validation_sample_max_abs_usd":maximum,
        "training_numeric_feature_parity_passed":True,
        "no_test_scoring":True,
        "serving_schema":"8_num_10_cat",
        "production_modified":False,
    }
    manifest["status"]="verified_for_live_historical_bid"
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (output/"verification.json").write_text(json.dumps(verification,indent=2)+"\n")
    print("US_BID_RESMLP_RELEASE_VERIFIED",json.dumps({
        "validation_CVaR95":manifest["original_validation"]["CVaR95_absolute_USD"],
        "validation_MAE":manifest["original_validation"]["MAE_USD"],
        "validation_RMSE":manifest["original_validation"]["RMSE_USD"],
        "sample_max_abs_usd":maximum,
        "profile_bytes":(output/"mileage_profiles.json").stat().st_size,
        "checkpoint_sha256":manifest["checkpoint_sha256"]}),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--winner",type=Path,required=True)
    parser.add_argument("--root",type=Path,default=Path("."))
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    build(a.source,a.winner,a.root,a.output)
if __name__=="__main__":
    main()
