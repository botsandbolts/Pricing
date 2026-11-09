"""Package the verified 600-tree European AFT winner with EXACT train-only vocab.

Recomputes feature encoding from the original, unmodified Hugging Face Parquet;
compares sampled training rows against the frozen Ray input artifact. Only a
successful parity check produces a deployable bundle. Aggregate market summaries
contain no IDs, individual listings or seller contact/location details.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.europe_price_ray import (INPUT_COLUMNS,CATEGORICAL,CAT_CAPS,
    FEATURES,FEATURE_TYPES,MIN_PRICE,MAX_PRICE,split_ids,encode,ids_sha)
from pricing.europe_model import encode_vehicle,estimate


def pack(raw_path,model_path,comparison_path,fit_dir,output):
    import xgboost as xgb
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    source=pq.ParquetFile(str(raw_path))
    columns=source.schema_arrow.names
    data=pq.read_table(str(raw_path),
           columns=[c for c in INPUT_COLUMNS if c in columns]).to_pandas()
    for col in INPUT_COLUMNS:
        if col not in data:
            data[col]=None
    raw_count=len(data)
    price=pd.to_numeric(data["price_eur"],errors="coerce").to_numpy(dtype=float)
    good=np.isfinite(price)&(price>=MIN_PRICE)&(price<=MAX_PRICE)
    frame=data.loc[good].reset_index(drop=True)
    original_id=np.flatnonzero(good).astype(np.int64)
    phases=split_ids(frame)
    manifest=json.loads((Path(fit_dir)/"manifest.json").read_text())
    if (raw_count!=2003645 or len(frame)!=2003371 or
        manifest["sha256_by_phase"]["train"]!=
              "b9b099c94c04cf276cb0cddceedb745ceb57fe02b0980e55d363b196ef398822" or
        ids_sha(original_id[phases=="train"])!=manifest["sha256_by_phase"]["train"] or
        ids_sha(original_id[phases=="validation"])!=manifest["sha256_by_phase"]["validation"] or
        ids_sha(original_id[phases=="test"])!=manifest["sha256_by_phase"]["test"]):
        raise AssertionError("Dataset/split SHA drift. Refusing to publish model.")
    train=phases=="train"
    vocabulary={}
    for key in CATEGORICAL:
        raw=frame[key].fillna("__missing__").astype(str).str.strip().str.lower()
        raw=raw.replace("","__missing__")
        vals=raw[train].value_counts().index[:CAT_CAPS.get(key,4000)].tolist()
        vocabulary[key]={value:i for i,value in enumerate(vals)}
        if len(vals)!=manifest["categorical_cardinalities"][key]:
            raise AssertionError("Training-only category count disagrees for "+key)
    X,cats=encode(frame,phases)
    with np.load(Path(fit_dir)/"train.npz",allow_pickle=False) as stored:
        xs=stored["x"][:256]
        ids=stored["row_id"][:256]
        yy=stored["y"][:256]
    trainpos=np.flatnonzero(train)[:256]
    np.testing.assert_array_equal(ids,original_id[trainpos])
    np.testing.assert_allclose(X[trainpos],xs,rtol=0.,atol=0.,equal_nan=True)
    np.testing.assert_allclose(price[original_id[trainpos]],yy,rtol=0.,atol=0.)
    parsed=pd.to_datetime(frame["first_seen_at"],errors="coerce",utc=True,
                          format="mixed")
    if not parsed.notna().any():
        raise AssertionError("Cannot establish the training advertising date")
    # Median SOURCE date, not today's date; no extrapolation is claimed.
    snapshot_date=str(parsed.dropna().median().date())
    schema={
      "trained_model":"aft_normal","train_only_vocabulary":True,
      "feature_names":FEATURES,"feature_types":FEATURE_TYPES,
      "category_vocabulary":vocabulary,
      "snapshot_date":snapshot_date,
      "snapshot_date_type":"median first_seen_at in dataset",
      "training_script":"scripts/europe_price_ray.py::encode",
      "training_split_sha256":manifest["sha256_by_phase"]["train"],
      "validation_split_sha256":manifest["sha256_by_phase"]["validation"],
      "test_split_sha256":manifest["sha256_by_phase"]["test"],
      "source_dataset":"KTHESAT/european-used-car-market-2m-listings",
      "source_licence":"CC BY-NC-SA 4.0",
      "raw_rows":raw_count,
      "eligible_rows":len(frame),
      "model_sha256":hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
    }
    comp=json.loads(Path(comparison_path).read_text())
    if (comp["winning_model"]["origin"]!="eu-ask-candidate-ray-8"
         or comp["winning_model"]["family"]!="aft_normal"
         or comp["winning_model"]["iterations"]!=600):
        raise AssertionError("Research winner identity did not match")
    shutil.copy2(model_path,output/"winner.ubj")
    shutil.copy2(comparison_path,output/"comparison.json")
    (output/"schema.json").write_text(json.dumps(schema,ensure_ascii=False,indent=2))
    booster=xgb.Booster()
    booster.load_model(str(output/"winner.ubj"))
    checked=0
    # Check 20 representative rows in the original heldout training
    # representation, excluding only rows with genuinely absent required data.
    for p in np.flatnonzero(train)[:1200]:
        record=frame.iloc[p].to_dict()
        if any(pd.isna(record.get(key)) for key in
               ("seller_country","make","model","reg_year","mileage_km","first_seen_at")):
            continue
        year=int(record["reg_year"])
        km=float(record["mileage_km"])
        if not 1950<=year<=2026 or not 0<=km<=2_000_000:
            continue
        if pd.notna(record.get("power_hp")) and not 20<=float(record["power_hp"])<=1200:
            continue
        reg=pd.to_datetime(record.get("first_registration"),errors="coerce",utc=True)
        record["reg_month"]=int(reg.month) if pd.notna(reg) else None
        seen=pd.to_datetime(record["first_seen_at"],errors="coerce",utc=True)
        if pd.isna(seen):
            continue
        record["first_seen_at"]=str(seen.date())
        row=encode_vehicle(record,schema)
        np.testing.assert_allclose(row,X[p:p+1],rtol=1e-6,atol=1e-6,equal_nan=True)
        native=xgb.DMatrix(X[p:p+1],feature_names=FEATURES,
                          feature_types=FEATURE_TYPES,enable_categorical=True)
        expected=float(booster.predict(native)[0])
        actual=estimate(record,bundle=(booster,schema,comp))
        np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=0.)
        checked+=1
        if checked>=20:
            break
    if checked<20:
        raise AssertionError("Insufficient eligible source rows for 20-way inference parity")
    # Aggregate non-identifying asking-price market evidence; never present it
    # as verified physical-car dedupe or transaction-price data.
    summary=frame[["seller_country","make","model","reg_year","price_eur","mileage_km"]].copy()
    for col in ("seller_country","make","model"):
        summary[col]=summary[col].fillna("__missing__").astype(str).str.strip().str.lower()
    summary["reg_year"]=pd.to_numeric(summary["reg_year"],errors="coerce")
    grouped=summary.groupby(["seller_country","make","model","reg_year"],
                             dropna=True,sort=False,observed=True).agg(
        count=("price_eur","size"),
        median_ask_eur=("price_eur","median"),
        median_km=("mileage_km","median")).reset_index()
    grouped=grouped.loc[grouped["count"]>=8].sort_values(
        ["seller_country","count"],ascending=[True,False])
    grouped.to_csv(output/"market_stats.csv.gz",index=False,compression="gzip")
    report={
      "dataset":"original, unchanged",
      "source_rows":raw_count,"eligible_rows":len(frame),
      "train_only_vocab_parity_rows":len(trainpos),
      "production_inference_parity_rows":checked,
      "train_sha":manifest["sha256_by_phase"]["train"],
      "snapshot_date":snapshot_date,
      "market_aggregates":len(grouped),
      "sealed_test_cvAR95_eur":comp["winning_model"]["test_once"]["cvar95_absolute_eur"],
      "model_sha256":schema["model_sha256"],
      "verified":True}
    (output/"release_audit.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser()
    for field in ("raw","model","comparison","fit","output"):
        ap.add_argument("--"+field,required=True,type=Path)
    args=ap.parse_args()
    pack(args.raw,args.model,args.comparison,args.fit,args.output)


if __name__=="__main__":
    main()
