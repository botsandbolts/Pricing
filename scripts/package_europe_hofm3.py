"""Package frozen, validation-selected European HOFM3 without retraining.

Extract the ORIGINAL train-only categorical ID vocabulary from saved fit tensors,
not via a new value-count order. Reconstruct the exact country-normalized fresh
cohort from the verified 2M original-derived Parquet, recover frozen IDs,
demonstrate numeric/price and HOFM3 prediction parity on multiple training rows,
and publish a checksum-verified self-contained inference bundle.

Do not substitute Eurostat forecast-based targets for observed price_eur.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.europe_neural_nov2026_prepare import make_recent_cohort
from scripts.europe_price_ray import (
    INPUT_COLUMNS, CATEGORICAL, NUMERIC, FEATURES, FEATURE_TYPES,
    split_ids, ids_sha,
)
from pricing.europe_model import encode_vehicle
from pricing.europe_hofm3 import (
    SOURCE_RUN, SOURCE_ARTIFACT, SUPPORTED_COUNTRIES,
    _prepared, _architecture,
)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def package(raw_path, fit_dir, model_path, comparison_path, output):
    import torch
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    raw_path=Path(raw_path)
    fit_dir=Path(fit_dir)
    source=pq.ParquetFile(str(raw_path))
    needed=list(dict.fromkeys(
        [x for x in INPUT_COLUMNS if x in source.schema_arrow.names]+[
            "last_verified_at","eligible_as_current_price_training_label",
        ]
    ))
    raw=pq.read_table(str(raw_path),columns=needed).to_pandas()
    if len(raw)!=2003645:
        raise AssertionError("Unexpected source size, refusing checkpoint publication.")
    frame,freshness=make_recent_cohort(raw)
    del raw
    if len(frame)!=1920996:
        raise AssertionError("Fresh listing cohort drift from 1,920,996 training rows.")
    manifest=json.loads((fit_dir/"manifest.json").read_text())
    comp=json.loads(Path(comparison_path).read_text())
    winner=comp["overall_champion"]
    if (manifest.get("target")!="price_eur"
        or manifest.get("synthetic_hicp_prices_used_as_training_labels") is not False
        or manifest.get("historical_model_labels_excluded") is not True
        or manifest.get("hypothetical_mileage_used_as_training_input") is not False
        or manifest.get("counts")!={"train":1536801,"validation":192149,"test":192046}
        or manifest.get("eligible_rows")!=1920996
        or manifest.get("feature_names")!=FEATURES
        or winner.get("origin")!="eu-nov26-neural-13"
        or winner.get("architecture")!="hofm3"
        or winner.get("loss")!="tail_mae_eur"
        or winner.get("kind")!="neural"
        or winner.get("sealed_test_sha256")!=manifest["sha256_by_phase"]["test"]
        or comp.get("compared_models")!=15):
        raise AssertionError("Frozen winner identity / fresh split provenance mismatch")
    phases=split_ids(frame)
    for phase in ("train","validation","test"):
        original=np.flatnonzero(phases==phase).astype(np.int64)
        if ids_sha(original)!=manifest["sha256_by_phase"][phase]:
            raise AssertionError("Country-normalized fingerprint split drift: "+phase)
    selected=phases=="train"
    positions=np.flatnonzero(selected)
    with np.load(fit_dir/"train.npz",allow_pickle=False) as fit:
        train_x=fit["x"].astype(np.float32,copy=False)
        train_y=fit["y"].astype(np.float64,copy=False)
        train_ids=fit["row_id"].astype(np.int64,copy=False)
    if train_x.shape!=(len(positions),len(FEATURES)):
        raise AssertionError("Frozen training feature matrix shape mismatch.")
    np.testing.assert_array_equal(train_ids,positions.astype(np.int64))
    original_price=pd.to_numeric(frame.iloc[positions]["price_eur"],errors="coerce")
    np.testing.assert_allclose(
        train_y,original_price.to_numpy(float),rtol=0,atol=0,
    )
    n=len(NUMERIC)
    vocabulary={}
    for j,key in enumerate(CATEGORICAL):
        text=frame.loc[selected,key].fillna("__missing__").astype(str).str.strip().str.lower()
        text=text.replace("","__missing__")
        codes=train_x[:,n+j]
        valid=np.isfinite(codes)
        integers=codes[valid].astype(np.int32)
        if not np.array_equal(integers.astype(np.float32),codes[valid]):
            raise AssertionError("Non-integer frozen category ID: "+key)
        pairs=pd.DataFrame({
            "category":text.to_numpy()[valid], "id":integers,
        }).drop_duplicates()
        if pairs["category"].duplicated().any() or pairs["id"].duplicated().any():
            raise AssertionError("Categorical label/code mapping not one-to-one: "+key)
        mapping=dict(zip(pairs["category"],pairs["id"].astype(int)))
        expected=manifest["categorical_cardinalities"][key]
        if len(mapping)!=expected or set(mapping.values())!=set(range(expected)):
            raise AssertionError("Incomplete or noncontiguous train-vocabulary: "+key)
        check=text.map(mapping).to_numpy(dtype=np.float32,na_value=np.nan)
        np.testing.assert_allclose(check,codes,rtol=0,atol=0,equal_nan=True)
        vocabulary[key]=mapping
    if set(vocabulary["seller_country"])!=set(SUPPORTED_COUNTRIES):
        raise AssertionError("Country alphabet is not exactly eight supported ISO2 codes.")
    source_date=pd.to_datetime(frame["first_seen_at"],
                               errors="coerce",utc=True,format="mixed")
    if source_date.notna().sum()!=len(frame):
        raise AssertionError("Fresh data has missing verified observation date.")
    snapshot=str(source_date.median().date())
    schema=dict(
        trained_model="hofm3",
        model_name="HOFM3 Tail-MAE",
        train_only_vocabulary=True,
        feature_names=FEATURES,
        feature_types=FEATURE_TYPES,
        category_vocabulary=vocabulary,
        snapshot_date=snapshot,
        snapshot_date_type="median last_verified_at, 2026 fresh observed ASK cohort",
        training_script="scripts/europe_price_ray.py::encode",
        source_training_target="original_price_eur",
        automatic_eurostat_adjustment=False,
        source_run_id=SOURCE_RUN,
        source_artifact_id=SOURCE_ARTIFACT,
        source_dataset="KTHESAT/european-used-car-market-2m-listings",
        source_licence="CC BY-NC-SA 4.0",
        raw_rows=2003645,
        eligible_rows=len(frame),
        training_split_sha256=manifest["sha256_by_phase"]["train"],
        validation_split_sha256=manifest["sha256_by_phase"]["validation"],
        test_split_sha256=manifest["sha256_by_phase"]["test"],
        source_date_earliest=freshness["min_source_date"],
        source_date_latest=freshness["max_source_date"],
        selected_by="validation CVaR95, MAE tiebreak",
    )
    pack=torch.load(model_path,map_location="cpu",weights_only=True)
    if (pack.get("config")!=winner["config"]
        or pack.get("manifest_train_sha")!=schema["training_split_sha256"]
        or pack.get("manifest_valid_sha")!=schema["validation_split_sha256"]
        or pack.get("test_scored") is not False):
        raise AssertionError("Neural tensor metadata contradicts comparison winner.")
    levels=[len(vocabulary[k]) for k in CATEGORICAL]
    from scripts.europe_neural_nov2026_ray import DeepTabular, Prepared
    reference=DeepTabular(levels,pack["config"],pack["train_median_log"])
    reference.load_state_dict(pack["state_dict"],strict=True)
    reference.eval()
    from pricing.europe_hofm3 import _architecture as production_arch
    production=production_arch(levels,pack["config"],pack["train_median_log"])
    production.load_state_dict(pack["state_dict"],strict=True)
    production.eval()
    norm={"train_mean":pack["train_mean"],"train_std":pack["train_std"]}
    parity_count=0
    for i,p in enumerate(positions[:700]):
        record=frame.iloc[p].to_dict()
        if any(pd.isna(record.get(k)) for k in (
            "seller_country","make","model","reg_year",
            "mileage_km","first_seen_at",
        )):
            continue
        if not (1950<=int(record["reg_year"])<=2026 and
                0<=float(record["mileage_km"])<=2000000):
            continue
        hp=record.get("power_hp")
        if pd.notna(hp) and not (20<=float(hp)<=1200):
            continue
        # Represent null optional data exactly as the original feature encoder.
        for optional in ("doors","seats","images_count","accident_free","power_hp"):
            if pd.isna(record.get(optional)):record[optional]=None
        reg=pd.to_datetime(record.get("first_registration"),errors="coerce",utc=True)
        record["reg_month"]=int(reg.month) if pd.notna(reg) else None
        seen=pd.to_datetime(record["first_seen_at"],errors="coerce",utc=True)
        record["first_seen_at"]=str(seen.date())
        encoded=encode_vehicle(record,schema)
        np.testing.assert_allclose(encoded,train_x[i:i+1],rtol=1e-6,
                                   atol=1e-6,equal_nan=True)
        ref_input=Prepared(train_x[i:i+1],train_y[i:i+1],
            np.asarray(pack["train_mean"],dtype=np.float32),
            np.asarray(pack["train_std"],dtype=np.float32),levels)
        prod_num,prod_cat=_prepared(encoded,norm,schema)
        torch.testing.assert_close(prod_num,ref_input.num,rtol=0,atol=0)
        torch.testing.assert_close(prod_cat,ref_input.cat,rtol=0,atol=0)
        with torch.no_grad():
            predicted=production(prod_num,prod_cat)
            baseline=reference(ref_input.num,ref_input.cat)
            torch.testing.assert_close(predicted,baseline,rtol=1e-6,atol=1e-6)
        parity_count+=1
        if parity_count>=30:break
    if parity_count<30:
        raise AssertionError("Insufficient exact inference/encoding parity examples.")

    for name,src in (("champion.pt",model_path),("comparison.json",comparison_path)):
        shutil.copy2(src,output/name)
    (output/"schema.json").write_text(
        json.dumps(schema,indent=2,ensure_ascii=False)+"\n")
    checkpoint_hash=sha256(output/"champion.pt")
    release={
        "model":"HOFM3 Tail-MAE",
        "source_run_id":SOURCE_RUN,
        "source_artifact_id":SOURCE_ARTIFACT,
        "source_trial":"eu-nov26-neural-13",
        "checkpoint_sha256":checkpoint_hash,
        "schema_training_sha256":schema["training_split_sha256"],
        "schema_validation_sha256":schema["validation_split_sha256"],
        "sealed_test_sha256":schema["test_split_sha256"],
        "release_status":"verified exact observed-ASK HOFM3 checkpoint",
        "market":"Europe ASK only",
        "source_dates":snapshot,
        "automatic_eurostat_adjustment":False,
        "published_by":"verified artifact packaging; never retrain",
    }
    (output/"release_manifest.json").write_text(json.dumps(release,indent=2)+"\n")
    stats=frame[["seller_country","make","model","reg_year","price_eur","mileage_km"]].copy()
    for col in ("seller_country","make","model"):
        stats[col]=stats[col].fillna("__missing__").astype(str).str.strip().str.lower()
    stats["reg_year"]=pd.to_numeric(stats["reg_year"],errors="coerce")
    market=stats.groupby(
        ["seller_country","make","model","reg_year"],dropna=True,observed=True
    ).agg(count=("price_eur","size"),
          median_ask_eur=("price_eur","median"),
          median_km=("mileage_km","median")).reset_index()
    market=market.loc[market["count"]>=8]
    market.to_csv(output/"market_stats.csv.gz",compression="gzip",index=False)
    audit={
        "verified":True,"source_run_id":SOURCE_RUN,"checkpoint_sha256":checkpoint_hash,
        "observed_ask_rows":len(frame),
        "training_rows":len(train_y),"validation_rows":manifest["counts"]["validation"],
        "test_rows":manifest["counts"]["test"],
        "iso2_countries":sorted(SUPPORTED_COUNTRIES),
        "exact_saved_train_feature_and_label_rows":len(train_y),
        "input_and_forward_parity_samples":parity_count,
        "snapshot_date":snapshot,
        "market_aggregates":len(market),
        "sealed_test_cvar95_eur":winner["test_once"]["cvar95_absolute_eur"],
        "sealed_test_mae_eur":winner["test_once"]["mae_eur"],
        "sealed_test_rmse_eur":winner["test_once"]["rmse_eur"],
        "no_synthetic_time_adjustment":True,
    }
    (output/"release_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2),flush=True)
    return audit


def main():
    a=argparse.ArgumentParser()
    for key in ("raw","fit","model","comparison","output"):
        a.add_argument("--"+key,required=True,type=Path)
    args=a.parse_args()
    package(args.raw,args.fit,args.model,args.comparison,args.output)


if __name__=="__main__":
    main()
