"""Verify and stage Granite DCNv2 AFT release for the SOLE production Ask.

Strict original row-ID, trained tensor and raw-to-live feature parity. No
test labels, no historical test metric substitution, no new model tuning.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from scripts.granite_ask_full_ray_4h import AskNetwork,price_predictions
from pricing.granite_aft_ask_live import (
    FOLDER, METRICS, SOURCE_RUN, SHARD, COHORT_SHA, EMBEDDING_SHA, ONNX_SHA,
    GraniteAFTAsking,available)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build(features,embedding,source,winner,root,output):
    features,embedding,source,winner,root,output=map(
        Path,(features,embedding,source,winner,root,output))
    zmeta=json.loads(features.with_suffix(".json").read_text())
    emeta=json.loads((embedding/"manifest.json").read_text())
    result=json.loads((winner/"result.json").read_text())
    config=result["configuration"]
    if not (
        zmeta["cohort_sha256"]==COHORT_SHA
        and zmeta.get("train_only_mileage_profiles") is True
        and zmeta["full_train"]==147777 and zmeta["validation"]==31668
        and zmeta["test"]==31682
        and result["shard"]==SHARD
        and result["loss"]=="aft" and result["family"]=="dcnv2"
        and result["best_epoch"]==76
        and result["source_sha256"]==COHORT_SHA
        and result["embedding_sha256"]==EMBEDDING_SHA
        and result["original_test_scored"] is False
        and emeta["embedding_file_sha256"]==EMBEDDING_SHA
        and emeta["model_onnx_sha256"]==ONNX_SHA
        and emeta["encoder_inference"]=="INT8 ONNX"
        and emeta["test_labels_used_in_encoder"] is False
    ):
        raise ValueError("Frozen IBM Granite AFT winning cohort does not match release")
    if not all(math.isclose(float(result["original_validation"][key]),value,abs_tol=.002)
               for key,value in METRICS.items()):
        raise ValueError("Saved Granite AFT validation metrics changed")
    if not abs(float(config["tail_strength"])-.35)<1e-6:
        raise ValueError("AFT training penalty differs from verified Ray study")
    if sha(embedding/"embeddings.npy")!=EMBEDDING_SHA:
        raise ValueError("Granite INT8 embeddings do not match source SHA")
    with np.load(features,allow_pickle=False) as data:
        z={k:data[k] for k in ("num","cat","y","phase","ids")}
    with np.load(source,allow_pickle=False) as data:
        old={k:data[k] for k in ("num","cat","y","phase","ids")}
    if (z["num"].shape!=(211127,10) or z["cat"].shape!=(211127,11)
            or old["num"].shape!=(211127,7)
            or not np.array_equal(z["num"][:,:7],old["num"])
            or not np.array_equal(z["cat"],old["cat"])
            or not np.array_equal(z["ids"],old["ids"])
            or not np.array_equal(z["phase"],old["phase"])
            or not np.array_equal(z["y"],old["y"])):
        raise ValueError("Full original Ask cohort changed during feature enhancement")
    phase=z["phase"].astype(str)
    assert (int((phase=="train").sum()),int((phase=="validation").sum()),
            int((phase=="test").sum()))==(147777,31668,31682)
    preproc_path=root/"models/neural-experimental-ask-bge/preprocessing.json"
    preproc=json.loads(preproc_path.read_text())
    assert preproc["cardinalities"]==zmeta["cardinalities"]
    output.mkdir(parents=True,exist_ok=True)
    shutil.copy2(winner/"weights.pt",output/"weights.pt")
    shutil.copy2(preproc_path,output/"preprocessing.json")
    shutil.copy2(features.parent/"mileage_profiles.json",output/"mileage_profiles.json")
    packed=torch.load(output/"weights.pt",map_location="cpu",weights_only=True)
    if (packed["source_sha256"]!=COHORT_SHA
            or packed["embedding_sha256"]!=EMBEDDING_SHA
            or packed["config"]!=config
            or packed["cards"]!=preproc["cardinalities"]
            or packed["original_test_scored"] is not False):
        raise ValueError("Weights/model configuration changed since original validation")
    model=AskNetwork(packed["cards"],float(packed["base"]),config)
    model.load_state_dict(packed["state_dict"],strict=True)
    model.eval()
    rows=np.flatnonzero(phase=="validation")
    assert len(rows)==31668
    archived=pd.read_csv(winner/"validation.csv.gz",dtype={"row_id":str})
    if (len(archived)!=31668 or archived["row_id"].duplicated().any()
            or not np.array_equal(archived["row_id"].astype(str).to_numpy(),
                                  z["ids"][rows].astype(str))):
        raise ValueError("Archived validation row IDs don't match ORIGINAL chronological validation")
    indices=np.linspace(0,len(rows)-1,128,dtype=int)
    emb=np.load(embedding/"embeddings.npy",mmap_mode="r")
    predicted=price_predictions(model,z["num"][rows[indices]],
               z["cat"][rows[indices]],emb[rows[indices]],batch=512)
    expected=archived.iloc[indices]["predicted_price"].to_numpy(dtype=float)
    diff=np.abs(predicted-expected)
    if not np.allclose(predicted,expected,rtol=3e-5,atol=.2):
        raise ValueError(f"Training-to-serve prediction parity failed: {diff.max()} USD")
    manifest={
        "status":"release_candidate_not_verified",
        "model":"DCNv2 + AFT",
        "training_objective":"aft_with_train_only_tail_penalty",
        "source_run_id":SOURCE_RUN,"source_shard":SHARD,
        "cohort_sha256":COHORT_SHA,"embedding_sha256":EMBEDDING_SHA,
        "encoder_onnx_sha256":ONNX_SHA,
        "encoder_model":"ibm-granite/granite-embedding-97m-multilingual-r2",
        "encoder_revision":"c61e626a6255c490879d0af885078b61929d51f6",
        "encoder_runtime":"pinned QInt8 ONNX CPU",
        "price_network_precision":"native PyTorch FP32",
        "original_training_rows":147777,"train_fit_rows":133000,
        "train_only_stopper_rows":14777,"test_scored":False,
        "best_epoch":76,"configuration":config,
        "cardinalities":preproc["cardinalities"],
        "validation":result["original_validation"],
        "mileage_profiles_training_only":True,
        "not_verified_2026_market_accuracy":True,
        "checkpoint_sha256":sha(output/"weights.pt"),
        "preprocessing_sha256":sha(output/"preprocessing.json"),
        "mileage_profile_sha256":sha(output/"mileage_profiles.json"),
    }
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (output/"verification.json").write_text(
        json.dumps({"status":"pending"})+"\n")
    # Production implementation, with the exact saved network, without downloading
    # the encoder: test its raw-fields -> 10 numeric + 11 categorical path.
    checkroot=output.parent.parent/"parity-root"
    checkfolder=checkroot/FOLDER
    checkfolder.mkdir(parents=True,exist_ok=True)
    for name in ("weights.pt","manifest.json","verification.json",
                 "preprocessing.json","mileage_profiles.json"):
        shutil.copy2(output/name,checkfolder/name)
    serving=GraniteAFTAsking(checkroot,_parity_test=True)
    original=pd.read_parquet(source.with_name("canonical-ask.parquet"))
    if len(original)!=211127 or not np.array_equal(
            original.row_id.astype(str).to_numpy(),z["ids"].astype(str)):
        raise ValueError("Raw original validation rows differ from frozen features")
    chosen=rows[indices]
    # Test the native preprocessing against frozen numeric and category tensors.
    valid_feature_rows=[]
    for index in chosen:
        row=original.iloc[int(index)]
        year=float(row["year"])
        miles=float(row["odometer"])
        if not (np.isfinite(year) and np.isfinite(miles)
                and 1900<=year<=2021 and 0<=miles<=2e6):
            continue
        fields={k:row[k] if k in original.columns and pd.notna(row[k])
                else "unknown" for k in serving.maps}
        fields.update(year=int(year),odometer=float(miles))
        raw=row["text"] if pd.notna(row["text"]) else ""
        num,cat,_=serving.encode_inputs(fields,{
            "raw":str(raw),
            "embedding":np.zeros((1,384),dtype=np.float32)})
        if not np.allclose(num[0],z["num"][index],atol=1e-4,rtol=1e-4):
            raise ValueError("LIVE field normalization differs from frozen train/validation at row "+
                             str(z["ids"][index])+"; max abs "+
                             str(np.max(np.abs(num[0]-z["num"][index]))))
        if not np.array_equal(cat[0],z["cat"][index]):
            raise ValueError("LIVE categorical encoding differs from train/validation")
        valid_feature_rows.append(str(z["ids"][index]))
    if len(valid_feature_rows)<60:
        raise ValueError("Insufficient full-cohort raw/live feature parity coverage")
    # All verification is complete before declaring a serving-ready checkpoint.
    validation={
        "status":"passed","source_run_id":SOURCE_RUN,
        "original_validation_prediction_parity_passed":True,
        "sampled_original_validation_predictions":len(indices),
        "validation_prediction_max_abs_usd":float(diff.max()),
        "live_feature_parity_passed":True,
        "live_feature_parity_rows":len(valid_feature_rows),
        "original_test_not_scored":True,
        "known_training_label_year":2021,
        "encoder_provenance_sha_verified":True,
        "no_competing_ask_backend":True,
    }
    manifest["status"]="verified_for_live_historical_ask"
    (output/"verification.json").write_text(json.dumps(validation,indent=2)+"\n")
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    # Test the actual fail-closed production gate.
    shutil.copy2(output/"verification.json",checkfolder/"verification.json")
    shutil.copy2(output/"manifest.json",checkfolder/"manifest.json")
    if not available(checkroot):
        raise ValueError("Release is not accepted by its production verification gate")
    print("GRANITE_AFT_RELEASE_VERIFIED",json.dumps({
        "validation":result["original_validation"],
        "raw_feature_parity_rows":len(valid_feature_rows),
        "prediction_max_abs_usd":float(diff.max()),
        "checkpoint_sha256":manifest["checkpoint_sha256"]}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features",required=True)
    p.add_argument("--embedding",required=True)
    p.add_argument("--source",required=True)
    p.add_argument("--winner",required=True)
    p.add_argument("--root",default=".")
    p.add_argument("--output",required=True)
    a=p.parse_args()
    build(a.features,a.embedding,a.source,a.winner,a.root,a.output)

if __name__=="__main__":
    main()
