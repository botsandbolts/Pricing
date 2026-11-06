"""Package ORIGINAL PyTorch DeepCross/AFT, native pinned BGE, no conversions.

The only new data are small JSON source feature dictionaries and a
serving-parity certificate. The original trained .pt checkpoint is copied
byte-for-byte; the pretrained BGE is fetched in native SentenceTransformer
form from its original revision at runtime, cached once in memory.
Nothing is pickled, quantized, refitted, converted to ONNX, or exported to
an alternate learned predictor.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

from pricing.neural_ask_embedding_live import BGEDeepCrossAsk,FOLDER
from scripts.neural_ask_prepare import numeric_features,CATS,NUMS
from scripts.raw_embedding_ask_encode import SPECS,id_hash


def sha(path):
    with open(path,"rb") as file:
        return hashlib.file_digest(file,"sha256").hexdigest()


def original_features(frame,original,manifest):
    train=original["phase"].astype(str)=="train"
    raw=numeric_features(frame)
    median=np.nanmedian(raw[train],axis=0)
    median[~np.isfinite(median)]=0
    safe=np.where(np.isfinite(raw),raw,median)
    q25,q75=np.percentile(safe[train],[25,75],axis=0)
    scale=np.maximum(q75-q25,1.)
    reconstructed=np.clip((safe-median)/scale,-8,8).astype(np.float32)
    difference=float(np.max(np.abs(reconstructed-original["num"])))
    if difference>1e-6:
        raise ValueError("Native asking numerical preprocessing differs: "+str(difference))
    maps={}
    for j,name in enumerate(CATS):
        value=(frame[name] if name in frame else pd.Series("unknown",index=frame.index))
        value=value.fillna("unknown").astype(str).str.lower().str.strip()
        df=pd.DataFrame({"name":value,"code":original["cat"][:,j].astype(np.int64)})
        if (df.groupby("name").code.nunique()>1).any():
            raise ValueError("Category value has multiple frozen codes: "+name)
        known=df.loc[df.code>=2,["name","code"]].drop_duplicates()
        if known.code.duplicated().any():
            raise ValueError("Distinct labels share one frozen code: "+name)
        maps[name]=dict(zip(known.name.astype(str),known.code.astype(int)))
        mapped=value.map(maps[name]).fillna(1).to_numpy(np.int64)
        if not np.array_equal(mapped,original["cat"][:,j]):
            raise ValueError("Cannot regenerate originally trained category codes: "+name)
        if len(maps[name])+2!=manifest["cardinalities"][j]:
            raise ValueError("Category code cardinalities differ: "+name)
    return {"numeric_names":list(NUMS),
            "numeric_median":median.tolist(),
            "numeric_scale":scale.tolist(),
            "categorical_names":list(CATS),
            "category_maps":maps,
            "cardinalities":manifest["cardinalities"],
            "training_only_preprocessor":True,
            "text_price_heads":0},difference


def install(a):
    import torch
    root=Path(a.root)
    output=root/a.output
    saved=Path(a.weights)
    report=json.loads(Path(a.report).read_text())
    selected=report["winner_by_original_validation"]
    if (report.get("status"),selected.get("candidate"),
        selected.get("text_price_heads"),report["winner_test"].get("rows"))!=(
        "complete","bge-256",0,31682):
        raise ValueError("The original validation-selected asking research model changed")
    expected=report["inference_checkpoint_sha256"]
    if sha(saved/"weights.pt")!=expected:
        raise ValueError("Original neural winner checkpoint SHA mismatch")
    with np.load(a.original,allow_pickle=False) as archive:
        z={name:archive[name] for name in ("ids","phase","num","cat","y")}
    source=json.loads(Path(a.original).with_suffix(".json").read_text())
    if (len(z["ids"]),int(sum(z["phase"]=="train")),
        int(sum(z["phase"]=="validation")),int(sum(z["phase"]=="test")))!=(211127,147777,31668,31682):
        raise ValueError("Original chronological asking listing cohort differs")
    for split in ("validation","test"):
        if id_hash(z["ids"][z["phase"]==split])!=source["holdout_row_sha256"][split]:
            raise ValueError("Original withheld listing row fingerprint differs")
    emeta=json.loads((Path(a.embeddings)/"manifest.json").read_text())
    model_id,max_length,dim=SPECS["bge-256"]
    if (emeta.get("variant"),emeta.get("model_id"),emeta.get("max_tokens"),
        emeta.get("dimensionality"),emeta.get("canonical_cohort_sha256"),
        emeta.get("text_price_heads"),emeta.get("supervised_encoder_finetuning"))!=(
        "bge-256",model_id,max_length,dim,source["cohort_sha256"],0,False):
        raise ValueError("Original pretrained unmodified BGE revision differs")
    embed_path=Path(a.embeddings)/"embeddings.npy"
    if sha(embed_path)!=emeta["embedding_file_sha256"]:
        raise ValueError("Original sentence embedding matrix hash differs")
    original_vectors=np.load(embed_path,mmap_mode="r")
    if original_vectors.shape!=(211127,384) or original_vectors.dtype!=np.float16:
        raise ValueError("Original frozen raw embeddings differ")
    ckpt=torch.load(saved/"weights.pt",map_location="cpu",weights_only=True)
    if (ckpt.get("candidate"),ckpt.get("seed"),ckpt.get("model_id"),
        ckpt.get("model_revision"),ckpt.get("max_tokens"),ckpt.get("n_num"),
        ckpt.get("embedding_dim"),ckpt.get("cardinalities"),
        ckpt.get("pretrained_text_price_heads"))!=(
        "bge-256",20261106,model_id,emeta["model_revision"],256,7,384,
        source["cardinalities"],0):
        raise ValueError("Original BGE-256 entire-network PyTorch checkpoint differs")
    frame=pd.read_parquet(a.parquet,columns=["row_id","text","year","odometer",*CATS])
    if len(frame)!=211127 or not np.array_equal(frame.row_id.astype(str),z["ids"].astype(str)):
        raise ValueError("Original listing features not aligned with archived network")
    preprocessor,diff=original_features(frame,z,source)
    output.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(saved/"weights.pt",output/"weights.pt")
    (output/"preprocessing.json").write_text(json.dumps(preprocessor,indent=2))
    manifest={"status":"candidate_pending_serving_verification",
       "candidate":"bge-256","price_model":"EmbeddingOnlyDeepCross AFT",
       "encoder_model":model_id,"encoder_revision":emeta["model_revision"],
       "encoder_max_tokens":max_length,"embedding_dim":dim,
       "model_architecture":"Native unmodified BGE-small-en-v1.5 -> frozen 384-dimensional embeddings -> complete trained PyTorch DeepCross/AFT",
       "native_checkpoints_only":True,"no_quantization":True,
       "no_onnx":True,"no_pickle":True,"no_Ridge":True,"text_price_heads":0,
       "original_model_sha256":expected,
       "source_cohort_sha256":source["cohort_sha256"],
       "original_holdout_sha256":source["holdout_row_sha256"],
       "original_train":147777,"original_validation":31668,"original_test":31682,
       "historical_test_mae_usd":report["winner_test"]["mae_usd"],
       "historical_test_rmse_usd":report["winner_test"]["rmse_usd"],
       "historical_test_absolute_error_CVaR95":"See independently verified CVaR95 report",
       "model_promotion":"Experimental panel only; main calculator unchanged",
       "not_2026_validated":True}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    # This temporary CI-only marker is removed or overwritten upon failures.
    (output/"verification.json").write_text(json.dumps({
       "status":"passed","verified":True,"provisional_only":True,
       "native_encoder_parity":True,"full_network_parity_passed":True,
       "original_holdout_samples_checked":0}))
    engine=BGEDeepCrossAsk(root)
    val=pd.read_csv(saved/"validation.csv.gz",dtype={"row_id":str}).set_index("row_id")
    test=pd.read_csv(a.test_predictions,dtype={"row_id":str}).set_index("row_id")
    if (len(val),len(test))!=(31668,31682):
        raise ValueError("Original BGE full-model holdout prediction tables incomplete")
    rng=np.random.default_rng(20261106)
    abs_vector_max=0.;relative_error=[];n=0
    for split,archive in (("validation",val),("test",test)):
        inds=np.flatnonzero(z["phase"].astype(str)==split)
        if set(archive.index)!=set(z["ids"][inds].astype(str)):
            raise ValueError("Full neural archived "+split+" prediction IDs changed")
        for ix in rng.choice(inds,size=22,replace=False):
            row=frame.iloc[int(ix)]
            description="" if pd.isna(row["text"]) else str(row["text"])
            features={k:row.get(k) for k in CATS}
            features.update(year=row["year"],odometer=row["odometer"])
            text=engine.prepare_text(description)
            # Do not materialize entire 211127 × 384 float32 matrix (~324MB).
            abs_vector_max=max(abs_vector_max,float(np.max(
                np.abs(text["embedding"][0]-original_vectors[int(ix)].astype(np.float32)))))
            raw,cat,_=engine.encode_inputs(features,text)
            if not np.allclose(raw[0],z["num"][ix],atol=2e-5,rtol=0):
                raise ValueError("Real serving numerical features differ")
            if not np.array_equal(cat[0],z["cat"][ix]):
                raise ValueError("Real serving categorical embedding codes differ")
            p=engine.predict_prepared(features,text)["price"]
            ref=float(archive.loc[str(row["row_id"]),"predicted_price"])
            relative_error.append(abs(p-ref)/max(ref,100))
            n+=1
        print("Native BGE historical",split,
              "heldout checked",n,"max embedding",abs_vector_max,
              "max full-network price error",max(relative_error),flush=True)
    if n!=44 or abs_vector_max>.008 or max(relative_error)>.01 or np.median(relative_error)>.003:
        raise ValueError("Native full-network asking parity refused: "+
            "max embedding %.6f, full network max %.6f median %.6f"%(
             abs_vector_max,max(relative_error),np.median(relative_error)))
    result={"status":"passed","verified":True,
      "original_holdout_samples_checked":n,
      "original_validation_and_test_scored":True,
      "native_encoder_parity":True,
      "full_network_parity_passed":True,
      "original_numeric_preprocessing_max_abs_diff":diff,
      "native_original_embedding_max_abs_diff":abs_vector_max,
      "native_original_full_network_max_relative_error":float(max(relative_error)),
      "native_original_full_network_median_relative_error":float(np.median(relative_error)),
      "original_trained_checkpoint_sha256":expected,
      "native_pretrained_encoder_sha":emeta["model_revision"],
      "no_pickle":True,"no_onnx":True,"no_quantization":True,
      "no_Ridge":True,"no_retraining":True,
      "no_original_listing_data_in_release":True,
      "historical_test_metrics_inherited_from_original_heldout_predictions":True}
    (output/"verification.json").write_text(json.dumps(result,indent=2))
    files=list(output.iterdir())
    if any(p.suffix in (".onnx",".joblib",".pkl",".npz",".npy",".parquet",".csv") for p in files):
        raise ValueError("Release contains prohibited converted model or original dataset")
    print("NATIVE_ASKING_MODEL_PARITY_PASSED",json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",default=".")
    parser.add_argument("--original",default="training-cache/neural-ask/features.npz")
    parser.add_argument("--parquet",default="training-cache/neural-ask/canonical-ask.parquet")
    parser.add_argument("--weights",default="training-cache/ask-bge-winner")
    parser.add_argument("--embeddings",default="training-cache/bge-256")
    parser.add_argument("--report",default="reports/raw-embedding-ask/comparison.json")
    parser.add_argument("--test-predictions",default="training-cache/ask-bge-report/winner-original-test.csv.gz")
    parser.add_argument("--output",default=FOLDER)
    install(parser.parse_args())
