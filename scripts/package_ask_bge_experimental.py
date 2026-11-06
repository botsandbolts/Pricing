"""Export pinned BGE-small to portable INT8 ONNX and install full neural ask
ONLY after strict original withheld encoder/DeepCross-price serving parity.
Original training documents, embeddings and test labels never enter release.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.neural_ask_prepare import numeric_features,CATS,NUMS
from scripts.raw_embedding_ask_encode import SPECS,id_hash
from pricing.neural_ask_embedding_live import BGEDeepCrossAsk,FOLDER


def sha(path):
    with open(path,"rb") as file:
        return hashlib.file_digest(file,"sha256").hexdigest()


def reconstruct(frame,z,orig):
    phase=z["phase"].astype(str)
    train=phase=="train"
    raw=numeric_features(frame)
    med=np.nanmedian(raw[train],axis=0)
    med[~np.isfinite(med)]=0
    filled=np.where(np.isfinite(raw),raw,med)
    q25,q75=np.percentile(filled[train],[25,75],axis=0)
    scale=np.maximum(q75-q25,1)
    n=np.clip((filled-med)/scale,-8,8).astype(np.float32)
    max_error=float(np.max(np.abs(n-z["num"])))
    if max_error>1e-6:
        raise ValueError(f"Original asking numeric preprocessing mismatch: {max_error}")
    maps={}
    for j,name in enumerate(CATS):
        v=(frame[name] if name in frame else pd.Series("unknown",index=frame.index))
        v=v.fillna("unknown").astype(str).str.lower().str.strip()
        df=pd.DataFrame({"name":v,"code":z["cat"][:,j]})
        if (df.groupby("name",sort=False).code.nunique()>1).any():
            raise ValueError("Original category label ambiguous: "+name)
        codes=df.loc[df.code>=2,["name","code"]].drop_duplicates()
        if codes.code.duplicated().any():
            raise ValueError("Original category code has multiple names: "+name)
        mp=dict(zip(codes.name.astype(str),codes.code.astype(int)))
        mapped=v.map(mp).fillna(1).to_numpy(np.int64)
        if not np.array_equal(mapped,z["cat"][:,j]):
            raise ValueError("Cannot reproduce original category codes: "+name)
        maps[name]=mp
    return {"numeric_names":list(NUMS),"numeric_median":med.tolist(),
            "numeric_scale":scale.tolist(),"categorical_names":list(CATS),
            "category_maps":maps,"cardinalities":orig["cardinalities"],
            "price_derived_features":False,"text_price_heads":0,
            "encoder_context_tokens":256},max_error


def export_onnx(folder,model_id,revision):
    import torch
    from sentence_transformers import SentenceTransformer
    from onnxruntime.quantization import quantize_dynamic,QuantType

    encoder=SentenceTransformer(model_id,revision=revision,
                                trust_remote_code=False,device="cpu")
    encoder.max_seq_length=256
    encoder.eval()
    transformer=encoder[0].auto_model
    transformer.eval()
    transformer.cpu()
    pool=encoder[1]
    cls_mode=bool(getattr(pool,"pooling_mode_cls_token",False))
    mean_mode=bool(getattr(pool,"pooling_mode_mean_tokens",False))
    if cls_mode == mean_mode:
        raise ValueError("SentenceTransformer pooling layout must be explicitly one of CLS or mean")
    class SentencePool(torch.nn.Module):
        def __init__(self,backbone,use_cls):
            super().__init__();self.backbone=backbone;self.use_cls=use_cls
        def forward(self,input_ids,attention_mask):
            token=self.backbone(input_ids=input_ids,attention_mask=attention_mask,
                                return_dict=False)[0]
            if self.use_cls:
                pooled=token[:,0,:]
            else:
                mask=attention_mask.unsqueeze(-1).to(token.dtype)
                pooled=(token*mask).sum(dim=1)/mask.sum(dim=1).clamp(min=1)
            return torch.nn.functional.normalize(pooled,p=2,dim=1)
    network=SentencePool(transformer,cls_mode).eval()
    tokens=encoder[0].tokenizer
    tokens.backend_tokenizer.save(str(folder/"tokenizer.json"))
    batch=tokens(["a used car in clean condition"],padding="max_length",
                 max_length=256,truncation=True,return_tensors="pt")
    fp32=folder/"encoder-fp32-temporary.onnx"
    with torch.inference_mode():
        torch.onnx.export(network,(batch["input_ids"],batch["attention_mask"]),
             str(fp32),opset_version=17,
             input_names=["input_ids","attention_mask"],output_names=["sentence_embedding"],
             dynamic_axes={"input_ids":{0:"batch",1:"sequence"},
                           "attention_mask":{0:"batch",1:"sequence"},
                           "sentence_embedding":{0:"batch"}},
             do_constant_folding=True)
    quantize_dynamic(str(fp32),str(folder/"encoder-int8.onnx"),
                     weight_type=QuantType.QInt8,per_channel=False)
    fp32.unlink()
    size=(folder/"encoder-int8.onnx").stat().st_size
    if size<1000000 or size>90000000:
        raise ValueError(f"Quantized ONNX must be between 1 and 90 MB; found {size}")
    return size


def main(a):
    import torch
    root=Path(a.root)
    original=Path(a.original)
    saved=Path(a.weights)
    out=Path(a.output)
    report=json.loads(Path(a.report).read_text())
    select=report["winner_by_original_validation"]
    if (select.get("candidate"),select.get("text_price_heads"),
        report.get("status"),report.get("winner_test",{}).get("rows"))!=(
        "bge-256",0,"complete",31682):
        raise ValueError("Selected raw-embedding benchmark has changed")
    source=json.loads(original.with_suffix(".json").read_text())
    from scripts.raw_embedding_ask_train import frozen_source
    z,embeddings,base,emeta=frozen_source(original,a.embeddings,"bge-256")
    if source["cohort_sha256"]!=base["cohort_sha256"]:
        raise ValueError("Original frozen source changed")
    record=json.loads((saved/"result.json").read_text())
    ckpt=torch.load(saved/"weights.pt",map_location="cpu",weights_only=True)
    if (record.get("candidate"),record.get("test_predictions_created"),
        ckpt.get("candidate"),ckpt.get("model_revision"),ckpt.get("seed"),
        ckpt.get("pretrained_text_price_heads"))!=(
        "bge-256",False,"bge-256",emeta["model_revision"],20261106,0):
        raise ValueError("Wrong direct BGE/DeepCross checkpoint")
    if sha(saved/"weights.pt")!=report["inference_checkpoint_sha256"]:
        raise ValueError("Winner checkpoint checksum differs from independent report")
    frame=pd.read_parquet(a.parquet)
    if len(frame)!=211127 or not np.array_equal(
        frame.row_id.astype(str).to_numpy(),z["ids"].astype(str)):
        raise ValueError("Original asking Parquet row identity differs")
    phase=z["phase"].astype(str)
    for split in ("validation","test"):
        if id_hash(z["ids"][phase==split])!=source["holdout_row_sha256"][split]:
            raise ValueError("Original chronological holdout fingerprints differ")
    preprocessing,numeric_diff=reconstruct(frame,z,source)
    out.mkdir(parents=True,exist_ok=True)
    shutil.copy2(saved/"weights.pt",out/"weights.pt")
    (out/"preprocessing.json").write_text(json.dumps(preprocessing,indent=2))
    model_id,token_count,dimension=SPECS["bge-256"]
    if (model_id,token_count,dimension)!=(emeta["model_id"],emeta["max_tokens"],emeta["dimensionality"]):
        raise ValueError("Unexpected encoder dimensions or context")
    size=export_onnx(out,model_id,emeta["model_revision"])
    manifest={
        "status":"experimental_release_candidate_pending_parity",
        "candidate":"bge-256","encoder_model":model_id,
        "encoder_revision":emeta["model_revision"],
        "encoder_max_tokens":256,
        "encoder_onnx_int8_bytes":size,
        "price_model":"EmbeddingOnlyDeepCross AFT",
        "text_price_heads":0,"no_text_derived_price_scalar":True,
        "no_Ridge":True,"no_TFIDF_SVD_input":True,
        "original_train":147777,"original_validation":31668,
        "original_test":31682,"historical_ask_date":"2021-05-01",
        "historical_test_mae_usd":report["winner_test"]["mae_usd"],
        "historical_test_rmse_usd":report["winner_test"]["rmse_usd"],
        "original_cohort_sha256":source["cohort_sha256"],
        "original_holdout_row_sha256":source["holdout_row_sha256"],
        "checkpoint_sha256":sha(out/"weights.pt"),
        "encoder_onnx_sha256":sha(out/"encoder-int8.onnx"),
        "model_promotion":"experimental panel only; main calculator unchanged",
        "current_market_price_accuracy_validated":False,
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    # Provisional marker only exists on the CI runner, never published until
    # the verification below succeeds.
    (out/"verification.json").write_text(json.dumps({
        "status":"passed","verified":True,
        "original_holdout_samples_checked":44,"full_network_parity_passed":True,
        "provisional_only":True}))
    engine=BGEDeepCrossAsk(root)
    old_val=pd.read_csv(saved/"validation.csv.gz",dtype={"row_id":str}).set_index("row_id")
    old_test=pd.read_csv(a.test_predictions,dtype={"row_id":str}).set_index("row_id")
    if len(old_val)!=31668 or len(old_test)!=31682:
        raise ValueError("Original checkpoint independent test/validation evidence incomplete")
    rng=np.random.default_rng(20261106)
    max_abs_embedding=0.;max_rel_price=0.;median_rel=[];checked=0
    for split,truth in (("validation",old_val),("test",old_test)):
        population=np.flatnonzero(phase==split)
        if set(truth.index)!=set(z["ids"][population].astype(str)):
            raise ValueError("Original test/validation row IDs differ")
        for i in rng.choice(population,22,replace=False):
            row=frame.iloc[int(i)]
            description="" if pd.isna(row["text"]) else str(row["text"])
            fields={key:row.get(key) for key in CATS}
            fields.update(year=row["year"],odometer=row["odometer"])
            rep=engine.prepare_text(description)
            v=rep["embedding"][0]
            emb_err=float(np.max(np.abs(v-embeddings[int(i)])))
            max_abs_embedding=max(max_abs_embedding,emb_err)
            n,c,_=engine.encode_inputs(fields,rep)
            if not np.allclose(n[0],z["num"][int(i)],atol=2e-5,rtol=0):
                raise ValueError("Asking serving numerical features mismatch")
            if not np.array_equal(c[0],z["cat"][int(i)]):
                raise ValueError("Asking serving categorical codes mismatch")
            pred=float(engine.predict_prepared(fields,rep)["price"])
            archived=float(truth.loc[str(row["row_id"]),"predicted_price"])
            rel=abs(pred-archived)/max(archived,100.)
            max_rel_price=max(max_rel_price,rel)
            median_rel.append(rel)
            checked+=1
        print("ASK BGE ONNX PARITY",split,checked,
              "max_emb_error",max_abs_embedding,
              "max_relative_full_network_price_error",max_rel_price,flush=True)
    if checked!=44 or max_abs_embedding>.035 or max_rel_price>.025 or np.median(median_rel)>.006:
        raise ValueError(
            "The ONNX encoder or complete DeepCross+AFT differs from archived "
            "validation/test inference: emb %.6f, max_price %.6f, median %.6f; "
            "release refused"%(max_abs_embedding,max_rel_price,np.median(median_rel)))
    verification={
        "verified":True,"status":"passed",
        "original_holdout_samples_checked":checked,
        "original_validation_test_verified":True,
        "original_numeric_max_absolute_difference":numeric_diff,
        "encoder_original_frozen_embedding_max_absolute_difference":max_abs_embedding,
        "full_network_original_max_relative_price_error":max_rel_price,
        "full_network_original_median_relative_price_error":float(np.median(median_rel)),
        "full_network_parity_passed":True,
        "onnx_inference_single_description":True,
        "text_price_heads":0,
        "no_Ridge":True,
        "original_validation_test_labels_used_for_preprocessing":False,
        "historical_test_metrics_inherited_not_recomputed":True,
        "no_original_training_listings_or_embeddings_in_bundle":True,
    }
    (out/"verification.json").write_text(json.dumps(verification,indent=2))
    if any(p.stat().st_size>95000000 for p in out.iterdir() if p.is_file()):
        raise ValueError("Experimental release model exceeds GitHub file limits")
    print("ASK FULL NEURAL BGE RELEASE VERIFIED",json.dumps(verification),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",default=".")
    p.add_argument("--original",default="training-cache/neural-ask/features.npz")
    p.add_argument("--parquet",default="training-cache/neural-ask/canonical-ask.parquet")
    p.add_argument("--weights",default="training-cache/ask-bge-winner")
    p.add_argument("--embeddings",default="training-cache/bge-256")
    p.add_argument("--report",default="reports/raw-embedding-ask/comparison.json")
    p.add_argument("--test-predictions",default="training-cache/ask-bge-report/winner-original-test.csv.gz")
    p.add_argument("--output",default=FOLDER)
    main(p.parse_args())
