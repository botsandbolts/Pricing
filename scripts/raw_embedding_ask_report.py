"""Select frozen raw sentence embeddings by validation MAE, then test ONE winner.

Selection never loads new candidate test predictions: training artifacts contain
only validation predictions. The original DeepCross+MiniLM scalar benchmark is
a reference, not an input to any candidate. After selection, the independent
report job downloads only the winner's embedding tensor and evaluates its one
checkpoint on the original test; optional independent-seed retrains are
reported on validation without reselecting a seed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from scripts.raw_embedding_ask_encode import SPECS
from scripts.raw_embedding_ask_train import EmbeddingOnlyDeepCross, frozen_source
from scripts.neural_ask_train import predict, metrics
from scripts.neural_ask_report import paired_interval


def reference(path):
    f = pd.read_csv(path, dtype={"row_id":str})
    if len(f)!=63350 or f.row_id.duplicated().any() or (
        set(f.split)!={"validation","test"}):
        raise ValueError("Original validation-selected DeepCross benchmark incomplete")
    if (sum(f.split=="validation"),sum(f.split=="test"))!=(31668,31682):
        raise ValueError("Wrong original benchmark holdout partitions")
    return f.set_index("row_id")


def audited_validation(row, orig):
    if len(row)!=len(orig) or row.row_id.duplicated().any():
        raise ValueError("Incomplete new validation predictions")
    if set(row.row_id)!=set(orig.index):
        raise ValueError("New validation vehicle identities differ")
    r=row.set_index("row_id").reindex(orig.index)
    if not r.split.eq("validation").all() or not np.allclose(
        r.observed_price.to_numpy(float),orig.observed_price.to_numpy(float),
        rtol=1e-6,atol=.05):
        raise ValueError("Observed validation prices differ")
    if not np.isfinite(r.predicted_price).all() or (r.predicted_price<=0).any():
        raise ValueError("Invalid new predictions")
    return r


def select(results, prior, existing_report, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ref=reference(prior)
    validation=ref.loc[ref.split=="validation"]
    baseline=json.loads(Path(existing_report).read_text())
    winner=baseline["winner_by_validation"]
    if (baseline["status"],winner["features"],winner["variant"])!=(
        "complete","minilm","dropout12"):
        raise ValueError("Wrong original benchmark comparison")
    files=sorted(Path(results).rglob("result.json"))
    if len(files)!=len(SPECS):
        raise ValueError(f"Expected {len(SPECS)} genuinely trained encoders, saw {len(files)}")
    seen=set();data=[]
    for f in files:
        record=json.loads(f.read_text())
        name=record.get("candidate")
        if name not in SPECS or name in seen:
            raise ValueError("Missing/duplicate registered raw embedding candidate")
        seen.add(name)
        if (record.get("status"),record.get("full_train_rows"),
            record.get("fitting_rows"),record.get("internal_early_stopper"),
            record.get("validation_rows"),record.get("training_seed"),
            record.get("text_price_heads"),record.get("price_derived_inputs"),
            record.get("test_predictions_created"))!=(
                "complete",147777,133000,14777,31668,20261106,0,False,False):
            raise ValueError("Wrong cohort or unsafe text feature in " + name)
        if (record.get("encoder"),record.get("max_tokens"),
            record.get("embedding_dimension"))!=SPECS[name]:
            raise ValueError("Encoder identity changed")
        check=record["batch1_parity"]
        if check["max_absolute_difference"]>.005 or check["minimum_cosine_similarity"]<.999:
            raise ValueError("Unreliable single-text serving encoder: "+name)
        raw=pd.read_csv(f.with_name("validation.csv.gz"),dtype={"row_id":str})
        aligned=audited_validation(raw,validation)
        y=aligned.observed_price.to_numpy(float)
        p=aligned.predicted_price.to_numpy(float)
        score=metrics(y,p)
        if abs(score["mae_usd"]-record["validation"]["mae_usd"])>.05:
            raise ValueError("Original score differs from independently calculated MAE")
        diff=np.abs(p-y)-np.abs(validation.predicted_price.to_numpy(float)-y)
        data.append({"candidate":name,"model":record["encoder"],
                     "revision":record["encoder_revision"],
                     "max_tokens":record["max_tokens"],
                     "embedding_dim":record["embedding_dimension"],
                     "mae_usd":score["mae_usd"],"rmse_usd":score["rmse_usd"],
                     "bias_usd":score["mean_bias_usd"],
                     "delta_vs_old_minilm_scalar_validation_usd":float(diff.mean()),
                     "best_epoch":record["best_epoch"],
                     "batch1_max_difference":check["max_absolute_difference"],
                     "batch1_min_cosine":check["minimum_cosine_similarity"],
                     "label_price_heads":0})
    board=pd.DataFrame(data).sort_values(["mae_usd","candidate"]).reset_index(drop=True)
    board["validation_rank"]=np.arange(1,len(board)+1)
    selected=str(board.iloc[0]["candidate"])
    if set(seen)!=set(SPECS):
        raise ValueError("Not all pretrained encoder variants compared")
    selected_data={
        "status":"selected_by_validation_only",
        "candidate":selected,
        "selected_validation_mae_usd":float(board.iloc[0]["mae_usd"]),
        "original_validation_mae_usd":float(winner["validation_mae_usd"]),
        "original_untouched_test_winner_mae_usd":float(winner["test_mae_usd"]),
        "selection_rule":"lowest original chronological validation MAE; no new candidate test predictions scored",
        "number_of_encoder_context_candidates":len(SPECS),
        "model_id":board.iloc[0]["model"],"revision":board.iloc[0]["revision"],
        "embedding_only":True,"text_price_heads":0,
        "production_changed":False}
    (out/"selection.json").write_text(json.dumps(selected_data,indent=2,allow_nan=False))
    board.to_csv(out/"validation-leaderboard.csv",index=False)
    print("VALIDATION_SELECTED",json.dumps(selected_data),flush=True)


def evaluate(selection_dir, checkpoint_dir, embedding_dir, original_path,
             prior, prior_report, robust_dir, output):
    selection_dir=Path(selection_dir)
    decision=json.loads((selection_dir/"selection.json").read_text())
    candidate=decision["candidate"]
    if candidate not in SPECS:
        raise ValueError("Invalid selected encoder")
    z, text, manifest, emeta=frozen_source(original_path,embedding_dir,candidate)
    weights=Path(checkpoint_dir)/"weights.pt"
    state=torch.load(weights,map_location="cpu",weights_only=True)
    if (state.get("candidate"),state.get("seed"),
        state.get("model_revision"),state.get("embedding_dim"),
        state.get("pretrained_text_price_heads"))!=(
            candidate,20261106,emeta["model_revision"],text.shape[1],0):
        raise ValueError("Checkpoint does not match validation-selected encoder")
    model=EmbeddingOnlyDeepCross(manifest["cardinalities"],7,text.shape[1])
    model.load_state_dict(state["state_dict"],strict=True)
    phases=z["phase"].astype(str)
    index=np.flatnonzero(phases=="test")
    if len(index)!=31682:
        raise ValueError("Original untouched test size changed")
    dataset=TensorDataset(
        torch.from_numpy(z["num"].astype(np.float32)),
        torch.from_numpy(z["cat"].astype(np.int64)),
        torch.from_numpy(text),
        torch.from_numpy(np.log(z["y"].astype(float)).astype(np.float32)))
    prediction=predict(model,dataset,index,768)
    baseline=reference(prior)
    baseline=baseline.loc[baseline.split=="test"]
    df=pd.DataFrame({"row_id":z["ids"][index].astype(str),
                     "split":"test",
                     "observed_price":z["y"][index].astype(float),
                     "predicted_price":prediction})
    if set(df.row_id)!=set(baseline.index):
        raise ValueError("Selected test vehicle identities do not match historical benchmark")
    df=df.set_index("row_id").reindex(baseline.index)
    y=df.observed_price.to_numpy(float)
    p=df.predicted_price.to_numpy(float)
    old=baseline.predicted_price.to_numpy(float)
    if not np.allclose(y,baseline.observed_price.to_numpy(float),rtol=1e-6,atol=.05):
        raise ValueError("Original test target changed")
    score=metrics(y,p)
    delta=np.abs(y-p)-np.abs(y-old)
    old_doc=json.loads(Path(prior_report).read_text())
    if abs(float(old_doc["winner_by_validation"]["test_mae_usd"])-
           float(np.abs(y-old).mean()))>.05:
        raise ValueError("Original frozen winner benchmark differs")
    seeds={}
    for name in ("42","1729"):
        path=Path(robust_dir)/name/"result.json"
        if not path.is_file():
            raise ValueError("Missing independent-seed reproducibility result: "+name)
        v=json.loads(path.read_text())
        if (v.get("candidate"),v.get("training_seed"),
            v.get("text_price_heads"),v.get("test_predictions_created"))!=(
            candidate,int(name),0,False):
            raise ValueError("Invalid independent robustness trial: "+name)
        seeds[name]={"validation_mae_usd":v["validation"]["mae_usd"],
                     "best_epoch":v["best_epoch"]}
    board=pd.read_csv(selection_dir/"validation-leaderboard.csv")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    results={
      "status":"complete",
      "study":"Direct pretrained embeddings -> learned projector -> DeepCross+AFT, no text price heads",
      "market":"ask","user_request":"Only embeddings. Price comes from full neural model.",
      "encoder_candidates":len(SPECS),
      "winner_by_original_validation":decision,
      "winner_test":score,
      "original_deepcross_minilm_scalar_test_mae_usd":float(np.abs(y-old).mean()),
      "winner_delta_test_mae_vs_prior_usd":float(delta.mean()),
      "paired_95pct_bootstrap_test_delta_usd":paired_interval(np.abs(y-p),np.abs(y-old)),
      "winner_fraction_better_than_prior_test":float((np.abs(y-p)<np.abs(y-old)).mean()),
      "independent_seed_validation_robustness":seeds,
      "three_seed_validation_mean_usd":float(np.mean([
         decision["selected_validation_mae_usd"],seeds["42"]["validation_mae_usd"],
         seeds["1729"]["validation_mae_usd"]])),
      "full_xgboost_published_unpaired":old_doc["full_xgboost_published_unpaired"],
      "heldout_rows":{"validation":31668,"test":31682},
      "encoder_serving_consistency":emeta["batch_consistency"],
      "embedding_snapshot":{"model":emeta["model_id"],
                            "revision":emeta["model_revision"],
                            "max_tokens":emeta["max_tokens"],
                            "stored_precision":emeta["saved_precision"],
                            "sha256":emeta["embedding_file_sha256"]},
      "inference_checkpoint_sha256":None,
      "selection":"All candidate training jobs return validation only; test evaluated for one validation winner",
      "production_promotion":False,
      "caveats":["Embedding model pretraining may contain unknown training corpora; no direct test-set label access here",
                 "Historical 2021 asking prices are not verified sale transactions",
                 "The original MiniLM scalar is a different feature representation, not an identical architecture",
                 "Model winner must pass complete Streamlit serving parity before any release"] }
    import hashlib
    with open(weights,"rb") as file:
        results["inference_checkpoint_sha256"]=hashlib.file_digest(file,"sha256").hexdigest()
    (output/"comparison.json").write_text(json.dumps(results,indent=2,allow_nan=False))
    board.to_csv(output/"validation-leaderboard.csv",index=False)
    df.reset_index().to_csv(output/"winner-original-test.csv.gz",index=False)
    (output/"README.md").write_text(
        "# Frozen raw-embedding DeepCross AFT asking-price study\n\n"
        "All 8 encoder/context variants use 7 original numeric features, "
        "11 original categorical embeddings and frozen **sentence embeddings only**. "
        "The only price-predicting parameters are in the learned projection + DeepCross AFT; "
        "no pretrained text-derived price head, MiniLM price scalar or TF-IDF/SVD input. "
        "All models selected by chronological original validation MAE, with the original "
        "test evaluated for ONE winner after selection. Batch-one serving consistency "
        "is verified against batch-sixteen embedding inference. Two additional training seeds "
        "are reported on validation without cherry-picking. No production promotion.\n")
    print("VALIDATION-WINNER-ORIGINAL-TEST",json.dumps({
        "candidate":candidate,"mae_usd":score["mae_usd"],
        "delta_vs_previous_model":results["winner_delta_test_mae_vs_prior_usd"]}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    s=sub.add_parser("select")
    s.add_argument("--results",default="training-cache/raw-results")
    s.add_argument("--prior",default="benchmark-prior/predictions.csv.gz")
    s.add_argument("--prior-report",default="reports/deepcross-aft-tuning/report.json")
    s.add_argument("--output",default="reports/raw-embedding-ask-stage")
    e=sub.add_parser("evaluate")
    e.add_argument("--selection",default="reports/raw-embedding-ask-stage")
    e.add_argument("--checkpoint",required=True)
    e.add_argument("--embeddings",required=True)
    e.add_argument("--original",default="training-cache/neural-ask/features.npz")
    e.add_argument("--prior",default="benchmark-prior/predictions.csv.gz")
    e.add_argument("--prior-report",default="reports/deepcross-aft-tuning/report.json")
    e.add_argument("--robust",default="training-cache/raw-robustness")
    e.add_argument("--output",default="reports/raw-embedding-ask")
    a=p.parse_args()
    if a.command=="select": select(a.results,a.prior,a.prior_report,a.output)
    else: evaluate(a.selection,a.checkpoint,a.embeddings,a.original,
                   a.prior,a.prior_report,a.robust,a.output)
