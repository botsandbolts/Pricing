"""Rank 29 Lemon FT/Huber trials on validation; test ONLY the chosen winner."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from scripts.ft_bid_tune import TRIALS,settings,model_for,frozen,tensors,ids_sha
from scripts.neural_ask_train import predict,seed_everything
from scripts.neural_bid_report import reference,_scores,paired_interval

def align_validation(df,baseline):
    if len(df)!=len(baseline) or df.row_id.duplicated().any():
        raise ValueError("Incomplete/duplicate original validation IDs")
    if set(df.row_id)!=set(baseline.index):
        raise ValueError("Validation IDs do not match frozen XGBoost")
    d=df.set_index("row_id").reindex(baseline.index)
    if not d.split.eq("validation").all() or not np.allclose(
        d.observed_price,baseline.observed_price,rtol=1e-6,atol=.05):
        raise ValueError("Validation prices or partitions differ")
    if not np.isfinite(d.predicted_price).all() or (d.predicted_price<=0).any():
        raise ValueError("Invalid validation predictions")
    return d

def run(results,source,xgb_path,full_path,out):
    z,manifest,fit,inner,val,test=frozen(source)
    baseline=reference(xgb_path)
    bx=baseline.loc[baseline.split=="validation"]
    bt=baseline.loc[baseline.split=="test"]
    if (len(bx),len(bt))!=(82380,83603) or (
        ids_sha(bx.index)!=manifest["holdout_row_sha256"]["validation"] or
        ids_sha(bt.index)!=manifest["holdout_row_sha256"]["test"]):
        raise ValueError("Original auction XGBoost holdouts do not match")
    full=json.loads(Path(full_path).read_text())
    if (full["scores"]["validation"]["n"],full["scores"]["test"]["n"])!=(82380,83603):
        raise ValueError("Published full-XGBoost sample sizes differ")
    files=sorted(Path(results).rglob("result.json"))
    if len(files)!=len(TRIALS):
        raise ValueError(f"Need all {len(TRIALS)} independent completed trials; found {len(files)}")
    rows=[];seen=set();paths={}
    truth=bx.observed_price.to_numpy(float);ref=bx.predicted_price.to_numpy(float)
    for f in files:
        d=json.loads(f.read_text())
        name=d.get("trial")
        if name not in TRIALS or name in seen:
            raise ValueError("Unexpected/duplicate trial")
        seen.add(name);paths[name]=f
        if (d.get("status"),d.get("market"),d.get("architecture"),d.get("loss"),
            d.get("cohort"),d.get("train_fit"),d.get("inner_rows"),
            d.get("test_evaluated"))!=(
                "completed","bid","fttransformer","huber","lemon",259334,28814,False):
            raise ValueError("Wrong cohort, architecture or test access")
        if d.get("config")!=settings(name) or d.get("no_hierarchy") is not True:
            raise ValueError("Unregistered configuration or hierarchy")
        if d.get("canonical_cohort_sha256")!=manifest["cohort_sha256"]:
            raise ValueError("Canonical auction cohort provenance differs")
        p=align_validation(pd.read_csv(f.with_name("validation.csv.gz"),dtype={"row_id":str}),bx)
        pred=p.predicted_price.to_numpy(float)
        score=_scores(truth,pred)
        if not np.isclose(score["mae_usd"],d["validation"]["mae_usd"],atol=.05):
            raise ValueError("Stored versus computed validation MAE mismatch")
        delta=np.abs(pred-truth)-np.abs(ref-truth)
        rows.append({"trial":name,"validation_mae_usd":score["mae_usd"],
                     "validation_rmse_usd":score["rmse_usd"],
                     "validation_bias_usd":score["mean_bias_usd"],
                     "internal_mae_usd":d["inner_mae_usd"],
                     "best_epoch":d["best_epoch"],
                     "delta_vs_frozen_xgb_lemon_validation_usd":float(delta.mean()),
                     "config":d["config"]})
    board=pd.DataFrame(rows).sort_values(["validation_mae_usd","trial"]).reset_index(drop=True)
    board["validation_rank"]=np.arange(1,len(board)+1)
    selected=board.iloc[0]["trial"];checkpoint=paths[selected].with_name("weights.pt")
    # TEST is accessed only after ranking ALL candidates by original validation.
    pack=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if pack["trial"]!=selected or pack["config"]!=settings(selected):
        raise ValueError("Selected checkpoint mismatches validation winner")
    seed_everything()
    net=model_for(selected,manifest["cardinalities"],z["num"].shape[1],z["txt"].shape[1])
    net.load_state_dict(pack["state"])
    ptest=predict(net,tensors(z),test,settings(selected)["batch"])
    observed=pd.DataFrame({"row_id":z["ids"][test].astype(str),
                           "observed_price":z["y"][test].astype(float),
                           "predicted_price":ptest}).set_index("row_id").reindex(bt.index)
    if observed.isna().any().any() or not np.allclose(
        observed.observed_price,bt.observed_price,rtol=1e-6,atol=.05):
        raise ValueError("Selected test outputs do not match original VIN rows and prices")
    y=observed.observed_price.to_numpy(float)
    pred=observed.predicted_price.to_numpy(float)
    other=bt.predicted_price.to_numpy(float)
    winner_test=_scores(y,pred)
    delta=np.abs(pred-y)-np.abs(other-y)
    report={"status":"complete","market":"bid","architecture":"fttransformer",
            "objective":"Huber on log historical auction sale price",
            "cohort":"frozen training-only 25% Lemon exclusion",
            "tested_variants":len(TRIALS),"test_evaluated_variants":1,
            "holdout_rows":{"validation":82380,"test":83603},
            "winner_by_original_validation":{
                "trial":selected,"validation_mae_usd":float(board.iloc[0]["validation_mae_usd"]),
                "config":settings(selected)},
            "winner_original_test":winner_test,
            "winner_test_delta_vs_xgboost_lemon_usd":float(delta.mean()),
            "winner_paired_test_ci95_delta_usd":paired_interval(
                np.abs(pred-y),np.abs(other-y)),
            "winner_test_fraction_beating_xgb_lemon":float((np.abs(pred-y)<np.abs(other-y)).mean()),
            "xgboost_lemon_matched":{
                "validation":_scores(bx.observed_price,bx.predicted_price),
                "test":_scores(bt.observed_price,bt.predicted_price)},
            "xgboost_full_published_aggregate":{
                "validation_mae_usd":full["scores"]["validation"]["mae_usd"],
                "test_mae_usd":full["scores"]["test"]["mae_usd"],
                "caveat":"same counts but not independently paired"},
            "selection_rule":"All 29 compared on original validation; ONLY validation winner evaluated on original test",
            "no_production_promotion":True,
            "validation_leaderboard":board.to_dict(orient="records")}
    output=Path(out);output.mkdir(parents=True,exist_ok=True)
    (output/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    board.to_csv(output/"validation-leaderboard.csv",index=False)
    pd.DataFrame({"row_id":bt.index,"split":"test","observed_price":y,
                  "predicted_price":pred}).to_csv(
                      output/"winner-test-predictions.csv.gz",index=False)
    (output/"README.md").write_text(
        "# FT-Transformer Huber Lemon wholesale bid hyperparameter search\n\n"
        "29 independently trained architecture/regularization variants. "
        "Original validation selects winner; test is evaluated for the selected "
        "ONE candidate only. Exact VIN IDs and prices audited against XGBoost Lemon. "
        "No deployment of experimental model.\n")
    print("SELECTED",json.dumps({"trial":selected,
        "validation_mae_usd":float(board.iloc[0]["validation_mae_usd"]),
        "test_mae_usd":winner_test["mae_usd"]}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--results",default="training-cache/ft-bid-tuning")
    p.add_argument("--features",default="training-cache/neural-bid/features.npz")
    p.add_argument("--reference",default="benchmark-reference/bid/predictions.csv.gz")
    p.add_argument("--full-xgboost",default="models/fivefold-live/bid/evaluation.json")
    p.add_argument("--output",default="reports/ft-bid-tuning")
    a=p.parse_args()
    run(a.results,a.features,a.reference,a.full_xgboost,a.output)
