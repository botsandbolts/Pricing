"""Full-vs-Lemon rank sweep: validate 26 candidates, test ONE winner."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from scripts.bid_fm_scale_train import TRIALS,FAMILIES,COHORTS,ScaledAuctionFM,prediction
from scripts.bid_fm_kernel_train import source,partition,_scores
from scripts.bid_fm_kernel_report import validation_frame,ALPHAS
from scripts.neural_bid_report import paired_interval,reference
from scripts.neural_bid_prepare import sha_ids


def run(results,features,xgb_heldout,xgb_full_report,output):
    z,prov=source(features)
    base=reference(xgb_heldout)
    if (len(base),sum(base.split=="validation"),sum(base.split=="test"))!=(165983,82380,83603):
        raise ValueError("Wrong XGBoost Lemon holdout cohort")
    val=base.loc[base.split=="validation"]
    test=base.loc[base.split=="test"]
    for split,frame in (("validation",val),("test",test)):
        if sha_ids(frame.index)!=prov["holdout_row_sha256"][split]:
            raise ValueError("Original VIN holdout SHA differs")
    full=json.loads(Path(xgb_full_report).read_text())
    if (full["scores"]["validation"]["n"],full["scores"]["test"]["n"])!=(82380,83603):
        raise ValueError("Wrong full XGBoost benchmark")
    entries=sorted(Path(results).rglob("result.json"))
    if len(entries)!=len(TRIALS):
        raise ValueError(f"Need all {len(TRIALS)} original-validation arms, saw {len(entries)}")
    truth=val.observed_price.to_numpy(float)
    xgbp=val.predicted_price.to_numpy(float)
    rows=[];blends=[];paths={};seen=set()
    for entry in entries:
        m=json.loads(entry.read_text())
        name=m.get("trial")
        if name not in TRIALS or name in seen:
            raise ValueError("Missing/duplicate prespecified architecture")
        seen.add(name)
        family,rank,cohort=TRIALS[name]
        fit,inner,_,_=partition(z,cohort)
        if (m.get("status"),m.get("trial"),m.get("market"),
            m.get("family"),m.get("rank"),m.get("cohort"),
            m.get("sample_size_training"),m.get("fitting_rows"),
            m.get("stopper_rows"),m.get("validation_rows"),
            m.get("untouched_test_rows"),m.get("test_evaluated"),
            m.get("no_hierarchy"),m.get("no_price_derived_inputs"),
            m.get("no_description"),m.get("cohort_sha256"),
            m.get("original_holdout_row_sha256"),m.get("cardinalities")) != (
            "completed",name,"bid",family,rank,cohort,
            len(fit)+len(inner),len(fit),len(inner),82380,83603,
            False,True,True,True,prov["cohort_sha256"],
            prov["holdout_row_sha256"],prov["cardinalities"]):
            raise ValueError("Training cohort or frozen provenance changed: "+name)
        aligned=validation_frame(entry.with_name("validation.csv.gz"),val)
        p=aligned.predicted_price.to_numpy(float)
        score=_scores(truth,p)
        if abs(score["mae_usd"]-m["validation"]["mae_usd"])>.05:
            raise ValueError("Stored and recomputed validation MAE disagree")
        rows.append({
            "trial":name,"family":family,"rank":rank,"cohort":cohort,
            "train_available_rows":len(fit)+len(inner),
            "fit_rows":len(fit),"stopper_rows":len(inner),
            "validation_mae_usd":score["mae_usd"],
            "validation_rmse_usd":score["rmse_usd"],
            "validation_bias_usd":score["bias_usd"],
            "delta_vs_xgboost_lemon_usd":
                score["mae_usd"]-_scores(truth,xgbp)["mae_usd"],
            "train_seconds":m["training_seconds"],
            "best_epoch":m["best_epoch"]})
        for alpha in ALPHAS:
            mix=(1-alpha)*xgbp+alpha*p
            blends.append({"trial":name,"family":family,"rank":rank,
                           "cohort":cohort,"fm_weight":alpha,
                           "xgb_weight":1-alpha,
                           "validation_mae_usd":_scores(truth,mix)["mae_usd"]})
        paths[name]=entry.parent
    if seen!=set(TRIALS):
        raise ValueError("Incomplete factorization architecture sweep")
    board=pd.DataFrame(rows).sort_values(
        ["validation_mae_usd","trial"],kind="stable").reset_index(drop=True)
    board["validation_rank"]=np.arange(1,len(board)+1)
    bboard=pd.DataFrame(blends).sort_values(
        ["validation_mae_usd","trial","fm_weight"],kind="stable").reset_index(drop=True)
    selected=str(board.iloc[0]["trial"])
    fm_weight=float(bboard.loc[bboard.trial==selected].iloc[0]["fm_weight"])
    selected_val_blend=float(bboard.loc[bboard.trial==selected].iloc[0]["validation_mae_usd"])
    family,rank,cohort=TRIALS[selected]
    checkpoint=torch.load(paths[selected]/"weights.pt",map_location="cpu",weights_only=True)
    if (checkpoint.get("trial"),checkpoint.get("family"),
        checkpoint.get("rank"),checkpoint.get("cohort"),
        checkpoint.get("cards"),checkpoint.get("seed")) != (
        selected,family,rank,cohort,prov["cardinalities"],20261106):
        raise ValueError("Wrong validation-selected model checkpoint")
    net=ScaledAuctionFM(
        prov["cardinalities"],family,rank,checkpoint["initial_log_price"])
    net.load_state_dict(checkpoint["state_dict"],strict=True)
    _,_,_,original_test=partition(z,cohort)
    ptest=prediction(net,z["num"].astype(np.float32),
                     z["cat"].astype(np.int64),original_test)
    held=pd.DataFrame({
        "row_id":z["ids"][original_test].astype(str),
        "split":"test","observed_price":z["y"][original_test].astype(float),
        "predicted_price":ptest,
    }).set_index("row_id").reindex(test.index)
    if held.isna().any().any() or not np.allclose(
        held.observed_price.to_numpy(float),
        test.observed_price.to_numpy(float),atol=.05,rtol=1e-6):
        raise ValueError("Original TEST row/target identity mismatch")
    target=held.observed_price.to_numpy(float)
    newp=held.predicted_price.to_numpy(float)
    oldp=test.predicted_price.to_numpy(float)
    mix=(1-fm_weight)*oldp+fm_weight*newp
    new_score=_scores(target,newp)
    old_score=_scores(target,oldp)
    blend_score=_scores(target,mix)
    family_rows=[]
    for architecture in FAMILIES:
        for training in COHORTS:
            item=board.loc[(board.family==architecture)&(board.cohort==training)].iloc[0]
            family_rows.append({
                "family":architecture,"cohort":training,
                "best_trial":str(item["trial"]),"rank":int(item["rank"]),
                "validation_mae_usd":float(item["validation_mae_usd"]),
                "delta_vs_xgboost_lemon_usd":float(item["delta_vs_xgboost_lemon_usd"])})
    report={
      "status":"complete","market":"bid",
      "study":"Scale five FM architectures over full versus frozen Lemon original 2015 wholesale auction sales",
      "n_models":len(TRIALS),
      "holdout_rows":{"original_train":384198,"lemon_train":288148,
                      "validation":82380,"test":83603},
      "selected_by_validation":{
          "trial":selected,"family":family,"rank":rank,"cohort":cohort,
          "mae_usd":float(board.iloc[0]["validation_mae_usd"])},
      "new_models_test_evaluated":1,
      "winner_test":new_score,
      "winner_delta_test_vs_paired_xgboost_lemon_usd":new_score["mae_usd"]-old_score["mae_usd"],
      "winner_delta_paired_ci95_usd":paired_interval(
          np.abs(target-newp),np.abs(target-oldp)),
      "xgboost_lemon_matched":{
          "validation":_scores(truth,xgbp),"test":old_score},
      "xgboost_full_aggregate_unpaired":{
          "validation_mae_usd":full["scores"]["validation"]["mae_usd"],
          "test_mae_usd":full["scores"]["test"]["mae_usd"]},
      "selected_winner_xgboost_blend":{
          "fm_weight":fm_weight,"xgboost_lemon_weight":1-fm_weight,
          "validation_mae_usd":selected_val_blend,
          "test":blend_score,
          "delta_test_vs_paired_xgboost_lemon_usd":blend_score["mae_usd"]-old_score["mae_usd"],
          "paired_ci95_usd":paired_interval(np.abs(target-mix),np.abs(target-oldp))},
      "best_any_fm_blend_validation_only":{
          "trial":str(bboard.iloc[0]["trial"]),
          "fm_weight":float(bboard.iloc[0]["fm_weight"]),
          "validation_mae_usd":float(bboard.iloc[0]["validation_mae_usd"])},
      "full_vs_lemon_by_family":family_rows,
      "no_test_driven_tuning":True,"no_production_change":True,
      "no_fit_on_in_sample_residuals":True,
      "canonical_cohort_sha256":prov["cohort_sha256"],
      "original_holdout_sha256":prov["holdout_row_sha256"],
      "validation_leaderboard":board.to_dict(orient="records"),
    }
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    (output/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    board.to_csv(output/"validation-leaderboard.csv",index=False)
    pd.DataFrame(family_rows).to_csv(output/"full-vs-lemon-leaders.csv",index=False)
    bboard.to_csv(output/"validation-xgboost-blends.csv",index=False)
    held.reset_index().to_csv(output/"winner-original-test.csv.gz",index=False)
    lines=[
       "# FM family rank scale: original wholesale auction data",
       "",
       "Five architectures, 26 registered full-vs-Lemon variants. Only original",
       "validation used to choose the winner, rank, cohort, and blend coefficient.",
       f"Winner: {selected}; original validation MAE USD {board.iloc[0]['validation_mae_usd']:.2f}.",
       f"Original test MAE USD {new_score['mae_usd']:.2f}; matched Lemon XGBoost USD {old_score['mae_usd']:.2f}.",
       f"Published full-data XGBoost test MAE USD {full['scores']['test']['mae_usd']:.2f}; aggregate unpaired.",
       "", "## Full versus Lemon, family best on original validation",
       "", "| Family | Cohort | Best variant | Rank | Validation USD MAE |",
       "|---|---|---|---:|---:|",
    ]
    for item in family_rows:
        lines.append(f"| {item['family']} | {item['cohort']} | {item['best_trial']} | {item['rank']} | {item['validation_mae_usd']:.2f} |")
    lines += ["","No models deployed; test is evaluated only for a single validation-selected model.",
              "Any XGBoost blend uses matched original validation predictions, not leaked in-sample residual targets."]
    (output/"README.md").write_text("\n".join(lines)+"\n")
    print("SCALED_FM_FINAL",json.dumps({
        "trial":selected,"val_usd":board.iloc[0]["validation_mae_usd"],
        "test_usd":new_score["mae_usd"],"xgb_lemon_test_usd":old_score["mae_usd"],
        "blend_test_usd":blend_score["mae_usd"]}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",default="training-cache/bid-fm-scale")
    p.add_argument("--features",default="training-cache/neural-bid/features.npz")
    p.add_argument("--reference",default="benchmark-reference/bid/predictions.csv.gz")
    p.add_argument("--full-xgboost",default="models/fivefold-live/bid/evaluation.json")
    p.add_argument("--output",default="reports/bid-fm-scale")
    a=p.parse_args()
    run(a.results,a.features,a.reference,a.full_xgboost,a.output)
