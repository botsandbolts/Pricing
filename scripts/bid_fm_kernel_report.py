"""Audited SVM-like/FMs against frozen XGBoost, winner test only.

Read all ten ORIGINAL validation prediction files; choose standalone FM/kernel
and validation-calibrated XGBoost blend BEFORE accessing original test. The
selected challenger is the ONLY new test-scored checkpoint. Never fake train
OOF residuals with in-sample XGBoost predictions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

from scripts.bid_fm_kernel_train import (
    TRIALS, AuctionFactorization, source, partition,
    fm_prediction, rff_features, y_price, _scores)
from scripts.neural_bid_prepare import sha_ids
from scripts.neural_bid_report import reference, paired_interval


ALPHAS=(0.,.1,.2,.35,.5,.75,1.)


def validation_frame(path,xgb):
    data=pd.read_csv(path,dtype={"row_id":str})
    if (len(data)!=len(xgb) or data.row_id.duplicated().any() or
        set(data.row_id)!=set(xgb.index)):
        raise ValueError("FM/SVR validation row identities missing/duplicated")
    a=data.set_index("row_id").loc[xgb.index]
    if not a.split.eq("validation").all() or not np.allclose(
        a.observed_price.to_numpy(float),
        xgb.observed_price.to_numpy(float),atol=.05,rtol=1e-6):
        raise ValueError("Original validation target or order differs")
    if not np.isfinite(a.predicted_price).all() or (a.predicted_price<=0).any():
        raise ValueError("Nonfinite or negative validation predictions")
    return a


def winner_test(trial, artifact, z,m,rows):
    kind,rank,_cohort=TRIALS[trial]
    if kind in ("fm","ffm"):
        checkpoint=torch.load(Path(artifact)/"weights.pt",map_location="cpu",weights_only=True)
        if (checkpoint.get("trial"),checkpoint.get("rank"),
            checkpoint.get("field_aware"),checkpoint.get("cards"))!=(
            trial,rank,kind=="ffm",m["cardinalities"]):
            raise ValueError("Chosen factorization checkpoint mismatches validation")
        net=AuctionFactorization(m["cardinalities"],rank,kind=="ffm",
                                  checkpoint["initial_log_price"])
        net.load_state_dict(checkpoint["state_dict"],strict=True)
        return fm_prediction(net,z["num"].astype(np.float32),
                             z["cat"].astype(np.int64),rows)
    bundle=joblib.load(Path(artifact)/"model.joblib")
    if bundle["trial"]!=trial or bundle["model"] not in (
        "RFF + epsilon-SVR","RFF + Huber"):
        raise ValueError("Chosen RFF model mismatches validation")
    transformed=rff_features(z["num"].astype(np.float32),
               z["cat"].astype(np.int64),
               bundle["onehot"],bundle["scaler"],bundle["rbf"],rows)
    return y_price(bundle["estimator"].predict(transformed)+bundle["target_log_center"])


def run(results,features,reference_path,full_path,output):
    z,provenance=source(features)
    xgb=reference(reference_path)
    if (len(xgb),int(sum(xgb.split=="validation")),int(sum(xgb.split=="test")))!=(
        165983,82380,83603):
        raise ValueError("Frozen Lemon XGBoost reference holdouts incomplete")
    validation=xgb.loc[xgb.split=="validation"]
    test=xgb.loc[xgb.split=="test"]
    for name,rows in (("validation",validation),("test",test)):
        if sha_ids(rows.index)!=provenance["holdout_row_sha256"][name]:
            raise ValueError("Original frozen XGBoost VIN-disjoint holdouts have changed")
    with open(full_path) as f:
        published=json.load(f)
    if (published["scores"]["validation"]["n"],published["scores"]["test"]["n"])!=(
        82380,83603):
        raise ValueError("Full XGBoost aggregate compared on different split size")
    files=sorted(Path(results).rglob("result.json"))
    if len(files)!=len(TRIALS):
        raise ValueError(f"All {len(TRIALS)} completed FM/kernel models required; got {len(files)}")
    actual=validation.observed_price.to_numpy(float)
    xgbp=validation.predicted_price.to_numpy(float)
    rows=[]; blends=[]; directory={};seen=set()
    for file in files:
        meta=json.loads(file.read_text())
        trial=meta.get("trial")
        if trial not in TRIALS or trial in seen:
            raise ValueError("Duplicate or unregistered trial")
        seen.add(trial)
        kind,rank,cohort=TRIALS[trial]
        fit,inner,_,_=partition(z,cohort)
        if (meta.get("status"),meta.get("market"),
            meta.get("model_family"),meta.get("rank_or_components"),
            meta.get("cohort"),meta.get("fitting_rows"),
            meta.get("stopper_rows"),meta.get("validation_rows"),
            meta.get("test_evaluated"),meta.get("no_hierarchy"),
            meta.get("price_derived_inputs")) != (
            "completed","bid",kind,rank,cohort,len(fit),len(inner),82380,False,True,False):
            raise ValueError("Changed auction cohort or unregistered fitting policy")
        if (meta.get("canonical_cohort_sha256")!=provenance["cohort_sha256"] or
            meta.get("holdout_row_sha256")!=provenance["holdout_row_sha256"] or
            meta.get("original_feature_cardinalities")!=provenance["cardinalities"]):
            raise ValueError("Model was fitted to different auction rows")
        aligned=validation_frame(file.with_name("validation.csv.gz"),validation)
        pred=aligned.predicted_price.to_numpy(float)
        score=_scores(actual,pred)
        if abs(score["mae_usd"]-meta["validation"]["mae_usd"])>.05:
            raise ValueError("Recomputed FM/kernel validation MAE differs from model log")
        rows.append(dict(trial=trial,family=kind,cohort=cohort,
                         rank_or_components=rank,
                         validation_mae_usd=score["mae_usd"],
                         validation_rmse_usd=score["rmse_usd"],
                         validation_bias_usd=score["bias_usd"],
                         elapsed_seconds=meta["wall_seconds"],
                         internal_mae_usd=meta["inner_mae_usd"],
                         best_epoch=meta["best_epoch"],
                         versus_xgb_lemon_validation_usd=score["mae_usd"]-_scores(actual,xgbp)["mae_usd"]))
        for alpha in ALPHAS:
            blended=(1-alpha)*xgbp+alpha*pred
            blends.append(dict(trial=trial,fm_kernel_weight=alpha,
                               xgboost_weight=1-alpha,
                               validation_mae_usd=_scores(actual,blended)["mae_usd"]))
        directory[trial]=file.parent
    if seen!=set(TRIALS):
        raise ValueError("Expected every pre-registered comparison arm")
    board=pd.DataFrame(rows).sort_values(["validation_mae_usd","trial"],kind="stable").reset_index(drop=True)
    board["validation_rank"]=np.arange(1,len(board)+1)
    ranked_blends=pd.DataFrame(blends).sort_values(
        ["validation_mae_usd","trial","fm_kernel_weight"],kind="stable").reset_index(drop=True)
    selected=str(board.iloc[0]["trial"])
    ensemble_trial=str(ranked_blends.iloc[0]["trial"])
    ensemble_alpha=float(ranked_blends.iloc[0]["fm_kernel_weight"])
    # Only one NEW model is evaluated on test: the validation-selected
    # CHALLENGER, not an untested blend winner from another challenger.
    # Select the blend parameter for this same selected challenger ONLY.
    allowed=ranked_blends.loc[ranked_blends.trial.eq(selected)]
    selected_alpha=float(allowed.iloc[0]["fm_kernel_weight"])
    # Evaluate test once for the validation-selected standalone challenger.
    _fit,_inner,_val,test_rows=partition(z,TRIALS[selected][2])
    pred=winner_test(selected,directory[selected],z,provenance,test_rows)
    if len(pred)!=len(test_rows):
        raise ValueError("Wrong selected original test prediction count")
    result=pd.DataFrame(dict(row_id=z["ids"][test_rows].astype(str),
        split="test",observed_price=z["y"][test_rows].astype(float),
        predicted_price=pred)).set_index("row_id").reindex(test.index)
    if result.isna().any().any() or not np.allclose(
        result.observed_price.to_numpy(float),
        test.observed_price.to_numpy(float),rtol=1e-6,atol=.05):
        raise ValueError("Test row/target changed after validation selection")
    actual_test=result.observed_price.to_numpy(float)
    yhat=result.predicted_price.to_numpy(float)
    xgb_test=test.predicted_price.to_numpy(float)
    blend_test=(1-selected_alpha)*xgb_test+selected_alpha*yhat
    test_scores=_scores(actual_test,yhat)
    blend_scores=_scores(actual_test,blend_test)
    xgb_scores=_scores(actual_test,xgb_test)
    result_report={
        "status":"complete","market":"bid",
        "study":"Scalable exact field-aware FM / standard FM / RFF epsilon-insensitive kernel alternative",
        "target":"Historical VIN-deduplicated 2014/15 wholesale completed auction prices; 2015-normalized USD",
        "n_trials":len(TRIALS),
        "original_holdouts":{"validation":82380,"test":83603},
        "candidate_train_cohorts":{"full":384198,"lemon":288148},
        "selected_by_original_validation":{"trial":selected,
             "validation_mae_usd":float(board.iloc[0]["validation_mae_usd"]),
             "family":TRIALS[selected][0],
             "rank_or_components":TRIALS[selected][1],
             "cohort":TRIALS[selected][2]},
        "test_scored_new_models":1,
        "winner_test":test_scores,
        "winner_test_difference_vs_lemon_xgboost_usd":test_scores["mae_usd"]-xgb_scores["mae_usd"],
        "winner_test_delta_paired_bootstrap_ci95_usd":paired_interval(
            np.abs(actual_test-yhat),np.abs(actual_test-xgb_test)),
        "winner_test_fraction_better_than_lemon_xgboost":float((
            np.abs(actual_test-yhat)<np.abs(actual_test-xgb_test)).mean()),
        "validation_selected_blend_of_same_winner":{
            "fm_kernel_weight":selected_alpha,
            "xgboost_weight":1-selected_alpha,
            "validation_mae_usd":float(allowed.iloc[0]["validation_mae_usd"]),
            "test":blend_scores,
            "test_difference_vs_lemon_xgboost_usd":blend_scores["mae_usd"]-xgb_scores["mae_usd"],
            "paired_bootstrap_ci95_delta_usd":paired_interval(
                np.abs(actual_test-blend_test),np.abs(actual_test-xgb_test))},
        "alternative_best_validation_blend_diagnostic_not_tested":{
             "trial":ensemble_trial,"fm_kernel_weight":ensemble_alpha,
             "validation_mae_usd":float(ranked_blends.iloc[0]["validation_mae_usd"])},
        "xgboost_lemon_exact_same_row_reference":{
            "validation":_scores(actual,xgbp),
            "test":xgb_scores},
        "xgboost_full_published_aggregate_unpaired":{
            "validation":published["scores"]["validation"]["mae_usd"],
            "test":published["scores"]["test"]["mae_usd"]},
        "no_true_fm_residual_regressor":True,
        "residual_note":"Validation-selected convex blend is the safe first residual-style check; training an FM on XGBoost residuals requires trustworthy train-only XGBoost OOF predictions, not in-sample errors.",
        "selection_rule":"All 10 models and allowed blend weights ranked using original validation; test evaluated for ONE standalone candidate and its prespecified selected weight.",
        "no_production_promotion":True,
        "validation_leaderboard":board.to_dict(orient="records"),
        "validation_blend_leaderboard":ranked_blends.to_dict(orient="records")}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/"comparison.json").write_text(json.dumps(result_report,indent=2,allow_nan=False))
    board.to_csv(output/"validation-leaderboard.csv",index=False)
    ranked_blends.to_csv(output/"validation-blends.csv",index=False)
    result.reset_index().to_csv(output/"winner-original-test.csv.gz",index=False)
    (output/"README.md").write_text(
        "# Scalable FM, FFM and RFF alternatives for historical wholesale auction prices\n\n"
        "Original VIN-disjoint chronological validation selects among 10 independently "
        "trained FM (rank16/32), exact field-aware FM (rank8) and RFF approximate "
        "epsilon-insensitive SVR / Huber regressor variants on full and Lemon training. "
        "Original untrimmed 82,380 validation and 83,603 test auctions are unchanged. "
        "All trial artifacts contain only validation predictions; independent report "
        "evaluates test for exactly ONE validation-selected winner. Separate "
        "validation-selected XGBoost/FМ blend tests possible complementary errors "
        "without falsely claiming fit-on-honest-OOF residual correction. Full "
        "XGBoost is an aggregate unpaired comparator, Lemon XGBoost is paired "
        "on identical original auction IDs. No production promotion.\n")
    print("FM_KERNEL_FINAL_SELECTION",json.dumps(dict(
        selected=selected,validation_mae=board.iloc[0]["validation_mae_usd"],
        test_mae=test_scores["mae_usd"],lemon_xgb_test=xgb_scores["mae_usd"],
        selected_blend_weight=selected_alpha,blend_test_mae=blend_scores["mae_usd"])),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",default="training-cache/bid-fm-kernel")
    p.add_argument("--features",default="training-cache/neural-bid/features.npz")
    p.add_argument("--reference",default="benchmark-reference/bid/predictions.csv.gz")
    p.add_argument("--full-xgboost",default="models/fivefold-live/bid/evaluation.json")
    p.add_argument("--output",default="reports/bid-fm-kernel")
    a=p.parse_args()
    run(a.results,a.features,a.reference,a.full_xgboost,a.output)
