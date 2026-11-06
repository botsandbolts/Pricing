"""Dollar-RMSE-first independent re-evaluation of original wholesale bid models.

Consumes archived VALIDATION predictions from existing FT-Transformer,
FM/kernel and 26-arm factorization sweeps. All model candidates and
XGBoost blend weights are selected using the SAME VIN-disjoint validation
set, on USD RMSE (not MAE). ONLY validation-selected checkpoint(s) are
evaluated on untouched original test. It does not retrain model weights;
that distinction is explicit in results and documentation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.neural_bid_prepare import sha_ids
from scripts.neural_bid_report import reference
from scripts.bid_fm_kernel_train import source,partition,rff_features,y_price
from scripts.bid_fm_kernel_report import validation_frame,ALPHAS,winner_test as kernel_predict
from scripts.bid_fm_scale_train import ScaledAuctionFM,prediction as scale_predict
from scripts.ft_bid_tune import frozen as ft_frozen,model_for,settings,tensors
from scripts.neural_ask_train import predict as neural_predict,seed_everything

STUDIES=("ft29","fm10","fm26")
TRIAL_COUNT={"ft29":29,"fm10":10,"fm26":26}
SEED=20261106


def score(y,p):
    y=np.asarray(y,dtype=np.float64)
    p=np.asarray(p,dtype=np.float64)
    if (y.shape!=p.shape or y.ndim!=1 or not len(y)
        or not np.isfinite(y).all() or not np.isfinite(p).all()
        or (y<=0).any() or (p<=0).any()):
        raise ValueError("Wrong target geometry or nonfinite/nonpositive prediction")
    signed=p-y
    err=np.abs(signed)
    return {
        "rows":int(len(y)),
        "rmse_usd":float(np.sqrt(np.mean(signed**2))),
        "mae_usd":float(np.mean(err)),
        "mean_signed_error_usd":float(np.mean(signed)),
        "absolute_error_p90_usd":float(np.percentile(err,90)),
        "absolute_error_p95_usd":float(np.percentile(err,95)),
        "absolute_error_p99_usd":float(np.percentile(err,99)),
        "absolute_error_p99_9_usd":float(np.percentile(err,99.9)),
        "worst_1_percent_average_absolute_error_usd":float(
            np.mean(np.sort(err)[-max(1,int(np.ceil(len(err)*.01))):])),
        "fraction_error_over_5000_usd":float(np.mean(err>5000)),
        "fraction_error_over_10000_usd":float(np.mean(err>10000)),
        "fraction_error_over_20000_usd":float(np.mean(err>20000)),
        "overestimation_rate":float(np.mean(signed>0)),
    }


def paired_rmse_interval(y,p,reference,seed=SEED,replications=350):
    """Paired bootstrap of RMSE difference, not bootstrap of MAE."""
    y=np.asarray(y,np.float64)
    a=np.square(np.asarray(p,np.float64)-y)
    b=np.square(np.asarray(reference,np.float64)-y)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid paired squared errors")
    rng=np.random.default_rng(seed)
    result=np.empty(replications,np.float64)
    for i in range(replications):
        ids=rng.integers(len(y),size=len(y))
        result[i]=np.sqrt(a[ids].mean())-np.sqrt(b[ids].mean())
    return list(map(float,np.percentile(result,[2.5,97.5])))


def specs(study):
    if study=="ft29":
        from scripts.ft_bid_tune import TRIALS
        return {name:("ft",None,"lemon") for name in TRIALS}
    if study=="fm10":
        from scripts.bid_fm_kernel_train import TRIALS
        return TRIALS
    if study=="fm26":
        from scripts.bid_fm_scale_train import TRIALS
        return TRIALS
    raise ValueError("Unknown study")


def validate_manifest(study,name,meta,source_meta,z):
    family,rank,cohort=specs(study)[name]
    if (meta.get("status"),meta.get("trial"),
        meta.get("market"),meta.get("test_evaluated"))!=(
        "completed",name,"bid",False):
        raise ValueError("Wrong original candidate or test already evaluated: "+name)
    if study=="ft29":
        fit,inner,_,_=partition(z,"lemon")
        from scripts.ft_bid_tune import settings
        if (meta.get("architecture"),meta.get("cohort"),
            meta.get("config"),meta.get("train_fit"),meta.get("inner_rows"),
            meta.get("canonical_cohort_sha256"),
            meta.get("holdout_row_sha256"))!=(
            "fttransformer","lemon",settings(name),len(fit),len(inner),
            source_meta["cohort_sha256"],source_meta["holdout_row_sha256"]):
            raise ValueError("FT source/configuration changed: "+name)
    elif study=="fm10":
        fit,inner,_,_=partition(z,cohort)
        if (meta.get("model_family"),meta.get("rank_or_components"),
            meta.get("cohort"),meta.get("fitting_rows"),
            meta.get("stopper_rows"),meta.get("validation_rows"),
            meta.get("canonical_cohort_sha256"),
            meta.get("holdout_row_sha256"),
            meta.get("price_derived_inputs"))!=(
            family,rank,cohort,len(fit),len(inner),82380,
            source_meta["cohort_sha256"],
            source_meta["holdout_row_sha256"],False):
            raise ValueError("FM/kernel source or original split changed: "+name)
    else:
        fit,inner,_,_=partition(z,cohort)
        if (meta.get("family"),meta.get("rank"),meta.get("cohort"),
            meta.get("fitting_rows"),meta.get("stopper_rows"),
            meta.get("validation_rows"),meta.get("untouched_test_rows"),
            meta.get("cohort_sha256"),meta.get("original_holdout_row_sha256"),
            meta.get("no_price_derived_inputs"))!=(
            family,rank,cohort,len(fit),len(inner),82380,83603,
            source_meta["cohort_sha256"],source_meta["holdout_row_sha256"],True):
            raise ValueError("Scaled FM source or original split changed: "+name)


def checkpoint_predictions(study,trial,folder,z,manifest):
    """The only model test inference allowed, after ALL validation choices."""
    family,rank,cohort=specs(study)[trial]
    test_idx=np.flatnonzero(z["phase"].astype(str)=="test")
    if len(test_idx)!=83603:
        raise ValueError("Original test VIN cohort changed")
    if study=="fm10":
        pred=kernel_predict(trial,folder,z,manifest,test_idx)
    elif study=="fm26":
        pack=torch.load(Path(folder)/"weights.pt",map_location="cpu",weights_only=True)
        if (pack.get("trial"),pack.get("family"),pack.get("rank"),
            pack.get("cohort"),pack.get("cards"),pack.get("seed"))!=(
            trial,family,rank,cohort,manifest["cardinalities"],SEED):
            raise ValueError("Scaled FM winner weights from wrong model")
        net=ScaledAuctionFM(manifest["cardinalities"],family,rank,
                            pack["initial_log_price"])
        net.load_state_dict(pack["state_dict"],strict=True)
        pred=scale_predict(net,z["num"].astype(np.float32),
                           z["cat"].astype(np.int64),test_idx)
    else:
        from scripts.ft_bid_tune import frozen,tensors
        full,original,fit,inner,val,test=frozen(
            "training-cache/neural-bid/features.npz")
        pack=torch.load(Path(folder)/"weights.pt",map_location="cpu",weights_only=True)
        if (pack.get("trial"),pack.get("config"),pack.get("cards"),
            pack.get("n_num"),pack.get("n_text"))!=(
            trial,settings(trial),manifest["cardinalities"],5,1):
            raise ValueError("FT checkpoint/configuration differs from validation")
        torch.manual_seed(SEED)
        net=model_for(trial,manifest["cardinalities"],5,1)
        net.load_state_dict(pack["state"],strict=True)
        pred=neural_predict(net,tensors(full),test,settings(trial)["batch"])
    if len(pred)!=83603 or not np.isfinite(pred).all() or (np.asarray(pred)<=0).any():
        raise ValueError("Selected test inference malformed")
    return pd.DataFrame({
        "row_id":z["ids"][test_idx].astype(str),
        "split":"test","observed_price":z["y"][test_idx].astype(float),
        "predicted_price":pred}).set_index("row_id")


def run(study,root,features,baseline_path,full_xgb_path,output):
    if study not in STUDIES:
        raise ValueError("Unregistered RMSE first study")
    z,manifest=source(features)
    ref=reference(baseline_path)
    if (len(ref),int(sum(ref.split=="validation")),
        int(sum(ref.split=="test")))!=(165983,82380,83603):
        raise ValueError("Original row-paired Lemon benchmark not found")
    val=ref.loc[ref.split=="validation"]
    test=ref.loc[ref.split=="test"]
    for label,view in (("validation",val),("test",test)):
        if sha_ids(view.index)!=manifest["holdout_row_sha256"][label]:
            raise ValueError("XGBoost benchmark and candidate original holdouts disagree")
    prior=json.loads(Path(full_xgb_path).read_text())
    if (prior["scores"]["validation"]["n"],
        prior["scores"]["test"]["n"])!=(82380,83603):
        raise ValueError("Full-data XGBoost aggregate reference has wrong test size")
    root=Path(root)
    trials=specs(study)
    files=sorted(root.rglob("result.json"))
    if len(files)!=TRIAL_COUNT[study]:
        raise ValueError(f"RMSE audit requires all {TRIAL_COUNT[study]} results; saw {len(files)}")
    y=val.observed_price.to_numpy(float)
    xgbp=val.predicted_price.to_numpy(float)
    rows=[];blends=[];seen=set();paths={}
    for path in files:
        record=json.loads(path.read_text())
        name=record.get("trial")
        if name not in trials or name in seen:
            raise ValueError("Duplicated/unknown archived RMSE candidate")
        seen.add(name)
        validate_manifest(study,name,record,manifest,z)
        prediction=validation_frame(path.with_name("validation.csv.gz"),val)
        guess=prediction.predicted_price.to_numpy(float)
        s=score(y,guess)
        # Protect against tampered prediction files: compare their documented MAE.
        if abs(s["mae_usd"]-record["validation"]["mae_usd"])>.05:
            raise ValueError("Candidate validation evidence does not match archived score")
        fam,rank,cohort=trials[name]
        rows.append({"trial":name,"family":fam,"rank_or_components":rank,
                     "cohort":cohort,**s})
        for alpha in ALPHAS:
            mix=(1-alpha)*xgbp+alpha*guess
            current=score(y,mix)
            blends.append({"trial":name,"cohort":cohort,"fm_neural_weight":float(alpha),
                           "xgboost_lemon_weight":float(1-alpha),
                           "rmse_usd":current["rmse_usd"],
                           "mae_usd":current["mae_usd"],
                           "absolute_error_p99_usd":current["absolute_error_p99_usd"]})
        paths[name]=path.parent
    if seen!=set(trials):
        raise ValueError("Incomplete registered validation candidate set")
    leaderboard=pd.DataFrame(rows).sort_values(["rmse_usd","trial"]).reset_index(drop=True)
    leaderboard["rmse_rank"]=np.arange(1,len(leaderboard)+1)
    blends_table=pd.DataFrame(blends).sort_values(
        ["rmse_usd","trial","fm_neural_weight"]).reset_index(drop=True)
    standalone=str(leaderboard.iloc[0]["trial"])
    blend_trial=str(blends_table.iloc[0]["trial"])
    blend_alpha=float(blends_table.iloc[0]["fm_neural_weight"])
    # BOTH test models are PRESELECTED ON VAL; at most two distinct models
    # receive test predictions; no test-scored model is used to change ranking.
    selected=tuple(dict.fromkeys((standalone,blend_trial)))
    predictions={}
    truth=test.observed_price.to_numpy(float)
    for trial in selected:
        held=checkpoint_predictions(study,trial,paths[trial],z,manifest)
        if (set(held.index)!=set(test.index) or
            not np.allclose(held.reindex(test.index).observed_price.to_numpy(float),
                            truth,rtol=1e-6,atol=.05)):
            raise ValueError("Test outputs differ from original VIN/label reference")
        predictions[trial]=held.reindex(test.index).predicted_price.to_numpy(float)
    xgbtest=test.predicted_price.to_numpy(float)
    test_standalone=score(truth,predictions[standalone])
    bp=(1-blend_alpha)*xgbtest+blend_alpha*predictions[blend_trial]
    test_blend=score(truth,bp)
    ref_score=score(truth,xgbtest)
    family_cohorts=[]
    for (family,cohort),items in leaderboard.groupby(["family","cohort"],dropna=False):
        first=items.iloc[0]
        family_cohorts.append({
            "family":family,"cohort":cohort,"trial":str(first["trial"]),
            "validation_rmse_usd":float(first["rmse_usd"]),
            "validation_mae_usd":float(first["mae_usd"])})
    outcome={
        "status":"complete","study":study,"market":"bid",
        "primary_selection_metric":"original_holdout_validation_RMSE_USD",
        "selection_loss":"square errors in DOLLARS, not log-price units",
        "trained_objectives_unchanged":True,
        "training_objective_warning":"Archived model weights were fitted using Huber on log dollars; this reanalysis selects them by USD RMSE but does not claim retraining with USD squared loss.",
        "original_auctions":{"train_full":384198,"train_lemon":288148,
                             "validation":82380,"test":83603},
        "evaluated_trials":len(trials),
        "selected_test_model_count":len(selected),
        "original_test_used_only_after_val_selection":True,
        "standalone_selected_on_validation":{
            "trial":standalone,
            "validation":{k:v for k,v in leaderboard.iloc[0].to_dict().items()
                          if k in ("rmse_usd","mae_usd","absolute_error_p95_usd",
                                   "absolute_error_p99_usd")},
            "test":test_standalone,
            "test_rmse_delta_vs_exact_xgb_lemon_usd":
                test_standalone["rmse_usd"]-ref_score["rmse_usd"],
            "paired_bootstrap_95pct_rmse_delta_usd":paired_rmse_interval(
                truth,predictions[standalone],xgbtest)},
        "blend_selected_on_validation":{
            "trial":blend_trial,
            "model_weight":blend_alpha,"xgboost_lemon_weight":1-blend_alpha,
            "validation":{"rmse_usd":float(blends_table.iloc[0]["rmse_usd"]),
                          "mae_usd":float(blends_table.iloc[0]["mae_usd"])},
            "test":test_blend,
            "test_rmse_delta_vs_exact_xgb_lemon_usd":
                test_blend["rmse_usd"]-ref_score["rmse_usd"],
            "paired_bootstrap_95pct_rmse_delta_usd":paired_rmse_interval(
                truth,bp,xgbtest,seed=SEED+13)},
        "matched_lemon_xgboost":{
            "validation":score(y,xgbp),"test":ref_score},
        "published_full_xgboost_aggregate_unpaired":{
            "validation":prior["scores"]["validation"],
            "test":prior["scores"]["test"],
            "caveat":"Full model uses more training auctions; not identical training policy or independently per-row paired here."},
        "prevent_catastrophic_errors":{
            "note":"RMSE penalizes large dollar errors but does not guarantee a maximum error bound; p95/p99/p99.9, top 1pct and $5k/$10k/$20k error rates are tracked independently.",
            "absolute_error_bands_USD":[5000,10000,20000]},
        "full_vs_lemon_by_family":family_cohorts,
        "no_production_promotion":True,
        "candidate_validation_leaderboard":leaderboard.to_dict(orient="records"),
        "blend_validation_leaderboard":blends_table.to_dict(orient="records"),
    }
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    (output/"comparison.json").write_text(json.dumps(outcome,indent=2,allow_nan=False))
    leaderboard.to_csv(output/"validation-rmse-leaderboard.csv",index=False)
    blends_table.to_csv(output/"validation-rmse-blends.csv",index=False)
    pd.DataFrame(family_cohorts).to_csv(output/"full-versus-lemon-rmse.csv",index=False)
    test_rows=pd.DataFrame({"row_id":test.index,"observed_price":truth,
                 "xgboost_lemon_price":xgbtest,
                 "rmse_selected_standalone_prediction":predictions[standalone],
                 "rmse_selected_blend_prediction":bp})
    test_rows.to_csv(output/"original-test-selected-predictions.csv.gz",index=False)
    notes=[
       "# USD RMSE-first bid re-analysis: "+study,
       "",
       "Original frozen VIN-disjoint cohort: 384,198 training sales,",
       "288,148 Lemon training sales, 82,380 validation, 83,603 test.",
       "Re-ranked existing weights using original validation USD RMSE.",
       "Weights were previously trained using Huber on LOG USD,",
       "**not newly retrained against dollar squared error**.",
       "",
       "Standalone validation RMSE winner: "+standalone,
       "Standalone original-test USD RMSE: %.2f" % test_standalone["rmse_usd"],
       "Blend validation RMSE winner: %s (weight %.2f)"%(blend_trial,blend_alpha),
       "Blend original-test USD RMSE: %.2f" % test_blend["rmse_usd"],
       "Paired Lemon XGBoost original-test USD RMSE: %.2f"%ref_score["rmse_usd"],
       "RMSE was NEVER computed on test to select a model.",
       "Model MAE, p95, p99, p99.9, worst 1pct and catastrophic dollar",
       "error rates are also reported. No automatic production promotion.",
       ""]
    (output/"README.md").write_text("\n".join(notes))
    print("RMSE_AUDIT_VERIFIED",json.dumps({
        "study":study,"val_model":standalone,"test_rmse":test_standalone["rmse_usd"],
        "val_blend_model":blend_trial,"alpha":blend_alpha,
        "blend_test_rmse":test_blend["rmse_usd"],
        "lemon_xgb_test_rmse":ref_score["rmse_usd"],
        "evaluated_test_models":len(selected)}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",choices=STUDIES,required=True)
    p.add_argument("--results",required=True)
    p.add_argument("--features",default="training-cache/neural-bid/features.npz")
    p.add_argument("--reference",default="benchmark-reference/bid/predictions.csv.gz")
    p.add_argument("--full-xgboost",default="models/fivefold-live/bid/evaluation.json")
    p.add_argument("--output",required=True)
    a=p.parse_args()
    run(a.study,a.results,a.features,a.reference,a.full_xgboost,a.output)
