"""Strict 36-arm DeepCross AFT hyperparameter benchmark against frozen past models.

Validation ranks only; test is used for heldout final audit and paired uncertainty,
not for choosing a configuration. Do not publish incomplete or mismatched jobs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.deepcross_aft_tune import CONFIGS, FEATURES, sha_ids
from scripts.neural_ask_report import _scores, paired_interval, reference


def aligned(path, baseline):
    data = pd.read_csv(path, dtype={"row_id":str})
    required = {"row_id","split","observed_price","predicted_price"}
    if not required.issubset(data) or data.row_id.duplicated().any() or len(data)!=len(baseline):
        raise ValueError("Missing/duplicated heldout prediction rows in " + str(path))
    data = data.set_index("row_id")
    if set(data.index)!=set(baseline.index):
        raise ValueError("Holdout row identities differ in " + str(path))
    data = data.loc[baseline.index]
    if not np.array_equal(data.split.to_numpy(),baseline.split.to_numpy()):
        raise ValueError("Holdout partition differs in " + str(path))
    if not np.allclose(data.observed_price,baseline.observed_price,atol=.05,rtol=1e-6):
        raise ValueError("Holdout price labels differ in " + str(path))
    if not np.isfinite(data.predicted_price.to_numpy(float)).all() or (data.predicted_price<=0).any():
        raise ValueError("Invalid predicted prices in " + str(path))
    return data


def choose_winner(frame):
    v=frame[frame.partition=="validation"].sort_values(
        ["mae_usd","variant","feature_variant"], kind="stable")
    if v.empty:
        raise ValueError("No validation results")
    return v.iloc[0]


def run(args):
    root=Path(args.results)
    files=sorted(root.rglob("result.json"))
    expected={(a,b) for a in CONFIGS for b in FEATURES}
    if len(files)!=len(expected):
        raise ValueError(f"Require exactly {len(expected)} successful tuning arms, found {len(files)}")
    xgb=reference(args.xgb)
    if len(xgb)!=63350 or int((xgb.split=="validation").sum())!=31668 or int((xgb.split=="test").sum())!=31682:
        raise ValueError("Original immutable asking holdouts missing")
    original=aligned(Path(args.original)/"predictions.csv.gz",xgb)
    mini=aligned(Path(args.minilm)/"predictions.csv.gz",xgb)
    published=json.loads(Path(args.full_xgb).read_text())
    if (published["scores"]["validation"]["n"],published["scores"]["test"]["n"])!=(31668,31682):
        raise ValueError("Original full XGBoost baseline counts drifted")
    results=[]
    seen=set()
    predictions={}
    for path in files:
        rec=json.loads(path.read_text())
        key=(rec.get("variant"),rec.get("feature_variant"))
        if key not in expected or key in seen or rec.get("status")!="completed":
            raise ValueError("Invalid, duplicate or incomplete training arm " + str(key))
        seen.add(key)
        if (rec.get("architecture"),rec.get("loss"),rec.get("original_train_rows"),rec.get("train_fit"))!=("deepcross","aft",147777,133000):
            raise ValueError("Wrong architecture, loss or training subset for "+str(key))
        if rec.get("no_hierarchy") is not True or rec.get("settings",{}).get("epochs")!=35:
            raise ValueError("Invalid benchmark setup")
        data=aligned(path.with_name("predictions.csv.gz"),xgb)
        predictions[key]=data
        for split in ("validation","test"):
            mask=xgb.split==split
            b=xgb.loc[mask]
            actual=data.loc[mask]
            if sha_ids(actual.index)!=rec["holdout_row_sha256"][split]:
                raise ValueError("Wrong original heldout fingerprint")
            y=actual.observed_price.to_numpy(float)
            p=actual.predicted_price.to_numpy(float)
            metric=_scores(y,p)
            saved=rec["original_validation" if split=="validation" else "original_test"]
            if abs(metric["mae_usd"]-saved["mae_usd"])>.05:
                raise ValueError("Saved metric differs from actual artifact")
            err=np.abs(p-y)
            ref=np.abs(b.predicted_price.to_numpy(float)-y)
            orig=np.abs(original.loc[mask,"predicted_price"].to_numpy(float)-y)
            minilm=np.abs(mini.loc[mask,"predicted_price"].to_numpy(float)-y)
            results.append(dict(variant=key[0],feature_variant=key[1],partition=split,
                validation_selection_only=True,best_epoch=rec["best_epoch"],
                train_fit=rec["train_fit"],**metric,
                delta_mae_vs_xgb_lemon_usd=float((err-ref).mean()),
                delta_mae_vs_previous_deepcross_usd=float((err-orig).mean()),
                delta_mae_vs_previous_minilm_deepcross_usd=float((err-minilm).mean()),
                pct_listings_better_than_xgb_lemon=float((err<ref).mean())))
    if seen!=expected or len(results)!=2*len(expected):
        raise ValueError("Missing tuned arms")
    frame=pd.DataFrame(results)
    vals=frame[frame.partition=="validation"].sort_values(["mae_usd","variant","feature_variant"],kind="stable")
    rank=vals[["variant","feature_variant"]].copy()
    rank["validation_rank"]=range(1,len(vals)+1)
    frame=frame.merge(rank,on=["variant","feature_variant"],validate="many_to_one")
    champion=choose_winner(frame)
    key=(champion.variant,champion.feature_variant)
    winner=predictions[key]
    comparisons={}
    for split in ("validation","test"):
        mask=xgb.split==split
        y=xgb.loc[mask].observed_price.to_numpy(float)
        champ=np.abs(winner.loc[mask,"predicted_price"].to_numpy(float)-y)
        variants={
            "previous_svd_deepcross_aft":original,
            "previous_minilm_deepcross_aft":mini,
            "xgboost_lemon_aft":xgb,
        }
        comparisons[split]={}
        for k,d in variants.items():
            errors=np.abs(d.loc[mask,"predicted_price"].to_numpy(float)-y)
            comparisons[split][k]={"reference_mae_usd":float(errors.mean()),
                 "winner_mae_delta_usd":float((champ-errors).mean()),
                 "paired_95pct_bootstrap_delta_usd":paired_interval(champ,errors),
                 "winner_better_fraction":float((champ<errors).mean())}
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    frame.sort_values(["validation_rank","partition"]).to_csv(out/"leaderboard.csv",index=False)
    report={"status":"complete","study":"DeepCross+AFT asking-price architecture and hyperparameter sweep",
        "selected_by":"Original chronological validation MAE only; never test",
        "total_configurations":len(expected),
        "architecture_variants":list(CONFIGS),
        "feature_families":list(FEATURES),
        "holdout_rows":{"validation":31668,"test":31682},
        "winner_by_validation":{"architecture":"deepcross","loss":"aft",
            "variant":champion.variant,"features":champion.feature_variant,
            "validation_mae_usd":float(champion.mae_usd),
            "test_mae_usd":float(frame[(frame.variant==champion.variant)&(frame.feature_variant==champion.feature_variant)&(frame.partition=="test")].iloc[0].mae_usd)},
        "paired_winner_comparisons":comparisons,
        "full_xgboost_published_unpaired":{"validation":published["scores"]["validation"],
            "test":published["scores"]["test"],
            "limitation":"Only aggregate data; per-listing prediction identity not verified"},
        "training_policy":{"full_training_cohort":147777,"actual_fit_rows":133000,
            "internal_temporal_stopper":14777,"maximum_epochs":35,
            "early_stopping_patience":7,"seed":20261106,
            "target":"exact historical Craigslist asking dollars, no censoring",
            "single_seed_caveat":"Exploratory hyperparameter sweep; validate top candidates across multiple seeds before deployment",
            "no_auto_deployment":True,
            "features":"same frozen 24 SVD text features; optional audited MiniLM 2-out-of-bag price scalar"},
        "results":frame.sort_values(["validation_rank","partition"]).to_dict("records")}
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    (out/"README.md").write_text(
        "# DeepCross/AFT tuning · original Craigslist asking cohort\n\n"
        f"36 runs = {len(CONFIGS)} architecture/hyperparameter configurations × two feature sets.\n\n"
        "Winner selected **strictly by original validation MAE**. The 31,682-row original test is "
        "reported for final auditing, not for tuning. Previous original and MiniLM DeepCross "
        "checkpoints and XGBoost Lemon are per-row matched. Full XGBoost is published aggregate only. "
        "All original weights are preserved as GitHub Actions artifacts. Nothing deployed automatically.\n")
    print(json.dumps({"winner":report["winner_by_validation"],
                      "test_compared_to":comparisons["test"]},indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--results",default="training-cache/deepcross-aft-tuning")
    p.add_argument("--xgb",default="benchmark-xgb/ask/predictions.csv.gz")
    p.add_argument("--original",default="benchmark-original")
    p.add_argument("--minilm",default="benchmark-minilm")
    p.add_argument("--full-xgb",default="models/fivefold-live/ask/evaluation.json")
    p.add_argument("--output",default="reports/deepcross-aft-tuning")
    run(p.parse_args())
