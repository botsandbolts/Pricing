"""Stage-two CVaR95 refinement: regression over all 200 validation results, then 25 Ask + 25 Bid joint trials.

The 200-trial run must have completed without missing/failing candidates.
Each proposal pairs hyperparameters and architectural settings ACTUALLY tested
in stage one. The regression estimates only the separate marginal effects:
the 50 joint fits test their interactions honestly. Neither stage consults TEST.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cvar95_historical import risk
from scripts.neural_200_sweep import GROUPS, N_TRIALS, ask, bid

MARKETS = ("ask","bid")
PER_MARKET = 25
ALPHAS = np.logspace(-2,4,16)


def _prior(path):
    doc=json.loads(Path(path).read_text())
    if (doc.get("status")!="complete" or doc.get("planned_trials")!=200
        or doc.get("completed_valid_trials")!=200 or doc.get("missing")
        or doc.get("invalid")):
        raise ValueError("Refinement requires a completely successful and audited 200-trial leaderboard")
    if any(doc.get("counts",{}).get(g)!=50 for g in GROUPS):
        raise ValueError("Original four 50-trial groups not complete")
    records=doc.get("experiments",[])
    if len(records)!=200:raise ValueError("Missing per-trial records for regression")
    seen=set()
    for r in records:
        key=(r.get("group"),r.get("trial"))
        if key in seen or key[0] not in GROUPS or not isinstance(key[1],int) or not 0<=key[1]<50:
            raise ValueError("Duplicate/unregistered training result")
        seen.add(key)
        if (r.get("status")!="completed" or r.get("test_evaluated") is not False
            or r.get("selection_split")!="original validation only"):
            raise ValueError("Original test used or training incomplete")
        m=r["validation"]
        expected=31668 if key[0].startswith("ask") else 82380
        if m.get("rows")!=expected or not np.isfinite(m["CVaR95_absolute_USD"]):
            raise ValueError("Untrusted or mismatched validation metric")
    if len(seen)!=200:raise ValueError("Original 200 unique identities required")
    hashes={}
    for market in MARKETS:
        group_records=[r for r in records if r["group"].startswith(market)]
        values={r["provenance"]["holdout_row_sha256"]["validation"] for r in group_records}
        cohorts={r["provenance"]["dataset_sha256"] for r in group_records}
        if len(values)!=1 or len(cohorts)!=1:
            raise ValueError("Mixed original validation vehicle cohort")
        hashes[market]={"validation":next(iter(values)),"dataset":next(iter(cohorts))}
    return doc,hashes


def _encode(items):
    values=pd.DataFrame([r["trial_configuration"] for r in items])
    if values.isna().any().any():raise ValueError("Training config has missing fields")
    for name in values:
        if name in ("lr","wd") and np.issubdtype(values[name].dtype,np.number):
            values[name]=np.log10(values[name].astype(float))
    for name in values:
        if values[name].dtype==object or values[name].dtype==bool:
            values[name]=values[name].astype(str)
    values=pd.get_dummies(values,dtype=float)
    return values.astype(float)


def _regression(rows):
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import KFold,cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    data=sorted(rows,key=lambda x:x["trial"])
    x=_encode(data)
    y=np.asarray([r["validation"]["CVaR95_absolute_USD"] for r in data],float)
    if len(y)!=50 or not np.isfinite(y).all():
        raise ValueError("Expected 50 finite original validation CVaR95 observations")
    pipeline=make_pipeline(StandardScaler(),RidgeCV(alphas=ALPHAS))
    cv=KFold(n_splits=5,shuffle=True,random_state=20261106)
    oof=cross_val_predict(pipeline,x,y,cv=cv)
    r2=float(r2_score(y,oof))
    oof_mae=float(mean_absolute_error(y,oof))
    pipeline.fit(x,y)
    pred=pipeline.predict(x)
    if not np.isfinite(pred).all():raise ValueError("Nonfinite surrogate estimates")
    # Negative OOF R² implies regression failed to outperform a constant;
    # rank ACTUAL observed configurations instead of pretending it predicts.
    useful=(r2>0.0)
    estimated=pred if useful else y
    coefficients=pipeline[-1].coef_
    effects=sorted(
        ({"feature":str(n),"standardized_usd_effect":float(v)}
         for n,v in zip(x.columns,coefficients)),
        key=lambda x:abs(x["standardized_usd_effect"]),reverse=True)[:10]
    return data,np.asarray(estimated),{
        "model":"Standardized RidgeCV with nested 5-fold OOF assessment",
        "observations":50,"cv_r2":r2,"cv_mae_usd":oof_mae,
        "observed_CVaR95_min":float(np.min(y)),
        "observed_CVaR95_median":float(np.median(y)),
        "ridge_alpha":float(pipeline[-1].alpha_),
        "surrogate_used_for_ranking":useful,
        "fallback":"observed original validation CVaR95" if not useful else None,
        "main_effects_only":True,
        "interactions_between_structure_and_hyperparameters_unobserved":True,
        "largest_standardized_coefficients":effects}


def _combine(market,records):
    hyper=[r for r in records if r["group"]==market+"_hparams"]
    structural=[r for r in records if r["group"]==market+"_structure"]
    h,hpred,hd=_regression(hyper)
    s,spred,sd=_regression(structural)
    # Correct study-to-study CVaR level shifts before combining; this is a
    # surrogate ranking, NOT an estimate of joint-fit CVaR95.
    hscale=max(float(np.std(hpred)),1.)
    sscale=max(float(np.std(spred)),1.)
    combinations=[]
    for hi,hr in enumerate(h):
        for si,sr in enumerate(s):
            score=((hpred[hi]-np.median(hpred))/hscale
                  +(spred[si]-np.median(spred))/sscale)
            combinations.append((float(score),hi,si))
    combinations.sort(key=lambda t:(t[0],h[t[1]]["trial"],s[t[2]]["trial"]))
    # Start with most promising and enforce diversity in BOTH source groups.
    chosen=[];huse={};suse={}
    for score,i,j in combinations:
        if len(chosen)==25:break
        if huse.get(i,0)>=3 or suse.get(j,0)>=3:continue
        hr,sr=h[i],s[j]
        selected={"trial":len(chosen),"market":market,
                  "hyperparameters":hr["trial_configuration"],
                  "structure":sr["trial_configuration"],
                  "source_hp_trial":hr["trial"],
                  "source_structure_trial":sr["trial"],
                  "surrogate_relative_rank_score":score,
                  "true_combined_validation_CVaR95_USD":None}
        chosen.append(selected)
        huse[i]=huse.get(i,0)+1;suse[j]=suse.get(j,0)+1
    if len(chosen)!=25 or len({(r["source_hp_trial"],r["source_structure_trial"])
                               for r in chosen})!=25:
        raise ValueError("Could not propose 25 diverse unique joint candidates")
    return chosen,{"hyperparameters":hd,"architecture":sd,
                   "selection":"minimize additive relative marginal CVaR95 surrogate; "
                   "cap reuse at three occurrences per source configuration",
                   "caution":"No Stage-1 joint hyperparameter/architecture interactions observed. "
                   "The 25 actual joint fits are needed to measure them."}


def create_plan(previous,output,parent_run):
    doc,hashes=_prior(previous)
    candidates={};diagnostics={}
    for market in MARKETS:
        candidates[market],diagnostics[market]=_combine(market,doc["experiments"])
    plan={
        "status":"ready",
        "upstream_run_id":str(parent_run),
        "upstream_report_sha256":hashlib.sha256(Path(previous).read_bytes()).hexdigest(),
        "upstream_status":"complete_200_of_200",
        "primary_metric":"original validation absolute USD CVaR95",
        "source_cohorts":hashes,
        "cross_market_models_never_mixed":True,
        "unseen_joint_combinations":True,
        "test_labels_accessed":False,
        "candidate_counts":{"ask":25,"bid":25},
        "candidates":candidates,
        "regression_diagnostics":diagnostics,
        "no_automatic_production_promotion":True}
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plan,indent=2,allow_nan=False))
    print("CVaR50_PLAN",json.dumps({"counts":plan["candidate_counts"],
                                  "diagnostics":diagnostics},allow_nan=False),flush=True)


def load_plan(path):
    plan=json.loads(Path(path).read_text())
    if plan.get("status")!="ready" or plan.get("upstream_status")!="complete_200_of_200":
        raise ValueError("Stage one incomplete; stage-two trials prohibited")
    for m in MARKETS:
        entries=plan.get("candidates",{}).get(m,[])
        if len(entries)!=25 or set(e["trial"] for e in entries)!=set(range(25)):
            raise ValueError("Exactly 25 registered "+m+" trials are required")
    return plan


def run_trial(plan_path,market,trial,args):
    plan=load_plan(plan_path)
    if market not in MARKETS or not 0<=trial<25:
        raise ValueError("Unregistered refinement trial")
    combination=plan["candidates"][market][trial]
    if combination["trial"]!=trial or combination["market"]!=market:
        raise ValueError("Refinement plan does not match requested trial")
    settings={"structure":combination["structure"],
              "hyperparameters":combination["hyperparameters"]}
    out=Path(args.output)
    if market=="ask":
        ask("ask_structure",trial,args.ask_features,args.ask_embeddings,out,
            args.max_seconds,refinement=settings)
    else:
        bid("bid_structure",trial,args.bid_features,args.bid_baseline,out,
            args.max_seconds,refinement=settings)


def report(plan_path,folder,output):
    plan=load_plan(plan_path)
    records={};issues=[]
    for file in sorted(Path(folder).rglob("result.json")):
        try:
            r=json.loads(file.read_text())
            group=r["group"]
            market=group.replace("_refined","")
            t=r["trial"]
            if group not in ("ask_refined","bid_refined") or not isinstance(t,int) or not 0<=t<25:
                raise ValueError("Unexpected refinement trial")
            key=(market,t)
            if key in records:raise ValueError("Duplicated candidate")
            intended=plan["candidates"][market][t]
            if r["trial_configuration"]!={"structure":intended["structure"],
                                         "hyperparameters":intended["hyperparameters"]}:
                raise ValueError("Candidate settings not identical to chosen regression proposal")
            if r.get("test_evaluated") is not False:
                raise ValueError("Original test used in training")
            actual_n=31668 if market=="ask" else 82380
            data=pd.read_csv(file.with_name("validation.csv.gz"),dtype={"row_id":str})
            if (len(data)!=actual_n or data.row_id.isna().any()
                or data.row_id.duplicated().any()
                or not data.split.eq("validation").all()):
                raise ValueError("Original validation sample identities incomplete")
            scores=risk(data.observed_price.to_numpy(float),
                        data.predicted_price.to_numpy(float))
            for metric in ("CVaR95_absolute_USD","MAE_USD","RMSE_USD"):
                if abs(scores[metric]-r["validation"][metric])>.05:
                    raise ValueError("Saved predictions disagree with reported "+metric)
            if r["provenance"]["holdout_row_sha256"]["validation"]!=plan["source_cohorts"][market]["validation"]:
                raise ValueError("Holdout fingerprint changed between stages")
            if not file.with_name("weights.pt").exists():
                raise ValueError("Missing native PyTorch checkpoint")
            records[key]=r
        except (OSError,ValueError,KeyError,TypeError) as exc:
            issues.append({"artifact":str(file),"error":str(exc)})
    missing=[{"market":m,"trial":i} for m in MARKETS for i in range(25)
             if (m,i) not in records]
    ranking={}
    for market in MARKETS:
        items=[r for (m,_),r in records.items() if m==market]
        items.sort(key=lambda x:(x["validation"]["CVaR95_absolute_USD"],
                                 x["validation"]["RMSE_USD"],x["trial"]))
        ranking[market]=[
            {"trial":r["trial"],"validation":r["validation"],
             "configuration":r["trial_configuration"]}
            for r in items]
    results={
        "status":"complete" if not missing and not issues and len(records)==50 else "partial_or_invalid",
        "completed_valid_trials":len(records),"planned_trials":50,
        "ask_count":len(ranking["ask"]),"bid_count":len(ranking["bid"]),
        "missing":missing,"invalid":issues,
        "metrics":"Original validation USD CVaR95, RMSE, MAE",
        "original_test_used":False,"date_2026_external_accuracy_validated":False,
        "production_changed":False,
        "next_steps":"Independently verify selected winner(s) on original untouched test, "
                     "native checkpoint serving parity, and deployment budgets.",
        "regression_diagnostics":plan["regression_diagnostics"],
        "ranked":ranking}
    dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    (dest/"results.json").write_text(json.dumps(results,indent=2,allow_nan=False))
    lines=["# Joint Ask/Bid follow-up: 50 additional models","",
           "Status: **"+results["status"]+"**",
           f"Successfully audited: **{len(records)}/50**",
           "24? No — exactly 25 Ask and exactly 25 Bid candidates.",
           "Regression: two independently cross-validated RidgeCV surrogates per "
           "market, trained on the original 200 validation CVaR95 results.",
           "All results are ORIGINAL VALIDATION only; no 2026 accuracy or automatic promotion.",
           "",
           "| Market | Successful | Best trial | CVaR95 (USD) | RMSE (USD) | MAE (USD) |",
           "|---|---:|---:|---:|---:|---:|"]
    for market in MARKETS:
        x=ranking[market]
        if x:
            leader=x[0];v=leader["validation"]
            lines.append(f"| {market} | {len(x)} | {leader['trial']} | "
                         f"{v['CVaR95_absolute_USD']:,.0f} | {v['RMSE_USD']:,.0f} | "
                         f"{v['MAE_USD']:,.0f} |")
        else:lines.append(f"| {market} | 0 | — | — | — | — |")
    if missing:lines.extend(["","Missing: "+str(missing)])
    if issues:lines.extend(["","Invalid: "+str(issues)])
    (dest/"README.md").write_text("\n".join(lines)+"\n")
    print("CVaR50_FINAL",results["status"],len(records),flush=True)
    if results["status"]!="complete":
        raise SystemExit(2)


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="command",required=True)
    pp=sub.add_parser("propose")
    pp.add_argument("--previous",required=True)
    pp.add_argument("--parent-run",required=True)
    pp.add_argument("--output",required=True)
    train=sub.add_parser("train")
    train.add_argument("--plan",required=True)
    train.add_argument("--market",choices=MARKETS,required=True)
    train.add_argument("--trial",required=True,type=int)
    train.add_argument("--output",required=True)
    train.add_argument("--ask-features",default="training-cache/neural-ask/features.npz")
    train.add_argument("--ask-embeddings",default="training-cache/bge-256")
    train.add_argument("--bid-features",default="training-cache/neural-bid/features.npz")
    train.add_argument("--bid-baseline",default="benchmark-reference/bid/predictions.csv.gz")
    train.add_argument("--max-seconds",type=float,default=3000)
    rr=sub.add_parser("report")
    rr.add_argument("--plan",required=True)
    rr.add_argument("--artifacts",required=True)
    rr.add_argument("--output",required=True)
    a=p.parse_args()
    if a.command=="propose":create_plan(a.previous,a.output,a.parent_run)
    elif a.command=="train":run_trial(a.plan,a.market,a.trial,a)
    else:report(a.plan,a.artifacts,a.output)


if __name__=="__main__":main()
