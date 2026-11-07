"""Overnight high-dimensional BOHB optimization with exact 06:00 CET cutoff.

Artifact-coordinated GitHub jobs: 4 Ask + 4 Bid at a time, six batches.
This is the HpBandSter BOHB configuration generator with successive-halving
promotions (4 -> 12 -> 36 epochs), not Optuna TPE with a different label.
No networked nameserver; each batch reconstructs the KDE from audited artifacts.

Promotions RE-FIT from scratch with the same deterministic native seed rather
than restoring optimizer state. Ask uses constant LR to give identical
per-epoch training behavior at different maximum budgets. The cost is repeated
work for promotions, explicitly noted in the reports.

No TEST data, no Streamlit production checkpoint writes, no automatic deploy.
"""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.hyperband_cvar95 import (
    audit_and_save_bid_blend, incumbent_comparison,
)

MARKETS=("ask","bid")
SLOTS=4
STAGES=30
BUDGETS=(6,18,54,162,216)
SEED=20261107
ROW_COUNTS={"ask":31668,"bid":82380}
DEADLINE=datetime(2026,11,8,5,0,0,tzinfo=timezone.utc)  # 06:00 Berlin CET
MIN_START_SECONDS=1800

def promotions_for(stage):
    return ((),(18,),(54,18),(162,54,18),(216,162,54))[stage%5]

def remaining_seconds():
    return (DEADLINE-datetime.now(timezone.utc)).total_seconds()



def search_space(market):
    """Thirty-plus BOHB dimensions; log-scaled learning dynamics and broad alternatives."""
    import ConfigSpace as CS
    if market not in MARKETS:raise ValueError("Bad market")
    s=CS.ConfigurationSpace(seed=SEED+(0 if market=="ask" else 1729))
    C=CS.CategoricalHyperparameter;L=CS.UniformFloatHyperparameter
    all=[
        L("lr",lower=.00008,upper=.004 if market=="ask" else .008,log=True),
        L("wd",lower=.000001,upper=.20,log=True),
        C("batch",[256,384,512,768,1024,1536] if market=="ask" else [512,1024,1536,2048,4096]),
        C("optimizer",["adamw","adam","radam","nadam","amsgrad"]),
        C("beta1",[.82,.88,.9,.94]),C("beta2",[.98,.995,.999]),
        C("schedule",["constant","cosine216","warm_restart","step_decay"]),
        C("clip",[.35,.7,1.,1.5,2.5,4.]),C("patience",[8,12,20,32]),
        C("loss",["aft","huber_log","charbonnier_log","logcosh_log",
          "mae_log","mse_log","expectile_log","huber_dollar",
          "hybrid_aft_dollar","cvar_dollar","soft_cvar_dollar",
          "focal_dollar","asymmetric_dollar","hybrid_log_dollar"]),
        C("beta",[.08,.2,.4,.75,1.2]),C("tau",[.25,.4,.5,.6,.75]),
        C("tail_fraction",[.05,.1,.2]),
        C("tail_strength",[0.,.03,.1,.25,.5,1.0]),
        C("usd_scale",[2500.,5000.,10000.,20000.]),
        C("tail_temperature",[.025,.075,.2,.5]),
        C("focal_gamma",[.5,1.,1.5,2.]),
        C("sigma_floor",[.03,.06,.12,.24]),
    ]
    if market=="ask":
        all += [
            C("projection",[48,64,96,128,192,256]),
            C("project_layers",[1,2,3,4]),
            C("proj_dropout",[0.,.05,.10,.18,.28]),
            C("cross_layers",[1,2,3,4,5,6]),
            C("rank",[8,16,24,32,48,64]),C("experts",[1,2,3,4]),
            C("width",[128,192,256,384,512]),
            C("blocks",[1,2,3,4,5]),
            C("dropout",[.05,.10,.18,.26,.36]),
            C("activation",["gelu","silu"]),
            C("output_dropout",[.05,.15,.25,.35]),
            C("embed_dropout",[.05,.12,.2,.3]),
            C("cross_scale",[.25,.5,1.,1.5]),
            C("noise",[0.,.004,.01,.02,.04]),
        ]
    else:
        all += [
            C("rank",[16,32]),C("width1",[64,96,128,192,256,384]),
            C("width2",[32,64,96,128,192]),
            C("depth",[1,2,3,4,5]),
            C("dropout",[.03,.10,.18,.28,.38]),
            C("activation",["gelu","silu"]),
            C("pairgate",[False,True]),
            C("blend_alpha",[.5,.6,.7,.8,.9,.95,1.]),
        ]
    s.add_hyperparameters(all)
    return s


def settings_from_params(market,params,budget):
    if budget not in BUDGETS:raise ValueError("Unregistered overnight fidelity")
    h=("lr","wd","batch","optimizer","beta1","beta2","schedule",
       "clip","patience","loss","beta","tau","tail_fraction",
       "tail_strength","usd_scale","tail_temperature","focal_gamma","sigma_floor")
    hp={key:params[key] for key in h}
    hp["epochs"]=budget
    if market=="ask":
        hp["noise"]=params["noise"]
        cfg={key:params[key] for key in (
            "projection","project_layers","proj_dropout","cross_layers",
            "rank","experts","width","blocks","dropout","activation",
            "output_dropout","embed_dropout","cross_scale")}
    elif market=="bid":
        hp["blend_alpha"]=params["blend_alpha"]
        cfg={key:params[key] for key in (
            "rank","width1","width2","depth","dropout","activation","pairgate")}
    else:raise ValueError("Unknown BOHB market")
    return {"hyperparameters":hp,"structure":cfg}


def read_history(path):
    if not path:
        return []
    doc=json.loads(Path(path).read_text())
    rows=doc["records"]
    if len(rows)!=len({(r["market"],r["stage"],r["slot"]) for r in rows}):
        raise ValueError("Repeated BOHB trial IDs")
    for r in rows:
        if (r["market"] not in MARKETS or r["state"] not in ("COMPLETE","FAILED")
            or r["budget"] not in BUDGETS or not isinstance(r["trial"],int)
            or r.get("test_evaluated") is not False):
            raise ValueError("Incomplete, unregistered or test-contaminated BOHB evidence")
        if r["state"]=="COMPLETE":
            if not math.isfinite(float(r["validation"]["CVaR95_absolute_USD"])):
                raise ValueError("Nonfinite BOHB CVaR95 metric")
        elif r.get("validation") is not None or not r.get("error"):
            raise ValueError("Failed experiments need error details, no fabricated metrics")
    for market in MARKETS:
        hashes={r["provenance"]["holdout_row_sha256"]["validation"]
                for r in rows if r["market"]==market and r["state"]=="COMPLETE"}
        if len(hashes)>1:
            raise ValueError("BOHB validation vehicle cohorts changed")
    return rows


def bohb_generator(market,history):
    from hpbandster.optimizers.config_generators.bohb import BOHB
    space=search_space(market)
    generator=BOHB(configspace=space,top_n_percent=40,
                   min_points_in_model=36,
                   random_fraction=.35,num_samples=96)
    # Load only completed, independently audited per-budget objective values.
    for row in sorted((r for r in history if r["market"]==market and r["state"]=="COMPLETE"),
                      key=lambda r:(r["stage"],r["slot"])):
        wrapper=SimpleNamespace(
            id=(row["stage"],row["slot"],0),
            kwargs={"config":row["params"],"budget":float(row["budget"])},
            result={"loss":float(row["validation"]["CVaR95_absolute_USD"])},
            exception=None,
        )
        generator.new_result(wrapper)
    return generator


def _identity(params):
    return json.dumps(params,sort_keys=True,separators=(",",":"))


def _promotion(market,history,target,planned):
    prev={18:6,54:18,162:54,216:162}[target]
    eligible=[r for r in history if r["market"]==market and r["state"]=="COMPLETE" and r["budget"]==prev]
    seen={(r["budget"],_identity(r["params"])) for r in history if r["market"]==market}
    seen.update((r["budget"],_identity(r["params"])) for r in planned)
    eligible=[r for r in eligible if (target,_identity(r["params"])) not in seen]
    eligible.sort(key=lambda r:(r["validation"]["CVaR95_absolute_USD"],r["trial"]))
    if not eligible:
        raise ValueError("No completed eligible candidate for BOHB promotion "+str(target))
    parent=eligible[0]
    return parent["params"],parent["origin_trial"],parent["trial"]


def propose(stage,history_path,output):
    if not 0 <= stage < STAGES:
        raise ValueError("BOHB batch must be between 0 and 29")
    history=read_history(history_path)
    if len(history)!=stage*SLOTS*len(MARKETS):
        raise ValueError("All earlier BOHB batch audits must pass before proposing")
    if remaining_seconds()<MIN_START_SECONDS:
        dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps({"status":"cutoff","stage":stage,
            "audited_trials":len(history),"deadline_utc":DEADLINE.isoformat(),
            "auto_deploy":False,"test_evaluated":False},indent=2))
        print("NIGHTMARE_CUTOFF",stage,len(history),flush=True)
        return
    np.random.seed(SEED+stage)
    proposals={}
    for market in MARKETS:
        generator=bohb_generator(market,history)
        proposed=[]
        existing={_identity(r["params"]) for r in history if r["market"]==market}
        for budget in promotions_for(stage):
            params,origin,parent=_promotion(market,history,budget,proposed)
            proposed.append({"params":params,"budget":budget,"origin_trial":origin,
                             "promoted_from_trial":parent,"source":"successive_halving"})
        for _ in range(SLOTS-len(proposed)):
            for _attempt in range(80):
                config,info=generator.get_config(budget=float(BUDGETS[0]))
                params={k:(v.item() if hasattr(v,"item") else v) for k,v in config.items()}
                if _identity(params) not in existing:
                    break
            else:
                raise ValueError("BOHB could not find unseen configuration")
            existing.add(_identity(params))
            proposed.append({"params":params,"budget":BUDGETS[0],
                             "origin_trial":stage*SLOTS+len(proposed),
                             "promoted_from_trial":None,
                             "source":"hpbandster_kde" if info.get("model_based_pick") else "hpbandster_prior"})
        for slot,p in enumerate(proposed):
            p.update({"market":market,"stage":stage,"slot":slot,
                      "trial":stage*SLOTS+slot})
            p["config"]=settings_from_params(market,p["params"],p["budget"])
        proposals[market]=proposed
    result={"status":"ready","stage":stage,"candidates":proposals,
            "research_only":True,"test_evaluated":False,"auto_deploy":False,
            "parallel":{"ask":4,"bid":4},"budgets_epochs":list(BUDGETS),
            "sampler":"HpBandSter BOHB good/bad KDE density ratio by fidelity",
            "promotion":"Successive halving: 6->18->54->162->216 epochs, same native seed, re-fit from scratch",
            "metric":"historical original-validation absolute USD CVaR95",
            "deadline_utc":DEADLINE.isoformat(),
            "long_epochs":216,"adaptive_search":True}
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2,allow_nan=False))
    print("BOHB_PROPOSE stage",stage,
          {m:[(p["source"],p["budget"]) for p in proposals[m]] for m in MARKETS},
          flush=True)


def train(plan_path,market,slot,output,args):
    from scripts import overnight_bohb_neural as native
    plan=json.loads(Path(plan_path).read_text())
    if (plan["status"]!="ready" or market not in MARKETS or not 0<=slot<SLOTS):
        raise ValueError("Invalid BOHB plan or worker")
    chosen=plan["candidates"][market][slot]
    if (chosen["market"],chosen["slot"],chosen["stage"])!=(market,slot,plan["stage"]):
        raise ValueError("Wrong worker for BOHB proposal")
    history=read_history(args.history)
    if len(history)!=plan["stage"]*SLOTS*len(MARKETS):
        raise ValueError("BOHB worker missing previous audited batch history")
    if chosen["promoted_from_trial"] is not None:
        parent=[r for r in history if r["market"]==market
                and r["trial"]==chosen["promoted_from_trial"]]
        if (len(parent)!=1 or parent[0]["params"]!=chosen["params"]
            or parent[0]["origin_trial"]!=chosen["origin_trial"]
            or parent[0]["budget"]!={18:6,54:18,162:54,216:162}[chosen["budget"]]):
            raise ValueError("BOHB promotion must reproduce audited preceding rung")
    if chosen["config"]!=settings_from_params(market,chosen["params"],chosen["budget"]):
        raise ValueError("BOHB optimizer proposal was modified")
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    seconds=min(float(args.max_seconds),remaining_seconds()-180)
    if seconds<120:
        raise RuntimeError("Too close to 06:00 Berlin cutoff for safe scoring")
    # Native trial index is the original seed for a promoted candidate.
    try:
        if market=="ask":
            native.ask("ask_structure",chosen["origin_trial"],args.ask_features,args.ask_embeddings,
                       root,seconds,refinement=chosen["config"])
        else:
            native.bid("bid_structure",chosen["origin_trial"],args.bid_features,args.bid_baseline,
                       root,seconds,refinement=chosen["config"])
            audit_and_save_bid_blend(root,args.bid_baseline)
    except (ValueError,RuntimeError,FloatingPointError,MemoryError) as exc:
        # A genuinely unstable network is an expected BOHB observation, not
        # permission to disrupt every other independent overnight candidate.
        # Fail closed on source provenance, immutable cohorts, and CSV parity.
        message=str(exc)
        security=("cohort","sha256","original","VIN-disjoint","manifest",
                  "baseline target mismatch","parity","heldout","frozen")
        if any(term.lower() in message.lower() for term in security):
            raise
        summary={**chosen,"state":"FAILED","validation":None,
                 "provenance":None,"test_evaluated":False,"auto_deploy":False,
                 "checkpoint_reused":False,"same_origin_seed":True,
                 "error":type(exc).__name__+": "+message[:1500]}
        (root/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
        print("NIGHTMARE_TRIAL_FAILED",market,chosen["stage"],slot,summary["error"],flush=True)
        return
    native_receipt=json.loads((root/"result.json").read_text())
    if native_receipt["trial_configuration"]!=chosen["config"] or native_receipt["test_evaluated"] is not False:
        raise ValueError("BOHB native checkpoint settings differ from registered study")
    if native_receipt["validation"]["rows"]!=ROW_COUNTS[market]:
        raise ValueError("Wrong original validation row count")
    summary={**chosen,"state":"COMPLETE","validation":native_receipt["validation"],
             "provenance":native_receipt["provenance"],"test_evaluated":False,
             "auto_deploy":False,
             "checkpoint_reused":False,"same_origin_seed":True}
    (root/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    print("BOHB_TRIAL",market,"stage",chosen["stage"],"slot",slot,
          "budget",chosen["budget"],"CVaR95",
          native_receipt["validation"]["CVaR95_absolute_USD"],flush=True)


def collect(plan_path,history_path,artifacts,output):
    import pandas as pd
    from scripts.cvar95_historical import risk
    plan=json.loads(Path(plan_path).read_text())
    stage=plan["stage"]
    history=read_history(history_path)
    if len(history)!=stage*SLOTS*len(MARKETS):
        raise ValueError("Previous BOHB batch is missing")
    gathered={}
    for file in sorted(Path(artifacts).rglob("summary.json")):
        r=json.loads(file.read_text())
        key=(r["market"],r["slot"])
        if key in gathered or r["stage"]!=stage:
            raise ValueError("Wrong/duplicate BOHB trial receipt")
        registered=plan["candidates"][key[0]][key[1]]
        for name in ("market","stage","slot","trial","origin_trial",
                     "promoted_from_trial","params","config","budget","source"):
            if r[name]!=registered[name]:
                raise ValueError("BOHB trained an unregistered configuration: "+name)
        if r["state"]=="FAILED":
            if r["validation"] is not None or not r.get("error"):
                raise ValueError("Failed BOHB candidate lacks verified error receipt")
            if file.with_name("validation.csv.gz").exists():
                raise ValueError("Failed BOHB candidate must not claim validation")
            gathered[key]=r
            continue
        if r["state"]!="COMPLETE":
            raise ValueError("Unexpected overnight BOHB trial state")
        native_receipt=json.loads(file.with_name("result.json").read_text())
        if (native_receipt["trial_configuration"]!=r["config"]
            or native_receipt["test_evaluated"] is not False):
            raise ValueError("BOHB result configuration/test integrity failed")
        if not file.with_name("weights.pt").is_file():
            raise ValueError("Native PyTorch checkpoint missing")
        prediction=pd.read_csv(file.with_name("validation.csv.gz"),dtype={"row_id":str})
        if (len(prediction)!=ROW_COUNTS[key[0]] or prediction.row_id.duplicated().any()
            or prediction.row_id.isna().any()
            or not prediction.split.eq("validation").all()):
            raise ValueError("Original validation vehicle identities invalid")
        actual=risk(prediction.observed_price.to_numpy(float),
                    prediction.predicted_price.to_numpy(float))
        for metric in ("CVaR95_absolute_USD","RMSE_USD","MAE_USD"):
            if abs(actual[metric]-r["validation"][metric])>.05:
                raise ValueError("BOHB original validation parity failed: "+metric)
        gathered[key]=r
    keys={(m,i) for m in MARKETS for i in range(SLOTS)}
    if set(gathered)!=keys:
        raise ValueError("Missing BOHB training results: "+str(sorted(keys-set(gathered))))
    all_records=history+[gathered[(m,i)] for m in MARKETS for i in range(SLOTS)]
    # Only the registered highest 216-epoch fidelity qualifies as a challenger.
    full={}
    for m in MARKETS:
        chosen=[r for r in all_records if r["market"]==m and r["state"]=="COMPLETE" and r["budget"]==216]
        chosen.sort(key=lambda r:(r["validation"]["CVaR95_absolute_USD"],
                                  r["validation"]["RMSE_USD"],r["trial"]))
        full[m]=[{"trial":r["trial"],"origin_trial":r["origin_trial"],
                  "budget":r["budget"],
                  **{metric:r["validation"][metric] for metric in
                     ("CVaR95_absolute_USD","RMSE_USD","MAE_USD")}}
                 for r in chosen]
    comparison=incumbent_comparison(all_records,full)
    artifact={"status":"complete","stages_completed":stage+1,
              "records":all_records,
              "full_budget_leaderboard":full,
              "compared_to_deployed_verified":comparison,
              "budget_distribution":{m:{str(b):sum(r["market"]==m and r["budget"]==b
                                      for r in all_records) for b in BUDGETS} for m in MARKETS},
              "optimizer":"HpBandSter BOHB KDE + 14 losses + 216-epoch promotion",
              "failed_counts":{m:sum(r["market"]==m and r["state"]=="FAILED" for r in all_records) for m in MARKETS},
              "deadline_utc":DEADLINE.isoformat(),
              "promotion_checkpoint_restoration":False,
              "training_restart_same_seed":True,
              "not_for_deployment":True,"test_evaluated":False}
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(artifact,indent=2,allow_nan=False))
    print("BOHB_COLLECT stage",stage,"models",len(all_records),
          "full_216_epochs", {m:len(full[m]) for m in MARKETS},
          "comparison",{m:comparison[m]["challenger_beats_incumbent_on_CVaR95"] for m in MARKETS},
          flush=True)


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("propose")
    q.add_argument("--stage",type=int,required=True)
    q.add_argument("--history")
    q.add_argument("--output",required=True)
    t=sub.add_parser("train")
    t.add_argument("--plan",required=True)
    t.add_argument("--history")
    t.add_argument("--market",required=True,choices=MARKETS)
    t.add_argument("--slot",required=True,type=int)
    t.add_argument("--output",required=True)
    t.add_argument("--ask-features",default="training-cache/neural-ask/features.npz")
    t.add_argument("--ask-embeddings",default="training-cache/bge-256")
    t.add_argument("--bid-features",default="training-cache/neural-bid/features.npz")
    t.add_argument("--bid-baseline",default="benchmark-reference/bid/predictions.csv.gz")
    t.add_argument("--max-seconds",type=float,default=3000)
    c=sub.add_parser("collect")
    c.add_argument("--plan",required=True)
    c.add_argument("--history")
    c.add_argument("--artifacts",required=True)
    c.add_argument("--output",required=True)
    a=p.parse_args()
    if a.command=="propose":propose(a.stage,a.history,a.output)
    elif a.command=="train":train(a.plan,a.market,a.slot,a.output,a)
    else:collect(a.plan,a.history,a.artifacts,a.output)


if __name__=="__main__":
    main()
