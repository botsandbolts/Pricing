"""BOHB experiment with real HpBandSter KDE proposals and Hyperband budgets.

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
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.hyperband_cvar95 import (
    audit_and_save_bid_blend, incumbent_comparison,
)

MARKETS=("ask","bid")
SLOTS=4
STAGES=6
BUDGETS=(4,12,36)
SEED=20261107
ROW_COUNTS={"ask":31668,"bid":82380}
PROMOTIONS={0:(),1:(12,),2:(36,12),3:(12,),4:(36,12),5:(36,12)}


def search_space(market):
    """Five dimensions so KDE can learn from modest GitHub Actions budgets."""
    import ConfigSpace as CS
    if market not in MARKETS:
        raise ValueError("Unknown market")
    space=CS.ConfigurationSpace(seed=SEED+(0 if market=="ask" else 1729))
    space.add_hyperparameters([
        CS.UniformFloatHyperparameter("lr",lower=.00025 if market=="ask" else .0005,
                                      upper=.0028 if market=="ask" else .007,log=True),
        CS.UniformFloatHyperparameter("wd",lower=.0001,upper=.13 if market=="ask" else .08,log=True),
        CS.CategoricalHyperparameter("batch",choices=[512,768,1024] if market=="ask" else [512,1024,2048]),
        CS.CategoricalHyperparameter("projection" if market=="ask" else "beta",
                                     choices=[64,96,128] if market=="ask" else [.1,.25,.4]),
        CS.CategoricalHyperparameter("dropout" if market=="ask" else "tail_lambda",
                                     choices=[.08,.12,.2] if market=="ask" else [0.,.05,.2]),
    ])
    return space


def settings_from_params(market,params,budget):
    if budget not in BUDGETS:
        raise ValueError("Unregistered BOHB training budget")
    if market=="ask":
        hp=dict(lr=.001,wd=.025,batch=768,noise=.008,clip=1.,
                patience=7,epochs=35,optimizer="adamw",schedule="constant")
        arch=dict(projection=96,project_layers=1,proj_dropout=.08,
                  cross_layers=3,rank=32,experts=1,width=256,blocks=2,
                  dropout=.12,activation="gelu")
    else:
        hp=dict(lr=.0025,wd=.005,batch=2048,beta=.25,
                tail_lambda=0.,clip=1.,patience=5,epochs=25)
        arch=dict(rank=16,width1=128,width2=64,depth=2,dropout=.16,
                  activation="gelu",pairgate=False)
    hp.update({k:params[k] for k in ("lr","wd","batch")})
    hp["epochs"]=budget
    if market=="ask":
        hp["schedule"]="constant"  # promotions reproduce same LR at earlier epochs
        arch["projection"]=int(params["projection"])
        arch["dropout"]=float(params["dropout"])
    else:
        hp["beta"]=float(params["beta"])
        hp["tail_lambda"]=float(params["tail_lambda"])
    return {"hyperparameters":hp,"structure":arch}


def read_history(path):
    if not path:
        return []
    doc=json.loads(Path(path).read_text())
    rows=doc["records"]
    if len(rows)!=len({(r["market"],r["stage"],r["slot"]) for r in rows}):
        raise ValueError("Repeated BOHB trial IDs")
    for r in rows:
        if (r["market"] not in MARKETS or r["state"]!="COMPLETE"
            or r["budget"] not in BUDGETS or not isinstance(r["trial"],int)
            or r.get("test_evaluated") is not False):
            raise ValueError("Incomplete, unregistered or test-contaminated BOHB evidence")
        if not math.isfinite(float(r["validation"]["CVaR95_absolute_USD"])):
            raise ValueError("Nonfinite BOHB CVaR95 metric")
    for market in MARKETS:
        hashes={r["provenance"]["holdout_row_sha256"]["validation"]
                for r in rows if r["market"]==market}
        if len(hashes)>1:
            raise ValueError("BOHB validation vehicle cohorts changed")
    return rows


def bohb_generator(market,history):
    from hpbandster.optimizers.config_generators.bohb import BOHB
    space=search_space(market)
    generator=BOHB(configspace=space,top_n_percent=33,
                   min_points_in_model=6,
                   random_fraction=.25,num_samples=64)
    # Load only completed, independently audited per-budget objective values.
    for row in sorted((r for r in history if r["market"]==market),
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
    prev={12:4,36:12}[target]
    eligible=[r for r in history if r["market"]==market and r["budget"]==prev]
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
        raise ValueError("BOHB batch must be between 0 and 5")
    history=read_history(history_path)
    if len(history)!=stage*SLOTS*len(MARKETS):
        raise ValueError("All earlier BOHB batch audits must pass before proposing")
    np.random.seed(SEED+stage)
    proposals={}
    for market in MARKETS:
        generator=bohb_generator(market,history)
        proposed=[]
        existing={_identity(r["params"]) for r in history if r["market"]==market}
        for budget in PROMOTIONS[stage]:
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
            proposed.append({"params":params,"budget":4,
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
            "promotion":"Successive halving: 4->12->36 epochs, same native initialization seed, re-fit from scratch",
            "metric":"historical original-validation absolute USD CVaR95"}
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2,allow_nan=False))
    print("BOHB_PROPOSE stage",stage,
          {m:[(p["source"],p["budget"]) for p in proposals[m]] for m in MARKETS},
          flush=True)


def train(plan_path,market,slot,output,args):
    from scripts import neural_200_sweep as native
    plan=json.loads(Path(plan_path).read_text())
    if (plan["status"]!="ready" or market not in MARKETS or not 0<=slot<SLOTS):
        raise ValueError("Invalid BOHB plan or worker")
    chosen=plan["candidates"][market][slot]
    if (chosen["market"],chosen["slot"],chosen["stage"])!=(market,slot,plan["stage"]):
        raise ValueError("Wrong worker for BOHB proposal")
    if chosen["config"]!=settings_from_params(market,chosen["params"],chosen["budget"]):
        raise ValueError("BOHB optimizer proposal was modified")
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    # Native trial index is the original seed for a promoted candidate.
    if market=="ask":
        native.ask("ask_structure",chosen["origin_trial"],args.ask_features,args.ask_embeddings,
                   root,args.max_seconds,refinement=chosen["config"])
    else:
        native.bid("bid_structure",chosen["origin_trial"],args.bid_features,args.bid_baseline,
                   root,args.max_seconds,refinement=chosen["config"])
        audit_and_save_bid_blend(root,args.bid_baseline)
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
    # Only FULL fidelity (36 epochs maximum) qualifies as a challenger.
    full={}
    for m in MARKETS:
        chosen=[r for r in all_records if r["market"]==m and r["budget"]==36]
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
              "optimizer":"HpBandSter BOHB KDE + successive halving promotions",
              "promotion_checkpoint_restoration":False,
              "training_restart_same_seed":True,
              "not_for_deployment":True,"test_evaluated":False}
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(artifact,indent=2,allow_nan=False))
    print("BOHB_COLLECT stage",stage,"models",len(all_records),
          "full_36_epochs", {m:len(full[m]) for m in MARKETS},
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
