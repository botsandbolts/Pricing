"""One-hour REAL BOHB on frozen Lemon75 DeepCross Ask with extended losses.

Uses the *real* HpBandSter BOHB good/bad KDE config generator, with explicit
successive-halving promotions at 4 -> 12 -> 36 training epochs. This is NOT
Optuna TPE mislabeled as BOHB. All research trials use only:
  99,749 ORIGINAL frozen-Lemon-retained TRAIN rows for gradients,
  11,083 retained TRAIN rows for BOHB selection and promotion.
The 23,751-row fixed original-reference Lemon-filtered VALIDATION is scored
only for the TRAIN-selected finalist. Original TEST is NEVER scored or used.
Models are native PyTorch and BGE vectors frozen/pinned. No deployment.

One independent GitHub runner searches the same DeepCross architecture
as the one-hour Ray Lemon winner. Research-only; no production deployment.
Fixed validation filtering uses observed prices: retrospective and optimistic.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from scripts.ask_interval_aft_likelihood import exact_ask_aft_nll
from scripts.ask25_joint_search import ResearchAskingNetwork
from scripts.cvar95_historical import risk
from scripts.neural_ask_train import predict
from scripts.raw_embedding_ask_train import frozen_source
from scripts.raw_embedding_ask_encode import id_hash
from scripts.ray_pbt_ask import architecture_params
from scripts.release_ray_lemon75_ask import reference_retained_mask
from scripts.overnight_bohb_losses import regression_loss

SEED=20261108
BUDGETS=(4,12,36)
KINDS=("aft","aft_huber","aft_cvar","aft_log_huber","aft_huber_cvar",
       "hybrid_aft_dollar","soft_cvar_dollar","asymmetric_dollar",
       "logcosh_log","hybrid_log_dollar")
EXPECTED={"fit":99749,"inner":11083,"lemon_validation":23751,"test":31682}
SOURCE_RUN=42485146978
SOURCE_RAY_FILTERED_CVAR95_USD=8959.618612064656
COHORT="frozen-lemon-oof-retained-75"

def _split_train_lemon(config):
    """Exactly the original pre-existing OOF Lemon mask; no validation leakage."""
    z,vec,source,encoder=frozen_source(
        config["features"],config["embeddings"],"bge-256")
    phase=z["phase"].astype(str)
    with np.load(config["features"],allow_pickle=False) as stored:
        if "lemon" not in stored.files:
            raise ValueError("Frozen original Lemon mask is missing")
        mask=stored["lemon"]
    if mask.dtype!=np.bool_ or mask.shape!=phase.shape:
        raise ValueError("Lemon selection geometry is not an original bool vector")
    train=np.flatnonzero(phase=="train")
    valid=np.flatnonzero(phase=="validation")
    test=np.flatnonzero(phase=="test")
    if (len(train),len(valid),len(test))!=(147777,31668,31682):
        raise ValueError("Original chronological splits changed")
    if not mask[valid].all() or not mask[test].all():
        raise ValueError("Original external holdout was unexpectedly pre-filtered")
    kept=train[mask[train]]
    if len(kept)!=110832 or len(train)-len(kept)!=36945:
        raise ValueError("Frozen Lemon retained 75% cohort is incorrect")
    early=11083
    fit,inner=kept[:-early],kept[-early:]
    assert len(fit)==99749 and len(inner)==11083
    values={}
    for name,indices in (("fit",fit),("inner",inner)):
        values[name]={
            "numeric":torch.from_numpy(z["num"][indices].astype(np.float32)),
            "category":torch.from_numpy(z["cat"][indices].astype(np.int64)),
            "vector":torch.from_numpy(vec[indices].astype(np.float32)),
            "log_target":torch.from_numpy(np.log(z["y"][indices]).astype(np.float32)),
            "truth":z["y"][indices].astype(np.float64),
        }
    meta={
        "cohort_sha256":source["cohort_sha256"],
        "holdout_row_sha256":source["holdout_row_sha256"],
        "embedding_sha256":encoder["embedding_file_sha256"],
        "cardinalities":source["cardinalities"],
        "fit_rows":len(fit),"train_only_inner_rows":len(inner),
        "original_train_rows":len(train),
        "lemon_retained_training_rows":len(kept),
        "lemon_excluded_training_rows":36945,
        "original_validation_rows":31668,
        "original_test_rows":31682,
        "original_test_scored":False,
    }
    return values,meta


def configspace(seed=SEED):
    import ConfigSpace as CS
    C=CS.CategoricalHyperparameter
    Fp=CS.UniformFloatHyperparameter
    cs=CS.ConfigurationSpace(seed=seed)
    cs.add_hyperparameters([
        C("loss",list(KINDS)),
        Fp("lr",lower=.00007,upper=.0018,log=True),
        Fp("wd",lower=.00015,upper=.08,log=True),
        C("batch",[384,512,768,1024]),
        C("dropout",[.04,.06,.10,.16,.24]),
        C("noise",[0.,.004,.008,.012]),
        C("clip",[.6,1.,2.]),
        Fp("sigma_floor",lower=.012,upper=.17,log=True),
        C("aft_weight",[.35,.50,.65,.80,1.]),
        C("tail_weight",[0.,.10,.25,.50,.80]),
        C("tail_fraction",[.05,.08,.10,.12,.15]),
        C("dollar_scale",[5000.,10000.,15000.,20000.]),
        C("log_huber_beta",[.15,.35,.60]),
        C("tail_strength",[0.,.1,.25,.5,.8,1.]),
        C("beta",[.15,.35,.60]),
        C("tau",[.35,.50,.65,.75]),
        C("tail_temperature",[.04,.1,.2]),
        C("optimizer",["adamw","adam","radam"]),
        C("schedule",["constant","cosine","warm_restart"]),
        C("beta1",[.85,.9,.95]),
    ])
    return cs


def normalise_config(config):
    result={}
    for k,v in dict(config).items():
        if isinstance(v,np.generic):v=v.item()
        result[str(k)]=v
    if result.get("loss") not in KINDS:
        raise ValueError("Unrecognized AFT mixture")
    if not .01<=float(result["sigma_floor"])<=.2:
        raise ValueError("AFT sigma floor outside audited interval")
    if not .3<=float(result["aft_weight"])<=1.:
        raise ValueError("AFT contribution must remain positive")
    if int(result["batch"]) not in (384,512,768,1024):
        raise ValueError("Unbounded minibatch size")
    if (result["optimizer"] not in ("adamw","adam","radam")
            or result["schedule"] not in ("constant","cosine","warm_restart")
            or not 0<=float(result["tail_strength"])<=1.
            or not 0<float(result["beta"])<=2
            or not 0<float(result["tau"])<1):
        raise ValueError("Extended BOHB settings outside safe bound")
    return result


def weighted_exact_aft_loss(raw, log_truth, hp):
    """Proper exact-event AFT plus robust supplements, never invented bounds.

    We drop target-dependent terms from the log-normal negative likelihood:
    this does not change its gradients and makes the mixing scale clearer.
    CVaR always uses the exact absolute USD metric on held-out data.
    """
    hp=normalise_config(hp)
    if raw.ndim!=2 or raw.shape[1]!=2 or raw.shape[0]!=log_truth.numel():
        raise ValueError("AFT Ask network must yield (batch,2) outputs")
    if hp["loss"] in ("hybrid_aft_dollar","soft_cvar_dollar",
                      "asymmetric_dollar","logcosh_log","hybrid_log_dollar"):
        settings={
            "loss":hp["loss"],"beta":float(hp["beta"]),
            "tau":float(hp["tau"]),
            "tail_fraction":float(hp["tail_fraction"]),
            "tail_strength":float(hp["tail_strength"]),
            "usd_scale":float(hp["dollar_scale"]),
            "tail_temperature":float(hp["tail_temperature"]),
            "sigma_floor":float(hp["sigma_floor"]),"focal_gamma":1.5,
        }
        return regression_loss(raw,log_truth,settings,"ask")
    target=log_truth.detach().exp()
    aft=exact_ask_aft_nll(raw,target,float(hp["sigma_floor"]))
    aft=aft-log_truth.mean()-.5*math.log(2*math.pi)
    mu=raw[:,0]
    delta=mu-log_truth
    dollar_residual=(
        mu.clamp(math.log(100.),math.log(1000000.)).exp()-target
    )/float(hp["dollar_scale"])
    huber=F.smooth_l1_loss(dollar_residual,torch.zeros_like(dollar_residual),beta=.5)
    log_huber=F.smooth_l1_loss(delta,torch.zeros_like(delta),
                               beta=float(hp["log_huber_beta"]))
    k=max(1,math.ceil(float(hp["tail_fraction"])*len(dollar_residual)))
    worst=dollar_residual.abs().topk(k).values.mean()
    mode=hp["loss"]
    a=float(hp["aft_weight"])
    t=float(hp["tail_weight"])
    if mode=="aft":
        value=aft
    elif mode=="aft_huber":
        value=a*aft+(1-a)*huber
    elif mode=="aft_cvar":
        value=aft+t*worst
    elif mode=="aft_log_huber":
        value=a*aft+(1-a)*log_huber
    else:
        value=a*aft+(1-a)*huber+t*worst
    if not bool(torch.isfinite(value).item()):
        raise ValueError("Nonfinite BOHB Ask AFT objective")
    return value


def bohb_generator(history, seed=SEED):
    """Actual HpBandSter KDE model, seeded with completed inner-only trials."""
    from hpbandster.optimizers.config_generators.bohb import BOHB
    space=configspace(seed)
    generator=BOHB(configspace=space,top_n_percent=40,
                   min_points_in_model=8,random_fraction=.30,
                   num_samples=64)
    for row in history:
        if row.get("status")!="complete":
            continue
        wrapper=SimpleNamespace(
            id=(int(row["trial"]),0,0),
            kwargs={"config":row["params"],"budget":float(row["epochs"])},
            result={"loss":float(row["inner_metrics"]["CVaR95_absolute_USD"])},
            exception=None,
        )
        generator.new_result(wrapper)
    return generator


def _unique_config(generator,history):
    used={json.dumps(x["params"],sort_keys=True) for x in history}
    for _ in range(100):
        config,_info=generator.get_config(budget=float(BUDGETS[0]))
        hp=normalise_config(config)
        if json.dumps(hp,sort_keys=True) not in used:
            return hp
    raise RuntimeError("BOHB sampler exhausted unique candidates")


def choose_next(history,generator):
    """Successive-halving promotion while allowing fresh KDE exploration.

    Each source config can move 3 -> 9 -> 27 only once; promotion consumes
    a separate trial and refits identically from the same seed.
    """
    counts={b:sum(x["epochs"]==b for x in history) for b in BUDGETS}
    # Reserve at least two initial candidates per promotion while allowing
    # the KDE to keep acquiring exploration evidence.
    for smaller,larger in ((12,36),(4,12)):
        eligible=[x for x in history if x.get("status")=="complete"
                  and x["epochs"]==smaller]
        promoted={json.dumps(x["params"],sort_keys=True) for x in history
                  if x["epochs"]==larger}
        if len(eligible)>=3 and counts[larger]<max(1,counts[smaller]//3):
            eligible=[x for x in eligible
                      if json.dumps(x["params"],sort_keys=True) not in promoted]
            if eligible:
                winning=min(eligible,
                    key=lambda z:z["inner_metrics"]["CVaR95_absolute_USD"])
                return dict(winning["params"]),larger,winning["trial"]
    return _unique_config(generator,history),BUDGETS[0],None


def _dataset(values):
    return TensorDataset(values["numeric"],values["category"],
                         values["vector"],values["log_target"])


def _trial(values, cards, arch, hp, budget, seed, deadline, max_single=310.):
    hp=normalise_config(hp)
    if budget not in BUDGETS:raise ValueError("Invalid BOHB resource fidelity")
    torch.set_num_threads(2)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    net=ResearchAskingNetwork(cards,384,arch)
    # Dropout is an allowed dynamic *probability*; architecture and shapes
    # are fixed within each runner.
    for m in net.modules():
        if isinstance(m,nn.Dropout):m.p=float(hp["dropout"])
    optimizer_class={
        "adamw":torch.optim.AdamW,"adam":torch.optim.Adam,
        "radam":torch.optim.RAdam}[hp["optimizer"]]
    optimizer=optimizer_class(net.parameters(),
        lr=float(hp["lr"]),weight_decay=float(hp["wd"]),
        betas=(float(hp["beta1"]),.999),foreach=False)
    scheduler=None
    if hp["schedule"]=="cosine":
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,T_max=BUDGETS[-1],eta_min=float(hp["lr"])*.05)
    elif hp["schedule"]=="warm_restart":
        scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,T_0=12,T_mult=2,eta_min=float(hp["lr"])*.05)
    ds=_dataset(values["fit"])
    b=int(hp["batch"]);size=len(ds)
    inner=_dataset(values["inner"])
    inner_ids=np.arange(len(inner))
    start=time.monotonic()
    best=float("inf");weights=None;best_epoch=0;metrics=None
    executed=0
    for epoch in range(1,budget+1):
        if time.monotonic()>=deadline-30:
            break
        net.train()
        order=torch.randperm(size,generator=torch.Generator().manual_seed(seed+epoch*17))
        for idx in order.split(b):
            num,cat,vec,truth=(v[idx] for v in ds.tensors)
            if float(hp["noise"])>0:
                num=num+torch.randn_like(num)*float(hp["noise"])
            pred=net(num,cat,vec)
            loss=weighted_exact_aft_loss(pred,truth,hp)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),float(hp["clip"]))
            optimizer.step()
            if time.monotonic()>min(deadline-20,start+max_single):
                break
        if scheduler is not None:
            scheduler.step()
        net.eval()
        pred_inner=predict(net,inner,inner_ids,1024)
        scored=risk(values["inner"]["truth"],pred_inner)
        executed=epoch
        c=float(scored["CVaR95_absolute_USD"])
        if c<best-.05:
            best=c;weights={k:v.detach().cpu().clone()
                             for k,v in net.state_dict().items()}
            best_epoch=epoch;metrics=scored
        print("BOHB_LEMON75_DEEPCROSS_EPOCH",json.dumps({
            "epoch":epoch,"budget":budget,"loss":hp["loss"],
            "architecture":arch["family"],"cvar95":round(c,3),
            "best_cvar95":round(best,3),
            "mae":round(float(scored["MAE_USD"]),3),
        }),flush=True)
        if time.monotonic()>min(deadline-20,start+max_single):
            break
    if weights is None or executed<2:
        raise RuntimeError("Incomplete AFT candidate; no valid weights")
    return {
        "status":"complete","epochs":budget,"epochs_trained":executed,
        "best_epoch":best_epoch,
        "inner_metrics":{k:float(metrics[k]) for k in
             ("CVaR95_absolute_USD","MAE_USD","RMSE_USD")},
        "weights":weights,"seconds":round(time.monotonic()-start,1),
    }


def _write_json(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False))


def run(args):
    from scripts.ray_pbt_ask import ARCHITECTURES
    if args.architecture!="deepcross":
        raise ValueError("BOHB Ask architecture invalid")
    if int(args.search_seconds)!=3600 or int(args.reserve_seconds)<100:
        raise ValueError("Search duration invalid or audit reserve too small")
    torch.set_num_threads(2)
    start=time.monotonic()
    deadline=start+int(args.search_seconds)
    stop_for_trials=deadline-int(args.reserve_seconds)
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    values,metadata=_split_train_lemon({"features":args.features,
                                  "embeddings":args.embeddings})
    arch=architecture_params(args.architecture)
    history=[];winner=None;seen=0
    summary={
        "status":"running","optimizer":"HpBandSter BOHB KDE + successive halving",
        "architecture":args.architecture,"search_seconds":args.search_seconds,
        "reserve_seconds":args.reserve_seconds,"trial_fidelities_epochs":list(BUDGETS),
        "selection":"frozen Lemon retained TRAIN-only internal 11,083 rows",
        "cleaned_external_validation_rows":23751,
        "cleaned_external_validation_scored_once":True,
        "filtered_validation_is_retrospective":True,
        "filtered_validation_uses_historical_prices":True,
        "search_space":"10 AFT and robust losses; real BOHB KDE; 18+ dimensions",
        "source_ray_study_run":SOURCE_RUN,
        "source_ray_retained_validation_CVaR95_USD":SOURCE_RAY_FILTERED_CVAR95_USD,
        "lemon_train_rows":110832,
        "training_cohort":COHORT,
        "original_validation_selection":False,
        "original_test_scored":False,"auto_deploy":False,
        "censored_rows":0,"target":"exact recorded seller asking prices",
        "original_split_counts":EXPECTED,
        "frozen_encoder_sha256":metadata["embedding_sha256"],
        "cohort_sha256":metadata["cohort_sha256"],
    }
    _write_json(root/"status.json",summary)
    print("BOHB_LEMON75_DEEPCROSS_STARTED",json.dumps(summary),flush=True)
    while time.monotonic()<stop_for_trials:
        gen=bohb_generator(history)
        hp,budget,parent=choose_next(history,gen)
        trial_id=len(history)
        record={"trial":trial_id,"params":hp,"epochs":budget,
                "promoted_from":parent}
        remaining=stop_for_trials-time.monotonic()
        if remaining<45:
            break
        try:
            result=_trial(values,metadata["cardinalities"],arch,hp,budget,
                SEED+trial_id if parent is None else SEED+parent,
                stop_for_trials,max_single=min(305.,max(20.,remaining-10)))
            record.update({k:v for k,v in result.items() if k!="weights"})
            if winner is None or (
                record["inner_metrics"]["CVaR95_absolute_USD"]<
                winner["record"]["inner_metrics"]["CVaR95_absolute_USD"]):
                winner={"record":record,"weights":result["weights"]}
                torch.save({"state_dict":winner["weights"],"params":hp,
                            "architecture":arch,"trial":trial_id,
                            "cohort_sha256":metadata["cohort_sha256"],
                            "embedding_sha256":metadata["embedding_sha256"],
                            "original_test_scored":False,"auto_deploy":False},
                           root/"weights.pt")
        except Exception as exc:
            record.update(status="failed",error=f"{type(exc).__name__}: {exc}")
            print("BOHB_LEMON75_DEEPCROSS_FAILED_TRIAL",json.dumps({
                "trial":trial_id,"error":record["error"]}),flush=True)
        history.append(record)
        _write_json(root/"history.json",{
            **summary,"status":"in_progress","trials":history,
            "best_inner_CVaR95_USD":None if winner is None else
             winner["record"]["inner_metrics"]["CVaR95_absolute_USD"],
        })
        print("BOHB_LEMON75_DEEPCROSS_TRIAL",json.dumps(record),flush=True)
    if winner is None:
        raise RuntimeError("No valid AFT BOHB trial before deadline")
    # Only the internal-selected winner is scored on frozen reference-cleaned
    # 75% of original validation. No original test row is predicted.
    z,vec,source,encoder=frozen_source(args.features,args.embeddings,"bge-256")
    if (source["cohort_sha256"]!=metadata["cohort_sha256"]
        or encoder["embedding_file_sha256"]!=metadata["embedding_sha256"]):
        raise RuntimeError("Validation cohort or BGE artifact changed")
    valid_all=np.flatnonzero(z["phase"].astype(str)=="validation")
    if len(valid_all)!=31668:
        raise RuntimeError("Original Ask validation changed")
    with np.load(args.features,allow_pickle=False) as stored:
        z["lemon"]=stored["lemon"]
    reference=pd.read_csv(args.reference,dtype={"row_id":str},
        usecols=["row_id","split","observed_price","residual_log"])
    filter75=reference_retained_mask(z,reference)
    val=valid_all[filter75]
    if len(val)!=23751:
        raise RuntimeError("Clean Lemon original validation did not retain 23,751")
    net=ResearchAskingNetwork(source["cardinalities"],384,arch)
    net.load_state_dict(winner["weights"],strict=True)
    net.eval()
    valset=TensorDataset(
        torch.from_numpy(z["num"][val].astype(np.float32)),
        torch.from_numpy(z["cat"][val].astype(np.int64)),
        torch.from_numpy(vec[val].astype(np.float32)),
        torch.from_numpy(np.log(z["y"][val]).astype(np.float32)))
    pred=predict(net,valset,np.arange(len(val)),1024)
    measured=risk(z["y"][val],pred)
    pd.DataFrame({"row_id":z["ids"][val].astype(str),"split":"lemon_validation",
                  "observed_price":z["y"][val].astype(float),
                  "predicted_price":pred}).to_csv(root/"validation.csv.gz",index=False)
    final={**summary,
        "status":"completed","elapsed_seconds":round(time.monotonic()-start,1),
        "trials_completed":sum(x["status"]=="complete" for x in history),
        "trials_failed":sum(x["status"]=="failed" for x in history),
        "winning_trial":winner["record"],
        "lemon_validation":{k:float(measured[k]) for k in
             ("CVaR95_absolute_USD","MAE_USD","RMSE_USD")},
        "lemon_validation_rows":int(len(val)),
        "original_validation_rows":31668,
        "excluded_original_validation_rows":7917,
        "retained_validation_selection_rule":"pre-Lemon OOF residual absolute log error; historical outcome-dependent",
        "lemon_validation_not_unbiased_future_accuracy":True,
        "original_validation_row_ids_sha256":id_hash(z["ids"][val].astype(str)),
        "original_test_rows_not_evaluated":31682,
        "research_only":True,"production_unchanged":True,
        "ray_lemon75_reference_CVaR95_USD":SOURCE_RAY_FILTERED_CVAR95_USD,
        "test_confirmation_required":True,
    }
    _write_json(root/"result.json",final)
    _write_json(root/"history.json",{
        **summary,"status":"completed","trials":history})
    print("BOHB_LEMON75_DEEPCROSS_FINAL",json.dumps({
        "architecture":args.architecture,"trials":final["trials_completed"],
        "lemon_validation":final["lemon_validation"],
        "best_internal":winner["record"]["inner_metrics"],
        "test_scored":False,"auto_deploy":False,
    }),flush=True)
    return final


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--architecture",choices=["deepcross"],required=True)
    p.add_argument("--reference",required=True,
                   help="Original immutable pre-Lemon Ask reference residuals")
    p.add_argument("--features",default="training-cache/neural-ask/features.npz")
    p.add_argument("--embeddings",default="training-cache/bge-256")
    p.add_argument("--output",required=True)
    p.add_argument("--search-seconds",type=int,default=3600)
    p.add_argument("--reserve-seconds",type=int,default=160)
    run(p.parse_args())


if __name__=="__main__":
    main()
