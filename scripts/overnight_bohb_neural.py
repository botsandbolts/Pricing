"""Overnight BOHB experimental neural engine, copied to isolate production/trials.

Search touches original TRAIN and original VALIDATION only. It never opens TEST.
No trained weights or deployment files are modified.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from scripts.cvar95_historical import risk
from scripts.overnight_bohb_losses import regression_loss, make_optimizer, make_scheduler

GROUPS = ("ask_hparams", "bid_hparams", "ask_structure", "bid_structure")
N_TRIALS = 50
SEED = 20261106


def spec(group, trial):
    if group not in GROUPS or not isinstance(trial, int) or not 0 <= trial < N_TRIALS:
        raise ValueError("Expected one of four groups and trial 0..49")
    seed = SEED + GROUPS.index(group) * 1000 + trial
    rng = np.random.default_rng(seed)
    if group == "ask_hparams":
        if trial == 0:
            return dict(lr=.001, wd=.025, batch=768, noise=.008, clip=1.,
                        patience=7, epochs=35, optimizer="adamw", schedule="cosine")
        return dict(lr=float(np.exp(rng.uniform(np.log(.00025), np.log(.0028)))),
            wd=float(np.exp(rng.uniform(np.log(.0004),np.log(.13)))),
            batch=int(rng.choice([384,512,768,1024,1536])),
            noise=float(rng.choice([0.,.004,.008,.014,.024])),
            clip=float(rng.choice([.35,.7,1.,1.5,2.5])),
            patience=int(rng.choice([4,6,8,10])),
            epochs=int(rng.choice([22,28,35,42])),
            optimizer=str(rng.choice(["adamw","adamw","adam"])),
            schedule=str(rng.choice(["cosine","cosine","constant"])))
    if group == "bid_hparams":
        if trial == 0:
            return dict(lr=.0025,wd=.005,batch=2048,beta=.25,
                        tail_lambda=0.,clip=1.,patience=5,epochs=25)
        return dict(lr=float(np.exp(rng.uniform(np.log(.0005),np.log(.007)))),
            wd=float(np.exp(rng.uniform(np.log(.0001),np.log(.08)))),
            batch=int(rng.choice([512,1024,1536,2048,4096])),
            beta=float(rng.choice([.1,.17,.25,.4,.65])),
            tail_lambda=float(rng.choice([0.,0.,.05,.1,.2,.35])),
            clip=float(rng.choice([.5,1.,2.])),
            patience=int(rng.choice([3,5,7,9])),
            epochs=int(rng.choice([18,22,25,30,36])))
    if group == "ask_structure":
        if trial == 0:
            return dict(projection=96, project_layers=1, proj_dropout=.08,
                cross_layers=3, rank=32, experts=1,width=256,blocks=2,
                dropout=.12,activation="gelu")
        return dict(projection=int(rng.choice([48,64,96,128,192])),
            project_layers=int(rng.choice([1,2,3])),
            proj_dropout=float(rng.choice([0.,.05,.08,.15,.23])),
            cross_layers=int(rng.choice([1,2,3,4,5])),
            rank=int(rng.choice([16,24,32,48,64])),
            experts=int(rng.choice([1,1,2,3])),
            width=int(rng.choice([128,192,256,384])),
            blocks=int(rng.choice([1,2,3,4])),
            dropout=float(rng.choice([.08,.12,.2,.3])),
            activation=str(rng.choice(["gelu","silu"])))
    if trial == 0:
        return dict(rank=16,width1=128,width2=64,depth=2,dropout=.16,
                    activation="gelu",pairgate=False)
    return dict(rank=int(rng.choice([16,32])),
        width1=int(rng.choice([64,96,128,192,256])),
        width2=int(rng.choice([32,48,64,96,128])),
        depth=int(rng.choice([1,2,3,4])),
        dropout=float(rng.choice([.05,.12,.16,.24,.32])),
        activation=str(rng.choice(["gelu","silu"])),
        pairgate=bool(rng.choice([False,True])))


def defaults(group):
    if group.startswith("ask"):
        return dict(lr=.001,wd=.025,batch=768,noise=.008,clip=1.,
                    patience=7,epochs=35,optimizer="adamw",schedule="cosine")
    return dict(lr=.0025,wd=.005,batch=2048,beta=.25,
                tail_lambda=0.,clip=1.,patience=5,epochs=25)


def build_ask(cards,embedding_dim,group,config):
    from scripts.raw_embedding_ask_train import EmbeddingOnlyDeepCross
    from scripts.deepcross_aft_tune import TunedDeepCross, config as original_config
    model = EmbeddingOnlyDeepCross(cards,7,embedding_dim)
    if group == "ask_structure":
        size = config["projection"]
        activation = nn.GELU if config["activation"]=="gelu" else nn.SiLU
        modules = [nn.LayerNorm(embedding_dim),nn.Linear(embedding_dim,size),
                   activation(),nn.LayerNorm(size)]
        for _ in range(config["project_layers"]-1):
            modules.extend([nn.Linear(size,size),activation(),nn.LayerNorm(size)])
        modules.append(nn.Dropout(config["proj_dropout"]))
        model.projector = nn.Sequential(*modules)
        settings = original_config("dropout12")
        settings.update({key:config[key] for key in
            ("cross_layers","rank","experts","width","blocks","dropout","activation",
              "cross_scale","embed_dropout","output_dropout")})
        model.price_network = TunedDeepCross(cards,7,size,settings)
    return model


class StructuredBid(nn.Module):
    def __init__(self,cards,initial_log_price,settings):
        super().__init__()
        from scripts.bid_fm_scale_train import ScaledAuctionFM
        from scripts.bid_fm_kernel_train import FIELD_COUNT
        rank=settings["rank"]
        self.core=ScaledAuctionFM(cards,"deepfm",rank,initial_log_price)
        n=FIELD_COUNT*rank
        activation=nn.GELU if settings["activation"]=="gelu" else nn.SiLU
        dims=[n]+[settings["width1"]]+[settings["width2"]]*max(settings["depth"]-1,0)
        layers=[nn.LayerNorm(n)]
        for left,right in zip(dims[:-1],dims[1:]):
            layers.extend([nn.Linear(left,right),activation(),nn.Dropout(settings["dropout"])])
        layers.append(nn.Linear(dims[-1],1))
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.core.deep=nn.Sequential(*layers)
        self.gates=nn.Parameter(torch.zeros(FIELD_COUNT,FIELD_COUNT)) if settings["pairgate"] else None

    def forward(self,num,cats):
        out=self.core(num,cats)
        if self.gates is not None:
            v=self.core.latent(num,cats)
            dots=torch.matmul(v,v.transpose(1,2))
            out=out+(dots*torch.triu(torch.tanh(self.gates),diagonal=1)).sum((1,2))
        return out


def _save(output,group,trial,settings,fit_report,prices,ids,y,net,metadata):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"row_id":ids.astype(str),"split":"validation",
                  "observed_price":y.astype(float),"predicted_price":prices}).to_csv(
                  output/"validation.csv.gz",index=False)
    record={"status":"completed","group":group,"trial":trial,
            "seed":SEED+GROUPS.index(group.replace("_refined","_structure"))*1000+trial,
            "trial_configuration":settings,
            "selection_metric":"validation absolute USD CVaR95",
            "selection_split":"original validation only",
            "test_evaluated":False,
            "original_model_unchanged":True,
            "validation":fit_report,"provenance":metadata}
    torch.save({"state_dict":net.state_dict(),"search_group":group,
                "trial":trial,"trial_configuration":settings,
                "source_provenance":metadata},output/"weights.pt")
    (output/"result.json").write_text(json.dumps(record,indent=2,allow_nan=False))
    print("CVaR95_TRIAL_COMPLETE",group,trial,json.dumps(fit_report),flush=True)


def ask(group,trial,original,embeddings,output,max_seconds,refinement=None):
    from scripts.raw_embedding_ask_train import frozen_source
    from scripts.neural_ask_train import seed_everything,predict,objective
    # BGE vectors were frozen and SHA256-verified before tuning; no encoder training.
    z,e,provenance,encoder_meta=frozen_source(original,embeddings,"bge-256")
    seed_everything(SEED+GROUPS.index(group)*1000+trial)
    phase=z["phase"].astype(str)
    tr=np.flatnonzero(phase=="train")
    fit,inner=tr[:-14777],tr[-14777:]
    val=np.flatnonzero(phase=="validation")
    if (len(fit),len(inner),len(val))!=(133000,14777,31668):
        raise ValueError("Original chronological asking holdouts changed")
    cfg=refinement["structure"] if refinement is not None else spec(group,trial)
    hp=refinement["hyperparameters"] if refinement is not None else (cfg if group=="ask_hparams" else defaults(group))
    ds=TensorDataset(torch.from_numpy(z["num"].astype(np.float32)),
        torch.from_numpy(z["cat"].astype(np.int64)),torch.from_numpy(e),
        torch.from_numpy(np.log(z["y"]).astype(np.float32)))
    model=build_ask(provenance["cardinalities"],e.shape[1],group,cfg)
    optimizer=make_optimizer(model,hp)
    scheduler=make_scheduler(optimizer,hp)
    loader=DataLoader(Subset(ds,fit.tolist()),batch_size=hp["batch"],
        shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(SEED+trial))
    best=math.inf;state=None;epoch_best=0;stale=0;start=time.monotonic()
    for epoch in range(1,hp["epochs"]+1):
        model.train()
        for num,cat,text,target in loader:
            optimizer.zero_grad(set_to_none=True)
            if hp["noise"]:num=num+torch.randn_like(num)*hp["noise"]
            raw=model(num,cat,text)
            loss=regression_loss(raw,target,hp,market="ask")
            if not torch.isfinite(loss):raise ValueError("Nonfinite AFT loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),hp["clip"])
            optimizer.step()
            if time.monotonic()-start>max_seconds:break
        if scheduler:scheduler.step()
        p=predict(model,ds,inner,hp["batch"])
        score=risk(z["y"][inner],p)["CVaR95_absolute_USD"]
        print("ASK_EPOCH",group,trial,epoch,round(score,2),flush=True)
        if score < best-.1:
            best=score;state=copy.deepcopy(model.state_dict())
            epoch_best=epoch;stale=0
        else:stale+=1
        if stale>=hp["patience"] or time.monotonic()-start>max_seconds:break
    if state is None:raise ValueError("No valid asking training checkpoint")
    model.load_state_dict(state)
    pv=predict(model,ds,val,hp["batch"])
    scores=risk(z["y"][val],pv)
    scores.update(best_epoch=epoch_best,trained_epochs=epoch,inner_stopping_CVaR95_USD=best,
                  elapsed_seconds=round(time.monotonic()-start,1))
    meta={"dataset_sha256":provenance["cohort_sha256"],
          "holdout_row_sha256":provenance["holdout_row_sha256"],
          "frozen_bge_sha256":encoder_meta["embedding_file_sha256"],
          "encoder":encoder_meta["model_id"],
          "training_rows":133000,"internal_stop_rows":14777,
          "validation_rows":31668,"no_description_price_head":True}
    _save(output,"ask_refined" if refinement is not None else group,trial,
          {"structure":cfg,"hyperparameters":hp} if refinement is not None else cfg,
          scores,pv,z["ids"][val],z["y"][val],model,meta)


def _matched_auction_baseline(path,ids,y):
    p=pd.read_csv(path,dtype={"row_id":str})
    req={"row_id","split","observed_price","predicted_price"}
    if not req.issubset(p.columns):raise ValueError("Original Lemon baseline columns missing")
    p=p.loc[p["split"].eq("validation")]
    if len(p)!=82380 or p.row_id.duplicated().any() or set(p.row_id)!=set(ids.astype(str)):
        raise ValueError("Unmatched original VIN-disjoint wholesale validation baseline")
    aligned=p.set_index("row_id").reindex(ids.astype(str))
    if not np.allclose(aligned.observed_price,y,rtol=1e-6,atol=.05):
        raise ValueError("Historical XGBoost baseline target mismatch")
    return aligned.predicted_price.to_numpy(float)


def bid(group,trial,features,baseline,output,max_seconds,refinement=None):
    from scripts.bid_fm_kernel_train import source,partition,y_price
    from scripts.bid_fm_scale_train import ScaledAuctionFM,prediction
    z,manifest=source(features)
    fit,inner,val,_test=partition(z,"full")
    if len(val)!=82380:raise ValueError("Wrong wholesale validation count")
    np.random.seed(SEED+GROUPS.index(group)*1000+trial)
    torch.manual_seed(SEED+GROUPS.index(group)*1000+trial)
    torch.set_num_threads(2)
    cfg=refinement["structure"] if refinement is not None else spec(group,trial)
    hp=refinement["hyperparameters"] if refinement is not None else (cfg if group=="bid_hparams" else defaults(group))
    numeric=z["num"].astype(np.float32);cat=z["cat"].astype(np.int64)
    y=z["y"].astype(np.float64)
    init=float(np.median(np.log(y[fit])))
    model=(ScaledAuctionFM(manifest["cardinalities"],"deepfm",16,init)
           if group=="bid_hparams" else StructuredBid(manifest["cardinalities"],init,cfg))
    optimizer=make_optimizer(model,hp)
    scheduler=make_scheduler(optimizer,hp)
    ds=TensorDataset(torch.from_numpy(numeric[fit]),torch.from_numpy(cat[fit]),
                     torch.from_numpy(np.log(y[fit]).astype(np.float32)))
    loader=DataLoader(ds,batch_size=hp["batch"],shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(SEED+trial))
    start=time.monotonic();best=math.inf;state=None;epoch_best=0;stale=0
    for epoch in range(1,hp["epochs"]+1):
        model.train()
        for num,categorical,target in loader:
            optimizer.zero_grad(set_to_none=True)
            raw=model(num,categorical)
            loss=regression_loss(raw,target,hp,market="bid")
            if not torch.isfinite(loss):raise ValueError("Nonfinite wholesale loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),hp["clip"])
            optimizer.step()
            if time.monotonic()-start>max_seconds:break
        if scheduler:scheduler.step()
        p=prediction(model,numeric,cat,inner,batch=8192) if group=="bid_hparams" else _bid_predict(model,numeric,cat,inner)
        score=risk(y[inner],p)["CVaR95_absolute_USD"]
        print("BID_EPOCH",group,trial,epoch,round(score,2),flush=True)
        if score<best-.1:
            best=score;state=copy.deepcopy(model.state_dict());epoch_best=epoch;stale=0
        else:stale+=1
        if stale>=hp["patience"] or time.monotonic()-start>max_seconds:break
    if state is None:raise ValueError("No wholesale training checkpoint")
    model.load_state_dict(state)
    pv=(prediction(model,numeric,cat,val) if group=="bid_hparams"
        else _bid_predict(model,numeric,cat,val))
    baseline_p=_matched_auction_baseline(baseline,z["ids"][val],y[val])
    # A single pre-registered blend is scored on the original VIN cohort.
    # BOHB registers the blend coefficient in TRAIN-ONLY optimization; do not
    # sweep the ORIGINAL VALIDATION set for an extra post-hoc winning blend.
    weights=np.array([float(hp["blend_alpha"])])
    if not .5<=weights[0]<=1.:raise ValueError("Bad pre-registered blend coefficient")
    choices=[(risk(y[val],alpha*pv+(1-alpha)*baseline_p),float(alpha))
             for alpha in weights]
    scores,alpha=min(choices,key=lambda pair:(pair[0]["CVaR95_absolute_USD"],pair[1]))
    scores.update(best_epoch=epoch_best,trained_epochs=epoch,neural_weight=alpha,
        lemon_xgb_weight=round(1-alpha,2),neural_only_CVaR95_USD=risk(y[val],pv)["CVaR95_absolute_USD"],
        inner_stopping_CVaR95_USD=best,elapsed_seconds=round(time.monotonic()-start,1))
    meta={"dataset_sha256":manifest["cohort_sha256"],
          "holdout_row_sha256":manifest["holdout_row_sha256"],
          "training_rows":384198,"validation_rows":82380,
          "baseline":"original matched Lemon XGBoost, unchanged",
          "selection":"validation blend CVaR95 only","test_used":False}
    _save(output,"bid_refined" if refinement is not None else group,trial,
          {"structure":cfg,"hyperparameters":hp} if refinement is not None else cfg,
          scores,pv,z["ids"][val],y[val],model,meta)


@torch.no_grad()
def _bid_predict(model,num,cat,rows,batch=8192):
    model.eval();out=[]
    for begin in range(0,len(rows),batch):
        idx=rows[begin:begin+batch]
        raw=model(torch.from_numpy(num[idx]),torch.from_numpy(cat[idx]))
        out.append(np.exp(np.clip(raw.detach().cpu().numpy(),np.log(100.),np.log(1000000.))))
    return np.concatenate(out)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--group",required=True,choices=GROUPS)
    p.add_argument("--trial",type=int,required=True)
    p.add_argument("--output",default="training-results/cvar200")
    p.add_argument("--ask-features",default="training-cache/neural-ask/features.npz")
    p.add_argument("--ask-embeddings",default="training-cache/bge-256")
    p.add_argument("--bid-features",default="training-cache/neural-bid/features.npz")
    p.add_argument("--bid-baseline",default="benchmark-reference/bid/predictions.csv.gz")
    p.add_argument("--max-seconds",type=float,default=3000)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    cfg=spec(a.group,a.trial)
    if a.dry_run:
        print(json.dumps({"group":a.group,"trial":a.trial,"config":cfg}))
        return
    if a.group.startswith("ask"):
        ask(a.group,a.trial,a.ask_features,a.ask_embeddings,a.output,a.max_seconds)
    else:
        bid(a.group,a.trial,a.bid_features,a.bid_baseline,a.output,a.max_seconds)


if __name__=="__main__":
    main()
