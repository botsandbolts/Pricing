"""25 independent joint neural structure and optimizer wholesale Bid trials.

The original deployed research model is 75% full-data DeepFM16 + 25%
Lemon XGBoost. We retrain novel architectures on the original FULL historical
2014-15 VIN-disjoint auction cohort; the original matched Lemon XGBoost model
is frozen and reused only for identical validation-row blends.

CVaR95 (mean top-5% absolute dollar error) is primary. Internal early stopping
on TRAIN ONLY, original validation once per candidate; original TEST never
evaluated during search. No production release or weights conversion.
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
from torch.utils.data import DataLoader,TensorDataset

from scripts.bid_fm_scale_train import ScaledAuctionFM
from scripts.bid_fm_kernel_train import (
    AuctionFactorization, FIELD_COUNT, partition, source, y_price,
)
from scripts.cvar95_historical import risk
from scripts.neural_bid_prepare import sha_ids

N_TRIALS=25
SEED=20261107


def trial_config(i):
    if not isinstance(i,int) or not 0<=i<N_TRIALS:
        raise ValueError("Expected Bid joint trial number 0..24")
    if i==0:
        return dict(family="deepfm",rank=16,width1=128,width2=64,
            depth=2,dropout=.16,activation="gelu",
            learning_rate=.0025,weight_decay=.005,batch_size=2048,
            huber_beta=.25,tail_weight=0.,clip=1.,patience=5,epochs=25,
            numeric_noise=0.,neural_weight=.75,warmup=0,schedule="constant")
    rng=np.random.default_rng(SEED+29*i)
    family=("deepfm_gate" if i in (4,8,12,17,22) else
            "deepfm_hofm3" if i in (6,13,19,24) else "deepfm")
    return dict(
        family=family,
        rank=int(rng.choice([8,12,16,24,32,48])),
        width1=int(rng.choice([64,96,128,192,256])),
        width2=int(rng.choice([32,48,64,96,128])),
        depth=int(rng.choice([1,2,3,4])),
        dropout=float(rng.choice([.04,.1,.16,.24,.3])),
        activation=str(rng.choice(["gelu","silu"])),
        learning_rate=float(np.exp(rng.uniform(np.log(.00045),np.log(.0055)))),
        weight_decay=float(np.exp(rng.uniform(np.log(.0005),np.log(.06)))),
        batch_size=int(rng.choice([1024,1536,2048,3072])),
        huber_beta=float(rng.choice([.12,.18,.25,.4,.6])),
        tail_weight=float(rng.choice([0.,0.,.02,.04,.07])),
        clip=float(rng.choice([.5,1.,1.5,2.])),
        patience=int(rng.choice([4,5,7,9])),
        epochs=int(rng.choice([20,25,30,35])),
        numeric_noise=float(rng.choice([0.,.003,.006,.01])),
        neural_weight=float(rng.choice([.6,.7,.75,.8,.85,.9,1.])),
        warmup=int(rng.choice([0,1,2])),
        schedule=str(rng.choice(["cosine","cosine","constant"])))


class StructuredWholesaleNet(nn.Module):
    """FM main/pairwise effects + nonlinear tower + optional gated/3-way effects."""
    def __init__(self,cards,init_log_price,cfg):
        super().__init__()
        rank=cfg["rank"]
        self.fm=AuctionFactorization(cards,rank,initial_log_price=init_log_price)
        self.family=cfg["family"]
        self.rank=rank
        activation=nn.GELU if cfg["activation"]=="gelu" else nn.SiLU
        size=FIELD_COUNT*rank
        dims=[size,cfg["width1"]]+[cfg["width2"]]*(cfg["depth"]-1)
        blocks=[nn.LayerNorm(size)]
        for a,b in zip(dims,dims[1:]):
            blocks.extend([nn.Linear(a,b),activation(),nn.Dropout(cfg["dropout"])])
        blocks.append(nn.Linear(dims[-1],1))
        nn.init.zeros_(blocks[-1].weight)
        nn.init.zeros_(blocks[-1].bias)
        self.tower=nn.Sequential(*blocks)
        self.pair_gate=(nn.Parameter(torch.zeros(FIELD_COUNT,FIELD_COUNT))
                        if self.family=="deepfm_gate" else None)
        self.third_strength=(nn.Parameter(torch.full((rank,),.015))
                             if self.family=="deepfm_hofm3" else None)

    def forward(self,num,cat):
        base=self.fm(num,cat)
        n=num.unsqueeze(-1)*self.fm.num_latent.unsqueeze(0)
        c=torch.stack([tab(cat[:,j]) for j,tab in enumerate(self.fm.cat_latent)],dim=1)
        latent=torch.cat((n,c),dim=1)
        estimate=base+self.tower(latent.flatten(start_dim=1)).squeeze(-1)
        if self.pair_gate is not None:
            weights=torch.triu(torch.tanh(self.pair_gate),diagonal=1)
            product=torch.matmul(latent,latent.transpose(1,2))
            estimate=estimate+(product*weights.unsqueeze(0)).sum((1,2))
        if self.third_strength is not None:
            a=latent.sum(dim=1)
            b=(latent*latent).sum(dim=1)
            c=(latent*latent*latent).sum(dim=1)
            rank_three=(a.pow(3)-3*a*b+2*c)/6
            estimate=estimate+(rank_three*self.third_strength).sum(dim=1)
        return estimate


def create_model(cards,initial,cfg):
    if cfg==trial_config(0):
        return ScaledAuctionFM(cards,"deepfm",16,initial)
    return StructuredWholesaleNet(cards,initial,cfg)


def train_loss(pred,log_actual,cfg):
    each=nn.functional.smooth_l1_loss(pred,log_actual,
                                      beta=cfg["huber_beta"],reduction="none")
    loss=each.mean()
    if cfg["tail_weight"]:
        pred_usd=torch.exp(pred.clamp(math.log(100),math.log(1e6)))
        true_usd=torch.exp(log_actual)
        absolute=torch.abs(pred_usd-true_usd)
        k=max(1,math.ceil(.1*len(absolute)))
        loss=loss+cfg["tail_weight"]*(torch.topk(absolute,k).values.mean()/10000.)
    if not torch.isfinite(loss):raise ValueError("Nonfinite wholesale training loss")
    return loss


@torch.no_grad()
def prediction(model,numeric,categories,indices,batch_size=8192):
    model.eval()
    results=[]
    for at in range(0,len(indices),batch_size):
        idx=indices[at:at+batch_size]
        x=torch.from_numpy(numeric[idx].astype(np.float32))
        cats=torch.from_numpy(categories[idx].astype(np.int64))
        value=model(x,cats).detach().cpu().numpy()
        results.append(y_price(value))
    return np.concatenate(results)


def matched_lemon(path,identities,actual):
    frame=pd.read_csv(path,dtype={"row_id":str})
    if not {"row_id","split","observed_price","predicted_price"}.issubset(frame):
        raise ValueError("Missing original XGBoost Lemon per-vehicle reference columns")
    frame=frame.loc[frame.split.eq("validation")].copy()
    ids=np.asarray(identities).astype(str)
    if len(frame)!=82380 or frame.row_id.duplicated().any() or set(frame.row_id)!=set(ids):
        raise ValueError("Original XGBoost Lemon validation row identities do not match")
    frame=frame.set_index("row_id").reindex(ids)
    if not np.allclose(frame.observed_price.to_numpy(float),actual,rtol=1e-6,atol=.05):
        raise ValueError("Original XGBoost and full auction observed dollars differ")
    p=frame.predicted_price.to_numpy(float)
    if not np.isfinite(p).all() or (p<=0).any():
        raise ValueError("Nonfinite or non-positive original Lemon XGBoost prediction")
    return p


def lr_scheduler(optimizer,cfg):
    if cfg["schedule"]=="constant":
        return None
    def factor(epoch):
        warm=cfg["warmup"]
        if warm and epoch<warm:
            return max(.2,(epoch+1)/warm)
        e=max(0,epoch-warm)
        total=max(1,cfg["epochs"]-warm)
        frac=min(1,e/total)
        return .08+.92*(1+math.cos(math.pi*frac))/2.
    return torch.optim.lr_scheduler.LambdaLR(optimizer,factor)


def run(i,features,baseline,out,max_seconds):
    cfg=trial_config(i)
    z,source_manifest=source(features)
    train_idx,inner,val,_=partition(z,"full")
    if len(val)!=82380 or len(train_idx)+len(inner)!=384198:
        raise ValueError("Original VIN-disjoint full wholesale training cohorts differ")
    torch.manual_seed(SEED+i)
    np.random.seed(SEED+i)
    torch.set_num_threads(2)
    numeric=z["num"].astype(np.float32)
    cats=z["cat"].astype(np.int64)
    labels=z["y"].astype(np.float64)
    lemon=matched_lemon(baseline,z["ids"][val],labels[val])
    med=float(np.median(np.log(labels[train_idx])))
    model=create_model(source_manifest["cardinalities"],med,cfg)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],
                                weight_decay=cfg["weight_decay"],foreach=False)
    schedule=lr_scheduler(optimizer,cfg)
    training=TensorDataset(torch.from_numpy(numeric[train_idx]),
        torch.from_numpy(cats[train_idx]),
        torch.from_numpy(np.log(labels[train_idx]).astype(np.float32)))
    batch=DataLoader(training,batch_size=cfg["batch_size"],shuffle=True,
                     num_workers=0,generator=torch.Generator().manual_seed(SEED+i))
    best=math.inf;weights=None;best_epoch=0;stale=0;finished=0
    start=time.monotonic();limited=False
    for epoch in range(1,cfg["epochs"]+1):
        model.train()
        for num,cat,target in batch:
            optimizer.zero_grad(set_to_none=True)
            if cfg["numeric_noise"]:
                num=num+torch.randn_like(num)*cfg["numeric_noise"]
            raw=model(num,cat)
            loss=train_loss(raw,target,cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),cfg["clip"])
            optimizer.step()
            if time.monotonic()-start>max_seconds:
                limited=True
                break
        if schedule:schedule.step()
        finished=epoch
        p=prediction(model,numeric,cats,inner)
        scored=risk(labels[inner],p)["CVaR95_absolute_USD"]
        print(json.dumps({"bid_trial":i,"epoch":epoch,
                          "training_only_CVaR95_USD":scored,
                          "elapsed_seconds":round(time.monotonic()-start,1)}),flush=True)
        if scored<best-.1:
            best=scored;weights=copy.deepcopy(model.state_dict())
            best_epoch=epoch;stale=0
        else:stale+=1
        if stale>=cfg["patience"] or limited:break
    if weights is None or finished<5:
        raise RuntimeError("Bid model stopped before it was adequately trained")
    model.load_state_dict(weights)
    neural=prediction(model,numeric,cats,val)
    blended=cfg["neural_weight"]*neural+(1-cfg["neural_weight"])*lemon
    result=risk(labels[val],blended)
    neural_metrics=risk(labels[val],neural)
    dest=Path(out)
    dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"row_id":z["ids"][val].astype(str),"split":"validation",
                  "observed_price":labels[val],"predicted_price":blended,
                  "neural_price":neural,"lemon_xgb_price":lemon}).to_csv(
                      dest/"validation.csv.gz",index=False)
    provenance={"dataset_sha256":source_manifest["cohort_sha256"],
                "holdout_row_sha256":source_manifest["holdout_row_sha256"],
                "original_full_train":384198,"train_fit":len(train_idx),
                "train_only_stopper":len(inner),"validation_rows":82380,
                "original_test_rows":83603,"original_test_used":False}
    torch.save({"state_dict":weights,"trial":i,"config":cfg,
                "original_model_family":"original native auction FM / DeepFM",
                "provenance":provenance},dest/"weights.pt")
    record={"status":"complete","market":"bid","trial":i,"config":cfg,
            "validation":result,"neural_only_validation":neural_metrics,
            "training_only_stopper_CVaR95_USD":best,
            "best_epoch":best_epoch,"completed_epochs":finished,
            "training_time_budget_limited":limited,
            "elapsed_seconds":round(time.monotonic()-start,1),
            "validation_blend_weight_fixed_before_training":cfg["neural_weight"],
            "test_evaluated":False,
            "original_test_and_production_weights_untouched":True,
            "provenance":provenance}
    (dest/"result.json").write_text(json.dumps(record,indent=2,allow_nan=False))
    print("BID25_VALIDATION",i,json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--trial",type=int,required=True)
    p.add_argument("--features",default="training-cache/neural-bid/features.npz")
    p.add_argument("--baseline",default="benchmark-reference/bid/predictions.csv.gz")
    p.add_argument("--output",default="training-results/bid25")
    p.add_argument("--max-seconds",type=float,default=4200.)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    cfg=trial_config(a.trial)
    if a.dry_run:
        print(json.dumps({"trial":a.trial,"config":cfg}))
    else:run(a.trial,a.features,a.baseline,a.output,a.max_seconds)


if __name__=="__main__":main()
