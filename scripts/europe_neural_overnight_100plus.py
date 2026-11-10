"""Overnight Ray Tune: 5 DNN families on fresh observed October-2026 EU asks.

15 independent CPU workers per wave in TWO sequential waves to respect
GitHub's six-hour per-job limit, through 11 Nov 2026 05:00 CET.
Long trials support up to 150 epochs; ASHA/early stopping can stop sooner.
Uses re-prepared, country-standardized, fresh-only 80/10/10 grouped holdout. Search sees train+validation
ONLY. All distributions are EXACT observed advertised asking prices, not
claimed sales or fabricated censoring intervals. No deployment performed.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.europe_price_ray import FEATURES, NUMERIC, CATEGORICAL, load_data, metrics

SEED=20261109
DEADLINE=dt.datetime(2026,11,11,4,0,0,tzinfo=dt.timezone.utc)
# Wednesday 05:00 CET (Europe/Amsterdam) == Wednesday 04:00 UTC.
EPOCH_CHOICES=(10,20,40,80,100,150)
FAMILIES=("resmlp","dcnv2","deepfm","hofm3","fttransformer")
LOSSES=(
    # Exact-event accelerated failure time (log-price distribution families)
    "aft_gaussian","aft_loglogistic","aft_laplace","aft_student_t","aft_weibull",
    # Distinct hybrid AFT + robust objectives, explicitly not true censoring
    "aft_huber","aft_huber_tail","aft_student_huber","aft_logistic_huber",
    "aft_huber_eur","aft_cvar_weighted",
    # Robust point and adaptive regression
    "huber_log","huber_eur","pseudo_huber_log","adaptive_huber_log",
    "cauchy_log","logcosh_log","charbonnier_log","mae_log","mse_log",
    # Distributional and asymmetric robust regression
    "quantile50","quantile_log","smooth_quantile_log","expectile_log",
    # Percentage/relative and worst-tail-aware robust regression
    "smape_eur","mape_eur","tail_huber","tail_mae_eur","tail_relative",
    "weighted_tail_huber",
)
LOG_MIN=math.log(100.)
LOG_MAX=math.log(1_000_000.)


def safe_config(c):
    if c["family"] not in FAMILIES or c["loss"] not in LOSSES:
        raise ValueError("Unregistered deep architecture or objective")
    if (int(c["dim"]) not in (16,24,32)
        or int(c["width"]) not in (96,160,256)
        or int(c["layers"]) not in (1,2,3)
        or int(c["rank"]) not in (8,16,24)
        or int(c["batch"]) not in (256,384,512,768,1024,1536)
        or int(c["epochs"]) not in EPOCH_CHOICES
        or not (.00005 <= float(c["lr"]) <= .003)
        or not (0 <= float(c["wd"]) <= .06)
        or not (0 <= float(c["dropout"]) <= .3)
        or not (0 <= float(c["noise"]) <= .025)
        or not (.4 <= float(c["clip"]) <= 2.0)
        or not (.08 <= float(c["sigma"]) <= 1.25)
        or not (.12 <= float(c["beta"]) <= 1.0)
        or not (.1 <= float(c["aft_weight"]) <= .8)
        or not (.005 <= float(c["tail_fraction"]) <= .15)
        or not (0 <= float(c["tail_strength"]) <= .6)
        or not (.05 <= float(c["tau"]) <= .95)
        or not (2. <= float(c["student_df"]) <= 20.)
        or not (.01 <= float(c["charbonnier_eps"]) <= .9)
        or not (.4 <= float(c["focal_gamma"]) <= 3.0)
        or not (5000 <= float(c["usd_scale"]) <= 30000)
        or not (.3 <= float(c["train_fraction"]) <= 1.0)
        or c["optimizer"] not in ("adamw","radam")
        or c["schedule"] not in ("constant","cosine")
        or c["activation"] not in ("gelu","silu")):
        raise ValueError("Out-of-range shape or crash-prone hyperparameter")
    if c["family"]=="fttransformer" and (c["batch"]>384 or c["dim"]<24 or c["layers"]>2):
        raise ValueError("Transformer CPU/memory cap violated")
    return dict(c)


def base_config(family):
    return dict(family=family,loss="aft_huber",dim=24,width=160,
                layers=2,rank=16,batch=512 if family!="fttransformer" else 256,
                epochs=20,lr=.0007,wd=.002,dropout=.12,noise=.005,
                clip=1.,sigma=.45,beta=.3,aft_weight=.45,
                tail_fraction=.05,tail_strength=.15,tau=.5,
                student_df=5.,charbonnier_eps=.2,focal_gamma=1.5,
                usd_scale=15000.,train_fraction=.60,
                optimizer="adamw",schedule="cosine",activation="gelu",seed=SEED)


def read_arrays(folder):
    folder=Path(folder)
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("deduplication_applied") is not False
        or manifest.get("market")!="ask" or manifest.get("censored_rows")!=0
        or manifest.get("feature_names")!=FEATURES
        or manifest.get("test_sealed_until_final_aggregation") is not True):
        raise ValueError("Not the frozen raw-European advertised ASK split")
    if (folder/"test.npz").exists():
        raise ValueError("Research workers may NOT access the sealed test split")
    if manifest["counts"]["train"]<1000000 or manifest["counts"]["validation"]<100000:
        raise ValueError("Expected full-cohort train/validation; no sampled replacement")
    train_x,train_y=load_data(folder,"train",manifest)
    valid_x,valid_y=load_data(folder,"validation",manifest)
    levels=[int(manifest["categorical_cardinalities"][k]) for k in CATEGORICAL]
    return manifest,levels,train_x,train_y,valid_x,valid_y


class Prepared:
    def __init__(self, x, y, mean, std, levels):
        n=len(NUMERIC)
        raw=x[:,:n].astype(np.float32)
        observed=np.isfinite(raw)
        replaced=np.where(observed,raw,mean)
        z=np.clip((replaced-mean)/std,-7.,7.).astype(np.float32)
        z=np.concatenate([z,observed.astype(np.float32)],axis=1)
        rawcats=x[:,n:].astype(np.float32)
        cats=np.where(np.isfinite(rawcats),rawcats,-1).astype(np.int64)+1
        for j,cap in enumerate(levels):
            cats[:,j]=np.where((cats[:,j]>=1)&(cats[:,j]<=cap),cats[:,j],0)
        self.num=torch.from_numpy(np.ascontiguousarray(z))
        self.cat=torch.from_numpy(np.ascontiguousarray(cats))
        self.log_y=torch.from_numpy(np.log(np.asarray(y,dtype=np.float32)))
        self.y=np.asarray(y,dtype=np.float64)

    def __len__(self):
        return len(self.y)


def stats_of_train(x):
    num=x[:,:len(NUMERIC)].astype(np.float64)
    valid=np.isfinite(num)
    count=valid.sum(axis=0)
    mean=np.divide(np.where(valid,num,0.).sum(axis=0),np.maximum(1,count))
    dev=np.where(valid,num-mean,0.)
    std=np.sqrt(np.divide((dev*dev).sum(axis=0),np.maximum(1,count)))
    std=np.maximum(std,1.)
    return mean.astype(np.float32),std.astype(np.float32)


def network_input(net,num,cat):
    catemb=torch.stack([embedding(cat[:,i])
                        for i,embedding in enumerate(net.embeddings)],dim=1)
    tokens=num[:,:len(NUMERIC)].unsqueeze(-1)*net.num_weight + net.num_bias
    return catemb,tokens


class ResidualBlock(nn.Module):
    def __init__(self,width,dropout,activation):
        super().__init__()
        act=nn.GELU if activation=="gelu" else nn.SiLU
        self.net=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),
                               act(),nn.Dropout(dropout),nn.Linear(width,width),
                               nn.Dropout(dropout))
    def forward(self,x):
        return x+.4*self.net(x)


class CrossBlock(nn.Module):
    def __init__(self,size,rank,dropout):
        super().__init__()
        self.v=nn.Linear(size,rank,bias=False)
        self.u=nn.Linear(rank,size,bias=True)
        self.norm=nn.LayerNorm(size)
        self.drop=nn.Dropout(dropout)
    def forward(self,original,current):
        delta=self.u(self.v(self.norm(current)))
        return current+self.drop(original*delta)


class DeepTabular(nn.Module):
    def __init__(self,levels,c,mean_log):
        super().__init__()
        self.c=safe_config(c)
        self.family=c["family"]
        self.dim=int(c["dim"])
        self.embeddings=nn.ModuleList(
            [nn.Embedding(level+1,self.dim) for level in levels])
        self.num_weight=nn.Parameter(torch.randn(len(NUMERIC),self.dim)*.01)
        self.num_bias=nn.Parameter(torch.randn(len(NUMERIC),self.dim)*.01)
        self.mean_log=float(mean_log)
        self.dropout=float(c["dropout"])
        width=int(c["width"]);layers=int(c["layers"])
        dim_in=len(NUMERIC)*2 + len(CATEGORICAL)*self.dim
        act=nn.GELU if c["activation"]=="gelu" else nn.SiLU
        if self.family=="fttransformer":
            transformer_layer=nn.TransformerEncoderLayer(
                d_model=self.dim,nhead=4,dim_feedforward=self.dim*2,
                dropout=self.dropout,activation="gelu",batch_first=True,
                norm_first=True)
            self.transformer=nn.TransformerEncoder(transformer_layer,num_layers=layers,
                                                   enable_nested_tensor=False)
            self.cls=nn.Parameter(torch.zeros(1,1,self.dim))
            self.head=nn.Sequential(nn.LayerNorm(self.dim),
                                    nn.Linear(self.dim,self.dim),act(),
                                    nn.Linear(self.dim,1))
        else:
            self.proj=nn.Sequential(nn.LayerNorm(dim_in),nn.Linear(dim_in,width),
                                    act(),nn.Dropout(self.dropout))
            if self.family=="resmlp":
                self.blocks=nn.Sequential(*[ResidualBlock(width,self.dropout,c["activation"])
                                            for _ in range(layers)])
            elif self.family=="dcnv2":
                self.crosses=nn.ModuleList(
                    [CrossBlock(dim_in,int(c["rank"]),self.dropout)
                     for _ in range(layers)])
                self.blocks=nn.Sequential(*[ResidualBlock(width,self.dropout,c["activation"])
                                            for _ in range(layers)])
                self.cross_head=nn.Linear(dim_in,1)
            elif self.family in ("deepfm","hofm3"):
                self.fm_head=nn.Linear(self.dim,1,bias=False)
                if self.family=="hofm3":
                    self.third_head=nn.Linear(self.dim,1,bias=False)
                self.blocks=nn.Sequential(*[ResidualBlock(width,self.dropout,c["activation"])
                                            for _ in range(layers)])
            self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,1))
        # Low-magnitude embeddings and FM projections avoid exploding initial
        # interaction outputs on frequent common makes/model-year combinations.
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight,std=.035)
        if self.family in ("deepfm","hofm3"):
            nn.init.zeros_(self.fm_head.weight)
            if self.family=="hofm3":
                nn.init.zeros_(self.third_head.weight)
        # Start every architecture from the observed training median log price.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self,num,cat):
        catemb,tokens=network_input(self,num,cat)
        if self.family=="fttransformer":
            all_tokens=torch.cat([tokens,catemb],dim=1)
            cls=self.cls.expand(len(num),-1,-1)
            encoded=self.transformer(torch.cat([cls,all_tokens],dim=1))
            raw=self.head(encoded[:,0]).squeeze(-1)
        else:
            full=torch.cat([num,catemb.flatten(1)],dim=1)
            hidden=self.proj(full)
            if self.family=="dcnv2":
                cross=full
                for layer in self.crosses:
                    cross=layer(full,cross)
                raw=self.head(self.blocks(hidden)).squeeze(-1)
                raw=raw+.1*self.cross_head(cross-full).squeeze(-1)
            else:
                hidden=self.blocks(hidden)
                raw=self.head(hidden).squeeze(-1)
                if self.family in ("deepfm","hofm3"):
                    fields=torch.cat([tokens,catemb],dim=1)
                    first=fields.sum(dim=1)
                    second=fields.square().sum(dim=1)
                    pairs=.5*(first.square()-second)
                    raw=raw+.1*self.fm_head(pairs).squeeze(-1)
                    if self.family=="hofm3":
                        third=fields.pow(3).sum(dim=1)
                        triples=(first.pow(3)-3*first*second+2*third)/6.
                        raw=raw+.01*self.third_head(triples).squeeze(-1)
        return self.mean_log+raw


def objective(mu,target,c):
    """Thirty bounded, differentiable training objectives on exact ASK prices.

    The five AFT families are exact-event likelihoods only. Censoring needs
    real observed interval bounds, which this dataset does not provide.
    """
    safe_config(c)
    residual=mu-target
    sigma=float(c["sigma"])
    beta=float(c["beta"])
    tau=float(c["tau"])
    kind=c["loss"]
    z=residual/sigma
    log_sigma=math.log(sigma)
    gaussian=(.5*z.square()+log_sigma).mean()
    logistic=(F.softplus(z)+F.softplus(-z)+log_sigma).mean()
    laplace=(z.abs()+log_sigma).mean()
    df=float(c["student_df"])
    student=(.5*(df+1.)*torch.log1p(z.square()/df)+log_sigma).mean()
    # Gumbel log-error, hence Weibull time-to-event distribution after exponentiation.
    # Clamp only the exponential argument against overflow.
    weibull_z=(-z).clamp(-12.,12.)
    weibull=(torch.exp(weibull_z)-weibull_z+log_sigma).mean()
    robust=F.smooth_l1_loss(mu,target,beta=beta)
    scale=float(c["usd_scale"])
    predicted=torch.exp(mu.clamp(LOG_MIN,LOG_MAX))
    actual=torch.exp(target)
    eur_error=(predicted-actual).abs()
    eur_residual=(predicted-actual)/scale
    eur_huber=F.smooth_l1_loss(eur_residual,torch.zeros_like(eur_residual),beta=.5)
    a=residual.abs()
    k=max(1,math.ceil(len(mu)*float(c["tail_fraction"])))
    tail_eur=eur_error.topk(k).values.mean()/scale
    w=float(c["aft_weight"])
    strength=float(c["tail_strength"])
    if kind=="aft_gaussian":
        result=gaussian
    elif kind=="aft_loglogistic":
        result=logistic
    elif kind=="aft_laplace":
        result=laplace
    elif kind=="aft_student_t":
        result=student
    elif kind=="aft_weibull":
        result=weibull
    elif kind in ("aft_huber","aft_huber_tail"):
        result=(1-w)*gaussian+w*robust
        if kind.endswith("_tail"):
            result=result+strength*tail_eur
    elif kind=="aft_student_huber":
        result=(1-w)*student+w*robust
    elif kind=="aft_logistic_huber":
        result=(1-w)*logistic+w*robust
    elif kind=="aft_huber_eur":
        result=(1-w)*gaussian+w*eur_huber
    elif kind=="aft_cvar_weighted":
        result=gaussian+strength*tail_eur
    elif kind in ("huber_log","tail_huber"):
        result=robust+(strength*tail_eur if kind=="tail_huber" else 0.)
    elif kind=="huber_eur":
        result=eur_huber
    elif kind=="pseudo_huber_log":
        result=(torch.sqrt(residual.square()+beta*beta)-beta).mean()
    elif kind=="adaptive_huber_log":
        cutoff=a.detach().topk(k).values.min().clamp_min(.05)
        weight=(1.+strength*(a.detach()/cutoff).clamp(max=5.))
        point=F.smooth_l1_loss(mu,target,beta=beta,reduction="none")
        result=(point*weight).sum()/weight.sum().clamp_min(1.)
    elif kind=="weighted_tail_huber":
        weight=(1.+strength*(eur_error.detach()/scale).clamp(max=5.).pow(float(c["focal_gamma"]))).detach()
        point=F.smooth_l1_loss(mu,target,beta=beta,reduction="none")
        result=(point*weight).sum()/weight.sum().clamp_min(1.)
    elif kind=="cauchy_log":
        result=torch.log1p(residual.square()/(beta*beta)).mean()
    elif kind=="logcosh_log":
        result=(a+F.softplus(-2*a)-math.log(2)).mean()
    elif kind=="charbonnier_log":
        epsilon=float(c["charbonnier_eps"])
        result=(torch.sqrt(residual.square()+epsilon*epsilon)-epsilon).mean()
    elif kind=="mae_log":
        result=a.mean()
    elif kind=="mse_log":
        result=residual.square().mean()
    elif kind=="quantile50":
        result=.5*a.mean()
    elif kind=="quantile_log":
        error=target-mu
        result=torch.maximum(tau*error,(tau-1.)*error).mean()
    elif kind=="smooth_quantile_log":
        error=target-mu
        result=(tau*F.softplus(error/beta)*beta+
                (1-tau)*F.softplus(-error/beta)*beta).mean()
    elif kind=="expectile_log":
        result=(torch.where(residual>=0,1.-tau,tau)*residual.square()).mean()
    elif kind=="smape_eur":
        result=(2.*eur_error/(predicted+actual+100.)).mean()
    elif kind=="mape_eur":
        result=(eur_error/actual.clamp_min(100.)).clamp(max=20.).mean()
    elif kind=="tail_mae_eur":
        result=eur_error.mean()/scale + strength*tail_eur
    elif kind=="tail_relative":
        relative=eur_error/actual.clamp_min(100.)
        result=relative.clamp(max=20.).mean()+strength*relative.topk(k).values.mean()
    else:
        raise ValueError("Unsupported neural objective "+kind)
    if not bool(torch.isfinite(result).item()):
        raise FloatingPointError("Nonfinite neural objective: "+kind)
    return result

@torch.no_grad()
def predict_model(model,data,batch=2048):
    model.eval()
    out=[]
    for i in range(0,len(data),batch):
        result=model(data.num[i:i+batch],data.cat[i:i+batch])
        out.append(torch.exp(result.clamp(LOG_MIN,LOG_MAX)).numpy())
    return np.concatenate(out).astype(np.float64)


def checkpoint(model,c,manifest,mean,std,median,epoch,val,path):
    saved=dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
               config=dict(c),train_mean=mean.tolist(),train_std=std.tolist(),
               train_median_log=float(median),epoch=int(epoch),
               validation=dict(val),manifest_train_sha=manifest["sha256_by_phase"]["train"],
               manifest_valid_sha=manifest["sha256_by_phase"]["validation"],
               test_scored=False)
    torch.save(saved,str(path))


def run_trial(config,source,out,reporter=None,deadline=None):
    c=safe_config(config)
    torch.set_num_threads(2)
    torch.manual_seed(int(c["seed"]))
    np.random.seed(int(c["seed"]))
    manifest,levels,train_x,train_y,valid_x,valid_y=read_arrays(source)
    mean,std=stats_of_train(train_x)
    median=float(np.median(np.log(train_y)))
    training=Prepared(train_x,train_y,mean,std,levels)
    valid=Prepared(valid_x,valid_y,mean,std,levels)
    del train_x,valid_x,train_y,valid_y
    model=DeepTabular(levels,c,median)
    optim=(torch.optim.AdamW if c["optimizer"]=="adamw" else torch.optim.RAdam)(
        model.parameters(),lr=float(c["lr"]),weight_decay=float(c["wd"]))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    highest=float("inf");bad_epochs=0
    best=None
    total_epochs=int(c["epochs"])
    for epoch in range(1,total_epochs+1):
        # Respect overnight cutoff even if Ray's time budget has almost expired.
        if deadline is not None and dt.datetime.now(dt.timezone.utc)>=deadline-dt.timedelta(seconds=45):
            break
        model.train()
        if c["schedule"]=="cosine":
            lr=float(c["lr"])*(.2+.8*.5*(1+math.cos(math.pi*(epoch-1)/total_epochs)))
            for group in optim.param_groups:group["lr"]=lr
        ids=torch.randperm(len(training))
        limit=max(100,int(len(ids)*float(c["train_fraction"])))
        ids=ids[:limit]
        total=0.
        batches=0
        for ix in ids.split(int(c["batch"])):
            if len(ix)<16:continue
            xx=training.num[ix]
            if float(c["noise"])>0:
                xx=xx+torch.randn_like(xx)*float(c["noise"])
            raw=model(xx,training.cat[ix])
            loss=objective(raw,training.log_y[ix],c)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),float(c["clip"]))
            optim.step()
            total+=float(loss.item())
            batches+=1
        predictions=predict_model(model,valid)
        score=metrics(valid.y,predictions)
        value=float(score["cvar95_absolute_eur"])
        if value<highest-.01:
            highest=value
            bad_epochs=0
            best=dict(architecture=c["family"],loss=c["loss"],config=dict(c),
                      validation=score,epoch=epoch,training_rows=int(limit),
                      train_total=len(training),validation_rows=len(valid),
                      test_scored=False,year_target="exact listed asking price")
            (out/"result.json").write_text(json.dumps(best,indent=2))
            checkpoint(model,c,manifest,mean,std,median,epoch,score,out/"winner.pt")
        else:
            bad_epochs+=1
        if reporter:
            reporter(dict(best_cvar95=highest,mae=float(score["mae_eur"]),
                          epoch=epoch,train_loss=total/max(1,batches)))
        # Longer schedules can need several epochs to recover from a poor
        # validation update. Still stop clearly unproductive trials early.
        patience=12 if total_epochs>=80 else 6
        if bad_epochs>=patience:break
    if best is None:
        raise RuntimeError("Neural trial produced no valid validation checkpoint")
    return best


def tune_worker(features,output,shard,wave,budget_min,deadline=None):
    import ray
    from ray import tune,train
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.basic_variant import BasicVariantGenerator
    family=FAMILIES[int(shard)%len(FAMILIES)]
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    features=Path(features).resolve()
    # One Ray worker per independent GitHub runner; 15 matrix jobs in parallel.
    started=dt.datetime.now(dt.timezone.utc)
    absolute_cutoff=DEADLINE
    deadline=min(absolute_cutoff,deadline) if deadline else absolute_cutoff
    remaining=(deadline-started).total_seconds()
    allowed=max(0,int(min(float(budget_min)*60,remaining-120)))
    if allowed<180:
        print("Overnight cutoff reached, no new training run",flush=True)
        (output/"cutoff.json").write_text(json.dumps(
            {"shard":shard,"wave":wave,"cutoff":True,"deadline_utc":str(deadline)}))
        return
    root=(Path("/tmp")/("eu_neural_ray_%02d_%d"%(int(shard),int(wave)))).resolve()
    (root/"ray").mkdir(parents=True,exist_ok=True)
    os.environ["RAY_TMPDIR"]=str(root/"ray")
    ray.init(num_cpus=2,include_dashboard=False,
             _temp_dir=str(root/"ray"),log_to_driver=False)
    search=dict(
        family=family,
        loss=tune.choice([
            "aft_gaussian","aft_loglogistic","aft_laplace","aft_student_t",
            "aft_weibull","aft_huber","aft_huber","aft_huber_tail",
            "aft_student_huber","aft_logistic_huber","aft_huber_eur",
            "aft_cvar_weighted","huber_log","huber_eur","pseudo_huber_log",
            "adaptive_huber_log","cauchy_log","logcosh_log","charbonnier_log",
            "mae_log","mse_log","quantile50","quantile_log",
            "smooth_quantile_log","expectile_log","smape_eur","mape_eur",
            "tail_huber","tail_mae_eur","tail_relative","weighted_tail_huber"
        ]),
        dim=tune.choice([24,32] if family=="fttransformer" else [16,24,32]),
        width=tune.choice([96,160,256]),
        layers=tune.choice([1,2] if family=="fttransformer" else [1,2,3]),
        rank=tune.choice([8,16,24]),
        batch=tune.choice([256,384] if family=="fttransformer"
                          else [384,512,768,1024,1536]),
        epochs=tune.choice(list(EPOCH_CHOICES)),
        lr=tune.loguniform(.00005,.003),
        wd=tune.loguniform(.00001,.05),
        dropout=tune.uniform(.03,.28),
        noise=tune.uniform(0.,.02),
        clip=tune.uniform(.5,1.8),
        sigma=tune.uniform(.12,1.1),
        beta=tune.uniform(.15,.9),
        aft_weight=tune.uniform(.15,.7),
        tail_fraction=tune.uniform(.02,.12),
        tail_strength=tune.uniform(0.,.5),
        tau=tune.uniform(.15,.85),
        student_df=tune.choice([2.5,3.,4.,6.,9.,15.]),
        charbonnier_eps=tune.uniform(.04,.75),
        focal_gamma=tune.uniform(.5,2.5),
        usd_scale=tune.choice([7500.,10000.,15000.,20000.,25000.]),
        train_fraction=tune.choice([.35,.55,.75,1.]),
        optimizer=tune.choice(["adamw","radam"]),
        schedule=tune.choice(["constant","cosine"]),
        activation=tune.choice(["gelu","silu"]),
        seed=SEED+int(shard)*100+int(wave)*1000,
    )
    def trainable(config):
        trial_id=train.get_context().get_trial_id()
        dest=output/"trials"/str(trial_id)
        run_trial(config,features,dest,reporter=train.report,deadline=deadline)
    algo=BasicVariantGenerator(random_state=SEED+int(shard)+int(wave)*100)
    sched=ASHAScheduler(max_t=max(EPOCH_CHOICES),grace_period=6,
                        reduction_factor=3)
    try:
        tune.run(trainable,config=search,num_samples=500,scheduler=sched,
                 search_alg=algo,metric="best_cvar95",mode="min",
                 resources_per_trial={"cpu":2},max_concurrent_trials=1,
                 time_budget_s=allowed,storage_path=str(root),name="neural",
                 raise_on_failed_trial=False,verbose=1)
    finally:
        ray.shutdown()
    candidates=[]
    for path in (output/"trials").glob("*/result.json"):
        try:
            summary=json.loads(path.read_text())
            if (path.parent/"winner.pt").is_file():
                candidates.append((summary,path))
        except (OSError,ValueError,KeyError):
            continue
    if not candidates:
        raise RuntimeError("No valid neural checkpoints; inspect Ray trial errors")
    candidates.sort(key=lambda t:(t[0]["validation"]["cvar95_absolute_eur"],
                                  t[0]["validation"]["mae_eur"]))
    winner,where=candidates[0]
    shutil.copy2(where.parent/"winner.pt",output/"winner.pt")
    (output/"result.json").write_text(json.dumps(
        {**winner,"wave":int(wave),"worker":int(shard)},indent=2))
    (output/"trials.json").write_text(json.dumps(
        [{**t,"trial":p.parent.name} for t,p in candidates],indent=2))
    print(json.dumps({"worker":shard,"wave":wave,
                      "architecture":family,"evaluated":len(candidates),
                      "winner":winner},indent=2),flush=True)


def smoke():
    torch.set_num_threads(2)
    levels=[4]*len(CATEGORICAL)
    num=torch.randn(24,len(NUMERIC)*2)
    cat=torch.randint(0,5,(24,len(CATEGORICAL)))
    target=torch.randn(24)*.2+9.5
    for family in FAMILIES:
        c=base_config(family)
        model=DeepTabular(levels,c,float(target.median()))
        for loss in LOSSES:
            c["loss"]=loss
            output=model(num,cat)
            loss_val=objective(output,target,c)
            model.zero_grad(set_to_none=True)
            loss_val.backward()
            if not bool(torch.isfinite(loss_val).item()) or not all(
                p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
            ):
                raise AssertionError("Nonfinite model/loss gradient: "+family+" "+loss)
        print("NETWORK_LOSS_SMOKE_OK",family,len(LOSSES),flush=True)
    c=base_config("fttransformer");c["batch"]=1536
    try:safe_config(c)
    except ValueError:pass
    else:raise AssertionError("Oversized transformer batch not rejected")
    c=base_config("resmlp");c["tau"]=1.2
    try:safe_config(c)
    except ValueError:pass
    else:raise AssertionError("Invalid expectile tau not rejected")


def ray_smoke():
    import tempfile
    import ray
    from ray import tune,train
    from ray.tune.schedulers import ASHAScheduler
    def report(config):
        for i in range(2):
            train.report(dict(best_cvar95=float(100-i),epoch=i+1))
    with tempfile.TemporaryDirectory() as root:
        ray.init(num_cpus=2,include_dashboard=False,log_to_driver=False)
        try:
            analysis=tune.run(report,config={"fixed":1},num_samples=1,
                              scheduler=ASHAScheduler(max_t=2,grace_period=1),
                              metric="best_cvar95",mode="min",
                              resources_per_trial={"cpu":1},
                              storage_path=root,name="preflight",
                              raise_on_failed_trial=True,verbose=0)
            if len(analysis.trials)!=1 or analysis.trials[0].status!="TERMINATED":
                raise AssertionError("Ray preflight trial did not finish")
        finally:ray.shutdown()
    print("NEURAL_RAY_ASHA_SMOKE_OK",flush=True)


def final_compare(root,features,output,baseline_root=None):
    """Compare neural and XGBoost on validation, test ONE overall winner."""
    root=Path(root);features=Path(features);output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((features/"manifest.json").read_text())
    neural=[]
    for path in root.glob("*/result.json"):
        ck=path.parent/"winner.pt"
        if not ck.is_file():continue
        summary=json.loads(path.read_text())
        if summary.get("test_scored") is not False:
            raise AssertionError("Neural trial accessed the sealed test")
        neural.append((summary,path,ck))
    if not neural:
        raise RuntimeError("No valid neural winner checkpoints found")
    neural.sort(key=lambda e:(e[0]["validation"]["cvar95_absolute_eur"],
                              e[0]["validation"]["mae_eur"]))
    baseline=[]
    if baseline_root is not None and Path(baseline_root).exists():
        for f in Path(baseline_root).glob("*/result.json"):
            model=f.parent/"winner.ubj"
            if not model.exists():continue
            item=json.loads(f.read_text())
            if item.get("test_scored") is not False:
                raise AssertionError("XGBoost baseline accessed sealed test")
            baseline.append((item,f,model))
    competitors=[]
    for item,path,checkpoint_path in neural:
        competitors.append((item["validation"],"neural",item,path,checkpoint_path))
    for item,path,checkpoint_path in baseline:
        competitors.append((item["validation"],"xgboost",item,path,checkpoint_path))
    competitors.sort(key=lambda v:(v[0]["cvar95_absolute_eur"],
                                    v[0]["mae_eur"]))
    val,kind,best,path,checkpoint_path=competitors[0]
    # Test data becomes visible only AFTER selection has been frozen.
    x,y=load_data(features,"test",manifest)
    if kind=="neural":
        pack=torch.load(str(checkpoint_path),map_location="cpu",weights_only=True)
        if pack["manifest_train_sha"]!=manifest["sha256_by_phase"]["train"]:
            raise ValueError("Neural checkpoint trained on wrong input cohort")
        if pack["manifest_valid_sha"]!=manifest["sha256_by_phase"]["validation"]:
            raise ValueError("Neural checkpoint selected on wrong validation cohort")
        levels=[int(manifest["categorical_cardinalities"][k]) for k in CATEGORICAL]
        model=DeepTabular(levels,pack["config"],pack["train_median_log"])
        model.load_state_dict(pack["state_dict"],strict=True)
        torch.set_num_threads(2)
        test=Prepared(x,y,np.array(pack["train_mean"],dtype=np.float32),
                      np.array(pack["train_std"],dtype=np.float32),levels)
        test_score=metrics(y,predict_model(model,test,batch=1024))
        shutil.copy2(checkpoint_path,output/"champion.pt")
        arch=best["architecture"]
        loss=best["loss"]
    else:
        import xgboost as xgb
        from scripts.europe_price_ray import dmatrix,predict_model as xgb_predict
        family=best["model_family"]
        model=xgb.Booster()
        model.load_model(str(checkpoint_path))
        test_dm=dmatrix(x,y,family,manifest)
        test_score=metrics(y,xgb_predict(model,test_dm,family))
        shutil.copy2(checkpoint_path,output/"champion.ubj")
        arch="XGBoost"
        loss=family
    chosen=dict(origin=path.parent.name,kind=kind,architecture=arch,
                loss=loss,validation=val,test_once=test_score,
                sealed_test_sha256=manifest["sha256_by_phase"]["test"],
                config=best["config"],neural_epoch=best.get("epoch"),
                xgboost_iterations=best.get("iterations"))
    report=dict(
        target="Fresh European original observed ASK, NOT index-forecast synthetic labels",
        deduplication_applied=False,censored_rows=0,
        features="Frozen numeric and categorical European tabular features",
        training_rows=manifest["counts"]["train"],
        validation_rows=manifest["counts"]["validation"],
        test_rows=manifest["counts"]["test"],
        search_deadline_utc="2026-11-11T04:00:00Z",
        epoch_search_options=list(EPOCH_CHOICES),
        max_epochs=max(EPOCH_CHOICES),
        cutoff_local="2026-11-11 05:00 CET",
        selected_by="validation CVaR95 (worst 5% absolute EUR error), then MAE",
        compared_models=len(competitors),
        total_neural_winners=len(neural),
        total_xgboost_baselines=len(baseline),
        neural_leaderboard=[dict(origin=p.parent.name,
                                architecture=v["architecture"],loss=v["loss"],
                                validation=v["validation"],config=v["config"],
                                epoch=v["epoch"],test_scored=False)
                            for v,p,_ in neural],
        xgboost_reference_validation=[dict(origin=p.parent.name,
                                family=v["model_family"],
                                validation=v["validation"],test_scored=False)
                            for v,p,_ in baseline],
        overall_champion=chosen,
        no_deployment=True,
    )
    (output/"comparison.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"overall_winner":chosen,"compared":len(competitors),
                      "neural":len(neural),"xgboost":len(baseline)},indent=2),flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="action",required=True)
    a=sub.add_parser("ray")
    a.add_argument("--features",required=True,type=Path)
    a.add_argument("--output",required=True,type=Path)
    a.add_argument("--shard",required=True,type=int)
    a.add_argument("--wave",required=True,type=int,choices=[1,2])
    a.add_argument("--budget-min",required=True,type=int)
    a=sub.add_parser("compare")
    a.add_argument("--root",required=True,type=Path)
    a.add_argument("--features",required=True,type=Path)
    a.add_argument("--output",required=True,type=Path)
    a.add_argument("--baselines",type=Path)
    sub.add_parser("smoke")
    sub.add_parser("ray-smoke")
    args=p.parse_args()
    if args.action=="ray":
        tune_worker(args.features,args.output,args.shard,args.wave,args.budget_min)
    elif args.action=="compare":
        final_compare(args.root,args.features,args.output,args.baselines)
    elif args.action=="smoke":smoke()
    else:ray_smoke()


if __name__=="__main__":
    main()
