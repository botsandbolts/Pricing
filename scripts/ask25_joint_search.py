"""Compact, CVaR95-first 25-trial Ask search on original 2021 listing cohorts.

Each trial JOINTLY varies structure and training hyperparameters. Original BGE
embeddings are frozen, SHA-verified and loaded from an existing artifact; no
encoder re-training, text-derived price scalar, model conversion or test peeking.
Trained native checkpoints are research-only, never deployed automatically.

Design references: DCN-V2 (low-rank / mixture of experts), strong tabular ResNet
baselines, AdamW, scheduled learning rates and train-only early stopping.
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
from scripts.deepcross_aft_tune import TunedDeepCross, config as dcn_config
from scripts.neural_ask_train import (
    CarEmbedding, ResidualBlock, objective, predict, seed_everything,
)
from scripts.raw_embedding_ask_train import (
    EmbeddingOnlyDeepCross, frozen_source,
)

TRIALS = 25
SEED = 20261107


def trial_config(index):
    if not isinstance(index, int) or not 0 <= index < TRIALS:
        raise ValueError("Trial index must be an integer between 0 and 24")
    # A direct comparison against the deployed *architecture* and optimizer.
    # It is re-fitted on the same original data, not the production checkpoint.
    if index == 0:
        return {
            "family": "dcnv2", "project_dim": 96, "project_layers": 1,
            "project_dropout": .08, "cross_layers": 3, "rank": 32,
            "experts": 1, "width": 256, "blocks": 2, "dropout": .12,
            "activation": "gelu", "learning_rate": .001, "weight_decay": .025,
            "beta1": .9, "eps": 1e-8, "batch_size": 768, "epochs": 35,
            "patience": 7, "numeric_noise": .008, "grad_clip": 1.,
            "tail_weight": 0., "warmup_epochs": 0, "lr_schedule": "cosine",
        }
    rng = np.random.default_rng(SEED + index * 31)
    family = "resmlp" if index in (5, 10, 15, 20, 24) else "dcnv2"
    return {
        "family": family,
        "project_dim": int(rng.choice([48,64,96,128])),
        "project_layers": int(rng.choice([1,1,2])),
        "project_dropout": float(rng.choice([.04,.08,.15])),
        "cross_layers": int(rng.choice([2,3,4])),
        "rank": int(rng.choice([16,24,32,48,64])),
        "experts": int(rng.choice([1,1,2,3])),
        "width": int(rng.choice([128,192,256,320])),
        "blocks": int(rng.choice([1,2,3])),
        "dropout": float(rng.choice([.08,.12,.2,.28])),
        "activation": str(rng.choice(["gelu","silu"])),
        "learning_rate": float(np.exp(rng.uniform(np.log(.0003),np.log(.002)))),
        "weight_decay": float(np.exp(rng.uniform(np.log(.001),np.log(.08)))),
        "beta1": float(rng.choice([.85,.9,.95])),
        "eps": float(rng.choice([1e-8,1e-7])),
        "batch_size": int(rng.choice([512,768,1024])),
        "epochs": int(rng.choice([24,30,35,40])),
        "patience": int(rng.choice([5,7,9])),
        "numeric_noise": float(rng.choice([0.,.004,.008,.015])),
        "grad_clip": float(rng.choice([.5,1.,1.5])),
        "tail_weight": float(rng.choice([0.,0.,.02,.04,.07])),
        "warmup_epochs": int(rng.choice([0,1,2])),
        "lr_schedule": str(rng.choice(["cosine","cosine","linear"])),
    }


class ResearchAskingNetwork(nn.Module):
    """One frozen text-vector projection plus trainable tabular pricing network."""
    def __init__(self,cards,embedding_dim,params):
        super().__init__()
        d=params["project_dim"]
        activation=nn.GELU if params["activation"]=="gelu" else nn.SiLU
        layers=[nn.LayerNorm(embedding_dim),nn.Linear(embedding_dim,d),
                activation(),nn.LayerNorm(d)]
        for _ in range(params["project_layers"]-1):
            layers.extend([nn.Linear(d,d),activation(),nn.LayerNorm(d)])
        layers.append(nn.Dropout(params["project_dropout"]))
        self.projector=nn.Sequential(*layers)
        self.family=params["family"]
        if self.family=="dcnv2":
            settings=dcn_config("dropout12")
            settings.update({
                "cross_layers": params["cross_layers"],
                "rank": params["rank"],
                "experts": params["experts"],
                "width": params["width"],
                "blocks": params["blocks"],
                "dropout": params["dropout"],
                "activation": params["activation"],
            })
            self.network=TunedDeepCross(cards,7,d,settings)
        elif self.family=="resmlp":
            self.input=CarEmbedding(cards,7,d,params["dropout"])
            w=params["width"]
            self.body=nn.Sequential(
                nn.Linear(self.input.width,w),nn.LayerNorm(w),activation(),
                *[ResidualBlock(w,params["dropout"])
                  for _ in range(params["blocks"])],
                nn.LayerNorm(w))
            self.out=nn.Linear(w,2)
        else:
            raise ValueError("Unsupported experiment architecture")

    def forward(self,numeric,category,embedding):
        projected=self.projector(embedding)
        if self.family=="dcnv2":
            return self.network(numeric,category,projected)
        return self.out(self.body(self.input(numeric,category,projected)))


def model_for(params,cards,text_dimension):
    if params==trial_config(0):
        return EmbeddingOnlyDeepCross(cards,7,text_dimension)
    return ResearchAskingNetwork(cards,text_dimension,params)


def training_loss(raw,log_y,tail_weight):
    base=objective(raw,log_y,"aft")
    if tail_weight:
        # Bounded dollar-tail penalty only on TRAIN examples. The unbiased
        # original VAL CVaR remains the primary selection criterion.
        estimated=torch.exp(torch.clamp(raw[:,0],
                            min=math.log(100.),max=math.log(1000000.)))
        observed=torch.exp(log_y)
        absolute=torch.abs(estimated-observed)
        count=max(1,int(math.ceil(.10*len(absolute))))
        # $10k normalization prevents scale from overwhelming log-AFT.
        base=base+tail_weight*(torch.topk(absolute,count).values.mean()/10000.)
    if not torch.isfinite(base):
        raise ValueError("Invalid nonfinite training loss")
    return base


def scheduler_for(optimizer,params):
    if params["warmup_epochs"]==0 and params["lr_schedule"]=="cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,T_max=params["epochs"],eta_min=.00006)
    warmup=params["warmup_epochs"]
    total=params["epochs"]
    def factor(epoch):
        if warmup and epoch<warmup:
            return max(.1,(epoch+1)/warmup)
        step=max(0,epoch-warmup)
        horizon=max(1,total-warmup)
        fraction=min(1.,step/horizon)
        remaining=1.-fraction
        if params["lr_schedule"]=="cosine":
            remaining=(1.+math.cos(math.pi*fraction))/2.
        return .08+.92*remaining
    return torch.optim.lr_scheduler.LambdaLR(optimizer,factor)


def train(index,original,embeddings,output,max_seconds):
    config=trial_config(index)
    z,text,source,encoder=frozen_source(original,embeddings,"bge-256")
    phases=z["phase"].astype(str)
    fit_all=np.flatnonzero(phases=="train")
    fit,inner=fit_all[:-14777],fit_all[-14777:]
    valid=np.flatnonzero(phases=="validation")
    if (len(fit),len(inner),len(valid))!=(133000,14777,31668):
        raise ValueError("Original temporal training/validation split was modified")
    seed_everything(SEED+index)
    dataset=TensorDataset(
        torch.from_numpy(z["num"].astype(np.float32)),
        torch.from_numpy(z["cat"].astype(np.int64)),
        torch.from_numpy(text.astype(np.float32)),
        torch.from_numpy(np.log(z["y"]).astype(np.float32)))
    model=model_for(config,source["cardinalities"],text.shape[1])
    optimizer=torch.optim.AdamW(model.parameters(),
        lr=config["learning_rate"],weight_decay=config["weight_decay"],
        betas=(config["beta1"],.999),eps=config["eps"],foreach=False)
    schedule=scheduler_for(optimizer,config)
    loader=DataLoader(Subset(dataset,fit.tolist()),
        batch_size=config["batch_size"],shuffle=True,num_workers=0,
        generator=torch.Generator().manual_seed(SEED+index))
    best=math.inf
    state=None
    best_epoch=0
    epochs_finished=0
    stale=0
    limited=False
    begin=time.monotonic()
    for epoch in range(1,config["epochs"]+1):
        model.train()
        for numerical,categories,vector,target in loader:
            optimizer.zero_grad(set_to_none=True)
            if config["numeric_noise"]:
                numerical=numerical+torch.randn_like(numerical)*config["numeric_noise"]
            raw=model(numerical,categories,vector)
            loss=training_loss(raw,target,config["tail_weight"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),config["grad_clip"])
            optimizer.step()
            if time.monotonic()-begin>max_seconds:
                limited=True
                break
        schedule.step()
        epochs_finished=epoch
        # The stopper is carved out of TRAIN, never the external validation set.
        prediction=predict(model,dataset,inner,config["batch_size"])
        inner_risk=risk(z["y"][inner],prediction)["CVaR95_absolute_USD"]
        print(json.dumps({"trial":index,"epoch":epoch,
              "inner_CVaR95_absolute_USD":inner_risk,
              "seconds":round(time.monotonic()-begin,1)}),flush=True)
        if inner_risk<best-.1:
            best=inner_risk
            state=copy.deepcopy(model.state_dict())
            best_epoch=epoch
            stale=0
        else:
            stale+=1
        if stale>=config["patience"] or limited:
            break
    if state is None or epochs_finished<5:
        raise RuntimeError("No valid adequately-trained candidate; fail instead of publishing partial results")
    model.load_state_dict(state)
    # Validation is scored ONCE per trial, after TRAIN-only checkpoint choice.
    pred=predict(model,dataset,valid,config["batch_size"])
    scores=risk(z["y"][valid],pred)
    destination=Path(output)
    destination.mkdir(parents=True,exist_ok=True)
    validation=pd.DataFrame({
        "row_id":z["ids"][valid].astype(str),
        "split":"validation",
        "observed_price":z["y"][valid].astype(float),
        "predicted_price":pred,
    })
    validation.to_csv(destination/"validation.csv.gz",index=False)
    provenance={
        "cohort_sha256":source["cohort_sha256"],
        "holdout_row_sha256":source["holdout_row_sha256"],
        "embedding_sha256":encoder["embedding_file_sha256"],
        "encoder_revision":encoder["model_revision"],
        "training_rows":len(fit),
        "train_only_early_stopping_rows":len(inner),
        "validation_rows":len(valid),
        "original_test_rows":31682,
        "test_used":False,
        "no_price_text_head":True,
    }
    checkpoint={
        "state_dict":state,
        "architecture":"BGE frozen embeddings to native PyTorch "+config["family"],
        "parameters":config,
        "seed":SEED+index,
        "trial":index,
        "provenance":provenance,
    }
    torch.save(checkpoint,destination/"weights.pt")
    record={
        "status":"completed","trial":index,
        "architecture":config["family"],
        "joint_structure_and_hyperparameters":config,
        "primary_metric":"original validation CVaR95 absolute USD",
        "original_validation":scores,
        "original_test_scored":False,
        "best_epoch":best_epoch,
        "epochs_completed":epochs_finished,
        "train_internal_CVaR95_USD":best,
        "time_budget_limited":limited,
        "elapsed_seconds":round(time.monotonic()-begin,1),
        "native_weights_only":True,
        "encoder_frozen":True,
        "research_only":True,
        "provenance":provenance,
    }
    (destination/"result.json").write_text(json.dumps(record,indent=2,allow_nan=False))
    print("ASK_25_VALIDATION",json.dumps({"trial":index,"risk":scores}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--trial",required=True,type=int)
    p.add_argument("--original",default="training-cache/neural-ask/features.npz")
    p.add_argument("--embeddings",default="training-cache/bge-256")
    p.add_argument("--output",default="training-results/ask25")
    p.add_argument("--max-seconds",type=float,default=4200)
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    cfg=trial_config(args.trial)
    if args.dry_run:
        print(json.dumps({"trial":args.trial,"config":cfg}))
    else:
        train(args.trial,args.original,args.embeddings,args.output,args.max_seconds)


if __name__=="__main__":
    main()
