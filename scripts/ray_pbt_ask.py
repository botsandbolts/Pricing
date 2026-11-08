"""Ray Tune Population-Based Training for historical Ask, native BGE embeddings only.

Two isolated architecture islands (ResMLP and DCNv2); each preserves shape
across PBT checkpoint inheritance. Ray tunes moving loss families, tail risk,
LR/decay, clipping, batch size, and dropout. Selection uses ONLY the last 14,777
TRAIN rows. One final incumbent comparison on ORIGINAL VALIDATION for the
within-training-selected winner; original test is NEVER scored or used.

Never auto-deploy. Production remains the previously verified Ask model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset

from scripts.ask25_joint_search import ResearchAskingNetwork, trial_config
from scripts.cvar95_historical import risk
from scripts.neural_ask_train import predict
from scripts.overnight_bohb_losses import regression_loss
from scripts.raw_embedding_ask_train import frozen_source
from scripts.raw_embedding_ask_encode import id_hash

SEED=20261108
ARCHITECTURES=("resmlp","deepcross")
LOSS_FAMILIES=("aft","huber_log","logcosh_log","hybrid_aft_dollar",
               "soft_cvar_dollar","asymmetric_dollar","hybrid_log_dollar")
MUTABLE=("lr","wd","batch","dropout","noise","clip","loss",
         "tail_strength","tail_fraction","tail_temperature","usd_scale",
         "beta","tau")
REQUIRED_COUNTS={"fit":133000,"inner":14777,"validation":31668,"test":31682}


def architecture_params(family):
    if family=="resmlp":
        result=trial_config(15).copy()
    elif family=="deepcross":
        result=trial_config(15).copy()
        result.update(family="dcnv2",project_dim=96,project_layers=2,
                      project_dropout=.10,cross_layers=3,rank=24,experts=2,
                      width=192,blocks=2,dropout=.12,activation="gelu")
    else:
        raise ValueError("Unregistered architecture: "+str(family))
    return result


def loss_hp(config):
    return {
        "loss":config["loss"],"beta":float(config["beta"]),
        "tau":float(config["tau"]),
        "tail_fraction":float(config["tail_fraction"]),
        "tail_strength":float(config["tail_strength"]),
        "usd_scale":float(config["usd_scale"]),
        "tail_temperature":float(config["tail_temperature"]),
        "sigma_floor":.035,"focal_gamma":1.5,
    }


def validate_config(config):
    if config.get("architecture") not in ARCHITECTURES:
        raise ValueError("Unregistered PBT architecture")
    if config["loss"] not in LOSS_FAMILIES:
        raise ValueError("Unsupported mutable loss")
    if not (0.00003<=config["lr"]<=.004
            and 0<=config["wd"]<=.3
            and config["batch"] in (384,512,768,1024)
            and 0<=config["dropout"]<=.4
            and 0<=config["noise"]<=.04
            and .2<=config["clip"]<=5):
        raise ValueError("Invalid PBT optimizer, noise or dropout mutation")
    # regression_loss validates remaining risk parameters.
    return True


def install_dynamic_hparams(model,optimizer,config):
    validate_config(config)
    for group in optimizer.param_groups:
        group["lr"]=float(config["lr"])
        group["weight_decay"]=float(config["wd"])
    # Shapes stay immutable; dropout probabilities can evolve in-place.
    for module in model.modules():
        if isinstance(module,nn.Dropout):
            module.p=float(config["dropout"])


def _safe_epoch_cvar(y,p):
    scores=risk(np.asarray(y,dtype=float),np.asarray(p,dtype=float))
    if not all(math.isfinite(float(scores[k])) for k in
               ("CVaR95_absolute_USD","MAE_USD","RMSE_USD")):
        raise RuntimeError("PBT internal error metrics are nonfinite")
    return scores


def _sha(path):
    with Path(path).open("rb") as fd:
        return hashlib.file_digest(fd,"sha256").hexdigest()


def _split_train(args):
    """Fingerprint complete source but expose TRAIN only to Ray actors."""
    z,vec,source,encoder=frozen_source(args["features"],args["embeddings"],"bge-256")
    phase=z["phase"].astype(str)
    tr=np.flatnonzero(phase=="train")
    fit=tr[:-14777]; inner=tr[-14777:]
    if (len(fit),len(inner),int((phase=="validation").sum()),
        int((phase=="test").sum()))!=(133000,14777,31668,31682):
        raise ValueError("Immutable Ask time split changed")
    values={}
    for side,index in (("fit",fit),("inner",inner)):
        values[side]={
            "numeric":torch.from_numpy(z["num"][index].astype(np.float32)),
            "category":torch.from_numpy(z["cat"][index].astype(np.int64)),
            "vector":torch.from_numpy(vec[index].astype(np.float32)),
            "log_target":torch.from_numpy(np.log(z["y"][index]).astype(np.float32)),
            "truth":z["y"][index].astype(np.float64),
        }
    meta={
        "cohort_sha256":source["cohort_sha256"],
        "holdout_row_sha256":source["holdout_row_sha256"],
        "embedding_sha256":encoder["embedding_file_sha256"],
        "encoder_revision":encoder["model_revision"],
        "cardinalities":source["cardinalities"],
        "fit_rows":133000,"train_only_inner_rows":14777,
        "original_validation_rows":31668,"original_test_rows":31682,
        "selection_uses_original_validation":False,
        "test_scored":False,
    }
    return values,meta


def _ray_trainable():
    """Import Ray lazily; ordinary compile/tests can run without Ray."""
    from ray import tune

    class AskPBT(tune.Trainable):
        def setup(self,config):
            validate_config(config)
            self.config=dict(config)
            self.architecture=config["architecture"]
            self.arch=architecture_params(self.architecture)
            # Restrict native Torch threading, particularly on two-trial CI.
            torch.set_num_threads(1)
            self.seed=SEED+int(config["seed"])
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            values,self.provenance=_split_train(config)
            self.fit=values["fit"]
            self.inner=values["inner"]
            self.epochs_per_step=int(config["epochs_per_step"])
            self.model=ResearchAskingNetwork(
                self.provenance["cardinalities"],384,self.arch)
            self.optimizer=torch.optim.AdamW(
                self.model.parameters(),lr=float(config["lr"]),
                weight_decay=float(config["wd"]),foreach=False)
            self.epoch=0
            self.best_inner=math.inf
            self.best_epoch=0
            self.best_weights=None
            self.best_config=None
            install_dynamic_hparams(self.model,self.optimizer,self.config)

        def reset_config(self,new_config):
            # PBT exploitation copies the donor's full configuration, seed
            # included. Incompatible architecture is the ONLY reason to
            # reject reset; donor RNG progression must also be inherited.
            if new_config["architecture"]!=self.architecture:
                return False
            self.config=dict(new_config)
            self.seed=SEED+int(new_config["seed"])
            install_dynamic_hparams(self.model,self.optimizer,self.config)
            return True

        def step(self):
            install_dynamic_hparams(self.model,self.optimizer,self.config)
            hp=loss_hp(self.config)
            fit=self.fit
            b=int(self.config["batch"])
            count=fit["numeric"].shape[0]
            for _ in range(self.epochs_per_step):
                self.epoch+=1
                self.model.train()
                generator=torch.Generator().manual_seed(self.seed+self.epoch*719)
                order=torch.randperm(count,generator=generator)
                for sub in order.split(b):
                    n=fit["numeric"][sub]
                    if float(self.config["noise"])>0:
                        n=n+torch.randn_like(n)*float(self.config["noise"])
                    raw=self.model(n,fit["category"][sub],fit["vector"][sub])
                    loss=regression_loss(raw,fit["log_target"][sub],hp,"ask")
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),float(self.config["clip"]))
                    self.optimizer.step()
            self.model.eval()
            inside=self.inner
            ds=TensorDataset(inside["numeric"],inside["category"],
                             inside["vector"],inside["log_target"])
            p=predict(self.model,ds,np.arange(len(inside["truth"])),1024)
            score=_safe_epoch_cvar(inside["truth"],p)
            value=float(score["CVaR95_absolute_USD"])
            if value<self.best_inner-.05:
                self.best_inner=value
                self.best_epoch=self.epoch
                self.best_weights={name:v.detach().clone()
                                   for name,v in self.model.state_dict().items()}
                self.best_config={k:self.config[k] for k in MUTABLE}
            print("ASK_RAY_PBT_EPOCH",json.dumps({
                "trial_id":self.trial_id,"family":self.architecture,
                "epoch":self.epoch,"cvar95":round(value,2),
                "mae":round(float(score["MAE_USD"]),2),
                "best_inner":round(self.best_inner,2),
                "loss":self.config["loss"],"lr":self.config["lr"]}),
                flush=True)
            return {
                "inner_CVaR95_absolute_USD":value,
                "inner_MAE_USD":float(score["MAE_USD"]),
                "inner_RMSE_USD":float(score["RMSE_USD"]),
                "best_inner_CVaR95_absolute_USD":float(self.best_inner),
                "best_epoch":int(self.best_epoch),
                "epochs_completed":int(self.epoch),
                "loss_family":self.config["loss"],
                "architecture":self.architecture,
                "validation_scored":False,"test_scored":False,
            }

        def save_checkpoint(self,checkpoint_dir):
            folder=Path(checkpoint_dir)
            folder.mkdir(parents=True,exist_ok=True)
            torch.save({
                "state_dict":self.model.state_dict(),
                "optimizer":self.optimizer.state_dict(),
                "epoch":self.epoch,"best_inner":self.best_inner,
                "best_epoch":self.best_epoch,
                "best_weights":self.best_weights,
                "best_config":self.best_config,
                "architecture":self.architecture,
                "cohort_sha256":self.provenance["cohort_sha256"],
                "embedding_sha256":self.provenance["embedding_sha256"],
            },folder/"model.pt")
            return str(folder)

        def load_checkpoint(self,checkpoint_dir):
            pack=torch.load(Path(checkpoint_dir)/"model.pt",
                            map_location="cpu",weights_only=True)
            if (pack["architecture"]!=self.architecture
                or pack["cohort_sha256"]!=self.provenance["cohort_sha256"]
                or pack["embedding_sha256"]!=self.provenance["embedding_sha256"]):
                raise ValueError("PBT exploit attempted incompatible weights/cohort")
            self.model.load_state_dict(pack["state_dict"],strict=True)
            self.optimizer.load_state_dict(pack["optimizer"])
            self.epoch=int(pack["epoch"])
            self.best_inner=float(pack["best_inner"])
            self.best_epoch=int(pack["best_epoch"])
            self.best_weights=pack["best_weights"]
            self.best_config=pack["best_config"]
            # Offspring gets inherited momenta, but its mutated learning
            # rate / regularization / loss are applied after optimizer restore.
            install_dynamic_hparams(self.model,self.optimizer,self.config)

    return AskPBT


def _pbt_space(tune,architecture,args):
    return {
        "architecture":architecture,
        "features":str(Path(args.features).resolve()),
        "embeddings":str(Path(args.embeddings).resolve()),
        "seed":tune.randint(1,1000000),
        "epochs_per_step":int(args.epochs_per_step),
        "lr":tune.loguniform(8e-5,1.8e-3),
        "wd":tune.loguniform(3e-4,.075),
        "batch":tune.choice([384,512,768,1024]),
        "dropout":tune.uniform(.06,.30),
        "noise":tune.choice([0.,.004,.008,.015]),
        "clip":tune.choice([.5,1.,1.5,2.]),
        "loss":tune.choice(list(LOSS_FAMILIES)),
        "tail_strength":tune.uniform(.02,1.2),
        "tail_fraction":tune.uniform(.035,.16),
        "tail_temperature":tune.uniform(.025,.3),
        "usd_scale":tune.choice([5000.,10000.,15000.,20000.]),
        "beta":tune.uniform(.1,.65),
        "tau":tune.uniform(.25,.75),
    }


def _pbt_mutations(tune):
    return {
        "lr":tune.loguniform(5e-5,2.5e-3),
        "wd":tune.loguniform(1e-4,.11),
        "batch":[384,512,768,1024],
        "dropout":tune.uniform(.04,.32),
        "noise":[0.,.004,.008,.015],
        "clip":[.5,1.,1.5,2.],
        "loss":list(LOSS_FAMILIES),
        "tail_strength":tune.uniform(.02,1.5),
        "tail_fraction":tune.uniform(.035,.20),
        "tail_temperature":tune.uniform(.025,.35),
        "usd_scale":[5000.,10000.,15000.,20000.],
        "beta":tune.uniform(.1,.8),
        "tau":tune.uniform(.25,.75),
    }


def publish_inner_selected(args,checkpoint_dir,result):
    """One and only one ORIGINAL-VALIDATION scoring for TRAIN-only winner."""
    architecture=args.architecture
    pack=torch.load(Path(checkpoint_dir)/"model.pt",
                    map_location="cpu",weights_only=True)
    if pack["best_weights"] is None or pack["best_epoch"]<3:
        raise ValueError("No complete trained and checkpointed PBT winner")
    z,emb,source,enc=frozen_source(args.features,args.embeddings,"bge-256")
    if pack["cohort_sha256"]!=source["cohort_sha256"]:
        raise ValueError("Winner inherited wrong historical cohort")
    if pack["embedding_sha256"]!=enc["embedding_file_sha256"]:
        raise ValueError("Winner inherited wrong pinned frozen BGE")
    phase=z["phase"].astype(str)
    valid=np.flatnonzero(phase=="validation")
    if len(valid)!=31668:
        raise ValueError("External historical validation split changed")
    model=ResearchAskingNetwork(source["cardinalities"],384,
                                architecture_params(architecture))
    model.load_state_dict(pack["best_weights"],strict=True)
    model.eval()
    ds=TensorDataset(torch.from_numpy(z["num"][valid].astype(np.float32)),
        torch.from_numpy(z["cat"][valid].astype(np.int64)),
        torch.from_numpy(emb[valid].astype(np.float32)),
        torch.from_numpy(np.log(z["y"][valid]).astype(np.float32)))
    validation_pred=predict(model,ds,np.arange(len(valid)),1024)
    scores=_safe_epoch_cvar(z["y"][valid],validation_pred)
    folder=Path(args.output).resolve()
    folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({
        "row_id":z["ids"][valid].astype(str),
        "split":"validation",
        "observed_price":z["y"][valid].astype(float),
        "predicted_price":validation_pred,
    }).to_csv(folder/"validation.csv.gz",index=False)
    checkpoint={
        "state_dict":pack["best_weights"],
        "architecture":architecture_params(architecture),
        "moving_hparams_at_best_epoch":pack["best_config"],
        "best_epoch":pack["best_epoch"],
        "train_internal_CVaR95_absolute_USD":pack["best_inner"],
        "cohort_sha256":source["cohort_sha256"],
        "holdout_row_sha256":source["holdout_row_sha256"],
        "frozen_bge_sha256":enc["embedding_file_sha256"],
        "source":"Ray Tune PBT original TRAIN only",
        "test_evaluated":False,
        "automatic_deployment":False,
    }
    torch.save(checkpoint,folder/"weights.pt")
    report={
        "status":"complete",
        "study":"ray-tune-population-based-training",
        "population":int(args.population),
        "architecture":architecture,
        "training_steps":int(args.steps),
        "epochs_per_step":int(args.epochs_per_step),
        "selected_using":"last 14,777 original TRAIN rows only",
        "selected_from_trial":result.path,
        "selected_best_epoch":int(pack["best_epoch"]),
        "trained_epochs_at_last_checkpoint":int(pack["epoch"]),
        "selected_train_internal_CVaR95_USD":float(pack["best_inner"]),
        "selected_mutable_hyperparameters":pack["best_config"],
        "external_original_validation":scores,
        "incumbent_original_validation_CVaR95_USD":18069.284887157904,
        "provenance":{
            "cohort_sha256":source["cohort_sha256"],
            "embedding_file_sha256":enc["embedding_file_sha256"],
            "holdout_row_sha256":source["holdout_row_sha256"],
            "original_validation_ids_sha256":id_hash(z["ids"][valid].astype(str)),
            "train_fit_rows":133000,"train_only_stopper_rows":14777,
            "external_validation_rows":31668,"test_rows_not_scored":31682,
        },
        "original_test_scored":False,
        "original_test_used_for_selection":False,
        "autodeploy":False,
        "native_pytorch_checkpoint":True,
        "reuse_native_bge_inference":True,
        "production_unchanged":True,
    }
    (folder/"result.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print("RAY_PBT_ASK_ORIGINAL_VALIDATION",json.dumps({
        "architecture":architecture,"validation":scores,
        "train_internal":float(pack["best_inner"]),
        "best_epoch":int(pack["best_epoch"]),
        "test_scored":False}),flush=True)
    return report


def run(args):
    from ray import tune,train
    import ray
    from ray.tune.schedulers import PopulationBasedTraining
    validate_run_settings(args)
    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    folder=Path(args.output).resolve()
    folder.mkdir(parents=True,exist_ok=True)
    # Explicit CPU ceiling prevents actor oversubscription and OOM on Actions.
    ray.init(num_cpus=int(args.workers),include_dashboard=False,
             log_to_driver=True)
    try:
        scheduler=PopulationBasedTraining(
            time_attr="training_iteration",metric="inner_CVaR95_absolute_USD",
            mode="min",perturbation_interval=int(args.perturb_steps),
            burn_in_period=int(args.perturb_steps),
            quantile_fraction=.33,resample_probability=.25,
            hyperparam_mutations=_pbt_mutations(tune),
            log_config=True,synch=False,
        )
        tuner=tune.Tuner(
            tune.with_resources(_ray_trainable(),resources={"cpu":1}),
            param_space=_pbt_space(tune,args.architecture,args),
            tune_config=tune.TuneConfig(
                scheduler=scheduler,num_samples=int(args.population),
                max_concurrent_trials=int(args.workers)),
            run_config=train.RunConfig(
                name="ray-pbt-ask-"+args.architecture,
                storage_path=str((folder/"ray-study").resolve()),
                stop={"training_iteration":int(args.steps)},
                failure_config=train.FailureConfig(max_failures=0),
                checkpoint_config=train.CheckpointConfig(
                    checkpoint_frequency=1,checkpoint_at_end=True,
                    num_to_keep=2,checkpoint_score_attribute=
                    "best_inner_CVaR95_absolute_USD",checkpoint_score_order="min"),
                verbose=1,
            ),
        )
        results=tuner.fit()
        failed=[r for r in results if r.error is not None]
        if failed:
            raise RuntimeError(f"{len(failed)} Ray Tune members failed; do not report partial study")
        best=results.get_best_result(metric="best_inner_CVaR95_absolute_USD",
                                     mode="min",scope="last")
        if best.checkpoint is None:
            raise RuntimeError("PBT best model has no inherited checkpoint")
        with best.checkpoint.as_directory() as cp:
            report=publish_inner_selected(args,cp,best)
        return report
    finally:
        ray.shutdown()


def validate_run_settings(args):
    if args.architecture not in ARCHITECTURES:
        raise ValueError("Unknown Ask island")
    if not 4<=args.population<=12 or not 1<=args.workers<=4:
        raise ValueError("Invalid population or CPU cap")
    if not 1<=args.epochs_per_step<=6 or not 1<=args.perturb_steps<=args.steps:
        raise ValueError("Invalid PBT scheduling cadence")
    if not 4<=args.steps<=50 or args.population<2*args.workers:
        raise ValueError("PBT needs a population large enough to evolve")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--architecture",required=True,choices=ARCHITECTURES)
    p.add_argument("--features",default="training-cache/neural-ask/features.npz")
    p.add_argument("--embeddings",default="training-cache/bge-256")
    p.add_argument("--output",required=True)
    p.add_argument("--population",type=int,default=6)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--steps",type=int,default=18)
    p.add_argument("--epochs-per-step",type=int,default=3)
    p.add_argument("--perturb-steps",type=int,default=3)
    a=p.parse_args()
    run(a)


if __name__=="__main__":
    main()
