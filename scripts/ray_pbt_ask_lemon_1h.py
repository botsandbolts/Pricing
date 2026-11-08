"""One-hour, test-isolated Ray PBT on frozen 75%-retained Lemon Ask rows.

The pre-existing Lemon mask was determined from group-OOF XGBoost residuals
in a separate, earlier study, without using original validation or test.
It is frozen in neural-ask-frozen-features and joined by canonical row order.
On the retained 110,832 TRAIN rows, the last 10% is the Ray TRAIN-only
stopper; the earlier 90% is used to fit neural models. These are conditional
internal scores, not an unbiased nested evaluation of the cleaning choice.
Original validation is scored once on a training-selected finalist.
Original test is never loaded into an actor or evaluated, and no deployment
takes place. Network weights are freshly initialized, not inherited from
the failed overnight five-hour study or the ten-minute experiment.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import scripts.ray_pbt_ask as core
from scripts.ray_pbt_ask_10m import patch_targeted_search,valid_neighborhood

TOTAL_TRAIN=147777
RETAINED_TRAIN=110832
ORIGINAL_VALIDATION=31668
ORIGINAL_TEST=31682
COHORT="frozen-lemon-oof-retained-75"
TRIMMED_TRAIN=TOTAL_TRAIN-RETAINED_TRAIN


def lemon_indices(phase, mask):
    phase=np.asarray(phase).astype(str)
    mask=np.asarray(mask)
    if mask.dtype != np.dtype("bool") or mask.shape != phase.shape:
        raise ValueError("Frozen Lemon mask must be a boolean vector matching canonical rows")
    original=np.flatnonzero(phase=="train")
    validation=np.flatnonzero(phase=="validation")
    test=np.flatnonzero(phase=="test")
    if (len(original),len(validation),len(test))!=(
        TOTAL_TRAIN,ORIGINAL_VALIDATION,ORIGINAL_TEST):
        raise ValueError("Original chronological split changed")
    if not mask[validation].all() or not mask[test].all():
        raise ValueError("Original validation and test must not be filtered")
    retained=original[mask[original]]
    if len(retained)!=RETAINED_TRAIN or len(original)-len(retained)!=TRIMMED_TRAIN:
        raise ValueError("Frozen OOF Lemon trim is not exactly the audited retained 75%")
    stopper=max(512,int(len(retained)*.10))
    fit,inside=retained[:-stopper],retained[-stopper:]
    if len(fit)<50000 or len(inside)<5000:
        raise ValueError("Insufficient Lemon training or internal stopper rows")
    if not (set(fit).isdisjoint(inside) and
            set(retained).isdisjoint(validation) and
            set(retained).isdisjoint(test)):
        raise ValueError("Train, internal, validation or test leakage")
    return fit,inside


def frozen_lemon_data(config):
    # Original frozen_source validates source, chronological holdout hashes,
    # embedding SHA and 384D BGE provenance before returning any feature data.
    z,vec,source,encoder=core.frozen_source(
        config["features"],config["embeddings"],"bge-256")
    with np.load(config["features"],allow_pickle=False) as stored:
        if "lemon" not in stored.files:
            raise ValueError("Missing frozen OOF Lemon selection mask")
        lemon=stored["lemon"]
    fit,inside=lemon_indices(z["phase"],lemon)
    values={}
    for side,index in (("fit",fit),("inner",inside)):
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
        "fit_rows":int(len(fit)),
        "train_only_inner_rows":int(len(inside)),
        "retained_train_rows":RETAINED_TRAIN,
        "trimmed_train_rows":TRIMMED_TRAIN,
        "original_validation_rows":ORIGINAL_VALIDATION,
        "original_test_rows":ORIGINAL_TEST,
        "selection_uses_original_validation":False,
        "test_scored":False,
    }
    return values,meta


def lemon_class():
    # Inherit real Ray Tune Trainable checkpoint/restore/optimizer/PBT
    # mechanics; only the training-data setup is replaced.
    Base=core._ray_trainable_original()
    class LemonAskPBT(Base):
        def setup(self,config):
            core.validate_config(config)
            self.config=dict(config)
            self.architecture=config["architecture"]
            self.arch=core.architecture_params(self.architecture)
            torch.set_num_threads(1)
            self.seed=core.SEED+int(config["seed"])
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            values,self.provenance=frozen_lemon_data(config)
            self.fit=values["fit"]
            self.inner=values["inner"]
            self.epochs_per_step=int(config["epochs_per_step"])
            self.model=core.ResearchAskingNetwork(
                self.provenance["cardinalities"],384,self.arch)
            self.optimizer=torch.optim.AdamW(
                self.model.parameters(),lr=float(config["lr"]),
                weight_decay=float(config["wd"]),foreach=False)
            self.epoch=0
            self.best_inner=math.inf
            self.best_epoch=0
            self.best_weights=None
            self.best_config=None
            core.install_dynamic_hparams(self.model,self.optimizer,self.config)
    return LemonAskPBT


def publish_lemon(args,checkpoint_dir,result):
    # Parent verifies canonical validation (31,668 unique IDs) and that
    # test was not used. Correct its full-training metadata for this variant.
    report=core._publish_inner_selected_original(args,checkpoint_dir,result)
    retained_inner=max(512,int(RETAINED_TRAIN*.10))
    fit_rows=RETAINED_TRAIN-retained_inner
    report["study"]="ray-tune-population-based-training-frozen-lemon-75"
    report["training_cohort"]=COHORT
    report["selected_using"]=(
        "Last 10% of frozen retained original TRAIN rows only;"
        " selection conditional on earlier OOF Lemon trimming")
    report["provenance"].update({
        "train_fit_rows":fit_rows,
        "train_only_stopper_rows":retained_inner,
        "original_train_rows":TOTAL_TRAIN,
        "lemon_retained_train_rows":RETAINED_TRAIN,
        "lemon_removed_train_rows":TRIMMED_TRAIN,
        "original_validation_rows":ORIGINAL_VALIDATION,
        "original_test_rows":ORIGINAL_TEST,
        "lemon_preexisting_group_oof_selection":True,
        "validation_test_filtered":False,
    })
    report["lemon_selection_caveat"]=(
        "A previously frozen train-only OOF residual filter used labels"
        " to choose training membership; internal scores are conditional"
        " on that selection, not unbiased nested CV."
        " Compare finalists only on the entire original validation set.")
    report["production_unchanged"]=True
    report["original_test_scored"]=False
    report["autodeploy"]=False
    root=Path(args.output)
    (root/"result.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    weight_file=root/"weights.pt"
    model=torch.load(weight_file,map_location="cpu",weights_only=True)
    model["training_cohort"]=COHORT
    model["retained_original_train_rows"]=RETAINED_TRAIN
    model["fit_train_rows"]=fit_rows
    model["internal_stopper_rows"]=retained_inner
    model["original_test_scored"]=False
    torch.save(model,weight_file)
    print("LEMON_75_RAY_ORIGINAL_VALIDATION",json.dumps({
        "architecture":args.architecture,
        "fit_rows":fit_rows,"train_inner_rows":retained_inner,
        "original_validation":report["external_original_validation"],
        "train_internal_CVaR95":report["selected_train_internal_CVaR95_USD"],
        "test_scored":False,"automatic_deployment":False,
    }),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture",required=True,choices=core.ARCHITECTURES)
    p.add_argument("--features",default="training-cache/neural-ask/features.npz")
    p.add_argument("--embeddings",default="training-cache/bge-256")
    p.add_argument("--output",required=True)
    p.add_argument("--population",type=int,default=6)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--steps",type=int,default=1800)
    p.add_argument("--epochs-per-step",type=int,default=2)
    p.add_argument("--perturb-steps",type=int,default=2)
    p.add_argument("--time-budget-seconds",type=int,default=3600)
    args=p.parse_args()
    if args.time_budget_seconds!=3600:
        raise ValueError("This experiment requires a full 3,600-second Ray budget")
    valid_neighborhood(args.architecture)
    # Proven safe in the successful ten-minute study: categorical-only PBT
    # mutations prevent tau>1, invalid tail fractions and unsafe scaling.
    patch_targeted_search(args.architecture)
    core._ray_trainable_original=core._ray_trainable
    core._publish_inner_selected_original=core.publish_inner_selected
    core._ray_trainable=lemon_class
    core.publish_inner_selected=publish_lemon
    print("LEMON_75_RAY_CONFIG",json.dumps({
        "architecture":args.architecture,
        "full_train":TOTAL_TRAIN,
        "retained_train":RETAINED_TRAIN,
        "removed":TRIMMED_TRAIN,
        "fit_rows":RETAINED_TRAIN-max(512,int(RETAINED_TRAIN*.1)),
        "internal_training_stopper_rows":max(512,int(RETAINED_TRAIN*.1)),
        "ray_budget_seconds":3600,"population":args.population,
        "frozen_raw_bge_embeddings":True,
        "original_validation_and_test_unfiltered":True,
        "original_test_evaluated":False,
        "automatic_model_deployment":False,
    }),flush=True)
    return core.run(args)


if __name__=="__main__":
    main()
