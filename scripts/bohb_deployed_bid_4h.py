"""Four-hour BOHB around the DEPLOYED full-non-Lemon DeepFM Bid model.

Ten simultaneous CPU trial slots: five independent BOHB search shards, with
two Ray trial workers per shard. Genuine BOHB = TuneBOHB + HyperBandForBOHB.
Maximum 250 training epochs per trial. NO Lemon trimming, NO bid blending,
NO original TEST scoring, NO deployment or altered serving weights.

Select by the inner 10%-of-TRAIN absolute dollar CVaR95; open original
validation once for each shard's selected checkpoint only, for reporting.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.ray_bid_9am import (
    BidNetwork, bound_config, read_source, prices, make_loss, _risk_exact,
)

EXPECTED_COHORT_SHA = "16290e5e7b7a2913edb47d2b7e34ffa6d4922bdd4269b729f705a53a4c135045"
EXPECTED_CHECKPOINT_SHA = "0526c178da90f737b6a204a039c65796432c7b29848b46e463d5bbe85db7037e"
SHARDS = 5
SLOTS_PER_SHARD = 2
MAX_EPOCHS = 250
METRIC = "inner_CVaR95_absolute_USD"
SEED = 546303


def reference(root):
    root = Path(root)
    p = root / "models/ray-bid-9am-winner/manifest.json"
    m = json.loads(p.read_text())
    weight = root / "models/ray-bid-9am-winner/weights.pt"
    observed = hashlib.sha256(weight.read_bytes()).hexdigest()
    if not (
        m.get("status") == "release_candidate_verified"
        and m.get("original_training_no_lemon") is True
        and m.get("no_blending") is True
        and m.get("original_test_scored") is False
        and m.get("cohort_sha256") == EXPECTED_COHORT_SHA
        and m.get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA == observed
        and m["configuration"]["family"] == "deepfm"
        and m["configuration"]["loss"] == "cvar95_weighted_huber"
    ):
        raise ValueError("Deployed FULL non-Lemon Bid identity changed")
    return m


def preflight(features, root):
    m = reference(root)
    z, meta, censor = read_source(features, with_bounds=True)
    if meta["cohort_sha256"] != EXPECTED_COHORT_SHA:
        raise ValueError("Research features differ from original deployed training source")
    if list(meta["cardinalities"]) != list(
        json.loads((Path(root)/"models/neural-experimental-bid/preprocessing.json").read_text())["cardinalities"]
    ):
        raise ValueError("Frozen category dictionary no longer matches live Bid model")
    phase = z["phase"].astype(str)
    train = np.flatnonzero(phase == "train")
    stop = max(512, int(len(train)*.1))
    if (len(train[:-stop]),len(train[-stop:])) != (345779,38419):
        raise ValueError("Original internal chronological train split changed")
    if (censor["explicit_intervals"] + censor["left_censored"]
            + censor["right_censored"]) != 0:
        raise ValueError("Study uses exact prices; verify any newly introduced censoring")
    if not meta.get("no_price_derived_inputs"):
        raise ValueError("Leaky model inputs")
    return m, meta


def space(tune, original, features):
    # Explicitly keep the deployed loss family and input contract.
    # Vary optimizer, FM factor rank, DeepFM geometry, robust objective,
    # regularization, training schedule and tail-risk emphasis.
    return {
        "features": str(Path(features).resolve()),
        "family": "deepfm",
        "loss": "cvar95_weighted_huber",
        "seed": SEED,
        "rank": tune.choice([16, 32]),
        "shape": tune.choice(["96x48x1", "128x64x2", "192x96x2",
                              "256x128x3", "128x64x3"]),
        "activation": tune.choice(["gelu", "silu"]),
        "dropout": tune.uniform(0.0, 0.30),
        "lr": tune.loguniform(0.00025, 0.0030),
        "wd": tune.loguniform(0.00001, 0.05),
        "batch": tune.choice([512, 1024, 2048]),
        "noise": tune.choice([0.0, 0.005, 0.012, 0.020]),
        "optimizer": tune.choice(["adamw", "radam"]),
        "schedule": tune.choice(["constant", "cosine", "step"]),
        "clip": tune.choice([0.5, 1.0, 2.0]),
        "beta": tune.choice([0.10, 0.20, 0.40, 0.60, 0.80]),
        "tail_frac": tune.choice([0.03, 0.05, 0.08, 0.10, 0.15]),
        "tail_weight": tune.uniform(0.05, 1.0),
        "usd_scale": tune.choice([5000.0, 10000.0, 20000.0, 30000.0]),
        # Not used by this deployed Huber loss; held fixed to avoid
        # wasting BOHB samples on nominally different, identical fits.
        "tau": float(original["tau"]),
        "sigma_floor": float(original["sigma_floor"]),
    }


def incumbent_point(manifest, features):
    c = dict(manifest["configuration"])
    c["features"] = str(Path(features).resolve())
    c["seed"] = SEED
    bound_config(c)
    return c


def _trainable_class():
    from scripts.ray_bid_9am import _trainable_class as original_class
    Base = original_class()

    class FullNonLemonBOHB(Base):
        def setup(self, config):
            if config["family"] != "deepfm" or config["loss"] != "cvar95_weighted_huber":
                raise ValueError("Only the deployed DeepFM/HUBER objective is admissible")
            super().setup(config)
            # One CPU core per training slot: 5 shards x 2 slots = 10 workers.
            torch.set_num_threads(1)

        def step(self):
            # Unlike the 09:00 Ray study, no hard-coded morning deadline.
            self.epoch += 1
            self.model.train()
            rng = torch.Generator().manual_seed(
                20261109 + int(self.c["seed"]) * 17 + self.epoch)
            indices = torch.randperm(len(self.y), generator=rng)
            start = time.monotonic()
            for batch in indices.split(int(self.c["batch"])):
                num = self.num[batch]
                if float(self.c["noise"]):
                    num = num + torch.randn_like(num) * float(self.c["noise"])
                self.optimizer.zero_grad(set_to_none=True)
                pred = self.model(num,self.cat[batch])
                loss = make_loss(pred,self.y[batch],
                    self.low[batch],self.high[batch],
                    self.model.sigma(self.c["sigma_floor"]),self.c)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         float(self.c["clip"]))
                self.optimizer.step()
            if self.schedule is not None:
                self.schedule.step()
            pred = prices(self.model,self.inner_num,self.inner_cat)
            risk = _risk_exact(self.inner_y,pred)
            value = float(risk["CVaR95_absolute_USD"])
            if not math.isfinite(value):
                raise FloatingPointError("BOHB CVaR95 is not finite")
            if value < self.best-0.05:
                self.best = value
                self.best_epoch = self.epoch
                self.best_state = {
                    k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()
                }
            print("BID_BOHB_4H_EPOCH",json.dumps({
                "trial":self.trial_id,"epoch":self.epoch,
                "inner_CVaR95_USD":round(value,2),
                "best_inner_CVaR95_USD":round(self.best,2),
                "inner_MAE_USD":round(float(risk["MAE_USD"]),2),
                "seconds":round(time.monotonic()-start,2),
                "family":"deepfm","loss":"cvar95_weighted_huber",
            }),flush=True)
            return {
                METRIC:value,
                "best_inner_CVaR95_absolute_USD":float(self.best),
                "inner_MAE_USD":float(risk["MAE_USD"]),
                "inner_RMSE_USD":float(risk["RMSE_USD"]),
                "best_epoch":int(self.best_epoch),
                "epochs_completed":int(self.epoch),
                "loss_family":self.c["loss"],"family":self.c["family"],
                "original_validation_scored":False,
                "original_test_scored":False,
            }

    return FullNonLemonBOHB


def smoke(features, root):
    from ray.tune.search.bohb import TuneBOHB
    from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
    from ray import tune
    manifest, meta = preflight(features, root)
    config = incumbent_point(manifest, features)
    search_space = space(tune,config,features)
    # Actual conversion to ConfigSpace is necessary: older Ray BOHB supplies
    # keyword q= when building ConfigSpace hyperparameters. This explicitly
    # rejects an incompatible ConfigSpace 1.x before a four-hour matrix runs.
    conversion_probe=TuneBOHB(metric=METRIC,mode="min",seed=20261109,
        points_to_evaluate=[config])
    if not conversion_probe.set_search_properties(METRIC,"min",search_space):
        raise RuntimeError("TuneBOHB could not register the complete search space")
    print("BID_BOHB_CONFIGSPACE_CONVERSION_OK",flush=True)
    # Actual import and construction gates for true Bayesian BOHB.
    scheduler = HyperBandForBOHB(time_attr="training_iteration",
        metric=METRIC,mode="min",max_t=MAX_EPOCHS,reduction_factor=3,
        stop_last_trials=False)
    searcher = TuneBOHB(metric=METRIC,mode="min",seed=20261109,
        points_to_evaluate=[config])
    if scheduler is None or searcher is None:
        raise AssertionError("Ray BOHB searcher/scheduler not constructed")
    torch.manual_seed(11)
    net = BidNetwork([8]*10,9.0,config)
    numeric = torch.randn(6,5)
    cats = torch.randint(0,7,(6,10))
    target = torch.randn(6)*.15+9
    value=make_loss(net(numeric,cats),target,target,target,
        net.sigma(config["sigma_floor"]),config)
    value.backward()
    if not bool(torch.isfinite(value)) or not all(
        p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()
    ):
        raise ValueError("Nonfinite BOHB training objective or gradients")
    print("BID_BOHB_PREFLIGHT_OK",json.dumps({
        "train":meta["full_train"],"validation":meta["validation"],
        "test":meta["test"],"workers":SHARDS*SLOTS_PER_SHARD,
        "max_epochs":MAX_EPOCHS,"search_algorithm":"TuneBOHB + HyperBandForBOHB",
        "deployed_reference_inner_CVaR95":manifest["retrained_internal_CVaR95_USD"]
    }),flush=True)


def run_shard(args):
    import ray
    from ray import tune
    from ray.tune.search.bohb import TuneBOHB
    from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
    from ray.tune.search import ConcurrencyLimiter

    if not 0 <= args.shard < SHARDS or not 60 <= args.budget_seconds <= 14400:
        raise ValueError("Invalid research shard or four-hour budget")
    report_dir=Path(args.output).resolve()
    report_dir.mkdir(parents=True,exist_ok=True)
    manifest, meta = preflight(args.features,args.root)
    fixed=incumbent_point(manifest,args.features)
    end_clock=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=args.budget_seconds)
    state={
        "status":"running","shard":args.shard,
        "algorithm":"Bayesian BOHB: TuneBOHB + HyperBandForBOHB",
        "concurrent_trials":2,"global_trial_slots":10,
        "max_epochs_per_trial":MAX_EPOCHS,
        "stop_at_utc":end_clock.isoformat(),
        "study_budget_seconds":int(args.budget_seconds),
        "dataset_sha256":meta["cohort_sha256"],
        "selected_on":"TRAIN chronological internal CVaR95 only",
        "production_modified":False,
        "original_test_scored":False,
        "original_validation_not_used_for_trial_selection":True,
        "no_lemon_filter":True,"no_xgboost_blending":True,
        "incumbent":{"reference_cv95_internal_USD":
                      float(manifest["retrained_internal_CVaR95_USD"]),
                     "reference_cv95_original_validation_USD":
                      float(manifest["original_validation"]["CVaR95_absolute_USD"])},
    }
    (report_dir/"status.json").write_text(json.dumps(state,indent=2)+"\n")
    # Ray's AF_UNIX sockets have a 107-byte Linux path limit; checkout paths
    # on GitHub Actions are long. Use a short /tmp root for Ray internals.
    short_ray_tmp=Path("/tmp")/("rb4h"+str(args.shard))
    short_ray_tmp.mkdir(parents=True,exist_ok=True)
    ray.init(num_cpus=2,include_dashboard=False,log_to_driver=True,
             _temp_dir=str(short_ray_tmp))
    try:
        scheduler=HyperBandForBOHB(
            time_attr="training_iteration",metric=METRIC,mode="min",
            max_t=MAX_EPOCHS,reduction_factor=3,stop_last_trials=False)
        bohb=TuneBOHB(metric=METRIC,mode="min",seed=20261109+args.shard*1009,
                      points_to_evaluate=[fixed])
        bohb=ConcurrencyLimiter(bohb,max_concurrent=2)
        tuner=tune.Tuner(
            tune.with_resources(_trainable_class(),{"cpu":1}),
            param_space=space(tune,fixed,args.features),
            tune_config=tune.TuneConfig(
                metric=METRIC,mode="min",
                scheduler=scheduler,search_alg=bohb,
                num_samples=2000,time_budget_s=args.budget_seconds,
            ),
            run_config=tune.RunConfig(
                name="four-hour-bid-bohb-shard-"+str(args.shard),
                storage_path=str((report_dir/"ray-experiment").resolve()),
                stop={"training_iteration":MAX_EPOCHS},
                failure_config=tune.FailureConfig(max_failures=1),
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_frequency=1,checkpoint_at_end=True,
                    num_to_keep=1,
                    checkpoint_score_attribute="best_inner_CVaR95_absolute_USD",
                    checkpoint_score_order="min"),
                verbose=1,
            ),
        )
        results=tuner.fit()
        candidates=[]
        for trial in results:
            if trial.error or trial.checkpoint is None or not trial.metrics:
                continue
            score=float(trial.metrics.get("best_inner_CVaR95_absolute_USD",math.inf))
            best_epoch=int(trial.metrics.get("best_epoch",0))
            if math.isfinite(score) and best_epoch>=1:
                candidates.append((score,trial))
        candidates.sort(key=lambda item:item[0])
        state.update(trials_reported=len(list(results)),
                     trials_with_valid_checkpoint=len(candidates),
                     finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        if not candidates:
            state["status"]="no_valid_trial"
            (report_dir/"status.json").write_text(json.dumps(state,indent=2)+"\n")
            raise RuntimeError("No BOHB trial produced a usable saved checkpoint")
        table=[]
        for score,trial in candidates[:50]:
            table.append({
                "best_inner_CVaR95_USD":score,
                "best_epoch":int(trial.metrics["best_epoch"]),
                "epochs_completed":int(trial.metrics.get("epochs_completed",0)),
                "configuration":dict(trial.config),
                "no_external_selection":True,
            })
        state["top_internal_trials"]=table
        state["best_inner_CVaR95_USD"]=candidates[0][0]
        state["status"]="complete"
        # Preserve the saved best STATE, not a weights artifact from the
        # last possibly-worse epoch (or a different non-deployed architecture).
        score, winner=candidates[0]
        with winner.checkpoint.as_directory() as folder:
            saved=torch.load(Path(folder)/"checkpoint.pt",
                             map_location="cpu",weights_only=False)
        if (saved.get("source_hash") != EXPECTED_COHORT_SHA or
                saved["config"].get("loss") != "cvar95_weighted_huber" or
                saved["config"].get("family") != "deepfm" or
                int(saved["best_epoch"]) < 1):
            raise ValueError("BOHB winner checkpoint does not match the training cohort")
        z,meta=read_source(args.features)
        phase=z["phase"].astype(str)
        exact_val=np.flatnonzero(phase=="validation")
        if len(exact_val)!=82380:
            raise ValueError("Original full validation count drifted")
        network=BidNetwork(meta["cardinalities"],
            float(np.median(np.log(z["y"][phase=="train"]))),winner.config)
        network.load_state_dict(saved["best_state"] or saved["state"],strict=True)
        observed=z["y"][exact_val].astype(np.float64)
        pred=prices(network,z["num"][exact_val],z["cat"][exact_val])
        validation=_risk_exact(observed,pred)
        state["winner_original_validation"]=validation
        state["winner_config"]=dict(winner.config)
        state["winner_best_epoch"]=int(saved["best_epoch"])
        state["winner_selected_on"]="TRAIN internal CVaR95, not original validation"
        (report_dir/"status.json").write_text(json.dumps(state,indent=2)+"\n")
        folder=report_dir/"selected"
        folder.mkdir(parents=True,exist_ok=True)
        torch.save({
            "state_dict":saved["best_state"] or saved["state"],
            "config":dict(winner.config),
            "cards":meta["cardinalities"],
            "source_sha256":meta["cohort_sha256"],
            "original_test_scored":False,
            "selected_by":"TRAIN internal CVaR95, not original validation",
        },folder/"weights.pt")
        (folder/"winner.json").write_text(json.dumps({
            "shard":args.shard,"internal_CVaR95_USD":score,
            "best_epoch":int(saved["best_epoch"]),
            "validation":validation,"configuration":dict(winner.config),
            "original_test_scored":False,"production_modified":False,
        },indent=2,allow_nan=False)+"\n")
        print("BID_BOHB_4H_SHARD_WINNER",json.dumps({
            "shard":args.shard,"inner_CVaR95_USD":round(score,2),
            "original_validation_CVaR95_USD":
                round(float(validation["CVaR95_absolute_USD"]),2),
            "best_epoch":int(saved["best_epoch"]),
            "original_test_scored":False,
        }),flush=True)
    finally:
        ray.shutdown()


def aggregate(root, output):
    root=Path(root)
    outputs=[]
    for p in sorted(root.glob("**/selected/winner.json")):
        r=json.loads(p.read_text())
        if r.get("original_test_scored") or r.get("production_modified"):
            raise ValueError("Research must not score TEST or mutate production")
        outputs.append(r)
    if not outputs:
        raise ValueError("No valid BOHB shard winners")
    outputs.sort(key=lambda v:v["internal_CVaR95_USD"])
    full={
        "status":"complete" if len(outputs)==SHARDS else "partial",
        "study":"2026-11-09 deployed full non-Lemon CVaR95 Huber DeepFM",
        "algorithm":"TuneBOHB + HyperBandForBOHB",
        "study_budget_seconds":14400,
        "max_concurrent_trainable_slots":10,
        "max_epochs":MAX_EPOCHS,
        "incumbent_reference":"models/ray-bid-9am-winner/manifest.json",
        "selected_by":"TRAIN-only internal CVaR95",
        "original_test_scored":False,
        "production_modified":False,
        "winner_by_train_internal":outputs[0],
        "all_shard_winners":outputs,
        "caution":"Original validation is reported but not used for winner selection; no deployment",
    }
    out=Path(output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(full,indent=2,allow_nan=False)+"\n")
    print("BID_BOHB_4H_ALL_RESULTS",json.dumps({
        "shards":len(outputs),
        "winner_internal_CVaR95":round(outputs[0]["internal_CVaR95_USD"],2),
        "winner_original_validation_CVaR95":round(
            outputs[0]["validation"]["CVaR95_absolute_USD"],2),
        "best_config":outputs[0]["configuration"],
        "production_modified":False,
    }),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features",
                   default="training-cache/neural-bid/features.npz")
    p.add_argument("--root",default=".")
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--budget-seconds",type=int,default=14400)
    p.add_argument("--output",
                   default="training-results/bohb-bid-4h/shard-0")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--aggregate")
    p.add_argument("--aggregate-output",default="reports/bohb-bid-4h/leaderboard.json")
    a=p.parse_args()
    if a.aggregate:
        aggregate(a.aggregate,a.aggregate_output)
    elif a.smoke:
        smoke(a.features,a.root)
    else:
        run_shard(a)


if __name__=="__main__":
    main()
