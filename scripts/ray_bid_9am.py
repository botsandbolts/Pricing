"""Ray Tune 09:00 Berlin full-cohort wholesale Bid risk search.

Research only. No Lemon row removal, blending, test scoring or deployment.
Uses original VIN-disjoint 2014/15 wholesale prices; only explicit provider
row-level bounds count as censoring. All search selection is TRAIN-only CVaR95.
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
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from scripts.bid_fm_scale_train import ScaledAuctionFM
from scripts.cvar95_historical import risk
from scripts.neural_bid_prepare import sha_ids

LOSSES = (
    "aft_gaussian", "aft_huber", "aft_loglogistic", "huber_log",
    "huber_usd", "quantile50", "smooth_quantile50", "mae_log",
    "mse_log", "logcosh_log", "charbonnier_log", "expectile_log",
    "cvar95_weighted_huber", "adaptive_tail_huber",
)
AFT_LOSSES = {"aft_gaussian", "aft_huber", "aft_loglogistic"}
EXPECTED = {"train": 384198, "validation": 82380, "test": 83603}
TZ = ZoneInfo("Europe/Berlin")
CEILING = math.log(1_000_000)
FLOOR = math.log(100)
SHAPES = {"96x48x1": (96, 48, 1), "128x64x2": (128, 64, 2),
          "192x96x2": (192, 96, 2), "256x128x3": (256, 128, 3),
          "128x64x3": (128, 64, 3)}


def bound_config(c):
    if c["loss"] not in LOSSES or c["family"] not in ("fm", "fieldgate", "hofm3", "deepfm"):
        raise ValueError("Unknown bounded bid objective or architecture")
    if int(c["rank"]) not in (16, 32) or c["shape"] not in SHAPES:
        raise ValueError("Unregistered safe architecture dimensions")
    if not (0 < float(c["tau"]) < 1 and .01 <= float(c["tail_frac"]) <= .15
            and 0 <= float(c["tail_weight"]) <= 1.0
            and 0 < float(c["lr"]) <= .003 and 0 <= float(c["wd"]) <= .05
            and 0 < float(c["beta"]) <= 1 and .02 <= float(c["sigma_floor"]) <= .2
            and int(c["batch"]) in (512, 1024, 2048)
            and 0 <= float(c["dropout"]) <= .30
            and 0 <= float(c["noise"]) <= .025
            and .4 <= float(c["clip"]) <= 2.0
            and 5000 <= float(c["usd_scale"]) <= 30000):
        raise ValueError("Hyperparameter out of proven bounded range")
    if c["optimizer"] not in ("adamw", "radam") or c["schedule"] not in (
            "constant", "cosine", "step"):
        raise ValueError("Invalid optimizer or scheduler")
    return dict(c)


class BidNetwork(nn.Module):
    def __init__(self, cards, init_price, config):
        super().__init__()
        self.core = ScaledAuctionFM(cards, config["family"], int(config["rank"]), init_price)
        if config["family"] == "deepfm":
            from scripts.bid_fm_kernel_train import FIELD_COUNT
            dim = FIELD_COUNT * int(config["rank"])
            w1, w2, depth = SHAPES[config["shape"]]
            act = nn.GELU if config["activation"] == "gelu" else nn.SiLU
            dims = [dim] + [w1] + [w2] * (depth - 1)
            blocks = [nn.LayerNorm(dim)]
            for before, after in zip(dims[:-1], dims[1:]):
                blocks.extend([nn.Linear(before, after), act(),
                               nn.Dropout(float(config["dropout"]))])
            blocks.append(nn.Linear(dims[-1], 1))
            nn.init.zeros_(blocks[-1].weight)
            nn.init.zeros_(blocks[-1].bias)
            self.core.deep = nn.Sequential(*blocks)
        self.sigma_raw = nn.Parameter(torch.tensor(-1.6, dtype=torch.float32))

    def forward(self, num, cat):
        return self.core(num, cat)

    def sigma(self, floor):
        return (F.softplus(self.sigma_raw) + float(floor)).clamp(min=.07, max=2.5)


def make_loss(mu, target_log, lower_log, upper_log, sigma, c):
    """True censor intervals for AFT only; never invent intervals or midpoints."""
    bound_config(c)
    exact = (lower_log == upper_log) & torch.isfinite(target_log)
    kind = c["loss"]
    diff = mu - target_log
    beta = float(c["beta"])
    exact_diff = diff[exact]
    if not bool(exact.any()):
        if kind not in AFT_LOSSES:
            raise ValueError("Point regression needs actual exact sale observations")
    robust = F.smooth_l1_loss(exact_diff, torch.zeros_like(exact_diff),
                              beta=beta) if bool(exact.any()) else mu.sum() * 0
    if kind in AFT_LOSSES:
        zlow = (lower_log - mu) / sigma
        zhigh = (upper_log - mu) / sigma
        if kind == "aft_loglogistic":
            nll_exact = F.softplus(zlow) + F.softplus(-zlow) + sigma.log()
            cdf_l = torch.sigmoid(zlow)
            cdf_u = torch.sigmoid(zhigh)
        else:
            nll_exact = .5 * zlow.square() + sigma.log()
            cdf_l = torch.special.ndtr(zlow)
            cdf_u = torch.special.ndtr(zhigh)
        p_interval = (cdf_u - cdf_l).clamp_min(1e-12)
        nll = torch.where(exact, nll_exact, -torch.log(p_interval)).mean()
        base = nll if kind != "aft_huber" else .60 * nll + .40 * robust
    else:
        if not bool(exact.any()):
            raise ValueError("No exact targets for non-AFT loss")
        a = exact_diff.abs()
        if kind == "huber_log":
            base = robust
        elif kind == "huber_usd":
            actual = torch.exp(target_log[exact])
            predicted = torch.exp(mu[exact].clamp(FLOOR, CEILING))
            dollars = (predicted - actual) / float(c["usd_scale"])
            base = F.smooth_l1_loss(dollars, torch.zeros_like(dollars), beta=.5)
        elif kind == "quantile50":
            base = .5 * a.mean()
        elif kind == "smooth_quantile50":
            base = .5 * (torch.sqrt(exact_diff.square() + .03**2) - .03).mean()
        elif kind == "mae_log":
            base = a.mean()
        elif kind == "mse_log":
            base = exact_diff.square().mean()
        elif kind == "logcosh_log":
            base = (a + F.softplus(-2 * a) - math.log(2)).mean()
        elif kind == "charbonnier_log":
            base = (torch.sqrt(exact_diff.square() + beta * beta) - beta).mean()
        elif kind == "expectile_log":
            tau = float(c["tau"])
            base = (torch.where(exact_diff >= 0, tau, 1-tau) *
                    exact_diff.square()).mean()
        elif kind in ("cvar95_weighted_huber", "adaptive_tail_huber"):
            actual = torch.exp(target_log[exact])
            pred = torch.exp(mu[exact].clamp(FLOOR, CEILING))
            usd = (pred - actual).abs() / float(c["usd_scale"])
            k = max(1, math.ceil(float(c["tail_frac"]) * len(usd)))
            if kind == "cvar95_weighted_huber":
                base = robust + float(c["tail_weight"]) * usd.topk(k).values.mean()
            else:
                cutoff = usd.detach().topk(k).values.min()
                weights = (1 + float(c["tail_weight"]) *
                           torch.clamp(usd.detach() / cutoff.clamp_min(.05), 0, 4))
                per = F.smooth_l1_loss(exact_diff, torch.zeros_like(exact_diff),
                                       beta=beta, reduction="none")
                base = (weights * per).sum() / weights.sum().clamp_min(1)
        else:
            raise ValueError("Unimplemented objective")
    if not bool(torch.isfinite(base).item()):
        raise FloatingPointError("Nonfinite bid training objective")
    return base


def _bounds(data, z):
    """Read only documented row-level provider bounds, never infer from round prices."""
    n = len(z["y"])
    exact = np.asarray(z["y"], dtype=np.float64)
    lower, upper = exact.copy(), exact.copy()
    p = Path(data).with_name("canonical-bid.parquet")
    summary = {"source": "observed exact wholesale transaction prices",
               "explicit_intervals": 0, "left_censored": 0,
               "right_censored": 0, "exact": n, "bounds_verified": False}
    if not p.exists():
        return lower, upper, summary
    import pyarrow.parquet as pq
    cols = set(pq.ParquetFile(p).schema_arrow.names)
    if not {"aft_lower_price", "aft_upper_price"}.issubset(cols):
        return lower, upper, summary
    frame = pd.read_parquet(p, columns=["row_id", "aft_lower_price", "aft_upper_price"])
    if len(frame) != n or not np.array_equal(frame.row_id.astype(str).to_numpy(),
                                               z["ids"].astype(str)):
        raise ValueError("Documented interval row IDs do not align with original cohort")
    low = frame.aft_lower_price.to_numpy(float)
    high = frame.aft_upper_price.to_numpy(float)
    if np.isnan(low).any() or np.isnan(high).any() or (low < 0).any() or (high <= 0).any() or (low > high).any():
        raise ValueError("Invalid documented AFT bounds")
    # Only exact vs well-defined intervals; zero is an open lower bound.
    exact_mask = np.isfinite(low) & (low == high)
    if not np.allclose(low[exact_mask], exact[exact_mask], rtol=1e-6, atol=.05):
        raise ValueError("Provider exact bounds disagree with observed sale target")
    lower, upper = low, high
    summary = {"source": "explicit aligned provider aft_lower_price/aft_upper_price",
               "explicit_intervals": int(((low < high) & (low > 0) & np.isfinite(high)).sum()),
               "left_censored": int((low == 0).sum()),
               "right_censored": int(np.isinf(high).sum()),
               "exact": int(exact_mask.sum()), "bounds_verified": True}
    return lower, upper, summary


def read_source(path, with_bounds=False):
    path = Path(path)
    m = json.loads(path.with_suffix(".json").read_text())
    if (m.get("market") != "bid" or not m.get("no_price_derived_inputs")
            or m.get("full_train") != EXPECTED["train"]
            or m.get("validation") != EXPECTED["validation"]
            or m.get("test") != EXPECTED["test"]):
        raise ValueError("Wrong frozen original full-cohort Bid evidence")
    with np.load(path, allow_pickle=False) as f:
        z = {k: f[k] for k in ("num", "cat", "y", "phase", "ids")}
    phase = z["phase"].astype(str)
    if len(phase) != 550181 or z["num"].shape != (550181,5) or z["cat"].shape != (550181,10):
        raise ValueError("Unexpected auction features or row counts")
    if len(np.unique(z["ids"])) != len(phase) or not np.isfinite(z["y"]).all() or (z["y"] <= 0).any():
        raise ValueError("Invalid or duplicate original observed auction labels")
    if (phase == "train").sum() != EXPECTED["train"]:
        raise ValueError("Full train was trimmed; Lemon exclusions forbidden")
    for split, count in EXPECTED.items():
        if int((phase == split).sum()) != count:
            raise ValueError("Partition count drift: " + split)
    for split in ("validation","test"):
        if sha_ids(z["ids"][phase == split]) != m["holdout_row_sha256"][split]:
            raise ValueError("Original holdout IDs differ: " + split)
    if with_bounds:
        low, high, summary = _bounds(path, z)
        z["lower_log"] = np.log(low).astype(np.float32)
        z["upper_log"] = np.log(high).astype(np.float32)
        return z, m, summary
    return z, m


@torch.no_grad()
def prices(model, num, cat, batch=8192):
    model.eval()
    out = []
    for b in range(0, len(num), batch):
        mu = model(torch.from_numpy(num[b:b+batch].astype(np.float32)),
                   torch.from_numpy(cat[b:b+batch].astype(np.int64)))
        out.append(np.exp(np.clip(mu.detach().numpy(), FLOOR, CEILING)))
    return np.concatenate(out).astype(np.float64)


def _risk_exact(y, p):
    d = risk(y, p)
    d["median_abs_error_USD"] = float(np.median(np.abs(y-p)))
    d["bias_USD"] = float(np.mean(p-y))
    return d


def smoke():
    cards = [8]*10
    num = torch.randn(32,5)
    cat = torch.randint(0, 7, (32,10))
    target = torch.randn(32)*.2+9.
    cfg = {"family":"deepfm","rank":16,"shape":"128x64x2",
           "activation":"gelu","dropout":.12,"optimizer":"adamw",
           "schedule":"constant","loss":"huber_log","beta":.25,
           "tau":.5,"tail_frac":.05,"tail_weight":.3,"lr":.001,
           "wd":.001,"batch":512,"noise":0.,"clip":1.0,
           "sigma_floor":.10,"usd_scale":10000}
    for family in ("fm", "fieldgate", "hofm3", "deepfm"):
        cfg["family"] = family
        net = BidNetwork(cards, 9.0, cfg)
        mu = net(num,cat)
        assert mu.shape == target.shape
        for loss in LOSSES:
            cfg["loss"] = loss
            net.zero_grad(set_to_none=True)
            result = make_loss(mu,target,target,target,net.sigma(cfg["sigma_floor"]),cfg)
            assert torch.isfinite(result), loss
            result.backward(retain_graph=True)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()), loss
        print("SMOKE_OK",family,len(LOSSES),flush=True)
    # Detect the exact previous PBT crash, tau escaping valid (0,1).
    cfg["tau"] = 1.2
    try:
        bound_config(cfg)
    except ValueError:
        print("INVALID_TAU_REJECTED",flush=True)
    else:
        raise AssertionError("Bad tau escaped its guard")
    return 0


def _trainable_class():
    from ray import tune

    class BidTrainable(tune.Trainable):
        def setup(self, config):
            self.c = bound_config(dict(config))
            torch.set_num_threads(2)
            torch.manual_seed(20261109 + int(config["seed"]))
            np.random.seed(20261109 + int(config["seed"]))
            z, meta, bounds = read_source(config["features"], with_bounds=True)
            phase = z["phase"].astype(str)
            training = np.flatnonzero(phase == "train")
            n_stop = max(512,int(len(training)*.1))
            fit, inner = training[:-n_stop], training[-n_stop:]
            assert (len(fit),len(inner)) == (345779,38419)
            self.source_hash = meta["cohort_sha256"]
            self.bounds = bounds
            self.num = torch.from_numpy(z["num"][fit].astype(np.float32))
            self.cat = torch.from_numpy(z["cat"][fit].astype(np.int64))
            self.y = torch.from_numpy(np.log(z["y"][fit]).astype(np.float32))
            self.low = torch.from_numpy(z["lower_log"][fit])
            self.high = torch.from_numpy(z["upper_log"][fit])
            exact_inner = z["lower_log"][inner] == z["upper_log"][inner]
            if int(exact_inner.sum()) < 500:
                raise ValueError("Too few genuinely observed internal sales for CVaR95")
            self.inner_num = z["num"][inner][exact_inner].astype(np.float32)
            self.inner_cat = z["cat"][inner][exact_inner].astype(np.int64)
            self.inner_y = z["y"][inner][exact_inner].astype(np.float64)
            self.model = BidNetwork(meta["cardinalities"],
                float(np.median(np.log(z["y"][fit]))), self.c)
            opt = torch.optim.AdamW if config["optimizer"] == "adamw" else torch.optim.RAdam
            self.optimizer = opt(self.model.parameters(), lr=float(config["lr"]),
                                 weight_decay=float(config["wd"]), foreach=False)
            self.schedule = None
            if config["schedule"] == "cosine":
                self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=250, eta_min=.00002)
            elif config["schedule"] == "step":
                self.schedule = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=36, gamma=.5)
            self.epoch=0
            self.best=math.inf
            self.best_epoch=0
            self.best_state=None

        def step(self):
            self.epoch+=1
            self.model.train()
            n=len(self.y)
            g=torch.Generator().manual_seed(20261109 + int(self.c["seed"])*17+self.epoch)
            idx=torch.randperm(n,generator=g)
            start=time.monotonic()
            for batch in idx.split(int(self.c["batch"])):
                if dt.datetime.now(TZ) >= dt.datetime(2026,11,9,8,56,tzinfo=TZ):
                    break
                num=self.num[batch]
                if float(self.c["noise"]):
                    num=num+torch.randn_like(num)*float(self.c["noise"])
                self.optimizer.zero_grad(set_to_none=True)
                pred=self.model(num,self.cat[batch])
                loss=make_loss(pred,self.y[batch],self.low[batch],self.high[batch],
                               self.model.sigma(self.c["sigma_floor"]),self.c)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),float(self.c["clip"]))
                self.optimizer.step()
            if self.schedule is not None:
                self.schedule.step()
            p=prices(self.model,self.inner_num,self.inner_cat)
            scores=_risk_exact(self.inner_y,p)
            value=float(scores["CVaR95_absolute_USD"])
            if value<self.best-.05:
                self.best=value
                self.best_epoch=self.epoch
                self.best_state={k:v.detach().cpu().clone()
                                 for k,v in self.model.state_dict().items()}
            print("BID_RAY_9AM_EPOCH",json.dumps({
                "trial":self.trial_id, "epoch":self.epoch, "loss":self.c["loss"],
                "family":self.c["family"],"CVaR95":round(value,2),
                "best_CVaR95":round(self.best,2),
                "MAE":round(float(scores["MAE_USD"]),2),
                "seconds":round(time.monotonic()-start,2)}),flush=True)
            return {"inner_CVaR95_absolute_USD":value,
                    "best_inner_CVaR95_absolute_USD":float(self.best),
                    "inner_MAE_USD":float(scores["MAE_USD"]),
                    "inner_RMSE_USD":float(scores["RMSE_USD"]),
                    "best_epoch":int(self.best_epoch),"epochs_completed":self.epoch,
                    "loss_family":self.c["loss"],"family":self.c["family"],
                    "test_scored":False,"validation_scored":False}

        def save_checkpoint(self, checkpoint_dir):
            target=Path(checkpoint_dir);target.mkdir(parents=True,exist_ok=True)
            torch.save({"state":self.model.state_dict(),
                "best_state":self.best_state,
                "optimizer":self.optimizer.state_dict(),
                "schedule":self.schedule.state_dict() if self.schedule else None,
                "epoch":self.epoch,"best_epoch":self.best_epoch,"best":self.best,
                "config":self.c,"source_hash":self.source_hash,
                "censoring":self.bounds},target/"checkpoint.pt")
            return str(target)

        def load_checkpoint(self, checkpoint_dir):
            pack=torch.load(Path(checkpoint_dir)/"checkpoint.pt",
                            map_location="cpu",weights_only=False)
            self.model.load_state_dict(pack["state"])
            self.optimizer.load_state_dict(pack["optimizer"])
            if self.schedule and pack["schedule"]:
                self.schedule.load_state_dict(pack["schedule"])
            self.epoch=int(pack["epoch"])
            self.best=float(pack["best"])
            self.best_epoch=int(pack["best_epoch"])
            self.best_state=pack["best_state"]

    return BidTrainable


def search(args):
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    import ray
    stop=dt.datetime(2026,11,9,8,55,tzinfo=TZ)
    seconds=int((stop-dt.datetime.now(TZ)).total_seconds())
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    z,meta,bounds=read_source(args.features,with_bounds=True)
    summary={"status":"not_started","shard":args.shard,"deadline_berlin":"2026-11-09T09:00+01:00",
        "training_budget_seconds":max(0,seconds),"loss_families":list(LOSSES),
        "cohort":"original FULL historical wholesale sales, not Lemon",
        "train_rows":EXPECTED["train"],"original_validation":EXPECTED["validation"],
        "original_test":EXPECTED["test"],"no_blending":True,
        "censoring_audit":bounds, "dataset_sha256":meta["cohort_sha256"],
        "validation_evaluated_once_per_loss_winner":True,"test_scored":False}
    if seconds < 120:
        summary["status"]="missed_deadline"
        (output/"summary.json").write_text(json.dumps(summary,indent=2))
        print("BID_RAY_DEADLINE_MISSED",json.dumps(summary),flush=True)
        return
    # Ten disjoint shards; first ten losses and last four all get coverage.
    candidates=[LOSSES[args.shard],LOSSES[(args.shard+10)%len(LOSSES)]]
    print("BID_RAY_SEARCH_BEGIN",json.dumps({"shard":args.shard,"losses":candidates,
        "max_minutes":round(seconds/60,1),"censoring":bounds}),flush=True)
    os.environ.setdefault("RAY_DEDUP_LOGS","0")
    ray.init(num_cpus=2,include_dashboard=False,log_to_driver=True)
    try:
        space = {
            "features":str(Path(args.features).resolve()),"seed":tune.randint(1,1000000),
            "loss":tune.grid_search(candidates),"family":tune.choice(
                ["fm","fieldgate","hofm3","deepfm"]),
            "rank":tune.choice([16,32]), "shape":tune.choice(list(SHAPES)),
            "activation":tune.choice(["gelu","silu"]),
            "dropout":tune.choice([.04,.10,.18,.25,.30]),
            "lr":tune.loguniform(.00025,.003),
            "wd":tune.loguniform(.00001,.05),
            "batch":tune.choice([512,1024,2048]),
            "noise":tune.choice([0.,.005,.012,.02]),
            "optimizer":tune.choice(["adamw","radam"]),
            "schedule":tune.choice(["constant","cosine","step"]),
            "clip":tune.choice([.5,1.,2.]),
            "beta":tune.choice([.1,.2,.4,.6]),
            "tau":tune.choice([.2,.35,.5,.65,.8]),
            "tail_frac":tune.choice([.03,.05,.08,.1]),
            "tail_weight":tune.choice([0.,.05,.15,.35,.6,1.]),
            "sigma_floor":tune.choice([.05,.10,.15]),
            "usd_scale":tune.choice([5000.,10000.,20000.,30000.]),
        }
        tuner=tune.Tuner(
            tune.with_resources(_trainable_class(),resources={"cpu":2}),
            param_space=space,
            tune_config=tune.TuneConfig(
                scheduler=ASHAScheduler(
                    time_attr="training_iteration",
                    metric="inner_CVaR95_absolute_USD",mode="min",
                    max_t=250,grace_period=4,reduction_factor=3),
                num_samples=120,max_concurrent_trials=1,
                time_budget_s=seconds),
            run_config=tune.RunConfig(
                name="bid-full-unfiltered-shard-"+str(args.shard),
                storage_path=str((output/"ray-study").resolve()),
                stop={"training_iteration":250},
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_frequency=1,checkpoint_at_end=True,
                    num_to_keep=1,
                    checkpoint_score_attribute="best_inner_CVaR95_absolute_USD",
                    checkpoint_score_order="min"),
                verbose=1))
        results=tuner.fit()
        allresults=list(results)
        summary["trials_total"]=len(allresults)
        summary["trials_failed"]=sum(bool(r.error) for r in allresults)
        summary["trials_ok"]=sum(not r.error for r in allresults)
        # Select only on internal TRAIN CVaR95, before opening original validation.
        chosen={}
        for r in allresults:
            if r.error or r.checkpoint is None or not r.metrics:
                continue
            loss=r.config["loss"]
            metric=float(r.metrics.get("best_inner_CVaR95_absolute_USD",float("inf")))
            if math.isfinite(metric) and (loss not in chosen or metric<chosen[loss][0]):
                chosen[loss]=(metric,r)
        val=np.flatnonzero(z["phase"].astype(str)=="validation")
        exact_val=(z["lower_log"][val]==z["upper_log"][val])
        val=val[exact_val]
        if len(val) < 500:
            raise ValueError("No valid original exact-price validation cohort")
        rows=[]
        for loss, (internal, r) in sorted(chosen.items()):
            with r.checkpoint.as_directory() as folder:
                saved=torch.load(Path(folder)/"checkpoint.pt",
                    map_location="cpu",weights_only=False)
            net=BidNetwork(meta["cardinalities"],
                float(np.median(np.log(z["y"][z["phase"].astype(str)=="train"]))),
                r.config)
            net.load_state_dict(saved["best_state"] or saved["state"])
            pred=prices(net,z["num"][val],z["cat"][val])
            metrics=_risk_exact(z["y"][val].astype(np.float64),pred)
            row={"loss":loss,"family":r.config["family"],
                 "inner_CVaR95_USD":internal,"best_epoch":int(saved["best_epoch"]),
                 "original_validation":metrics,"configuration":r.config,
                 "source_sha256":meta["cohort_sha256"],
                 "training_full_non_lemon":True,"test_scored":False}
            rows.append(row)
            out=output/("winner-"+loss)
            out.mkdir(parents=True,exist_ok=True)
            torch.save({"state_dict":saved["best_state"] or saved["state"],
                "config":r.config,"cards":meta["cardinalities"],
                "source_sha256":meta["cohort_sha256"],"censoring":bounds,
                "original_test_scored":False},out/"weights.pt")
            pd.DataFrame({"row_id":z["ids"][val].astype(str),
                "split":"validation","observed_price":z["y"][val],
                "predicted_price":pred}).to_csv(out/"validation.csv.gz",index=False)
            (out/"result.json").write_text(json.dumps(row,indent=2,allow_nan=False))
            print("BID_RAY_9AM_ORIGINAL_VALIDATION",json.dumps({
                "loss":loss,"shard":args.shard,"inner_CVaR95_USD":round(internal,2),
                "validation_CVaR95_USD":round(metrics["CVaR95_absolute_USD"],2),
                "validation_MAE_USD":round(metrics["MAE_USD"],2)}),flush=True)
        summary["status"]="complete" if rows else "no_completed_trial"
        summary["loss_winners"]=sorted(rows,
            key=lambda v:v["original_validation"]["CVaR95_absolute_USD"])
        (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    finally:
        ray.shutdown()


def aggregate(path,output):
    root=Path(path)
    files=sorted(root.glob("**/summary.json"))
    rows=[]
    audits=[]
    for f in files:
        entry=json.loads(f.read_text())
        audits.append({k:entry.get(k) for k in ("shard","status","trials_ok","trials_failed","censoring_audit")})
        rows.extend(entry.get("loss_winners",[]))
    rows.sort(key=lambda x:x["original_validation"]["CVaR95_absolute_USD"])
    report={"study":"Ray Tune FULL non-Lemon bid until 2026-11-09 09:00 CET",
            "metric":"CVaR95 absolute USD worst 5%","shards":audits,
            "loss_families_covered":sorted(set(r["loss"] for r in rows)),
            "leaderboard":rows,"validation_is_model_selection_not_test":True,
            "original_test_scored":False,"no_deployment":True}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(report,indent=2,allow_nan=False))
    print("BID_RAY_AGGREGATED",json.dumps({"shards":len(audits),
        "losses":report["loss_families_covered"],
        "best":rows[0] if rows else None},default=str),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--aggregate")
    p.add_argument("--shard",type=int,choices=range(10),default=0)
    p.add_argument("--features",default="training-cache/neural-bid/features.npz")
    p.add_argument("--output",default="training-results/ray-bid-9am")
    args=p.parse_args()
    if args.smoke:
        return smoke()
    if args.aggregate:
        return aggregate(args.aggregate,args.output)
    return search(args)


if __name__ == "__main__":
    main()
