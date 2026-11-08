"""Ten-minute targeted Ray Tune PBT around the prior five-hour Ask island winners.

The failed five-hour study did not publish usable model weights. We therefore
REFIT new native PyTorch models (not checkpoints from that study), seeding the
search near the best internal holdout configurations:
    ResMLP: AFT / hybrid log-dollar, LR ~1.7e-3 or ~6.8e-5.
    DeepCross: AFT, LR ~2.8e-4.
PBT config mutations are CATEGORICAL only, strictly inside the proven ranges
of the underlying loss, notably 0 < tau < 1 and .01 <= tail_fraction <= .30.
This prevents Ray's previous multiplicative mutation pushing tau above 1.

The original 2021 133000 / 14777 / 31668 / 31682 split, immutable BGE-256
vectors, research-only status, and external test isolation are preserved.
"""
import argparse
import json
from pathlib import Path

import scripts.ray_pbt_ask as core


def nearby(architecture):
    if architecture=="resmlp":
        return dict(
            lr=[.000065,.000085,.00018,.00065,.00140,.00172],
            wd=[.0002,.002,.019,.049],
            loss=["aft","hybrid_log_dollar","hybrid_aft_dollar"],
            dropout=[.025,.04,.065,.095],
            batch=[512,768,1024],
            noise=[0.,.004,.008],
            clip=[.7,1.,1.5],
        )
    if architecture=="deepcross":
        return dict(
            lr=[.00019,.00023,.00028,.00035,.0005],
            wd=[.0006,.0012,.003,.0052],
            loss=["aft","logcosh_log","hybrid_aft_dollar"],
            dropout=[.06,.10,.14],
            batch=[512,768,1024],
            noise=[0.,.004,.008],
            clip=[.5,1.,1.5],
        )
    raise ValueError("Unknown targeted architecture "+str(architecture))


def safe_options(architecture):
    choices=nearby(architecture).copy()
    choices.update(
        tail_strength=[0.,.10,.25,.50,.70],
        tail_fraction=[.05,.08,.12,.16],
        tail_temperature=[.04,.10,.20],
        usd_scale=[5000.,10000.,15000.,20000.],
        beta=[.15,.30,.50,.70],
        tau=[.35,.50,.65,.75],
    )
    return choices


def valid_neighborhood(architecture):
    options=safe_options(architecture)
    if set(options)!=set(core.MUTABLE):
        raise ValueError("Targeted PBT must mutate exactly known safe loss dimensions")
    for name,items in options.items():
        if not isinstance(items,list) or not items:
            raise ValueError("All PBT mutations must be finite categorical lists")
        if name=="tau" and not all(0<x<1 for x in items):
            raise ValueError("Invalid tau, must be inside (0,1)")
        if name=="tail_fraction" and not all(.01<=x<=.30 for x in items):
            raise ValueError("Invalid CVaR fraction")
        if name=="tail_strength" and not all(0<=x<=2 for x in items):
            raise ValueError("Invalid loss strength")
        if name=="tail_temperature" and not all(.005<=x<=2 for x in items):
            raise ValueError("Invalid tail temperature")
        if name=="usd_scale" and not all(1000<=x<=50000 for x in items):
            raise ValueError("Invalid dollar normalization")
    return options


def patch_targeted_search(architecture):
    options=valid_neighborhood(architecture)
    old_space=core._pbt_space

    def centered_space(tune,family,args):
        if family!=architecture:
            raise ValueError("Cross-architecture focus not allowed")
        parameters=old_space(tune,family,args)
        for key,items in options.items():
            parameters[key]=tune.choice(items)
        return parameters

    def bounded_mutations(tune):
        # Crucial: ALL categorical. No tune.uniform/loguniform here;
        # Ray PBT's 0.8x / 1.2x fallback never leaves valid domains.
        return options

    core._pbt_space=centered_space
    core._pbt_mutations=bounded_mutations


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--architecture",required=True,choices=core.ARCHITECTURES)
    p.add_argument("--features",default="training-cache/neural-ask/features.npz")
    p.add_argument("--embeddings",default="training-cache/bge-256")
    p.add_argument("--output",required=True)
    p.add_argument("--population",type=int,default=4)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--steps",type=int,default=600)
    p.add_argument("--epochs-per-step",type=int,default=2)
    p.add_argument("--perturb-steps",type=int,default=2)
    p.add_argument("--time-budget-seconds",type=int,default=600)
    args=p.parse_args()
    patch_targeted_search(args.architecture)
    print("RAY_ASK_TEN_MINUTE_FOCUS",json.dumps({
       "architecture":args.architecture,
       "architecture_neighborhood":nearby(args.architecture),
       "loss_mutations":"bounded categorical only; tau cannot exceed 1",
       "wall_clock_training_seconds":args.time_budget_seconds,
       "resume_previous_failed_study":False,
       "frozen_embeddings":True,"original_test_scored":False,
       "production_unchanged":True,
    }),flush=True)
    core.run(args)


if __name__=="__main__":
    main()
