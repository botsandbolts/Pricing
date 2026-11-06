"""Contracts for the 200 -> regression -> 50 dependency, with synthetic data."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.neural_200_sweep import GROUPS, spec, build_ask, StructuredBid
from scripts.neural_50_refine import _prior, create_plan, load_plan
from scripts.neural_200_report import assemble


def synthetic_prior():
    records=[]
    groups={}
    for group in GROUPS:
        for trial in range(50):
            market=group.split("_",1)[0]
            base=18000. if market=="ask" else 8500.
            # Both group-specific regression tasks have a real but noisy slope.
            score=base+trial*13.+(trial%7)*27.+(200 if group.endswith("structure") else 0)
            records.append({
                "status":"completed","group":group,"trial":trial,
                "trial_configuration":spec(group,trial),
                "test_evaluated":False,"selection_split":"original validation only",
                "validation":{"rows":31668 if market=="ask" else 82380,
                    "CVaR95_absolute_USD":score,"RMSE_USD":score*.4,
                    "MAE_USD":score*.2},
                "provenance":{"dataset_sha256":market+"-cohort",
                    "holdout_row_sha256":{"validation":market+"-val-sha",
                                             "test":market+"-test-sha"}}
                })
        groups[group]=50
    return {"status":"complete","planned_trials":200,
            "completed_valid_trials":200,"missing":[],"invalid":[],
            "counts":groups,"experiments":records}


def test_all_200_and_identical_cohorts_required(tmp_path):
    prior=synthetic_prior();p=tmp_path/"prior.json"
    p.write_text(json.dumps(prior))
    doc,source=_prior(p)
    assert len(doc["experiments"])==200
    assert source["ask"]["validation"]=="ask-val-sha"
    prior["experiments"][0]["provenance"]["holdout_row_sha256"]["validation"]="wrong"
    p.write_text(json.dumps(prior))
    with pytest.raises(ValueError,match="Mixed original validation"): _prior(p)


def test_partial_200_cannot_trigger_refinement(tmp_path):
    prior=synthetic_prior()
    for modification in ("missing","failed"):
        current=json.loads(json.dumps(prior))
        if modification=="missing":
            current["completed_valid_trials"]=199
        else:
            current["experiments"][0]["status"]="failed"
        p=tmp_path/(modification+".json")
        p.write_text(json.dumps(current))
        with pytest.raises(ValueError):_prior(p)


def test_regression_yields_exactly_25_joints_per_market(tmp_path):
    p=tmp_path/"prior.json";out=tmp_path/"plan.json"
    p.write_text(json.dumps(synthetic_prior()))
    create_plan(p,out,42270078516)
    plan=load_plan(out)
    assert plan["candidate_counts"]=={"ask":25,"bid":25}
    assert plan["test_labels_accessed"] is False
    for market in ("ask","bid"):
        group=plan["candidates"][market]
        assert len(group)==25
        assert len(set((r["source_hp_trial"],r["source_structure_trial"])
                       for r in group))==25
        assert all(r["true_combined_validation_CVaR95_USD"] is None for r in group)
        assert all("lr" in r["hyperparameters"] and "rank" in r["structure"]
                   for r in group)
        assert plan["regression_diagnostics"][market]["architecture"]["observations"]==50


def test_structural_candidates_are_real_torch_models():
    ask_cfg=spec("ask_structure",27)
    ask=build_ask([9]*11,384,"ask_structure",ask_cfg)
    out=ask(torch.zeros(2,7),torch.ones(2,11,dtype=torch.long),torch.zeros(2,384))
    assert out.shape==(2,2) and torch.isfinite(out).all()
    bid_cfg=spec("bid_structure",35)
    bid=StructuredBid([9]*10,9.,bid_cfg)
    out=bid(torch.zeros(2,5),torch.ones(2,10,dtype=torch.long))
    assert out.shape==(2,) and torch.isfinite(out).all()
