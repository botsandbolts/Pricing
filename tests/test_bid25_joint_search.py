"""Bid25 preflight: joint structure + hyperparameter diversity and original holdout contract."""
import json

import numpy as np
import pytest
import torch

from scripts.bid25_joint_search import (
    N_TRIALS,trial_config,create_model,train_loss,lr_scheduler,
)
from scripts.bid25_joint_report import collect


def test_exactly_25_deterministic_distinct_bid_candidates():
    assert N_TRIALS==25
    configs=[trial_config(i) for i in range(N_TRIALS)]
    assert configs==[trial_config(i) for i in range(N_TRIALS)]
    assert len({json.dumps(c,sort_keys=True) for c in configs})==25
    assert configs[0]["family"]=="deepfm" and configs[0]["rank"]==16
    assert configs[0]["neural_weight"]==.75
    assert sum(x["family"]=="deepfm_gate" for x in configs)==5
    assert sum(x["family"]=="deepfm_hofm3" for x in configs)==4
    assert all(0<x["learning_rate"]<.006 and x["epochs"]>=20 for x in configs)
    assert all(.6<=x["neural_weight"]<=1. for x in configs)
    assert all(0<=x["tail_weight"]<=.07 for x in configs)
    for invalid in (-1,25,1.5,"4"):
        with pytest.raises(ValueError):trial_config(invalid)


@pytest.mark.parametrize("trial",[0,1,4,6,8,13,19,22,24])
def test_native_bid_structure_forward_backward_and_loss(trial):
    cfg=trial_config(trial)
    model=create_model([10]*10,9.,cfg)
    num=torch.randn(5,5)
    cat=torch.ones(5,10,dtype=torch.long)
    output=model(num,cat)
    assert output.shape==(5,) and torch.isfinite(output).all()
    target=torch.tensor([8.1,8.4,8.6,9.0,9.4])
    loss=train_loss(output,target,cfg)
    assert torch.isfinite(loss).item()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"])
    schedule=lr_scheduler(optimizer,cfg)
    assert schedule is None or hasattr(schedule,"step")


def test_loss_tail_cannot_make_simple_example_nonfinite():
    cfg=trial_config(1)
    pred=torch.tensor([8.5,9.,8.,9.4],requires_grad=True)
    truth=torch.tensor([8.1,9.2,8.5,9.1])
    zero={**cfg,"tail_weight":0.}
    large={**cfg,"tail_weight":.07}
    assert torch.isfinite(train_loss(pred,truth,large))
    assert train_loss(pred,truth,large)>=train_loss(pred,truth,zero)


def test_no_unsupported_missing_trainings_pass_as_complete(tmp_path):
    audit=collect(tmp_path)
    assert audit["status"]=="partial_or_invalid"
    assert audit["successful_audited_trials"]==0
    assert audit["missing"]==list(range(25))
    assert audit["automatic_deployment"] is False
