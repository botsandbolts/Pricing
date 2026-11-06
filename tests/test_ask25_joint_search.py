"""Synthetic, local-only tests for the 25-Ask resource-bounded neural search."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.ask25_joint_search import (
    TRIALS,model_for,training_loss,scheduler_for,trial_config
)
from scripts.ask25_joint_report import collect


def test_registers_exactly_25_distinct_joint_configs():
    assert TRIALS==25
    specs=[trial_config(i) for i in range(TRIALS)]
    assert specs==[trial_config(i) for i in range(TRIALS)]
    assert len({json.dumps(x,sort_keys=True) for x in specs})==25
    assert specs[0]["family"]=="dcnv2"
    assert sum(x["family"]=="resmlp" for x in specs)==5
    assert all(0<x["learning_rate"]<.003 and .0<=x["tail_weight"]<=.1 for x in specs)
    assert all(x["epochs"]>=24 and x["batch_size"]<=1024 for x in specs)
    with pytest.raises(ValueError):
        trial_config(25)
    with pytest.raises(ValueError):
        trial_config(-1)


@pytest.mark.parametrize("trial",[0,1,5,10,15,20,24])
def test_network_is_native_and_trains_real_aft_parameters(trial):
    config=trial_config(trial)
    model=model_for(config,[8]*11,384)
    num=torch.randn(4,7)
    cat=torch.ones(4,11,dtype=torch.long)
    emb=torch.randn(4,384)
    result=model(num,cat,emb)
    assert result.shape==(4,2)
    assert torch.isfinite(result).all()
    target=torch.log(torch.tensor([3000.,5000.,8000.,12000.]))
    loss=training_loss(result,target,config["tail_weight"])
    assert torch.isfinite(loss).item()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"])
    scheduler=scheduler_for(optimizer,config)
    assert scheduler is not None


def test_tail_loss_is_finite_and_preserves_zero_weight_baseline():
    predicted=torch.tensor([[8.5,-.4],[9.,-.8],[8.7,.2],[10.,-.1]],requires_grad=True)
    target=torch.tensor([8.1,9.3,8.5,9.5])
    primary=training_loss(predicted,target,0.)
    nudged=training_loss(predicted,target,.04)
    assert torch.isfinite(primary).item()
    assert torch.isfinite(nudged).item()
    assert nudged>=primary


def test_leaderboard_marks_missing_experiments_explicitly(tmp_path):
    report=collect(tmp_path)
    assert report["status"]=="partial_or_invalid"
    assert report["successful_audited_trials"]==0
    assert report["missing_trials"]==list(range(25))
    assert report["best_trial"] is None
