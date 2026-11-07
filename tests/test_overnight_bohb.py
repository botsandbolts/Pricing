"""Overnight BOHB search-contract, deadline, optimizer and loss regression tests."""
import json
import math
import pytest
import torch

from scripts import overnight_bohb as b
from scripts.overnight_bohb_losses import (
    LOSS_NAMES, regression_loss, make_optimizer, make_scheduler,
)


@pytest.fixture(autouse=True)
def during_window(monkeypatch):
    monkeypatch.setattr(b,"remaining_seconds",lambda:100000.)


def _fake_records(plan):
    rows=[]
    for market in b.MARKETS:
        for c in plan["candidates"][market]:
            x=dict(c)
            x.update(state="COMPLETE",validation={
               "CVaR95_absolute_USD":float((20000 if market=="ask" else 10000)+100*c["slot"]),
               "MAE_USD":float(3300+c["slot"]),
               "RMSE_USD":float(6500+c["slot"])},
               provenance={"holdout_row_sha256":{"validation":market+"-immutable"}},
               test_evaluated=False,auto_deploy=False)
            rows.append(x)
    return rows


def test_wide_parameter_spaces_are_actual_model_inputs():
    assert b.BUDGETS==(6,18,54,162,216)
    assert b.STAGES==30 and b.SLOTS==4
    for market,min_dimensions in (("ask",32),("bid",26)):
        space=b.search_space(market)
        cfg=space.sample_configuration().get_dictionary()
        assert len(cfg)>=min_dimensions
        model=b.settings_from_params(market,cfg,216)
        assert model["hyperparameters"]["epochs"]==216
        assert model["hyperparameters"]["loss"] in LOSS_NAMES
        assert model["hyperparameters"]["optimizer"] in ("adamw","adam","radam","nadam","amsgrad")
        assert model["hyperparameters"]["schedule"] in ("constant","cosine216","warm_restart","step_decay")
        if market=="ask":
            assert model["structure"]["cross_layers"] in [1,2,3,4,5,6]
            assert "cross_scale" in model["structure"]
            assert "noise" in model["hyperparameters"]
        else:
            assert model["hyperparameters"]["blend_alpha"] in [.5,.6,.7,.8,.9,.95,1.]
            assert "pairgate" in model["structure"]


def test_boosted_dollar_tail_losses_all_finite_with_gradients():
    y=torch.linspace(math.log(4000),math.log(80000),32)
    for market in b.MARKETS:
        for name in LOSS_NAMES:
            p=(torch.stack((y+.08*torch.cos(torch.arange(32)), torch.full_like(y,.1)),dim=-1)
               if market=="ask" else y+.08*torch.cos(torch.arange(32)))
            p=p.detach().requires_grad_(True)
            hp={"loss":name,"beta":.25,"tau":.65,"tail_fraction":.05,
                "tail_strength":.25,"usd_scale":5000.,"tail_temperature":.1,
                "focal_gamma":1.,"sigma_floor":.06}
            v=regression_loss(p,y,hp,market)
            assert torch.isfinite(v), (market,name)
            v.backward()
            assert p.grad is not None and torch.isfinite(p.grad).all(),(market,name)


def test_optimizers_and_all_schedulers_step():
    for opt in ("adamw","adam","radam","nadam","amsgrad"):
        for mode in ("constant","cosine216","warm_restart","step_decay"):
            net=torch.nn.Linear(3,1)
            hp={"optimizer":opt,"lr":.0008,"wd":.01,"beta1":.9,"beta2":.999,
                "schedule":mode}
            optimizer=make_optimizer(net,hp)
            scheduler=make_scheduler(optimizer,hp)
            out=net(torch.ones(6,3)).square().mean()
            out.backward()
            optimizer.step()
            if scheduler:scheduler.step()


def test_first_batch_and_promotion_inherit_exact_seed(tmp_path):
    a=tmp_path/"a.json"
    b.propose(0,None,a)
    p0=json.loads(a.read_text())
    assert p0["status"]=="ready" and p0["long_epochs"]==216
    for market in b.MARKETS:
        assert len(p0["candidates"][market])==4
        assert all(c["budget"]==6 for c in p0["candidates"][market])
    history={"records":_fake_records(p0)}
    h=tmp_path/"history0.json";h.write_text(json.dumps(history))
    next_plan=tmp_path/"b.json"
    b.propose(1,h,next_plan)
    p1=json.loads(next_plan.read_text())
    for market in b.MARKETS:
        promoted=[c for c in p1["candidates"][market] if c["budget"]==18]
        assert len(promoted)==1
        parent=p0["candidates"][market][0]
        assert promoted[0]["origin_trial"]==parent["origin_trial"]
        assert promoted[0]["params"]==parent["params"]
        assert promoted[0]["config"]["hyperparameters"]["epochs"]==18
    assert b.promotions_for(4)==(216,162,54)
    assert b.promotions_for(29)==(216,162,54)


def test_deadline_stops_proposals_without_fabricating_metrics(tmp_path,monkeypatch):
    monkeypatch.setattr(b,"remaining_seconds",lambda:2.)
    output=tmp_path/"stopped.json"
    b.propose(0,None,output)
    doc=json.loads(output.read_text())
    assert doc["status"]=="cutoff"
    assert doc["audited_trials"]==0
    assert doc["test_evaluated"] is False


def test_bid_blend_receipt_is_pre_registered():
    p=b.search_space("bid").sample_configuration().get_dictionary()
    hp=b.settings_from_params("bid",p,162)["hyperparameters"]
    assert .5<=hp["blend_alpha"]<=1
    assert hp["loss"] in LOSS_NAMES
    with pytest.raises(ValueError,match="fidelity"):
        b.settings_from_params("ask",{},300)


def test_failed_unstable_trial_is_recorded_but_not_treated_as_a_win(tmp_path):
    plan=tmp_path/"p.json"
    b.propose(0,None,plan)
    p0=json.loads(plan.read_text())
    rows=_fake_records(p0)
    rows[0]["state"]="FAILED"
    rows[0]["validation"]=None
    rows[0]["error"]="ValueError: Nonfinite BOHB objective"
    rows[0]["provenance"]=None
    history_file=tmp_path/"h0.json"
    history_file.write_text(json.dumps({"records":rows}))
    recovered=b.read_history(history_file)
    assert len(recovered)==8
    next_plan=tmp_path/"p1.json"
    b.propose(1,history_file,next_plan)
    p1=json.loads(next_plan.read_text())
    assert p1["candidates"]["ask"][0]["origin_trial"]!=rows[0]["origin_trial"]


def test_all_failed_first_rung_can_fall_back_to_new_configs(tmp_path):
    plan=tmp_path/"p0.json"
    b.propose(0,None,plan)
    rows=_fake_records(json.loads(plan.read_text()))
    for row in rows:
        if row["market"]=="ask":
            row.update(state="FAILED",validation=None,provenance=None,
                       error="Nonfinite experimental loss")
    history=tmp_path/"h.json"
    history.write_text(json.dumps({"records":rows}))
    next_plan=tmp_path/"p1.json"
    b.propose(1,history,next_plan)
    p=json.loads(next_plan.read_text())
    assert all(row["budget"]==6 for row in p["candidates"]["ask"])
    assert any(row["budget"]==18 for row in p["candidates"]["bid"])
