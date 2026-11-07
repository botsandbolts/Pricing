"""BOHB-specific preflight regression checks, with synthetic CVaR95 evidence."""
import json
from pathlib import Path

import pytest

from scripts import bohb_cvar95 as b


def _completed(plan):
    records=[]
    for market in b.MARKETS:
        for item in plan["candidates"][market]:
            row=dict(item)
            row.update({
                "state":"COMPLETE",
                "validation":{
                    "CVaR95_absolute_USD": float(
                        (18000 if market=="ask" else 8500)+item["slot"]*120),
                    "MAE_USD": float(3000+item["slot"]),
                    "RMSE_USD": float(6000+item["slot"]),
                },
                "provenance":{"holdout_row_sha256":{
                    "validation": "ask-fixed" if market=="ask" else "bid-fixed"}},
                "test_evaluated":False,
                "auto_deploy":False,
            })
            records.append(row)
    return records


def test_genuine_hpbandster_proposal_and_no_optuna_tpe(tmp_path):
    from hpbandster.optimizers.config_generators.bohb import BOHB
    for market in b.MARKETS:
        generator=b.bohb_generator(market,[])
        assert isinstance(generator, BOHB)
        assert len(b.search_space(market).get_hyperparameters()) == 5
    path=tmp_path/"batch0.json"
    b.propose(0,None,path)
    plan=json.loads(path.read_text())
    assert plan["auto_deploy"] is False
    assert plan["parallel"] == {"ask":4,"bid":4}
    assert plan["budgets_epochs"] == [4,12,36]
    for market in b.MARKETS:
        entries=plan["candidates"][market]
        assert len(entries)==4
        assert all(x["budget"]==4 and x["source"].startswith("hpbandster")
                   for x in entries)
        assert len({json.dumps(x["params"],sort_keys=True) for x in entries})==4
        assert all(x["config"]["hyperparameters"]["epochs"]==4 for x in entries)


def test_promote_best_rung_with_same_origin_seed_and_cv_fingerprint(tmp_path):
    initial=tmp_path/"batch0.json"
    b.propose(0,None,initial)
    plan0=json.loads(initial.read_text())
    history=_completed(plan0)
    path0=tmp_path/"h0.json"
    path0.write_text(json.dumps({"records":history}))
    next_plan=tmp_path/"batch1.json"
    b.propose(1,path0,next_plan)
    plan1=json.loads(next_plan.read_text())
    for market in b.MARKETS:
        chosen=plan1["candidates"][market]
        assert sorted(x["budget"] for x in chosen)==[4,4,4,12]
        promoted=next(x for x in chosen if x["budget"]==12)
        best=plan0["candidates"][market][0]
        assert promoted["params"]==best["params"]
        assert promoted["origin_trial"]==best["origin_trial"]
        assert promoted["promoted_from_trial"]==best["trial"]
        assert promoted["config"]["hyperparameters"]["epochs"]==12
        assert len({json.dumps(x["params"],sort_keys=True)
                    for x in chosen if x["budget"]==4})==3
    h1=history+_completed(plan1)
    path1=tmp_path/"h1.json"
    path1.write_text(json.dumps({"records":h1}))
    p2=tmp_path/"batch2.json"
    b.propose(2,path1,p2)
    plan2=json.loads(p2.read_text())
    for market in b.MARKETS:
        assert sorted(x["budget"] for x in plan2["candidates"][market])==[4,4,12,36]


def test_boosting_training_budgets_not_test_labels():
    assert b.STAGES==6 and b.SLOTS==4
    assert b.PROMOTIONS[0]==()
    assert b.PROMOTIONS[1]==(12,)
    assert b.PROMOTIONS[2]==(36,12)
    for market in b.MARKETS:
        for budget in b.BUDGETS:
            param = {"lr":0.001,"wd":0.01,"batch":512}
            param.update({"projection":96,"dropout":.12}
                         if market=="ask" else {"beta":.25,"tail_lambda":.05})
            cfg=b.settings_from_params(market,param,budget)
            assert cfg["hyperparameters"]["epochs"]==budget
            assert cfg["hyperparameters"]["lr"]==.001
            if market=="ask":
                assert cfg["hyperparameters"]["schedule"]=="constant"
            assert "structure" in cfg
    with pytest.raises(ValueError, match="budget"):
        b.settings_from_params("bid",{"lr":.001},99)


def test_refuse_missing_round_or_changed_holdout(tmp_path):
    with pytest.raises(ValueError,match="earlier BOHB batch audits"):
        b.propose(2,None,tmp_path/"bad.json")
    initial=tmp_path/"p0.json"
    b.propose(0,None,initial)
    rows=_completed(json.loads(initial.read_text()))
    rows[0]["provenance"]["holdout_row_sha256"]["validation"]="different"
    file=tmp_path/"drift.json"
    file.write_text(json.dumps({"records":rows}))
    with pytest.raises(ValueError,match="vehicle cohorts changed"):
        b.read_history(file)
