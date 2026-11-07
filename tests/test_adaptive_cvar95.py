"""No training data or production serving weights required for search-contract QA."""
import json
from pathlib import Path

import optuna
import pytest

from scripts import adaptive_cvar95 as a


def test_first_plan_has_four_ask_four_bid_and_unique_proposals(tmp_path):
    out = tmp_path / "plan.json"
    a.propose(0, None, out)
    doc = json.loads(out.read_text())
    assert doc["concurrent_workers"] == 8
    assert (doc["ask_workers"], doc["bid_workers"]) == (4, 4)
    assert doc["auto_deploy"] is False and doc["test_evaluated"] is False
    for market in a.MARKETS:
        candidates = doc["candidates"][market]
        assert len(candidates) == 4
        assert set(x["trial"] for x in candidates) == set(range(4))
        assert len({json.dumps(x["params"],sort_keys=True) for x in candidates}) == 4
        for proposal in candidates:
            assert proposal["config"] == a._config(proposal["params"])
            assert all(k.startswith(("hp_", "arch_")) for k in proposal["params"])


def test_previous_batch_reconstructs_tpe_and_halving_state(tmp_path):
    initial = tmp_path / "first.json"
    a.propose(0, None, initial)
    candidates = json.loads(initial.read_text())["candidates"]
    history = []
    for market in a.MARKETS:
        for proposal in candidates[market]:
            slot = proposal["slot"]
            history.append({
                **proposal, "state": "COMPLETE", "validation": {
                    "CVaR95_absolute_USD": float(10000 + slot*100),
                    "RMSE_USD": float(6000+slot), "MAE_USD": float(4000+slot)},
                "provenance": {"holdout_row_sha256": {"validation": market+"-cohort"}},
                "intermediates": {"1": 12000.0, "4": 11200.0, "8": 10500.0},
                "rungs": {},
            })
    previous = tmp_path / "history.json"
    previous.write_text(json.dumps({"records": history}))
    second = tmp_path / "second.json"
    a.propose(1, previous, second)
    planned = json.loads(second.read_text())["candidates"]
    for market in a.MARKETS:
        assert set(x["trial"] for x in planned[market]) == set(range(4, 8))
        study = a._restore_study(market, history)
        assert len(study.trials) == 4
        assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        assert len(set(json.dumps(x["params"],sort_keys=True)
                       for x in planned[market])) == 4


def test_study_rehydrates_pruned_arms_and_rejects_missing_batch(tmp_path):
    initial = tmp_path / "plan.json"
    a.propose(0, None, initial)
    proposal = json.loads(initial.read_text())["candidates"]["ask"][0]
    prior = [{**proposal, "state": "PRUNED", "validation": None,
              "provenance": {}, "intermediates": {"4": 10000.},
              "rungs": {"completed_rung_0": 10000.}}]
    restored = a._restore_study("ask", prior)
    assert restored.trials[0].state == optuna.trial.TrialState.PRUNED
    assert restored.trials[0].intermediate_values[4] == 10000.
    history_file = tmp_path / "incomplete.json"
    history_file.write_text(json.dumps({"records":prior}))
    with pytest.raises(ValueError,match="every previous"):
        a.propose(1,history_file,tmp_path/"bad.json")


def test_no_cross_market_holdout_drift(tmp_path):
    plan = tmp_path / "plan.json"
    a.propose(0, None, plan)
    arm = json.loads(plan.read_text())["candidates"]["ask"][0]
    results = [{**arm, "state":"COMPLETE",
                "validation":{"CVaR95_absolute_USD":1000.},
                "provenance":{"holdout_row_sha256":{"validation":hash_}},
                "intermediates":{}, "rungs":{}}
               for hash_ in ("one","two")]
    results[1]["slot"] = 1
    previous=tmp_path/"invalid.json"
    previous.write_text(json.dumps({"records":results}))
    with pytest.raises(ValueError, match="Mismatched holdout"):
        a._read_history(previous)
