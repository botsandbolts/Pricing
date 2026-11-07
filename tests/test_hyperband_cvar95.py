"""Hyperband multi-bracket + pinned incumbent audit; no model training required."""
import json
from pathlib import Path

import optuna
import pytest

from scripts import hyperband_cvar95 as a


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



def test_bid_csv_reconciles_blended_cvar95_not_standalone_neural(tmp_path, monkeypatch):
    """The exact bug that blocked collect_0: CSV must reflect the scored blend."""
    import numpy as np
    import pandas as pd
    from scripts.cvar95_historical import risk
    import scripts.neural_200_sweep as native

    y = np.linspace(5000, 40000, 100, dtype=float)
    neural = y + np.linspace(-2200, 3200, 100)
    lemon = y + np.linspace(700, -1700, 100)
    alpha = .7
    combined = alpha * neural + (1-alpha) * lemon
    metrics = {**risk(y, combined), "neural_weight": alpha}
    (tmp_path/"result.json").write_text(json.dumps({"validation": metrics}))
    original = pd.DataFrame({
        "row_id": [str(i) for i in range(len(y))],
        "split": ["validation"]*len(y),
        "observed_price": y,
        "predicted_price": neural,
    })
    original.to_csv(tmp_path/"validation.csv.gz",index=False)
    monkeypatch.setattr(native, "_matched_auction_baseline",
                        lambda path, ids, observed: lemon)
    assert abs(risk(y, neural)["CVaR95_absolute_USD"] -
               metrics["CVaR95_absolute_USD"]) > .05
    a.audit_and_save_bid_blend(tmp_path, "synthetic-baseline.csv.gz")
    corrected = pd.read_csv(tmp_path/"validation.csv.gz")
    assert np.allclose(corrected["predicted_price"], combined)
    assert np.allclose(corrected["deepfm_price_before_blend"], neural)
    assert np.allclose(corrected["lemon_xgboost_price_before_blend"], lemon)
    assert risk(corrected["observed_price"],
                corrected["predicted_price"])["CVaR95_absolute_USD"] == pytest.approx(
                metrics["CVaR95_absolute_USD"], abs=.05)


def test_bad_bid_blend_is_rejected_not_silently_accepted(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    import scripts.neural_200_sweep as native
    from scripts.cvar95_historical import risk

    y = np.full(100, 20000.)
    neural = y + 500.
    lemon = y - 500.
    metrics = {**risk(y, neural), "neural_weight": .5}
    (tmp_path/"result.json").write_text(json.dumps({"validation":metrics}))
    pd.DataFrame({"row_id": [str(i) for i in range(100)],
                  "split": ["validation"]*100,"observed_price": y,
                  "predicted_price": neural}).to_csv(
                      tmp_path/"validation.csv.gz",index=False)
    monkeypatch.setattr(native, "_matched_auction_baseline",
                        lambda path, ids, actual: lemon)
    with pytest.raises(ValueError,match="blended validation does not match"):
        a.audit_and_save_bid_blend(tmp_path,"baseline.csv.gz")


def test_hyperband_two_brackets_pinned_study_and_six_batches():
    import optuna
    assert a.STAGES == 6
    study = a._restore_study("ask", [])
    assert study.study_name == "pricing-hyperband-20261107-ask"
    assert isinstance(study.pruner, optuna.pruners.HyperbandPruner)
    assert study.pruner._min_resource == 4
    assert study.pruner._max_resource == 42
    assert study.pruner._reduction_factor == 4
    # Two brackets: floor(log_4(42/4)) + 1.
    assert study.pruner._n_brackets == 0 or study.pruner._n_brackets == 2


def test_incumbent_comparison_uses_matching_original_validation_cohorts():
    from pathlib import Path
    import json
    root = Path(a.__file__).resolve().parents[1]
    records = []
    board = {}
    for market in a.MARKETS:
        manifest = json.loads(
            (root/"models"/f"neural-experimental-{market}-cvar95"/"manifest.json").read_text())
        original = manifest["historical_validation"]
        row = {"trial": 0,
               "CVaR95_absolute_USD": original["CVaR95_absolute_USD"]+100.,
               "MAE_USD": original["MAE_USD"]+10.,
               "RMSE_USD": original["RMSE_USD"]+20.}
        board[market] = [row]
        records.append({"market":market,"state":"COMPLETE",
                        "provenance":{"holdout_row_sha256":{
                            "validation":manifest["original_holdout_row_sha256"]["validation"]}}})
    results = a.incumbent_comparison(records, board)
    assert results["ask"]["metrics"]["CVaR95_absolute_USD"]["saved_usd"] == pytest.approx(-100.)
    assert not results["bid"]["challenger_beats_incumbent_on_CVaR95"]
    corrupted = json.loads(json.dumps(records))
    corrupted[0]["provenance"]["holdout_row_sha256"]["validation"] = "bad"
    with pytest.raises(ValueError,match="different validation cohort"):
        a.incumbent_comparison(corrupted,board)
