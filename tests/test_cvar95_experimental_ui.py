"""Accessible, honest CVaR95 scientific evidence and experimental-only controls."""
import json
from pathlib import Path
import pytest
from pricing.cvar95_experiment_science import evidence,model_kpis,all_validation,PAPERS


def test_verified_historical_tail_card_and_wrong_status_fail_closed(tmp_path):
    assert evidence(tmp_path)=={}
    folder=tmp_path/"reports/cvar95/previously-selected"
    folder.mkdir(parents=True)
    k=lambda title:dict(model=title,CVaR95_absolute_USD=10000.,
        RMSE_USD=3000.,MAE_USD=1500.,VaR95_absolute_USD=4000.)
    f=folder/"comparison.json"
    d={"status":"complete","metric":"CVaR95 absolute dollar error",
       "original_holdout_sizes":{"wholesale":83603,"asking":31682},
       "test_used_for_reselection":False,"no_retraining":True,
       "bid":[k("fm26 / validated_blend")],
       "ask":[k("BGE-256 + full DeepCross/AFT")]}
    f.write_text(json.dumps(d))
    assert evidence(tmp_path)["status"]=="complete"
    # Historical MAE/RMSE-selected test scores do not impersonate
    # the new validation-CVaR95-selected engine.
    assert model_kpis(tmp_path,"bid") is None
    assert model_kpis(tmp_path,"ask") is None
    d["test_used_for_reselection"]=True
    f.write_text(json.dumps(d))
    assert evidence(tmp_path)=={}


def test_science_links_are_original_peer_reviewed_papers():
    assert len(PAPERS)>=4
    for author,title,journal,url,meaning in PAPERS:
        assert author and title and journal and meaning
        assert url.startswith("https://doi.org/10.")


def test_rejects_unverified_new_cvaar_champion(tmp_path):
    with pytest.raises(ValueError):all_validation(tmp_path,"cars")
    f=tmp_path/"reports/cvar95/all-candidates/ask/comparison.json"
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"status":"complete","market":"ask",
         "primary_metric":"validation_absolute_dollar_CVaR95",
         "training_objectives_unchanged":True,"models_chosen_by_test":True}))
    assert all_validation(tmp_path,"ask")=={}
    f.write_text(json.dumps({"status":"complete","market":"ask",
         "primary_metric":"validation_absolute_dollar_CVaR95",
         "training_objectives_unchanged":True,"models_chosen_by_test":False}))
    assert all_validation(tmp_path,"ask")["market"]=="ask"


def test_ui_accessibility_and_main_calculator_isolated():
    code=Path("pricing/neural_experimental_ui.py").read_text()
    science=Path("pricing/cvar95_experiment_science.py").read_text()
    app=Path("app.py").read_text()
    assert "if _workspace=='🧬 Experimental models':" in app
    assert "from pricing.neural_experimental_ui import render" in app
    assert "st.tabs" in code and "st.form" in code
    assert 'aria-label="Experimental vehicle pricing models"' in code
    assert "CVaR95 · PRIMARY" in code
    assert "RMSE · secondary" in code and "MAE · secondary" in code
    assert "st.warning" in code
    assert "st.download_button" in code
    assert "CVar" not in code
    assert "VaR95" in science
    assert "not automatically" in science.lower() or "not automatic" in science.lower()
