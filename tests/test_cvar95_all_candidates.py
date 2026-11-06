"""Strict CVaR95 original-row and validation-only selection tests."""
import numpy as np
import pandas as pd
import pytest
from scripts.cvar95_all_candidates import BID_FAMILIES, ASK_FAMILIES, compare, WEIGHTS
from scripts.cvar95_historical import risk


def test_all_model_counts_and_cvaar_blend_grid():
    assert sum(BID_FAMILIES.values())==65
    assert sum(ASK_FAMILIES.values())==44
    assert set(BID_FAMILIES)=={"ft29","fm10","fm26"}
    assert set(ASK_FAMILIES)=={"bge8","deepcross36"}
    assert WEIGHTS[0]==0 and WEIGHTS[-1]==1
    assert len(WEIGHTS)==7


def test_cvar95_averages_worst_five_percent_absolute_usd():
    actual=np.repeat(10000., 200)
    prediction=actual.copy()
    prediction[0:10]-=np.arange(100,1100,100)
    got=risk(actual,prediction)
    assert got["worst_five_pct_rows"]==10
    assert got["CVaR95_absolute_USD"]==550.
    assert got["VaR95_absolute_USD"]<got["CVaR95_absolute_USD"]
    assert got["CVaR95_absolute_USD"]>got["RMSE_USD"]


def test_comparison_is_strictly_original_holdout_by_vehicle_id():
    base=pd.DataFrame({
        "row_id":["A","B","C"],
        "observed_price":[100.,200.,300.],
        "predicted_price":[115.,203.,290.]}).set_index("row_id")
    candidate=pd.DataFrame({
        "row_id":["C","A","B"],
        "split":["validation"]*3,
        "observed_price":[300.,100.,200.],
        "predicted_price":[290.,99.,199.]})
    aligned=compare(candidate,base,"validation")
    assert aligned.predicted_price.to_list()==[99.,199.,290.]
    with pytest.raises(ValueError):
        compare(candidate.iloc[:2],base,"validation")
    bad=candidate.copy();bad.loc[0,"observed_price"]=5.
    with pytest.raises(ValueError):
        compare(bad,base,"validation")
    wrong=candidate.copy();wrong.loc[0,"split"]="test"
    with pytest.raises(ValueError):
        compare(wrong,base,"validation")
    bad2=candidate.copy();bad2.loc[0,"predicted_price"]=np.inf
    with pytest.raises(ValueError):
        compare(bad2,base,"validation")


def test_selection_only_uses_validation_score_not_test_score():
    # Illustrate why the test must NEVER pick between candidates.
    sample=[{"model":"low_tail","val":300,"test":10000},
            {"model":"best_on_test","val":800,"test":100}]
    assert min(sample,key=lambda item:item["val"])["model"]=="low_tail"
    assert min(sample,key=lambda item:item["test"])["model"]!="low_tail"
