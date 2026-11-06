"""Mathematical contracts for CVaR95 of original absolute dollar prediction errors."""
import numpy as np
import pandas as pd
import pytest
from scripts.cvar95_historical import risk,consistent


def test_cvar95_exact_worst_5pct_mean_not_rmse():
    actual=np.ones(100)*1000
    pred=actual.copy()
    pred[-5:]=[800,900,400,200,500]
    r=risk(actual,pred)
    assert r["rows"]==100
    assert r["worst_five_pct_rows"]==5
    assert r["CVaR95_absolute_USD"]==float(np.mean([200,100,600,800,500]))
    assert r["CVaR95_absolute_USD"]>r["MAE_USD"]
    assert r["VaR95_absolute_USD"]<=r["CVaR95_absolute_USD"]


def test_cvar95_nondivisible_size_includes_ceil_tail_count():
    original=np.ones(101)*1000
    predictions=original.copy()
    predictions[-6:]=[900,800,700,600,500,400]
    r=risk(original,predictions)
    assert r["worst_five_pct_rows"]==6
    assert r["CVaR95_absolute_USD"]==350.


def test_cvar95_rejects_invalid_prices():
    with pytest.raises(ValueError):risk([100],[np.nan])
    with pytest.raises(ValueError):risk([100],[-1])
    with pytest.raises(ValueError):risk([100,200],[100])
    with pytest.raises(ValueError):risk([],[])


def test_original_test_identity_and_label_alignment():
    left=pd.DataFrame({"row_id":["a","b"],"observed_price":[100,200]})
    right=left.iloc[::-1].copy()
    a,b=consistent(left,right)
    assert a.index.tolist()==b.index.tolist()
    assert b.observed_price.tolist()==[100,200]
    with pytest.raises(ValueError):
        consistent(left,right.iloc[:1])
    with pytest.raises(ValueError):
        consistent(left,pd.DataFrame({"row_id":["a","b"],"observed_price":[200,200]}))
