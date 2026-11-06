"""Auction neural: frozen identity, target, no-price-input and split tests."""
import numpy as np
import pandas as pd
import pytest
from scripts.neural_bid_prepare import numeric_features, lemon_membership, sha_ids

def sample():
    return pd.DataFrame({
      "row_id":["a","b","c","d"],
      "vehicle_group":["VIN-1","VIN-2","VIN-3","VIN-4"],
      "split":["calibration","calibration","validation","test"],
      "observed_price":[2500.,3200.,4300.,5400.],
      "year":[2011,2012,2009,2010],
      "date":["2015-01-01","2015-01-02","2015-02-01","2015-03-01"],
      "odometer":[50000.,70000.,89000.,63000.],
      "condition":[2.,3.,5.,4.]})

def test_numeric_cannot_learn_sale_target():
    frame=sample(); before=numeric_features(frame)
    frame["observed_price"]*=10
    np.testing.assert_allclose(before,numeric_features(frame))
    assert before.shape==(4,5) and np.isfinite(before).all()

def test_holdout_fingerprint_is_stable():
    assert sha_ids(["a","b"])==sha_ids(["b","a"])
    assert sha_ids(["a","b"])!=sha_ids(["a","c"])

def test_lemon_cannot_remove_original_holdouts(monkeypatch):
    import scripts.neural_bid_prepare as module
    monkeypatch.setattr(module,"EXPECTED_LEMON",1)
    monkeypatch.setattr(module,"EXPECTED",{"train":2,"validation":1,"test":1})
    original=sample()
    phase=np.array(["train","train","validation","test"])
    keep=lemon_membership(original,original.iloc[[0,2,3]],phase)
    assert keep.tolist()==[True,False,True,True]
    with pytest.raises(ValueError,match="validation/test"):
        lemon_membership(original,original.iloc[[0,3]],phase)

def test_lemon_rejects_different_targets(monkeypatch):
    import scripts.neural_bid_prepare as module
    monkeypatch.setattr(module,"EXPECTED_LEMON",1)
    monkeypatch.setattr(module,"EXPECTED",{"train":2,"validation":1,"test":1})
    original=sample();trimmed=original.iloc[[0,2,3]].copy()
    trimmed.loc[trimmed.row_id.eq("d"),"observed_price"]+=10
    with pytest.raises(ValueError,match="target drift"):
        lemon_membership(original,trimmed,np.array(["train","train","validation","test"]))
