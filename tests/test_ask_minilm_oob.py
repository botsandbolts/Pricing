"""Prove two-out-of-bag feature cannot read its own target or heldout labels."""
import numpy as np
import pytest
from scripts.ask_minilm_augment import two_oob

def test_genuine_two_oob():
    x=np.random.default_rng(1).normal(size=(24,6)).astype(np.float32)
    y=np.exp(np.random.default_rng(2).uniform(7,11,size=24))
    fit=np.arange(15)
    value,fold,heads=two_oob(x,y,fit)
    assert len(heads)==3 and set(fold[fit])=={0,1,2}
    assert np.all(fold[15:]==-1)
    assert np.isfinite(value).all() and (value>0).all()
    for i in [0,4,9]:
        different=y.copy();different[i]*=2.
        alternate,_,_=two_oob(x,different,fit)
        assert alternate[i]==pytest.approx(value[i],rel=1e-10)
    changed=y.copy();changed[15:]=1e7
    held,_,_=two_oob(x,changed,fit)
    np.testing.assert_array_equal(value,held)

def test_duplicate_fit_indexes_rejected():
    x=np.ones((12,4),dtype=np.float32)
    y=np.ones(12)*10000
    with pytest.raises(ValueError,match="Invalid FIT"):
        two_oob(x,y,[0,1,2,3,4,5,6,7,7])
