"""Test target/bound scale, unseen categories, calibration direction, and persistence."""
import numpy as np
import pandas as pd
import pytest
from pricing.model_suite import bounds, comparisons, encode, fit, predict, NAMES, predictors


def test_aft_bounds_are_dollars_and_validate_censoring():
    lo,hi=bounds([10000,20000]);assert lo.tolist()==[10000,20000];assert np.array_equal(lo,hi)
    lo,hi=bounds([10000,20000],[0,15000],[10000,np.inf]);assert np.isinf(hi[1])
    with pytest.raises(ValueError):bounds([10000], [20000], [10000])
    with pytest.raises(ValueError):bounds([0])


def test_ridge_pair_direction_and_bias():
    v=pd.DataFrame({'ask':np.arange(100)*100.+3000,'bid':np.arange(100)*50.+1000})
    t=pd.DataFrame({'ask':np.arange(20)*200.+3100,'bid':np.arange(20)*100.+1050})
    r=comparisons(v,t)
    p=next(p for p in r['ridge_pairs'] if p['response']=='ask')
    assert p['beta']==pytest.approx(2.,rel=.02)
    assert p['intercept_usd']==pytest.approx(1000,abs=100)
    assert r['spearman']['ask']['bid']==pytest.approx(1.)
    assert p['bias_predictor_minus_response_usd']<0
    assert r['ridge_fit_partition']=='validation'


def test_models_positive_with_unseen_category():
    x=pd.DataFrame({'age':np.arange(150)%12,'model':['a','b','c']*50})
    schema={'model':['a','b','c']};x=encode(x,schema);y=30000-np.asarray(x.age)*1000
    unseen=encode(pd.DataFrame({'age':[4],'model':['new-model']}),schema)
    for name in NAMES:
        model=fit(name,x,y);p=predict(model,name,unseen)
        assert len(p)==1 and np.isfinite(p).all() and p[0]>0


def test_ridge_calibration_uses_validation_only():
    v=pd.DataFrame({'a':np.arange(100)+100.});t=pd.DataFrame({'a':np.arange(30)+500.})
    a=comparisons(v,t,(2*v.a+50,2*t.a+50))
    b=comparisons(v,t,(2*v.a+50,8*t.a+1000))
    assert a['ridge_observed_price_calibration'][0]['beta']==b['ridge_observed_price_calibration'][0]['beta']


def test_split_model_integrity(tmp_path):
    from pricing.suite_storage import split_model,read_model_file
    path=tmp_path/'model.ubj.gz';raw=b'model-weights'*100
    path.write_bytes(raw);spec=split_model(path,part_bytes=200)
    assert not path.exists() and read_model_file(path)==raw
    (tmp_path/spec['parts'][0]['file']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):read_model_file(path)


def test_auction_grade_reaches_inference_features():
    frame=pd.DataFrame([{'year':2015,'odometer':90000,'as_of':'2021-05-01','auction_condition':4.}])
    x,valid,_=predictors(frame,'bid')
    assert valid.all() and x.auction_grade.iloc[0]==4.
