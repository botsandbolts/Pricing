import numpy as np
import pandas as pd
import pytest
from pricing.hierarchical import experiment,validate,predict


def fixture():
    rng=np.random.default_rng(7);n=300
    return pd.DataFrame(dict(row_id=[str(i) for i in range(n)],vehicle_group=[str(i) for i in range(n)],
        split=['calibration']*200+['validation']*50+['test']*50,
        prediction_kind=['oof']*200+['heldout']*100,base_fold=[i%3 for i in range(n)],
        manufacturer=['a' if i%2 else 'b' for i in range(n)],model=['m'+str(i%4) for i in range(n)],
        age=rng.uniform(1,20,n),observed_price=10000*np.exp(.2+rng.normal(0,.02,n)),
        base_price=10000.,base_model_id='assessment-model-v1',price_basis='historical-common-USD'))


def test_correction_and_untouched_evaluation():
    d=fixture();b,r,p=experiment(d,'bid')
    assert r['cohorts']['test']['hierarchical']['rows']==50
    assert r['cohorts']['test']['hierarchical']['mae_usd'] < r['cohorts']['test']['base']['mae_usd']
    assert len(p)==50 and np.isfinite(p.corrected_price).all()
    changed=d.copy();changed.loc[changed.split=='test','observed_price']*=10
    other,_,_=experiment(changed,'bid')
    assert b==other # test labels never fit the hierarchy or its hyperparameters
    assert len(b['fingerprint'])==64


def test_unknown_categories_pool_back_and_effects_sum():
    b,_,p=experiment(fixture(),'ask')
    q=p.iloc[:1].copy();q['manufacturer']='unseen';q['model']='unseen'
    result=predict(b,q).iloc[0]
    assert result.make_n==0 and result.model_n==0
    assert result.make_effect==0 and result.model_effect==0
    assert result.delta_log==pytest.approx(result.global_effect+result.make_effect+result.model_effect+result.age_effect)
    assert result.corrected_price==pytest.approx(result.base_price*np.exp(result.delta_log))


def test_provenance_and_group_leakage_guards():
    for change in ['in_sample','group_leak','bad_price','no_folds']:
        d=fixture()
        if change=='in_sample':d.loc[0,'prediction_kind']='in_sample'
        elif change=='group_leak':d.loc[250,'vehicle_group']='0'
        elif change=='bad_price':d.loc[0,'base_price']=0
        else:d['base_fold']=0
        with pytest.raises(ValueError):validate(d)


def test_database_schema_and_consent(monkeypatch):
    import sqlite3
    import streamlit as st
    from unittest.mock import patch
    from pricing.analytics import SCHEMA
    from pricing.hierarchical_store import record
    db=sqlite3.connect(':memory:');db.execute('PRAGMA foreign_keys=ON')
    for sql in SCHEMA:db.execute(sql)
    for sql in SCHEMA:db.execute(sql) # idempotent migration
    assert 'hierarchical_estimations' in {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    with patch.dict(st.session_state,{},clear=True),patch('pricing.analytics._store') as store:
        assert record('bid',{}, {},'request') is False
        store.assert_not_called()
