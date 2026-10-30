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


def test_consented_audit_deduplicates_and_expiry_respects_foreign_keys():
    import sqlite3,uuid
    from datetime import datetime,timezone,timedelta
    from unittest.mock import patch
    import streamlit as st
    from pricing.analytics import TursoStore,CONSENT_VERSION
    from pricing.hierarchical_store import record
    class Store(TursoStore):
        def __init__(self):
            super().__init__('libsql://test.turso.io','test');self.db=sqlite3.connect(':memory:');self.db.execute('PRAGMA foreign_keys=ON')
        def execute(self,sql,args=()):return self.db.execute(sql,args)
        def atomic(self,statements):
            for sql,args in statements:self.execute(sql,args)
    store=Store();c=dict(choice='accepted',version=CONSENT_VERSION,
        consent_at=datetime.now(timezone.utc).isoformat(),visitor_id=str(uuid.uuid4()),session_id=str(uuid.uuid4()))
    bundle,_,rows=experiment(fixture(),'bid')
    location=dict(country='DE',city='Hamburg',region=None,geo_provider=None)
    with patch.dict(st.session_state,{'_analytics_consent':c},clear=True),patch('pricing.analytics.configured',return_value=True),patch('pricing.analytics._store',return_value=store),patch('pricing.analytics._location',return_value=location):
        assert record('bid',rows.iloc[0],bundle,'same_request')
        assert record('bid',rows.iloc[0],bundle,'same_request')
    assert store.execute('SELECT count(*) FROM hierarchical_estimations').fetchone()[0]==1
    assert store.execute('SELECT city FROM hierarchical_estimations').fetchone()[0]=='Hamburg'
    old=(datetime.now(timezone.utc)-timedelta(days=100)).isoformat()
    store.execute('UPDATE visitors SET last_seen=?',[old]);store.purge()
    assert store.execute('SELECT count(*) FROM visitors').fetchone()[0]==1
    store.execute('UPDATE hierarchical_estimations SET timestamp=?',[old]);store.purge()
    assert store.execute('SELECT count(*) FROM hierarchical_estimations').fetchone()[0]==0
    assert store.execute('SELECT count(*) FROM visitors').fetchone()[0]==0


@pytest.mark.parametrize('side',['ask','bid'])
def test_catboost_residual_generator_excludes_heldout_labels(side):
    from pricing.catboost_residuals import generate
    d=fixture();d['odometer']=50000
    first,meta=generate(d,side,iterations=8)
    second=d.copy();second.loc[second.split=='test','observed_price']*=2
    other,_=generate(second,side,iterations=8)
    np.testing.assert_allclose(first.base_price,other.base_price)
    assert first.loc[first.split=='calibration','base_fold'].nunique()==3
    assert first.loc[first.split!='calibration','prediction_kind'].eq('heldout').all()
    np.testing.assert_allclose(first.signed_residual_log,np.log(first.observed_price/first.base_price))
    assert meta['text_branch'] is False


def test_completed_panel_keeps_export_data_separate_from_widget_state():
    from streamlit.testing.v1 import AppTest
    source="from pricing.hierarchical_ui import render\nrender('.')"
    app=AppTest.from_string(source)
    app.session_state['studio_tabs']='🌳 Hierarchical model'
    app.session_state['hierarchy_experiment']=experiment(fixture(),'ask')
    app.session_state['hierarchy_residual_export']='synthetic smoke test'
    app.run(timeout=30)
    assert not app.exception
