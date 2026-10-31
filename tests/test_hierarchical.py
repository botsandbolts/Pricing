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


def test_original_bundle_loads_without_upload(tmp_path):
    import gzip,json
    from pricing.original_residual_results import load
    b,r,rows=experiment(fixture(),'ask')
    path=tmp_path/'models'/'forest-original'/'ask';path.mkdir(parents=True)
    with gzip.open(path/'model.json.gz','wt') as f:json.dump(b,f)
    (path/'evaluation.json').write_text(json.dumps(r))
    rows.to_csv(path/'examples.csv.gz',index=False)
    loaded=load(tmp_path,'ask')
    assert loaded[0]['fingerprint']==b['fingerprint']
    assert loaded[1]['cohorts']['test']['base']['rows']==len(rows)
    assert len(loaded[2])==len(rows)


def test_xgboost_parallel_forest_and_legacy_wrapper_agree():
    import warnings
    from xgboost import XGBRegressor,XGBRFRegressor
    x=np.arange(120,dtype=float).reshape(40,3);y=np.log1p(x[:,0])
    params=dict(max_depth=3,learning_rate=1.,subsample=.8,colsample_bynode=.8,reg_lambda=10.,random_state=42,n_jobs=2,tree_method='hist')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',FutureWarning)
        old=XGBRFRegressor(n_estimators=8,**params).fit(x,y)
    new=XGBRegressor(n_estimators=1,num_parallel_tree=8,**params).fit(x,y)
    np.testing.assert_allclose(old.predict(x),new.predict(x),rtol=1e-6)
    assert new.get_booster().num_boosted_rounds()==1
    assert len(new.get_booster().get_dump())==8


def test_forest_original_generator_excludes_test_targets():
    from pricing.forest_residuals import generate
    d=fixture();d['odometer']=np.linspace(10000,160000,len(d))
    source=d.drop(columns=['base_price','base_fold','prediction_kind','base_model_id'])
    first,meta=generate(source,'bid')
    source.loc[source.split=='test','observed_price']*=2
    second,_=generate(source,'bid')
    np.testing.assert_allclose(first.base_price,second.base_price)
    assert meta['trees']==200 and meta['boosting_rounds']==1


def test_complete_forest_form_and_two_price_submit():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_string('''
from pricing.forest_form import render
import streamlit as st
fields,notes,date,allow,submitted=render()
if submitted:st.json(fields)
''').run()
    assert not app.exception
    labels={x.label for x in app.text_input}
    assert {'Make','Model / trim','US state abbreviation'}<=labels
    assert len(app.selectbox)>=9
    assert any('Seller description' in x.label for x in app.text_area)
    submit=next(b for b in app.button if b.label=='Estimate asking + auction prices')
    submit.click().run()
    assert not app.exception and len(app.json)==1


def test_live_forest_handles_missing_grade_and_unseen_categories():
    from pricing.forest_live import predict
    from pricing.forest_residuals import CATEGORICAL
    from pricing.hierarchical import fit
    class NativeModel:
        def predict(self,x):
            assert x.auction_grade.dtype.kind=='f' and x.auction_grade.isna().all()
            assert x.manufacturer.isna().all()
            return np.array([np.log(10000.)])
    meta={'category_schema':{c:['known'] for c in CATEGORICAL},'features':['age','log_miles','auction_grade']+CATEGORICAL}
    fields={'manufacturer':'unseen','model':'unknown','odometer':100000,'auction_grade':None}
    result,unknown=predict(NativeModel(),meta,fit(fixture()),fields,5.)
    assert result['base_price']==pytest.approx(10000.)
    assert result['model_n']==0 and 'manufacturer' in unknown


def test_compact_forest_categorical_and_missing_routes():
    from pricing.forest_compact import CompactForest
    import pandas as pd
    arrays={'base':np.array([10],np.float32),'starts':np.array([0,3]),'feature':np.array([0,-1,-1]),'value':np.array([0,1,2],np.float32),'yes':np.array([1,0,0]),'no':np.array([2,0,0]),'missing':np.array([2,0,0]),'cat_offset':np.zeros(3,int),'cat_length':np.array([1,0,0]),'bits':np.array([4],np.uint8)}
    x=pd.DataFrame({'make':pd.Categorical(['c','b',None],categories=['a','b','c'])})
    np.testing.assert_array_equal(CompactForest(arrays).predict(x),[11,12,12])


def test_joint_bayesian_posterior_and_independent_interval_calibration():
    from pricing.bayesian_hierarchy import fit, predict as bayes_predict, assess, interval
    d=fixture();cal=d[d.split=='calibration'];val=d[d.split=='validation'];test=d[d.split=='test']
    meta,draws=fit(cal,warmup=16,draws=25)
    assert meta['method']=='joint_gibbs_hierarchical_gaussian_log_residuals'
    assert len(draws['global'])==50 and len(draws['make'])==50
    summary,_,est=assess(meta,draws,val,test)
    assert 0<=summary['test_coverage']<=1 and summary['validation_rows']==50
    assert (est.predictive_sd_log>0).all()
    low,high=interval(est,summary['validation_multiplier'])
    assert np.all(low<high)
    changed=d.copy();changed.loc[changed.split=='test','observed_price']*=7
    other,other_draws=fit(changed[changed.split=='calibration'],warmup=16,draws=25)
    assert meta==other
    for key in draws:np.testing.assert_array_equal(draws[key],other_draws[key])
    # An unseen model draws from the hierarchical population, widening its range.
    unknown=test.iloc[:1].copy();unknown['manufacturer']='not-a-make';unknown['model']='not-a-model'
    pred=bayes_predict(meta,draws,unknown).iloc[0]
    assert pred.model_n==0 and pred.make_n==0
    assert pred.predictive_sd_log>0
