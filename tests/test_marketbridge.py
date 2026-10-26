from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from marketbridge.structural import StructuralTransfer,prepare,hierarchy,IndexPanel
from marketbridge.engine import MarketComparison
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def structural():return StructuralTransfer(ROOT/'models/structural-transfer')
@pytest.fixture(scope='module')
def engine():return MarketComparison(ROOT)
def car():return dict(year=2014,manufacturer='Toyota',model='Camry',odometer=95000,state='ca',type='sedan',transmission='automatic',condition='excellent',as_of='2022-05-01')

def test_same_model_year_has_different_age_and_constant_market_value(structural):
    f=car();a=structural.predict_frame(pd.DataFrame([{**f,'as_of':'2015-05-01'},f]))
    assert a.iloc[1].vehicle_age_years-a.iloc[0].vehicle_age_years==pytest.approx(7.,abs=.003)
    assert a.iloc[1].auction_reference_2015_usd<a.iloc[0].auction_reference_2015_usd
    np.testing.assert_allclose(a.transported_auction_price_usd,a.auction_reference_2015_usd*a.market_index_factor)
    assert a.index_month_used.tolist()==['2015-03','2022-03']

def test_monotone_mileage_and_grade(structural):
    f=car();f['auction_condition']=3.
    a=structural.predict_frame(pd.DataFrame([{**f,'odometer':m} for m in [10000,65000,150000,250000]]))
    assert (np.diff(a.auction_reference_2015_usd)<=1e-7).all()
    a=structural.predict_frame(pd.DataFrame([{**f,'auction_condition':g} for g in [1.,2.,3.,4.,5.]]))
    assert (np.diff(a.auction_reference_2015_usd)>=-1e-7).all()

def test_no_raw_modelyear_feature_and_condition_not_guessed(structural):
    assert 'year' not in structural.model.feature_name()
    x,_,_,_=prepare(pd.DataFrame([car(),{**car(),'auction_condition':4}]))
    assert pd.isna(x.auction_grade.iloc[0]);assert x.auction_grade.iloc[1]==4

def test_outcome_text_and_prices_do_not_change_predictions(engine):
    f=car();r=engine.predict_frame(pd.DataFrame([f,{**f,'price':999999,'sale_price':1,'text':'SOLD FOR $999999','mmr':1e6}]))
    for col in ['transported_auction_price_usd','asking_price_usd','asking_with_auction_structure_usd']:
        assert r[col].iloc[0]==r[col].iloc[1]

def test_future_index_missing_date_and_invalid_rows_are_preserved(engine):
    f=car();r=engine.predict_frame(pd.DataFrame([{**f,'row_id':'a'},{**f,'as_of':'2028-05-01','row_id':'b'},{**f,'as_of':None,'row_id':'c'},{**f,'odometer':-1,'row_id':'d'}],index=[9,2,8,3]))
    assert r.index.tolist()==[9,2,8,3];assert r.row_id.tolist()==list('abcd')
    assert r.transported_auction_price_usd.iloc[1:].isna().all()
    assert r.status.iloc[1]=='market_index_unavailable';assert r.status.iloc[2]=='missing_valuation_date'
    assert not r.retail_sale_calibrated.any();assert not r.target_coverage_validated.any()

def test_decomposition_reconstructs_transport(structural):
    steps=structural.decompose(car());r=structural.predict_frame(pd.DataFrame([car()])).iloc[0]
    assert steps[-1]['price_usd']==pytest.approx(r.transported_auction_price_usd)
    assert steps[0]['price_usd']*np.exp(sum(v['log_effect'] for v in steps))==pytest.approx(r.transported_auction_price_usd)

def test_unknown_generation_and_hierarchical_fallback(structural):
    f={**car(),'year':2021,'model':'unknownnewmodel'};r=structural.predict_frame(pd.DataFrame([f])).iloc[0]
    assert np.isfinite(r.transported_auction_price_usd);assert not r.model_known;assert not r.generation_supported
    assert r.model_training_count==0

def test_partial_pooling_shrinks_small_groups():
    keys=pd.DataFrame({'make':['a']*101,'model':['a::x']*100+['a::y']});y=np.r_[np.zeros(100),1.]
    h=hierarchy(keys,y);assert 0<h['model']['a::y']<.1

def test_deployed_bundle_and_asking_experiment_version(engine):
    report=json.loads((ROOT/'models/transfer-asking/evaluation.json').read_text())
    assert report['structural_model_id']==engine.structural.model_id
    source=json.loads((ROOT/'models/structural-transfer/evaluation.json').read_text())
    assert report['known_upstream_vin_exclusions']==source['known_target_vin_overlaps_excluded']
    assert report['test_rows']==31777
    assert report['target_kind']=='asking_price'
    assert engine.enhanced.model.feature_name()[-8:]==['vehicle_age_years','auction_reference_2015_usd','transported_auction_price_usd','relative_model_value_factor','model_training_count','mileage_vs_age_typical','one_year_age_effect_pct','market_index_factor']
