"""Behavioral regression checks for the larger ensemble."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from marketbridge.structural import StructuralTransfer
from marketbridge.robust import RobustAsking,engineer
from pricing.tabular import fit_schema,CATEGORICAL
ROOT=Path(__file__).resolve().parents[1]

def car():return {'year':2014,'manufacturer':'toyota','model':'camry','odometer':95000,'state':'ca','type':'sedan','transmission':'automatic','as_of':'2021-05-01'}

@pytest.fixture(scope='module')
def model():
    path=ROOT/'models/robust-asking'
    if not (path/'manifest.json').exists():pytest.skip('Larger ensemble is not ready')
    return RobustAsking(path,StructuralTransfer(ROOT/'models/structural-transfer'))

def test_member_weights_are_a_convex_usd_ensemble(model):
    frame=pd.DataFrame([car(),{**car(),'odometer':125000}])
    members=model.member_predictions(frame);weights=np.array([m['weight'] for m,_ in model.members])
    assert weights.min()>0;assert weights.sum()==pytest.approx(1)
    np.testing.assert_allclose(model.predict_frame(frame),members@weights)
    assert np.isfinite(members).all() and (members>=0).all()

def test_no_outcome_or_description_leakage(model):
    frame=pd.DataFrame([car(),{**car(),'price':150000,'sale_price':1,'text':'sold for $100000','mmr':999999}])
    values=model.predict_frame(frame);assert values[0]==values[1]

def test_unknown_categories_and_missing_mileage_are_supported(model):
    f=car();f.update(manufacturer='unseenmake',model='unseenmodel',odometer=None)
    assert np.isfinite(model.predict_frame(pd.DataFrame([f]))).all()

def test_composite_counts_use_fitted_vocabularies_only(model):
    f=pd.DataFrame([car(),{**car(),'model':'never_seen_model_in_training'}])
    x,fs=engineer(f,model.schema,model.structural,model.feature_schema)
    assert x.make_model.cat.codes.iloc[1]==-1
    assert x.make_model_log_count.iloc[1]==0
    assert fs==model.feature_schema

def test_bundle_records_selection_and_benchmark_limits(model):
    report=json.loads((ROOT/'models/robust-asking/evaluation.json').read_text())
    assert report['seed_repetitions']==[42,137,777]
    assert report['target_kind']=='asking_price'
    assert 'earlier project experiments' in report['test_note']
    assert report['structural_model_id']==model.structural.model_id
    assert report['rows']==211844


def test_member_explanations_reconstruct_on_their_own_scales(model):
    from marketbridge.robust_explanations import explain_member
    for i in range(len(model.members)):
        result=explain_member(model,pd.DataFrame([car()]),i)
        assert result['additivity_error']<1e-6
        assert result['member_price_usd']>=0


def test_shared_engine_uses_promotion_flag_and_keeps_baseline(model):
    from marketbridge.engine import MarketComparison
    engine=MarketComparison(ROOT);out=engine.predict_frame(pd.DataFrame([car()])).iloc[0]
    if engine.robust.manifest['promoted']:
        assert out.asking_model_kind=='regularized_lightgbm_ensemble'
        assert out.asking_price_usd==pytest.approx(out.robust_asking_price_usd)
        assert out.asking_model_id==engine.ids['robust_asking']
    else:
        assert out.asking_model_kind=='original_tabular_baseline'
        assert out.asking_price_usd==out.baseline_asking_price_usd


def test_isolation_filter_excludes_labels_and_is_reproducible():
    from marketbridge.robust import isolation_inliers
    rng=np.random.default_rng(5)
    frame=pd.DataFrame({'year':rng.normal(2015,2,300),
                        'odometer':rng.normal(60000,5000,300),
                        'transported_auction_price_usd':rng.normal(15000,1000,300)})
    frame.loc[299,['year','odometer']]=[1900,9000000]
    keep,report=isolation_inliers(frame)
    altered=frame.assign(price=np.arange(300)*1000000, sale_price=-1, text='outcome')
    again,other=isolation_inliers(altered)
    pd.testing.assert_series_equal(keep,again)
    assert report==other
    assert not keep.loc[299]
    assert 0<report['removed_rows']<len(frame)
    assert 'transported_auction_price_usd' in report['features']
    assert 'price' not in report['features']


def test_filtered_bundle_requires_completed_passing_assessment(tmp_path):
    from marketbridge.engine import best_asking_bundle
    original=tmp_path/'models/robust-asking'
    filtered=tmp_path/'models/robust-asking-isolation'
    assert best_asking_bundle(tmp_path)==original
    filtered.mkdir(parents=True)
    for status,promoted,expected in [('pending',True,original),('ready',False,original),('ready',True,filtered)]:
        (filtered/'manifest.json').write_text(json.dumps({'status':status,'promoted':promoted}))
        assert best_asking_bundle(tmp_path)==expected


def test_trained_isolation_bundle_is_consistent():
    path=ROOT/'models/robust-asking-isolation'
    if not (path/'manifest.json').exists():pytest.skip('Filtered bundle not installed')
    manifest=json.loads((path/'manifest.json').read_text())
    report=json.loads((path/'evaluation.json').read_text())
    assert manifest['training_fingerprint']==report['training_fingerprint']
    assert manifest['trained_rows']==manifest['outlier_filter']['retained_rows']
    assert report['outlier_filter']['input_rows']+report['validation_rows']+report['test_rows']==report['rows']
    assert report['outlier_filter']['retained_rows']==report['train_rows']
    assert report['outlier_filter']['removed_rows']>0
    assert 'price' not in report['outlier_filter']['features']
    filtered=RobustAsking(path,StructuralTransfer(ROOT/'models/structural-transfer'))
    assert len(filtered.members)==report['ensemble_members']
    frame=pd.DataFrame([car(),{**car(),'price':999999,'description':'asking for $999999'}])
    predictions=filtered.predict_frame(frame)
    assert np.isfinite(predictions).all() and (predictions>0).all()
    assert predictions[0]==predictions[1]
    from marketbridge.robust_explanations import explain_member
    assert explain_member(filtered,frame.iloc[:1],0)['additivity_error']<1e-6
