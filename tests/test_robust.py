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
    if model.manifest['promoted']:
        assert out.asking_model_kind=='regularized_lightgbm_ensemble'
        assert out.asking_price_usd==pytest.approx(out.robust_asking_price_usd)
        assert out.asking_model_id==engine.ids['robust_asking']
    else:
        assert out.asking_model_kind=='original_tabular_baseline'
        assert out.asking_price_usd==out.baseline_asking_price_usd
