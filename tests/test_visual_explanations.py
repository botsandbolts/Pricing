"""Contracts for price-scale explanations, derived inputs and chart semantics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from pricing.visual_explanations import GROUPS,ModelView,clean_frame,load_reference,permutation_shap,analyse
from pricing import explanation_plots as plots

ROOT=Path(__file__).resolve().parents[1]

def test_grouped_permutations_reconstruct_non_linear_prediction_without_hybrid_identities():
    background=pd.DataFrame({'make':['a','b'],'model':['aa','bb'],'mileage':[10.,20.]})
    query=pd.DataFrame({'make':['c'],'model':['cc'],'mileage':[35.]})
    original=background.copy(deep=True)
    def predict(frame):
        assert all((a,b) in {('a','aa'),('b','bb'),('c','cc')} for a,b in zip(frame.make,frame.model))
        return frame.mileage.to_numpy()**2+frame.make.map({'a':10,'b':20,'c':30}).to_numpy()*frame.mileage.to_numpy()
    result=permutation_shap(predict,query,background,pairs=8,groups={'Identity':('make','model'),'Mileage':('mileage',)})
    assert result['prediction']==predict(query)[0]
    assert result['base']==pytest.approx(predict(background).mean())
    assert result['base']+result['values'].sum()==pytest.approx(predict(query)[0])
    assert (result['se']>=0).all()
    pd.testing.assert_frame_equal(background,original)

def test_additive_dollar_function_has_exact_feature_allocations():
    query=pd.DataFrame({'year':[2015.],'odometer':[95000.]})
    background=pd.DataFrame({'year':[2010.,2020.],'odometer':[50000.,150000.]})
    predict=lambda x:5000+(x.year.to_numpy()-2000)*1000-x.odometer.to_numpy()*.02
    r=permutation_shap(predict,query,background,pairs=4,groups={'Year':('year',),'Mileage':('odometer',)})
    np.testing.assert_allclose(r['values'],[0,100],atol=1e-8)
    np.testing.assert_allclose(r['se'],0,atol=1e-8)
    assert r['error']<1e-8

def test_reference_decodes_the_existing_real_parity_fixtures():
    reference,provenance=load_reference()
    assert len(reference)==30
    assert 'not a random' in provenance['sampling']
    assert 'price' not in reference and 'text' not in reference and 'VIN' not in reference
    from pricing.tabular import features
    schema=json.loads((ROOT/'models/studio-baseline/schema.json').read_text())
    expected=np.array(json.loads((ROOT/'dist/baseline-fixtures.json').read_text())['rows'],float)
    x=features(reference,schema)
    for name in schema:x[name]=x[name].cat.codes
    np.testing.assert_allclose(x.to_numpy(float),expected,equal_nan=True)
    cleaned=clean_frame(reference.assign(price=999999,text='secret',VIN='secret'))
    pd.testing.assert_frame_equal(reference,cleaned)

@pytest.fixture(scope='module')
def analysis_bundle():
    from pricing.inference import PricePipeline
    pipeline=PricePipeline(ROOT/'models/robust-asking')
    reference,_=load_reference()
    car={'year':2015,'odometer':95000,'manufacturer':'toyota','model':'corolla se',
         'type':'sedan','condition':'unknown','fuel':'gas','transmission':'automatic',
         'drive':'unknown','title_status':'clean','cylinders':'unknown','paint_color':'unknown','state':'ca'}
    a=analyse(pipeline,car,None,reference,pairs=4)
    return pipeline,car,a

def test_full_ensemble_dollars_and_native_member_units_are_distinct(analysis_bundle):
    pipeline,car,a=analysis_bundle
    assert a['local']['prediction']==pytest.approx(pipeline.predict('',car)['price'])
    assert a['local']['error']<1e-5
    assert a['native']['error']<1e-5
    assert a['native']['unit']=='log(1 + price)'
    assert a['local']['evaluated_rows']>1000//2  # Covers batched coalition prediction.
    assert np.isfinite(a['local']['values']).all()
    # Mileage changes must recompute dependent features through the real pipeline.
    scenario={**car,'odometer':float(a['ice']['mileage'][0])}
    assert a['ice']['price'][0]==pytest.approx(pipeline.predict('',scenario)['price'])
    assert np.isfinite(a['heatmap']['price']).all()

def test_all_graphs_preserve_their_scale_and_reconstruction(analysis_bundle):
    _,car,a=analysis_bundle
    native=a['native']
    charts=[plots.waterfall(a['local']),plots.beeswarm(native),plots.dependence(native,'odometer','year'),
            plots.decision(a['decision'],a['comparison']),plots.mileage_curve(a['ice'],car['odometer'],a['local']['prediction']),
            plots.interaction_grid(a['heatmap'])]
    for fig in charts:
        assert len(fig.data)>0
        assert len(fig.to_json())>100
    assert 'USD' in charts[0].layout.xaxis.title.text
    assert 'log(1 + price)' in charts[1].layout.xaxis.title.text
    for i,trace in enumerate(charts[3].data):
        assert trace.x[-1]==pytest.approx(a['decision']['prediction'][i])
    np.testing.assert_allclose(charts[5].data[0].z,a['heatmap']['price'])
