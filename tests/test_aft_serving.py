"""The served AFT price and every graph must explain the same log-price function."""
import numpy as np
import pandas as pd
from pricing.inference import PricePipeline
from pricing.visual_explanations import ModelView,analyse,load_reference


def test_installed_aft_and_explanations_match():
    pipeline=PricePipeline('models/price-model-suite/ask')
    fields={'year':2015,'manufacturer':'toyota','model':'corolla','odometer':95000,
        'type':'sedan','transmission':'automatic','state':'ca','fuel':'gas'}
    result=pipeline.predict('Maintained sedan with service records and clean interior.',fields,explain=True)
    assert result['price']>0 and result['explanation']['target']=='log'
    assert result['explanation']['additivity_error']<1e-4
    e=result['explanation']
    assert np.isclose(np.exp(e['raw_prediction']),result['price'],rtol=1e-6)
    view=ModelView(pipeline,result['predicted_text_price'])
    assert view.unit=='log(price)'
    assert np.isclose(view.predict(pd.DataFrame([fields]))[0],result['price'])
    ref,_=load_reference()
    graphs=analyse(pipeline,fields,result['predicted_text_price'],ref.iloc[:4],pairs=2)
    assert np.isclose(graphs['local']['prediction'],result['price'])
    assert graphs['native']['error']<1e-4
    assert graphs['unit']=='log(price)'
    assert np.isfinite(graphs['heatmap']['price']).all()
