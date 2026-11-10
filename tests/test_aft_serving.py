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


def test_saved_aft_reconciles_training_vocabulary_without_retraining():
    """A common car plus an unseen trim must not trip the XGBoost recoder.

    The price and native explanation share the very same normalized input.
    A never-trained label is missing, not a silently fabricated known model.
    """
    pipeline=PricePipeline('models/price-model-suite/ask')
    aft=pipeline.aft
    fields={'year':2015,'manufacturer':'toyota','model':'corolla','odometer':95000,
            'type':'sedan','transmission':'automatic','state':'ca','fuel':'gas'}
    assert 'model' in aft._trained_categories
    known=aft.encoded(pd.DataFrame([fields]))
    trained=set(aft._trained_categories['model'])
    assert set(known['model'].cat.categories)==set(aft.schema['model'])
    unseen={**fields,'model':'unseen-2026-novel-model-not-in-training'}
    encoded=aft.encoded(pd.DataFrame([unseen]))
    assert pd.isna(encoded['model'].iloc[0])
    assert set(encoded['model'].cat.categories)==set(aft.schema['model'])
    # Both standard and explicit-code paths keep the exact original split codes.
    assert list(aft._dmatrix(pd.DataFrame([fields])).feature_types)[4]=='c'
    # Keep the original training weights; no re-fitting in the serving path.
    old=pipeline.predict('Maintained sedan',fields,explain=True)
    assert np.isfinite(old['price']) and old['price']>0
    assert old['explanation']['additivity_error']<1e-4
