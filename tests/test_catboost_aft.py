import numpy as np
import pandas as pd
import pytest
pytest.importorskip('catboost')
from catboost import CatBoostRegressor,Pool
from scripts.train_catboost_aft import metrics,residual_summary


def test_aft_dollar_bounds_restore_log_location():
    x=pd.DataFrame({'age':np.arange(100)%10,'make':['a','b']*50})
    y=np.full(100,10000.)
    model=CatBoostRegressor(loss_function='SurvivalAft:dist=Normal;scale=1.0',allow_const_label=True,iterations=200,learning_rate=.2,depth=2,thread_count=2,verbose=False,allow_writing_files=False)
    model.fit(Pool(x,label=np.column_stack([y,y]),cat_features=['make']))
    raw=model.predict(x,prediction_type='RawFormulaVal')
    assert np.allclose(np.exp(raw),y,rtol=.001)


def test_residual_sign_and_dollar_metrics():
    r=metrics(np.array([100.,400.]),np.array([80.,200.]))
    assert r['mae_usd']==110. and r['bias_observed_minus_predicted_usd']==110.
    assert residual_summary(np.array([-2.,0.,2.]))['mean']==0.
