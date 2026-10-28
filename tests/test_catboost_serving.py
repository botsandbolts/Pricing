import gzip,json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor,Pool
from pricing.model_suite import predictors
from pricing.catboost_aft_pipeline import CatBoostAFTPipeline


def test_catboost_aft_serving_and_native_shap(tmp_path):
    raw=pd.DataFrame([{'year':2005+i%15,'odometer':20000+i*1000,'manufacturer':['ford','toyota'][i%2],'model':['focus','corolla'][i%2],'as_of':'2021-05-01'} for i in range(100)])
    x,valid,_=predictors(raw,'ask');cats=list(x.select_dtypes('object'))
    y=np.exp(10-.03*x.age.to_numpy()-.05*x.log_miles.to_numpy())
    m=CatBoostRegressor(loss_function='SurvivalAft:dist=Normal;scale=1.0',iterations=25,depth=3,thread_count=2,allow_writing_files=False,verbose=False)
    m.fit(Pool(x,label=np.c_[y,y],cat_features=cats));m.save_model(str(tmp_path/'m.cbm'))
    (tmp_path/'catboost_aft.cbm.gz').write_bytes(gzip.compress((tmp_path/'m.cbm').read_bytes()))
    (tmp_path/'schema.json').write_text(json.dumps({c:sorted(x[c].unique().tolist()) for c in cats}))
    (tmp_path/'manifest.json').write_text(json.dumps({'status':'ready','pipeline':'catboost-aft','side':'ask','feature_names':list(x),'categorical_columns':cats,'text_feature':False}))
    p=CatBoostAFTPipeline(tmp_path);r=p.predict('',raw.iloc[0].to_dict(),explain=True)
    assert r['price']>0 and r['explanation']['additivity_error']<1e-5
    assert r['explanation']['importance_kind']=='CatBoost PredictionValuesChange'
    assert abs(sum(f['gain_pct'] for f in r['explanation']['features'])-100)<1e-6
