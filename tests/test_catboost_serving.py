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


def test_catboost_text_signal_is_required_and_changes_price(tmp_path,monkeypatch):
    import pytest,sys,types
    raw=pd.DataFrame([{'year':2015,'odometer':95000,'manufacturer':'toyota','model':'corolla','as_of':'2021-05-01'} for _ in range(80)])
    x,_,_=predictors(raw,'ask');x['predicted_text_price']=np.tile([5000.,25000.],40)
    cats=list(x.select_dtypes('object'));y=x.predicted_text_price.to_numpy()
    m=CatBoostRegressor(loss_function='SurvivalAft:dist=Normal;scale=1.0',iterations=80,depth=3,learning_rate=.2,thread_count=2,allow_writing_files=False,verbose=False)
    m.fit(Pool(x,label=np.c_[y,y],cat_features=cats));folder=tmp_path/'models'/'combined'/'ask';folder.mkdir(parents=True)
    model_file=folder/'m.cbm';m.save_model(str(model_file));(folder/'catboost_aft.cbm.gz').write_bytes(gzip.compress(model_file.read_bytes()))
    text_path=tmp_path/'models'/'minilm-cleanlab'/'text-model';text_path.mkdir(parents=True);(text_path/'provenance.json').write_text(json.dumps({'fingerprint':'fixture'}))
    monkeypatch.setitem(sys.modules,'pricing.text',types.SimpleNamespace(load_text=lambda *a:('model','tokenizer'),predict_text=lambda model,tok,texts,device:np.array([25000. if 'excellent' in t else 5000. for t in texts])))
    (folder/'schema.json').write_text(json.dumps({c:sorted(x[c].unique().tolist()) for c in cats}))
    (folder/'manifest.json').write_text(json.dumps({'status':'ready','pipeline':'catboost-aft','side':'ask','feature_names':list(x),'categorical_columns':cats,'text_feature':True,'text_model_path':'minilm-cleanlab/text-model','text_model_fingerprint':'fixture'}))
    p=CatBoostAFTPipeline(folder);car=raw.iloc[0].to_dict()
    low=p.predict('needs repairs',car,explain=True);high=p.predict('excellent service history',car,explain=True)
    from pricing.inference import PricePipeline
    outer=PricePipeline(folder)
    assert outer.text_model is not None
    assert outer.predict('excellent service history',car)['price']==high['price']
    from pricing.residual_ui import estimate
    with_text=raw.iloc[:1].assign(text='excellent service history')
    assert estimate(p,p.schema,p.manifest,with_text,'ask')[0]==high['price']
    assert high['price']>low['price']*2
    assert high['predicted_text_price']==25000.
    assert high['explanation']['additivity_error']<1e-5
    assert 'predicted_text_price' in [f['feature'] for f in high['explanation']['features']]
    assert len(p.predict_frame(raw.iloc[:2],[5000.,25000.]))==2
    for invalid in [None,float('nan'),-1,[5000.,25000.]]:
        with pytest.raises(ValueError):p.predict_frame(raw.iloc[:1],invalid)
