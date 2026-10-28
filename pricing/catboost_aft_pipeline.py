"""Serve real CatBoost SurvivalAft models and native log-price TreeSHAP."""
import gzip,json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor,Pool
from pricing.suite_storage import read_model_file
from pricing.model_suite import predictors,apply_category_pooling


class CatBoostAFTPipeline:
    def __init__(self,path,device='cpu'):
        self.path=Path(path);self.device=device;self.manifest=json.loads((self.path/'manifest.json').read_text())
        if self.manifest.get('status')!='ready' or self.manifest.get('pipeline')!='catboost-aft':raise ValueError('Ready CatBoost AFT bundle required')
        self.schema=json.loads((self.path/'schema.json').read_text());self.model=CatBoostRegressor();self.model.load_model(blob=gzip.decompress(read_model_file(self.path/'catboost_aft.cbm.gz')))
        self.names=self.model.feature_names_;self.text_model=self.tokenizer=None
        if self.manifest.get('text_feature'):
            from pricing.text import load_text
            text_path=self.path.parents[1]/self.manifest['text_model_path']
            provenance=json.loads((text_path/'provenance.json').read_text())
            if provenance['fingerprint']!=self.manifest['text_model_fingerprint']:
                raise ValueError('CatBoost/MiniLM provenance mismatch')
            self.text_model,self.tokenizer=load_text(text_path,device)
        if self.names!=self.manifest['feature_names']:raise ValueError('CatBoost feature schema mismatch')
    def encoded(self,frame,text_price=None):
        raw=frame.copy()
        if 'as_of' not in raw:raw['as_of']='2021-05-01'
        x,valid,_=predictors(raw,self.manifest['side'])
        if not valid.all():raise ValueError('Unsupported vehicle year / valuation date')
        if self.manifest.get('text_feature'):
            values=np.asarray(text_price,dtype=float)
            if values.ndim==0:values=np.repeat(values,len(x))
            if values.shape!=(len(x),) or not np.isfinite(values).all() or (values<0).any():
                raise ValueError('One finite nonnegative MiniLM text price per car is required')
            x['predicted_text_price']=values
        x=apply_category_pooling(x,self.manifest.get('category_pooling',{}))
        for c in self.manifest['categorical_columns']:x[c]=x[c].fillna('unknown').astype(str)
        return x[self.names]
    def predict_frame(self,frame,text_price=None):
        raw=np.asarray(self.model.predict(self.encoded(frame,text_price),prediction_type='RawFormulaVal',thread_count=4),float)
        if not np.isfinite(raw).all():raise ValueError('Nonfinite CatBoost output')
        return np.exp(np.clip(raw,-700,np.log(1e7)))
    def native(self,frame,text_price=None):
        x=self.encoded(frame,text_price);pool=Pool(x,cat_features=self.manifest['categorical_columns']);v=np.asarray(self.model.get_feature_importance(pool,type='ShapValues',thread_count=4),float)
        raw=np.asarray(self.model.predict(x,prediction_type='RawFormulaVal',thread_count=4),float);error=float(np.max(abs(v.sum(axis=1)-raw)))
        if error>1e-5:raise ValueError('CatBoost native SHAP reconstruction failed')
        return {'x':x,'values':v[:,:-1],'base':v[:,-1],'prediction':raw,'error':error,'names':self.names,'unit':'log(price)','member':'CatBoost normal AFT','weight':1.}
    def predict(self,text,fields,explain=False):
        text_price=None
        if self.text_model is not None:
            from pricing.text import predict_text
            text_price=float(predict_text(self.text_model,self.tokenizer,[text],self.device)[0])
        frame=pd.DataFrame([fields]);price=float(self.predict_frame(frame,text_price)[0]);result={'price':price,'predicted_text_price':text_price,'range':None,'model_name':('MiniLM + CatBoost normal AFT' if self.text_model is not None else 'CatBoost normal AFT')}
        if explain:
            n=self.native(frame,text_price);importance=np.asarray(self.model.get_feature_importance(type='PredictionValuesChange',thread_count=4),float);den=float(importance.sum()) or 1.;raw=float(n['prediction'][0]);base=float(n['base'][0])
            result['explanation']={'method':'CatBoost native TreeSHAP on AFT log-price margin','importance_kind':'CatBoost PredictionValuesChange','target':'log','base':base,'raw_prediction':raw,'reconstructed':float(base+n['values'][0].sum()),'additivity_error':n['error'],'clipped':raw>np.log(1e7),'trees':self.model.tree_count_,'features':[{'feature':name,'value':None if pd.isna(n['x'].iloc[0][name]) else (str(n['x'].iloc[0][name]) if name in self.schema else float(n['x'].iloc[0][name])),'contribution':float(n['values'][0][i]),'gain_pct':float(100*importance[i]/den)} for i,name in enumerate(self.names)]}
        return result
