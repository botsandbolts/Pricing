"""Serve the trained asking AFT median, its MiniLM signal, and native log-price SHAP."""
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pricing.suite_storage import read_model_file
from pricing.model_suite import encode,predictors,predict,apply_category_pooling


class AFTPricePipeline:
    def __init__(self,path,device='cpu'):
        import xgboost as xgb
        self.path=Path(path);self.device=device
        self.manifest=json.loads((self.path/'manifest.json').read_text())
        if self.manifest.get('status')!='ready' or self.manifest.get('side')!='ask':
            raise ValueError('Completed asking AFT bundle required')
        self.schema=json.loads((self.path/'schema.json').read_text())
        self.model=xgb.Booster()
        self.model.load_model(bytearray(gzip.decompress(read_model_file(self.path/'xgb_aft_normal.ubj.gz'))))
        self.names=self.model.feature_names
        if self.names!=self.manifest['feature_names']:raise ValueError('AFT feature schema mismatch')
        # XGBoost >=3.1 remembers which categorical labels the fitted booster
        # actually saw. Our saved schema covers the wider original cohort and
        # can contain labels absent from the final fitting subset. Passing the
        # wider pandas Categorical vocabulary makes modern XGBoost raise:
        # "Found a category not in the training set" even on a valid request.
        # Restrict pandas categories to the booster-owned training vocabulary.
        # Unseen values become genuine missing values, NEVER arbitrary codes.
        exported=self.model.get_categories(export_to_arrow=True).to_arrow()
        self._trained_categories={
            name:values.to_pylist() for name,values in exported if values is not None
        }
        for name,levels in self._trained_categories.items():
            if name not in self.schema:continue
            if not levels:raise ValueError('AFT trained categorical vocabulary is empty: '+name)
        self.text_model=self.tokenizer=None
        if self.manifest['text_feature']:
            from pricing.text import load_text
            text_path=self.path.parents[1]/'minilm-cleanlab/text-model'
            provenance=json.loads((text_path/'provenance.json').read_text())
            expected=self.manifest.get('text_model_fingerprint')
            if expected and provenance['fingerprint']!=expected:raise ValueError('AFT text head provenance mismatch')
            self.text_model,self.tokenizer=load_text(text_path,device)

    def encoded(self,frame,text_price=None):
        raw=frame.copy()
        if 'as_of' not in raw:raw['as_of']='2021-05-01'
        x,valid,_=predictors(raw,'ask')
        if not valid.all():raise ValueError('AFT requires a valid vehicle year and valuation date')
        if self.manifest['text_feature']:
            if text_price is None or not np.isfinite(text_price):raise ValueError('Finite description-price feature required')
            x['predicted_text_price']=float(text_price)
        # Preserve ORIGINAL training-time category ordering. Reordering codes
        # to a more limited vocabulary would silently change every split.
        encoded=encode(apply_category_pooling(x,self.manifest.get('category_pooling',{})),self.schema)
        for name,levels in self._trained_categories.items():
            if name in encoded:
                encoded[name]=encoded[name].where(encoded[name].isin(levels))
        return encoded[self.names]

    def _dmatrix(self,frame,text_price=None):
        """Version-stable categorical inference with original saved category codes.

        XGBoost 3.2 may reject unused labels carried in a pandas categorical
        dictionary from a 3.4-trained model. Explicit categorical feature types
        plus ORIGINAL saved training codes preserve every trained split while
        avoiding that incompatible name-recoding path. Unknown names are NaN.
        """
        import xgboost as xgb
        encoded=self.encoded(frame,text_price)
        numeric=encoded.copy()
        for name in self.schema:
            if name in numeric:
                codes=encoded[name].cat.codes.astype(float)
                numeric[name]=codes.where(codes>=0,np.nan)
        return xgb.DMatrix(numeric,enable_categorical=True,nthread=4,
                           feature_types=['c' if name in self.schema else 'q'
                                          for name in self.names])

    def predict_frame(self,frame,text_price=None):
        value=np.asarray(self.model.predict(self._dmatrix(frame,text_price)),float)
        if not np.isfinite(value).all():raise ValueError('Nonfinite AFT estimate')
        return np.clip(value,0,1e7)

    def native(self,frame,text_price=None):
        import xgboost as xgb
        x=self.encoded(frame,text_price);dm=self._dmatrix(frame,text_price)
        values=np.asarray(self.model.predict(dm,pred_contribs=True),float)
        margin=np.asarray(self.model.predict(dm,output_margin=True),float)
        error=float(np.max(abs(values.sum(axis=1)-margin)))
        if error>1e-4:raise ValueError('AFT TreeSHAP failed its reconstruction check')
        return {'x':x,'values':values[:,:-1],'base':values[:,-1],'prediction':margin,
            'error':error,'names':self.names,'unit':'log(price)','member':'XGBoost normal AFT','weight':1.}

    def predict(self,text,fields,explain=False):
        text_price=None
        if self.text_model is not None:
            from pricing.text import predict_text
            text_price=float(predict_text(self.text_model,self.tokenizer,[text],self.device)[0])
        frame=pd.DataFrame([fields]);price=float(self.predict_frame(frame,text_price)[0])
        result={'price':price,'predicted_text_price':text_price,'range':None,'model_name':'XGBoost normal AFT'}
        if explain:
            n=self.native(frame,text_price);x=n['x'];gain=self.model.get_score(importance_type='total_gain');total=sum(gain.values()) or 1.
            raw=float(n['prediction'][0]);base=float(n['base'][0]);contributions=n['values'][0]
            result['explanation']={'method':'XGBoost native TreeSHAP on AFT log-price margin',
                'target':'log','base':base,'raw_prediction':raw,'reconstructed':float(base+sum(contributions)),
                'additivity_error':n['error'],'clipped':bool(np.exp(np.clip(raw,-700,700))>1e7),
                'trees':self.model.num_boosted_rounds(),'features':[
                    {'feature':name,'value':None if pd.isna(x.iloc[0][name]) else
                     (int(x[name].cat.codes.iloc[0]) if name in self.schema else float(x.iloc[0][name])),
                     'contribution':float(contributions[i]),'gain_pct':100*float(gain.get(name,0))/total}
                    for i,name in enumerate(self.names)]}
        return result
