"""Load one completed bundle; calculate the text feature dynamically on every request."""
import json
import gzip
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from pricing.text import load_text, predict_text
from pricing.tabular import features


class PricePipeline:
    def __init__(self,path,device='cpu'):
        path=Path(path)
        self.manifest=json.loads((path/'manifest.json').read_text())
        if self.manifest.get('status')!='ready' or self.manifest.get('pipeline') not in ['fine-tuned-e5-base+lightgbm','tabular-lightgbm']:
            raise ValueError('A completed E5-base + LightGBM bundle is required')
        self.device=device
        self.text_model=self.tokenizer=None
        if self.manifest['pipeline']=='fine-tuned-e5-base+lightgbm':
            self.text_model,self.tokenizer=load_text(path/'text-model',device)
        self.schema=json.loads((path/'schema.json').read_text())
        if (path/'lightgbm.txt').exists():
            self.lgb=lgb.Booster(model_file=str(path/'lightgbm.txt'))
        else:
            with gzip.open(path/'lightgbm.txt.gz','rt') as f:
                self.lgb=lgb.Booster(model_str=f.read())
        if self.lgb.feature_name()!=self.manifest['feature_names']:
            raise ValueError('LightGBM feature schema mismatch')

    def predict(self,text,fields):
        # Description enters E5 alone; no serialization of form fields into its input.
        text_price=float(predict_text(self.text_model,self.tokenizer,[text],self.device)[0]) if self.text_model else None
        frame=pd.DataFrame([fields])
        for col in ['year','odometer']+list(self.schema):
            if col not in frame:frame[col]=None
        x=features(frame,self.schema,None if text_price is None else [text_price])
        raw=float(self.lgb.predict(x)[0])
        scaled=raw*self.manifest.get('target_std',1.)+self.manifest.get('target_mean',0.)
        transform=self.manifest.get('target_transform','log1p')
        if transform not in ['log1p','dollars']:raise ValueError('Unknown target transform')
        final=float(np.expm1(np.clip(scaled,0,np.log1p(1e7)))) if transform=='log1p' else float(np.clip(scaled,0,1e7))
        log_price=np.log1p(final)
        b=self.manifest['band80_log_radius']
        invert=lambda z:float(np.expm1(np.clip(z,0,np.log1p(1e7))))
        return {'predicted_text_price':text_price,'price':final,
                'range':[invert(log_price-b),invert(log_price+b)]}

