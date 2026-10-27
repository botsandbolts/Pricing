"""Prediction-scale permutation Shapley estimates and native member TreeSHAP.

No price labels are read. Grouped raw-input masking recomputes derived features.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from pricing.tabular import NUMERIC, CATEGORICAL, features

FIELDS=NUMERIC+CATEGORICAL
GROUPS={
    'Make, model & body':('manufacturer','model','type'),
    'Model year':('year',), 'Mileage':('odometer',),
    'Condition':('condition',),
    'Engine & drivetrain':('fuel','cylinders','transmission','drive'),
    'Title status':('title_status',), 'Paint colour':('paint_color',),
    'Location':('state',),
}
LABELS={'year':'Model year','odometer':'Mileage (miles)','manufacturer':'Make',
        'model':'Model / trim','type':'Body type','paint_color':'Paint colour',
        'predicted_text_price':'Description-price signal','vehicle_age_years':'Vehicle age',
        'transported_auction_price_usd':'CarBids auction equivalent',
        'auction_reference_2015_usd':'CarBids reference value'}

def label(name):return LABELS.get(name,name.replace('_',' ').capitalize())

def clean_frame(frame):
    """Strict predictor allowlist; uploaded outcomes and personal fields are ignored."""
    if not isinstance(frame,pd.DataFrame) or frame.empty:raise ValueError('Provide at least one reference car')
    x=frame.reindex(columns=FIELDS).copy().reset_index(drop=True)
    for c in NUMERIC:
        x[c]=pd.to_numeric(x[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
    for c in CATEGORICAL:
        x[c]=x[c].astype('string').str.lower().str.strip().replace('',pd.NA)
    return x

def load_reference(path=None):
    path=Path(path or Path(__file__).resolve().parents[1]/'models/explanations/reference-cars.json')
    data=json.loads(path.read_text())
    return clean_frame(pd.DataFrame(data['cars'])),data['provenance']

class ModelView:
    """Expose the actual served dollar function and a selected native tree model."""
    def __init__(self,pipeline,text_price=None,member_index=0):
        self.pipeline=pipeline
        self.ensemble=getattr(pipeline,'ensemble',None)
        if self.ensemble is None and getattr(pipeline,'_ensemble',None) is not None:
            self.ensemble=pipeline._ensemble.ensemble
        self.text_price=text_price
        self.aft=getattr(pipeline,'aft',None)
        if self.aft is not None:
            self.model=self.aft.model;self.member=None;self.weight=1.
            self.target='log';self.mean=0.;self.std=1.
            self.member_name='XGBoost normal AFT';self.schema=self.aft.schema
            self.conditional_text=self.aft.manifest['text_feature']
            if self.conditional_text and (text_price is None or not np.isfinite(text_price)):
                raise ValueError('A submitted description-price signal is required for AFT')
            return
        if self.ensemble is not None:
            self.member,self.model=self.ensemble.members[member_index]
            self.target=self.member['target'];self.mean=self.member['mean'];self.std=self.member['std']
            self.member_name=self.member['file'];self.weight=self.member['weight']
            self.schema=self.ensemble.schema
        else:
            self.model=pipeline.lgb;self.member=None;self.weight=1.
            self.target=pipeline.manifest.get('target_transform','log1p')
            self.mean=pipeline.manifest.get('target_mean',0.);self.std=pipeline.manifest.get('target_std',1.)
            self.member_name='Selected LightGBM model';self.schema=pipeline.schema
        self.conditional_text='predicted_text_price' in self.model.feature_name()
        if self.conditional_text and (text_price is None or not np.isfinite(text_price)):
            raise ValueError('A submitted description-price signal is required for this model')
    @property
    def unit(self):return 'log(price)' if self.target=='log' else 'log(1 + price)' if self.target=='log1p' else 'USD'
    def encoded(self,frame):
        if self.aft is not None:return self.aft.encoded(clean_frame(frame),self.text_price)
        frame=clean_frame(frame)
        if self.ensemble is not None:
            from marketbridge.robust import engineer
            frame['as_of']='2021-05-01'
            return engineer(frame,self.schema,self.ensemble.structural,self.ensemble.feature_schema)[0]
        text=np.full(len(frame),self.text_price) if self.conditional_text else None
        return features(frame,self.schema,text)
    def predict(self,frame):
        if self.aft is not None:return self.aft.predict_frame(clean_frame(frame),self.text_price)
        frame=clean_frame(frame)
        if self.ensemble is not None:
            frame['as_of']='2021-05-01'
            return np.asarray(self.ensemble.predict_frame(frame),float)
        score=np.asarray(self.model.predict(self.encoded(frame),num_threads=4))*self.std+self.mean
        return np.expm1(np.clip(score,0,np.log1p(1e7))) if self.target=='log1p' else np.clip(score,0,1e7)
    def native(self,frame):
        if self.aft is not None:return self.aft.native(clean_frame(frame),self.text_price)
        x=self.encoded(frame)
        raw=np.asarray(self.model.predict(x,pred_contrib=True,num_threads=4),float)
        contributions=raw[:,:-1]*self.std;base=raw[:,-1]*self.std+self.mean
        prediction=np.asarray(self.model.predict(x,num_threads=4))*self.std+self.mean
        error=float(np.max(np.abs(base+contributions.sum(axis=1)-prediction)))
        if error>1e-5:raise ValueError('Native TreeSHAP failed its reconstruction check')
        return {'x':x,'values':contributions,'base':base,'prediction':prediction,'error':error,
                'names':list(x.columns),'unit':self.unit,'member':self.member_name,'weight':self.weight}

def permutation_shap(predict,query,background,pairs=8,seed=42,groups=None):
    """Antithetic permutation Shapley estimates in predict() output units.

    Background is fixed. SE reflects permutation-order Monte Carlo variation,
    not background selection, model uncertainty, or a prediction interval.
    """
    groups=groups or GROUPS
    columns=[c for fields in groups.values() for c in fields]
    if len(set(columns))!=len(columns):raise ValueError('Explanation groups must not overlap')
    query=query.reindex(columns=columns).iloc[:1].copy()
    background=background.reindex(columns=columns).reset_index(drop=True).copy()
    if len(background)==0 or pairs<2:raise ValueError('Use reference cars and at least two permutation pairs')
    g=len(groups);n=len(background);rng=np.random.default_rng(seed);frames=[];orders=[]
    for _ in range(pairs):
        order=rng.permutation(g)
        for path in [order,order[::-1]]:
            state=background.copy();frames.append(state.copy());orders.append(path.copy())
            for j in path:
                for field in list(groups.values())[j]:state[field]=query.iloc[0][field]
                frames.append(state.copy())
    predictions=np.asarray(predict(pd.concat(frames,ignore_index=True)),float).reshape(pairs,2,g+1,n)
    if not np.isfinite(predictions).all():raise ValueError('Model produced nonfinite prices')
    deltas=np.diff(predictions,axis=2);allocated=np.zeros((pairs,2,g,n))
    for k,order in enumerate(orders):
        for step,j in enumerate(order):allocated[k//2,k%2,j,:]=deltas[k//2,k%2,step,:]
    per_pair=allocated.mean(axis=(1,3));values=per_pair.mean(axis=0)
    se=per_pair.std(axis=0,ddof=1)/np.sqrt(pairs)
    base=float(predictions[:,:,0,:].mean());prediction=float(np.asarray(predict(query))[0])
    error=float(abs(base+values.sum()-prediction))
    if error>max(1e-6,abs(prediction)*1e-9):raise ValueError('Dollar explanation failed reconstruction')
    return {'names':list(groups),'values':values,'se':se,'base':base,'prediction':prediction,
            'error':error,'orders':2*pairs,'background_rows':n,'evaluated_rows':len(frames)*n,
            'method':'Grouped interventional permutation Shapley estimates on the complete served USD function',
            'seed':seed}

def analyse(pipeline,fields,text_price,reference,member_index=0,pairs=8):
    reference=clean_frame(reference)
    if len(reference)>128:reference=reference.iloc[np.linspace(0,len(reference)-1,128,dtype=int)].reset_index(drop=True)
    query=clean_frame(pd.DataFrame([fields]));view=ModelView(pipeline,text_price,member_index)
    indices=np.linspace(0,len(reference)-1,min(8,len(reference)),dtype=int)
    local=permutation_shap(view.predict,query,reference.iloc[indices],pairs=pairs)
    native=view.native(reference)
    # Compare with nearby reference cars by year and log-mileage, not outcomes.
    year=reference.year.fillna(reference.year.median()).fillna(2010)
    miles=reference.odometer.clip(lower=0).fillna(reference.odometer.median()).fillna(0)
    qyear=float(query.year.fillna(2010).iloc[0]);qmiles=float(query.odometer.fillna(0).iloc[0])
    distance=((year-qyear)/10)**2+((np.log1p(miles)-np.log1p(max(0,qmiles)))/2)**2
    peers=reference.iloc[np.argsort(distance.to_numpy())[:4]]
    comparison=pd.concat([query,peers],ignore_index=True);decision=view.native(comparison)
    # True model sensitivities in dollars, with derived features recomputed each time.
    low,high=np.nanquantile(miles,[.05,.95]);high=max(high,low+1000)
    mileage_grid=np.unique(np.r_[np.linspace(max(0,low),min(2e6,high),21),qmiles])
    scenarios=pd.concat([query]*len(mileage_grid),ignore_index=True);scenarios['odometer']=mileage_grid
    ice={'mileage':mileage_grid,'price':view.predict(scenarios)}
    center=int(np.clip(qyear,1900,2021));years=np.arange(max(1900,center-4),min(2021,center+4)+1)
    grid_miles=np.linspace(max(0,low),min(2e6,high),9)
    grid=pd.concat([query]*(len(years)*len(grid_miles)),ignore_index=True)
    grid['year']=np.repeat(years,len(grid_miles));grid['odometer']=np.tile(grid_miles,len(years))
    heatmap={'years':years,'mileage':grid_miles,'price':view.predict(grid).reshape(len(years),len(grid_miles))}
    return {'local':local,'native':native,'decision':decision,'comparison':comparison,
            'ice':ice,'heatmap':heatmap,'reference':reference,'conditional_text':view.conditional_text,
            'member_index':member_index,'unit':view.unit}
