"""Generate group-held-out residuals with a fresh, tabular CatBoost assessment.

Never use an all-phase deployment refit to manufacture an honest test. The web
pilot excludes MiniLM; a nested text refit would be required for end-to-end OOF.
"""
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from pricing.hierarchical import validate

CATEGORICAL=['manufacturer','model','type','transmission','state','fuel','drive','title_status']
OBSERVATION_COLUMNS=['row_id','vehicle_group','split','manufacturer','model','age','odometer','observed_price','price_basis']


def generate(frame,side,*,iterations=250):
    from catboost import CatBoostRegressor,Pool
    if side not in {'ask','bid'}:raise ValueError('Choose ask or bid.')
    if len(frame)>10000:raise ValueError('CPU pilot supports at most 10,000 rows. Use offline training for larger cohorts.')
    missing=set(OBSERVATION_COLUMNS)-set(frame)
    if missing:raise ValueError('Missing observation columns: '+', '.join(sorted(missing)))
    source=frame.reset_index(drop=True).copy()
    source['odometer']=pd.to_numeric(source.odometer,errors='raise')
    if not source.odometer.between(0,2000000).all():raise ValueError('Invalid mileage.')
    train=source.split.eq('calibration')
    source['base_price']=1.;source['base_model_id']='pending_assessment'
    source['prediction_kind']=np.where(train,'oof','heldout')
    source['base_fold']=pd.factorize(source.vehicle_group)[0]%3
    d=validate(source)
    train_idx=np.flatnonzero(train);held_idx=np.flatnonzero(~train)
    if source.loc[train,'vehicle_group'].nunique()<3:raise ValueError('Need at least three calibration vehicle groups.')
    x=pd.DataFrame({'age':d.age,'log_miles':np.log1p(source.odometer)})
    for col in CATEGORICAL:x[col]=source[col].fillna('unknown').astype(str).str.strip().str.lower() if col in source else 'unknown'
    loss='SurvivalAft:dist=Normal;scale=1.0' if side=='ask' else 'Quantile:alpha=0.5'
    params=dict(loss_function=loss,iterations=iterations,depth=6,learning_rate=.05,
        l2_leaf_reg=10,random_seed=42,thread_count=2,allow_writing_files=False,verbose=False)
    y=d.observed_price.to_numpy(float)
    def fit(idx):
        labels=np.column_stack([y[idx],y[idx]]) if side=='ask' else np.log(y[idx])
        return CatBoostRegressor(**params).fit(Pool(x.iloc[idx],label=labels,cat_features=CATEGORICAL))
    def pred(model,idx):return np.exp(model.predict(x.iloc[idx],prediction_type='RawFormulaVal'))
    predictions=np.full(len(d),np.nan);fold_ids=np.full(len(d),-1)
    for fold,(a,b) in enumerate(GroupKFold(3).split(train_idx,groups=d.vehicle_group.iloc[train_idx])):
        fit_idx=train_idx[a];out_idx=train_idx[b]
        predictions[out_idx]=pred(fit(fit_idx),out_idx);fold_ids[out_idx]=fold
    predictions[held_idx]=pred(fit(train_idx),held_idx)
    digest=hashlib.sha256((x.iloc[train_idx].to_json()+json.dumps(params,sort_keys=True)+y[train_idx].tobytes().hex()).encode()).hexdigest()
    d['base_model_id']='catboost-tabular-assessment-'+digest
    d['base_price']=predictions;d['base_fold']=fold_ids
    d['signed_residual_usd']=d.observed_price-d.base_price
    d['signed_residual_log']=np.log(d.observed_price/d.base_price)
    validate(d)
    return d,{'method':'fresh_group_cross_fitted_tabular_catboost','side':side,'parameters':params,
        'features':list(x),'calibration_folds':3,'text_branch':False,
        'warning':'Separate tabular assessment, not the installed MiniLM/five-stage deployment model. No outlier trimming. Held-out rows never fit the base model or hierarchy.'}
