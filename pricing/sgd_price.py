"""Group-cross-fitted SGD-Huber log-price baseline on original vehicle facts."""
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from pricing.hierarchical import validate
from pricing.sgd_residuals import fit as fit_regression,predict as predict_regression,CATEGORIES,describe

CATEGORICAL=CATEGORIES
OBSERVATION_COLUMNS=['row_id','vehicle_group','split','manufacturer','model','age','odometer','observed_price','price_basis']


def fit_price(frame):
    copy=frame.copy();copy['base_price']=1.
    center=float(np.log(copy.observed_price).mean())
    copy['observed_price']=copy.observed_price/np.exp(center)
    model=fit_regression(copy)
    model.named_steps['regress'].intercept_+=center
    return model


def predict_price(model,frame):
    copy=frame.copy();copy['base_price']=1.
    return predict_regression(model,copy)


def generate(frame,side,*,max_rows=1000000,model_path=None):
    if side not in {'ask','bid'}:raise ValueError('Choose ask or bid.')
    source=frame.reset_index(drop=True).copy()
    source['base_price']=1.;source['base_model_id']='pending_sgd_assessment'
    train=source.split.eq('calibration')
    source['prediction_kind']=np.where(train,'oof','heldout')
    source['base_fold']=pd.factorize(source.vehicle_group)[0]%3
    d=validate(source,max_rows=max_rows)
    train_idx=np.flatnonzero(train);held_idx=np.flatnonzero(~train)
    predictions=np.full(len(d),np.nan);fold_ids=np.full(len(d),-1)
    for fold,(a,b) in enumerate(GroupKFold(3).split(train_idx,groups=d.vehicle_group.iloc[train_idx])):
        print(f'SGD-Huber {side} OOF fold {fold+1}/3',flush=True)
        predictions[train_idx[b]]=predict_price(fit_price(d.iloc[train_idx[a]]),d.iloc[train_idx[b]])
        fold_ids[train_idx[b]]=fold
    print(f'SGD-Huber {side} training-only assessment refit',flush=True)
    final=fit_price(d.iloc[train_idx]);predictions[held_idx]=predict_price(final,d.iloc[held_idx])
    if model_path:
        import joblib
        joblib.dump(final,model_path,compress=3)
    digest=hashlib.sha256(d.iloc[train_idx].to_json().encode()).hexdigest()
    d['base_model_id']='sgd-huber-log-price-'+digest;d['base_price']=predictions;d['base_fold']=fold_ids
    d['signed_residual_usd']=d.observed_price-d.base_price
    d['signed_residual_log']=np.log(d.observed_price/d.base_price)
    return d,{'method':'fresh_group_cross_fitted_sgd_huber_log_price','side':side,
        'model':describe(final),'calibration_folds':3,'text_branch':False,
        'warning':'Separate SGD-Huber log-price assessment on the full eligible original cohort. No CatBoost or random forest in this experiment. MiniLM and five-stage production cleaning are excluded. Held-out labels never fit the base model or hierarchy.'}
