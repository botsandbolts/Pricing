"""Sparse, robust regression of group-held-out base-model log-price residuals."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

CATEGORIES=['manufacturer','model','type','transmission','state','fuel','drive','title_status','cylinders','paint_color']
NUMERIC=['age','log_miles','mileage_missing','log_base_price','auction_grade','grade_missing']


def features(frame):
    out=pd.DataFrame(index=frame.index)
    out['age']=pd.to_numeric(frame.age,errors='raise')
    miles=pd.to_numeric(frame.get('odometer',pd.Series(np.nan,index=frame.index)),errors='coerce')
    out['mileage_missing']=miles.isna().astype(float)
    out['log_miles']=np.log1p(miles.clip(lower=0).fillna(0))
    grade=pd.to_numeric(frame.get('auction_grade',pd.Series(np.nan,index=frame.index)),errors='coerce')
    out['grade_missing']=grade.isna().astype(float)
    out['auction_grade']=grade.fillna(0)
    out['log_base_price']=np.log(pd.to_numeric(frame.base_price,errors='raise'))
    for c in CATEGORIES:
        out[c]=frame[c].fillna('unknown').astype(str).str.strip().str.lower() if c in frame else 'unknown'
    return out


def fit(frame):
    encoder=ColumnTransformer([
        ('numeric',StandardScaler(),NUMERIC),
        ('categorical',OneHotEncoder(min_frequency=20,max_categories=50,
          handle_unknown='infrequent_if_exist',sparse_output=True),CATEGORIES)
    ],sparse_threshold=1.)
    estimator=SGDRegressor(loss='huber',epsilon=.1,penalty='elasticnet',alpha=.001,
        l1_ratio=.05,max_iter=3000,tol=1e-5,learning_rate='invscaling',eta0=.01,
        power_t=.25,average=True,early_stopping=False,random_state=42)
    model=Pipeline([('encode',encoder),('regress',estimator)])
    model.fit(features(frame),np.log(frame.observed_price/frame.base_price))
    return model


def predict(model,frame):
    raw=np.asarray(model.predict(features(frame)),float)
    if not np.isfinite(raw).all():raise ValueError('Nonfinite SGD correction.')
    return np.exp(np.clip(np.log(frame.base_price.to_numpy(float))+raw,0,np.log(1e7)))


def describe(model):
    estimator=model.named_steps['regress'];encoder=model.named_steps['encode']
    names=encoder.get_feature_names_out();coef=estimator.coef_
    order=np.argsort(np.abs(coef))[-20:][::-1]
    return {'method':'sgd_huber_log_residuals','loss':'huber','epsilon_log':.1,
            'min_category_count':20,'max_categories_per_field':50,
            'scaling':'numeric StandardScaler fitted on calibration only',
            'unknown_categories':'infrequent bucket if fitted; otherwise all-zero encoding',
            'parameters':estimator.get_params(),'feature_count':len(names),
            'iterations_completed':int(estimator.n_iter_),
            'largest_coefficients':[{'feature':str(names[i]),'coefficient':float(coef[i])} for i in order],
            'warning':'Coefficients concern log residuals; numerical coefficients use standardized units. Not causal effects. USD outputs are clipped to 1–10 million.'}
