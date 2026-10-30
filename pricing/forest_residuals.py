"""One-round XGBoost random forests on log prices; group-held-out evaluation."""
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from pricing.hierarchical import validate
from pricing.sgd_price import CATEGORICAL,OBSERVATION_COLUMNS


def generate(frame,side,*,max_rows=1000000,model_path=None):
    from xgboost import XGBRegressor
    if side not in {'ask','bid'}:raise ValueError('Choose ask or bid.')
    source=frame.reset_index(drop=True).copy();train=source.split.eq('calibration')
    source['base_price']=1.;source['base_model_id']='pending_xgbrf_assessment'
    source['prediction_kind']=np.where(train,'oof','heldout');source['base_fold']=pd.factorize(source.vehicle_group)[0]%3
    d=validate(source,max_rows=max_rows)
    x=pd.DataFrame({'age':d.age,'log_miles':np.log1p(pd.to_numeric(source.odometer,errors='coerce'))})
    x['auction_grade']=pd.to_numeric(source.get('auction_grade',pd.Series(np.nan,index=source.index)),errors='coerce')
    for col in CATEGORICAL:x[col]=source[col].fillna('unknown').astype(str).str.strip().str.lower() if col in source else 'unknown'
    params=dict(objective='reg:squarederror',n_estimators=1,num_parallel_tree=200,max_depth=8,learning_rate=1.,
        min_child_weight=20,reg_lambda=10.,subsample=.8,colsample_bynode=.8,
        random_state=42,n_jobs=4,tree_method='hist',max_bin=128,
        enable_categorical=True,max_cat_to_onehot=4)
    y=np.log(d.observed_price.to_numpy(float))
    def encode(idx,schema):
        encoded=x.iloc[idx].copy()
        for col,levels in schema.items():encoded[col]=pd.Categorical(encoded[col],categories=levels)
        return encoded
    def fit(idx):
        schema={c:sorted(x[c].iloc[idx].unique().tolist()) for c in CATEGORICAL}
        return XGBRegressor(**params).fit(encode(idx,schema),y[idx]),schema
    def pred(fitted,idx):
        model,schema=fitted
        return np.exp(np.clip(model.predict(encode(idx,schema)),0,np.log(1e7)))
    tr=np.flatnonzero(train);held=np.flatnonzero(~train);predictions=np.full(len(d),np.nan);fold_ids=np.full(len(d),-1)
    for fold,(a,b) in enumerate(GroupKFold(3).split(tr,groups=d.vehicle_group.iloc[tr])):
        print(f'XGBoost RF {side} OOF fold {fold+1}/3',flush=True)
        predictions[tr[b]]=pred(fit(tr[a]),tr[b]);fold_ids[tr[b]]=fold
    print(f'XGBoost RF {side} training-only refit',flush=True)
    final,schema=fit(tr);predictions[held]=pred((final,schema),held)
    if model_path:final.save_model(str(model_path))
    digest=hashlib.sha256((x.iloc[tr].to_json()+y[tr].tobytes().hex()+json.dumps(params,sort_keys=True)).encode()).hexdigest()
    d['base_model_id']='xgbrf-log-price-'+digest;d['base_price']=predictions;d['base_fold']=fold_ids
    d['signed_residual_usd']=d.observed_price-d.base_price;d['signed_residual_log']=np.log(d.observed_price/d.base_price)
    return d,{'method':'fresh_group_cross_fitted_xgboost_random_forest','side':side,'parameters':params,
        'category_schema':schema,'features':list(x),'calibration_folds':3,'text_branch':False,
        'boosting_rounds':final.get_booster().num_boosted_rounds(),'trees':len(final.get_booster().get_dump()),
        'warning':'Separate XGBoost random forest log-price assessment: 200 parallel trees, one boosting round. Full eligible original cohorts; no residual trimming or held-out label fitting. Not the production MiniLM model.'}
