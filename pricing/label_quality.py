"""Training-only Cleanlab regression auditing shared by asking and auction models.

The audit uses vehicle facts and observed log prices, never previously computed
stacking features. This avoids indirect target leakage through text OOF models.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

ALGORITHM = 'Cleanlab'
FORBIDDEN = {'price','sale_price','sellingprice','mmr','bid','highest_bid','predicted_text_price',
             'text','description','phase','row_id','date','source_url','price_evidence','log_real_price'}


class AuditRegressor(RegressorMixin, BaseEstimator):
    """Cloneable categorical LightGBM, including Cleanlab uncertainty refits."""
    def __init__(self, categorical=(), seed=42, threads=4, trees=200):
        self.categorical=categorical;self.seed=seed;self.threads=threads;self.trees=trees

    def fit(self, X, y, sample_weight=None):
        from lightgbm import LGBMRegressor
        self.model_=LGBMRegressor(objective='huber',alpha=.9,n_estimators=self.trees,
            learning_rate=.05,num_leaves=31,min_child_samples=10,reg_lambda=10.,
            random_state=self.seed,n_jobs=self.threads,verbosity=-1,deterministic=True,
            force_col_wise=True)
        self.model_.fit(X,y,sample_weight=sample_weight,categorical_feature=list(self.categorical))
        self.n_features_in_=np.asarray(X).shape[1]
        return self

    def predict(self, X):return self.model_.booster_.predict(X,num_threads=self.threads)


def cleanlab_inliers(frame, prices, path, row_ids, seed=42, threads=4,
                     folds=3, n_boot=3, trees=200):
    import cleanlab
    from cleanlab.regression.learn import CleanLearning
    frame=frame.copy().reset_index(drop=True)
    y=np.asarray(prices,float);ids=np.asarray(row_ids).astype(str)
    if frame.empty or frame.columns.duplicated().any() or FORBIDDEN.intersection(frame.columns):
        raise ValueError('Cleanlab audit requires an explicit vehicle-predictor allowlist without targets, text or stacked predictions')
    if y.ndim!=1 or len(y)!=len(frame) or len(ids)!=len(y) or len(set(ids))!=len(ids):
        raise ValueError('Unique row identities and one price per training car required')
    if len(y)<max(12,folds*3) or folds<2 or n_boot<0 or trees<1 or not np.isfinite(y).all() or (y<=0).any():
        raise ValueError('Need sufficient positive finite training prices and valid Cleanlab settings')
    schema={};columns=[];categorical=[]
    for i,column in enumerate(frame):
        values=frame[column]
        if not pd.api.types.is_numeric_dtype(values):
            values=values.astype('string').str.lower().str.strip()
            vocabulary=sorted(values.dropna().unique().tolist());schema[column]=vocabulary
            values=pd.Series(pd.Categorical(values,categories=vocabulary).codes,dtype=float)
            categorical.append(i)
        else:values=pd.to_numeric(values,errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(-1.)
        columns.append(values.to_numpy(dtype=float))
    X=np.column_stack(columns)
    settings={'seed':seed,'threads':threads,'folds':folds,'n_boot':n_boot,'trees':trees,
              'cleanlab_version':cleanlab.__version__,'numpy_version':np.__version__,'pandas_version':pd.__version__}
    fingerprint=hashlib.sha256(X.tobytes()+y.tobytes()+json.dumps(ids.tolist()).encode()+
        json.dumps({'settings':settings,'columns':list(frame),'schema':schema},sort_keys=True).encode()+Path(__file__).read_bytes()).hexdigest()
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    csv=path/'label-audit.csv';meta=path/'label-audit.json'
    if meta.exists():
        report=json.loads(meta.read_text())
        if report['fingerprint']!=fingerprint:raise ValueError('Cleanlab cache belongs to different data/code/configuration; use a new output directory')
        if hashlib.sha256(csv.read_bytes()).hexdigest()!=report['audit_sha256']:raise ValueError('Cleanlab audit checksum mismatch')
        audit=pd.read_csv(csv,dtype={'row_id':str})
        if audit.row_id.tolist()!=ids.tolist():raise ValueError('Cleanlab audit identities differ')
        return np.flatnonzero(audit.kept.to_numpy(dtype=bool)),report
    state=np.random.get_state()
    try:
        learner=CleanLearning(AuditRegressor(tuple(categorical),seed,threads,trees),
            cv_n_folds=folds,n_boot=n_boot,include_aleatoric_uncertainty=True,seed=seed)
        issues=learner.find_label_issues(X,np.log1p(y))
    finally:np.random.set_state(state)
    if len(issues)!=len(y) or not np.isfinite(issues[['label_quality','predicted_label']].to_numpy()).all():
        raise ValueError('Cleanlab returned invalid audit scores')
    keep=~issues.is_label_issue.to_numpy(dtype=bool)
    if keep.sum()<2:raise ValueError('Cleanlab retained too few cars; no fallback to unfiltered training')
    audit=issues.copy();audit.insert(0,'row_id',ids)
    audit['observed_price']=y;audit['audit_predicted_price']=np.expm1(np.clip(audit.predicted_label,0,np.log1p(1e7)))
    audit['kept']=keep;audit.to_csv(csv,index=False)
    report={'algorithm':ALGORITHM,'method':ALGORITHM,'fingerprint':fingerprint,
        'input':'structured vehicle fields + observed training log1p price',
        'features':list(frame),'input_rows':len(y),'retained_rows':int(keep.sum()),
        'removed_rows':int((~keep).sum()),'removed':int((~keep).sum()),
        'removed_fraction':float((~keep).mean()),'estimated_noise_fraction':float(learner.k),
        'settings':settings,'audit_sha256':hashlib.sha256(csv.read_bytes()).hexdigest(),
        'scope':'Cleanlab internal CV and uncertainty fits use this fitting pool only. Validation/test are never filtered. Suspected errors are not verified errors.',
        'stacking_safety':'No text-price predictions or target-trained CarBids features enter the audit.'}
    (path/'audit-schema.json').write_text(json.dumps(schema,indent=2))
    meta.write_text(json.dumps(report,indent=2));return np.flatnonzero(keep),report
