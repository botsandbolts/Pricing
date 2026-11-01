"""Offline-fitted mixed-type chained imputation used only for retrieval vectors."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from lightgbm import LGBMClassifier,LGBMRegressor

class CategoricalLGBM(LGBMClassifier):
    def fit(self,X,y,**kwargs):
        kwargs.pop('categorical_feature',None)
        return super().fit(X,y,categorical_feature=list(range(np.asarray(X).shape[1])),**kwargs)

FIELDS=('manufacturer','model','type','fuel','transmission','drive','title_status')

def missing(value):return str(value or '').strip().lower() in ('','unknown','nan','none','__missing__')
def numeric(rows):
    out=[]
    for row in rows:
        values=[]
        for field in ('year','odometer'):
            try:value=float(row.get(field));value=value if np.isfinite(value) else np.nan
            except (ValueError,TypeError):value=np.nan
            values.append(value)
        out.append([values[0],np.log1p(values[1]) if np.isfinite(values[1]) and values[1]>=0 else np.nan])
    return np.asarray(out,dtype=float)

def fingerprint(rows):
    values=[{k:r.get(k) for k in ('id',)+FIELDS+('year','odometer')} for r in rows]
    return hashlib.sha256(json.dumps(values,sort_keys=True,default=str).encode()).hexdigest()

class RetrievalImputer:
    def fit(self,rows):
        self.levels={f:sorted({str(r.get(f)).strip().lower() for r in rows if not missing(r.get(f))}) for f in FIELDS}
        x=self._codes(rows);self.active=np.flatnonzero(np.isfinite(x).any(axis=0));self.categorical=None
        if len(self.active)>=2:
            values=x[:,self.active].copy()
            # Optional complete columns need a fitted model for missing query values.
            # Hide one known cell only in optional columns; original observations are restored at transform.
            for j,index in enumerate(self.active):
                if FIELDS[index] not in ('manufacturer','model') and np.isfinite(values[:,j]).all():values[0,j]=np.nan
            estimator=CategoricalLGBM(n_estimators=20,num_leaves=7,max_depth=3,min_child_samples=15,reg_lambda=10.,cat_smooth=20.,verbosity=-1,n_jobs=2,random_state=42)
            self.categorical=IterativeImputer(estimator=estimator,initial_strategy='most_frequent',max_iter=3,tol=.01,skip_complete=True,sample_posterior=False,random_state=42).fit(values)
        values=numeric(rows);self.numeric_active=np.flatnonzero(np.isfinite(values).any(axis=0));self.numerical=None
        if len(self.numeric_active)==2:
            train=values.copy()
            if np.isfinite(train[:,1]).all():train[0,1]=np.nan
            self.numerical=IterativeImputer(estimator=LGBMRegressor(n_estimators=35,num_leaves=7,max_depth=3,min_child_samples=15,reg_lambda=10.,verbosity=-1,n_jobs=2,random_state=42),initial_strategy='median',max_iter=5,skip_complete=True,sample_posterior=False,min_value=np.array([1900.,0.]),max_value=np.array([2030.,np.log1p(2000000)]),random_state=42).fit(train)
        self.numeric_medians=np.array([np.nanmedian(values[:,j]) if np.isfinite(values[:,j]).any() else (2015. if j==0 else 0.) for j in range(2)])
        return self
    def _codes(self,rows):
        out=np.full((len(rows),len(FIELDS)),np.nan)
        for j,f in enumerate(FIELDS):
            lookup={v:i for i,v in enumerate(self.levels[f])}
            for i,r in enumerate(rows):out[i,j]=lookup.get(str(r.get(f) or '').strip().lower(),np.nan)
        return out
    def transform(self,rows):
        codes=self._codes(rows);filled=codes.copy()
        if self.categorical is not None:filled[:,self.active]=self.categorical.transform(codes[:,self.active])
        num=numeric(rows);values=self.numerical.transform(num) if self.numerical is not None else np.where(np.isfinite(num),num,self.numeric_medians)
        values=np.where(np.isfinite(num),num,values)
        categories=[];estimated=[]
        for i,r in enumerate(rows):
            cats=[];fields=[]
            for j,f in enumerate(FIELDS):
                if not missing(r.get(f)):value=str(r[f]).strip().lower()
                elif np.isfinite(filled[i,j]) and self.levels[f]:
                    code=int(filled[i,j])
                    value=self.levels[f][code] if 0<=code<len(self.levels[f]) else '__missing__';fields.append(f)
                else:value='__missing__'
                cats.append(value)
            for j,f in enumerate(('year','odometer')):
                if not np.isfinite(num[i,j]):fields.append(f)
            categories.append(cats);estimated.append(fields)
        values[:,1]=np.expm1(np.clip(values[:,1],0,np.log1p(2000000)))
        if not np.isfinite(values).all():raise ValueError('Incomplete imputation matrix')
        return categories,values.astype(np.float32),estimated

class FrozenChain:
    """Portable inference of sklearn's fitted chained schedule; no refitting."""
    def __init__(self,data):
        import lightgbm as lgb
        self.initial=np.asarray(data['initial'],dtype=float)
        self.minimum=data.get('minimum');self.maximum=data.get('maximum')
        self.steps=[(x['feature'],np.asarray(x['neighbours'],dtype=int),lgb.Booster(model_str=x['model']),x.get('classes')) for x in data['steps']]
    def transform(self,values):
        values=np.asarray(values,dtype=float);mask=~np.isfinite(values)
        completed=np.where(mask,self.initial,values).copy()
        for feature,neighbours,model,classes in self.steps:
            selected=mask[:,feature]
            if not selected.any():continue
            prediction=model.predict(completed[selected][:,neighbours],num_threads=2)
            if classes is not None:
                classes=np.asarray(classes)
                if len(classes)==1:prediction=np.full(selected.sum(),classes[0])
                elif len(classes)==2:prediction=classes[(np.asarray(prediction)>.5).astype(int)]
                else:prediction=classes[np.argmax(prediction,axis=1)]
            if self.minimum is not None:prediction=np.clip(prediction,self.minimum[feature],self.maximum[feature])
            completed[selected,feature]=prediction
        return completed

def chain_data(chain):
    if chain is None:return None
    return {'initial':chain.initial_imputer_.statistics_.tolist(),
            'minimum':np.asarray(chain.min_value).tolist() if isinstance(chain.min_value,np.ndarray) else None,
            'maximum':np.asarray(chain.max_value).tolist() if isinstance(chain.max_value,np.ndarray) else None,
            'steps':[{'feature':int(x.feat_idx),'neighbours':x.neighbor_feat_idx.tolist(),'model':x.estimator.booster_.model_to_string(),
                      'classes':x.estimator.classes_.tolist() if hasattr(x.estimator,'classes_') else None} for x in chain.imputation_sequence_]}

def export(imputer):
    return {'levels':imputer.levels,'active':imputer.active.tolist(),'numeric_active':imputer.numeric_active.tolist(),
            'numeric_medians':imputer.numeric_medians.tolist(),'categorical':chain_data(imputer.categorical),'numerical':chain_data(imputer.numerical)}

def restore(data):
    obj=RetrievalImputer.__new__(RetrievalImputer)
    obj.levels=data['levels'];obj.active=np.asarray(data['active'],dtype=int);obj.numeric_active=np.asarray(data['numeric_active'],dtype=int)
    obj.numeric_medians=np.asarray(data['numeric_medians'],dtype=float)
    obj.categorical=FrozenChain(data['categorical']) if data['categorical'] else None
    obj.numerical=FrozenChain(data['numerical']) if data['numerical'] else None
    return obj

def load(root,side,rows):
    import gzip
    folder=Path(root)/'data/retrieval-imputation';meta=folder/(side+'.json');file=folder/(side+'.json.gz')
    if not meta.exists() or not file.exists():return None
    info=json.loads(meta.read_text())
    if info['reference_sha256']!=fingerprint(rows):return None
    if file.stat().st_size>8*1024*1024 or hashlib.sha256(file.read_bytes()).hexdigest()!=info['model_sha256']:raise ValueError('Invalid imputation artifact')
    return restore(json.load(gzip.open(file,'rt')))
