"""Shared chained imputation of a missing ask-text price signal in auction records."""
import gzip,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from lightgbm import LGBMRegressor
from pricing.retrieval_imputation import RetrievalImputer,export,restore,FrozenChain,chain_data

class NumericLGBM(LGBMRegressor):
    def fit(self,X,y,**kwargs):
        kwargs.pop('categorical_feature',None)
        return super().fit(X,y,categorical_feature=list(range(3,np.asarray(X).shape[1])),**kwargs)

def identity(asks,bids):
    return hashlib.sha256(json.dumps([asks,bids],sort_keys=True,default=str).encode()).hexdigest()

def safe_log(values):
    values=np.asarray(values,dtype=float);out=np.full(len(values),np.nan);valid=np.isfinite(values)&(values>0)
    out[valid]=np.log(values[valid]);return out

class DescriptionPriceImputer:
    def fit(self,rows,prices_2021,description_prices_2021):
        self.base=RetrievalImputer().fit(rows)
        cats,nums,_=self.base.transform(rows)
        self.levels=[sorted(set(row[j] for row in cats)) for j in range(len(cats[0]))]
        self.known={str(r['id']):float(v) for r,v in zip(rows,description_prices_2021) if np.isfinite(v) and v>0}
        matrix=self._matrix(rows,prices_2021,description_prices_2021)
        if not np.isfinite(matrix[:,3]).any():raise ValueError('Observed asking-description prices are required to impute the missing auction signal')
        if np.isfinite(matrix[:,1]).all():matrix[0,1]=np.nan
        lows=np.r_[1900.,0.,0.,0.,np.zeros(len(self.levels))]
        highs=np.r_[2030.,np.log1p(2000000),np.log(1e7),np.log(1e7),[max(1,len(v)-1) for v in self.levels]]
        self.chain=IterativeImputer(estimator=NumericLGBM(n_estimators=30,num_leaves=7,max_depth=3,min_child_samples=20,reg_lambda=10.,verbosity=-1,n_jobs=2,random_state=42),max_iter=5,initial_strategy='median',skip_complete=True,sample_posterior=False,random_state=42,min_value=lows,max_value=highs).fit(matrix)
        return self
    def _matrix(self,rows,prices,description_prices):
        cats,nums,_=self.base.transform(rows)
        code=np.empty((len(rows),len(self.levels)),dtype=float)
        for j,levels in enumerate(self.levels):
            lookup={v:i for i,v in enumerate(levels)}
            code[:,j]=[lookup.get(row[j],0) for row in cats]
        return np.column_stack([nums[:,0],np.log1p(nums[:,1]),safe_log(prices),safe_log(description_prices),code])
    def transform(self,rows,prices_2021,description_prices_2021=None):
        if description_prices_2021 is None:description_prices_2021=np.array([self.known.get(str(r.get('id')),np.nan) for r in rows])
        known=np.asarray(description_prices_2021,dtype=float)
        values=self.chain.transform(self._matrix(rows,prices_2021,known))[:,3]
        out=np.exp(np.clip(values,0,np.log(1e7)))
        observed=np.isfinite(known)&(known>0);out[observed]=known[observed]
        return out,~observed
    def data(self):return {'base':export(self.base),'levels':self.levels,'known':self.known,'chain':chain_data(self.chain)}
    @classmethod
    def from_data(cls,data):
        obj=cls.__new__(cls);obj.base=restore(data['base']);obj.levels=data['levels'];obj.known=data['known'];obj.chain=FrozenChain(data['chain']);return obj

def load(root,asks,bids):
    root=Path(root);folder=root/'data/retrieval-imputation';meta=folder/'description-price.json';file=folder/'description-price.json.gz'
    if not meta.exists() or not file.exists():return None
    report=json.loads(meta.read_text())
    if report['reference_sha256']!=identity(asks,bids):return None
    heads=root/'models/xgboost-hierarchy/ask/minilm-heads.json'
    if hashlib.sha256(heads.read_bytes()).hexdigest()!=report['heads_sha256']:return None
    if file.stat().st_size>8*1024*1024 or hashlib.sha256(file.read_bytes()).hexdigest()!=report['model_sha256']:raise ValueError('Invalid description-price imputer artifact')
    return DescriptionPriceImputer.from_data(json.load(gzip.open(file,'rt')))
