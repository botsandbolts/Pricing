"""Price-assisted completion; return only vehicle features, never the price column."""
import gzip, hashlib, json
from pathlib import Path
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from lightgbm import LGBMClassifier, LGBMRegressor
from pricing.retrieval_imputation import RetrievalImputer, FIELDS, numeric, missing, export, restore, chain_data, FrozenChain, fingerprint

class ContextClassifier(LGBMClassifier):
    def fit(self, X, y, **kwargs):
        kwargs.pop('categorical_feature', None)
        return super().fit(X, y, categorical_feature=list(range(max(0, X.shape[1]-3))), **kwargs)

class ContextRegressor(LGBMRegressor):
    def fit(self, X, y, **kwargs):
        kwargs.pop('categorical_feature', None)
        return super().fit(X, y, categorical_feature=list(range(2, X.shape[1])), **kwargs)

class PriceContextImputer:
    def fit(self, rows, prices):
        self.base=RetrievalImputer().fit(rows)
        self.context={str(r['id']):float(v) for r,v in zip(rows,prices) if np.isfinite(v) and v>0}
        logged=np.log(np.asarray(list(self.context.values())))
        if not len(logged):raise ValueError('No supported adjusted reference prices for imputation')
        self.price_median=float(np.median(logged))
        codes=self.base._codes(rows);self.active=np.flatnonzero(np.isfinite(codes).any(axis=0))
        nums=numeric(rows);self.medians=np.array([np.nanmedian(nums[:,j]) if np.isfinite(nums[:,j]).any() else v for j,v in enumerate([2015.,np.log1p(95000)])])
        context=self._context(rows);continuous=np.column_stack([np.where(np.isfinite(nums),nums,self.medians),context])
        train=np.column_stack([codes[:,self.active],continuous]);self.categorical=None
        if len(self.active)>1:
            for j,k in enumerate(self.active):
                if k>=2 and np.isfinite(train[:,j]).all():train[0,j]=np.nan
            self.categorical=IterativeImputer(estimator=ContextClassifier(n_estimators=20,num_leaves=7,max_depth=3,min_child_samples=15,reg_lambda=10,verbosity=-1,n_jobs=2,random_state=42),max_iter=3,initial_strategy='most_frequent',skip_complete=True,random_state=42).fit(train)
        completed=self._categories(rows)
        train=np.column_stack([nums,context,completed])
        if np.isfinite(train[:,1]).all():train[0,1]=np.nan
        low=np.r_[1900.,0.,0.,np.zeros(len(self.active))]
        high=np.r_[2030.,np.log1p(2000000),np.log(1e7),[max(1,len(self.base.levels[FIELDS[k]])-1) for k in self.active]]
        self.numerical=IterativeImputer(estimator=ContextRegressor(n_estimators=30,num_leaves=7,max_depth=3,min_child_samples=15,reg_lambda=10,verbosity=-1,n_jobs=2,random_state=42),max_iter=5,initial_strategy='median',skip_complete=True,min_value=low,max_value=high,random_state=42).fit(train)
        return self
    def _context(self,rows):
        values=np.asarray([r.get('_imputation_price',self.context.get(str(r.get('id')),np.nan)) for r in rows],dtype=float)
        out=np.full(len(rows),self.price_median);valid=np.isfinite(values)&(values>0);out[valid]=np.log(values[valid]);return out
    def _categories(self,rows):
        codes=self.base._codes(rows)[:,self.active];nums=numeric(rows)
        matrix=np.column_stack([codes,np.where(np.isfinite(nums),nums,self.medians),self._context(rows)])
        if self.categorical is not None:return self.categorical.transform(matrix)[:,:len(self.active)]
        return np.nan_to_num(codes)
    def transform(self,rows):
        codes=self._categories(rows);nums=numeric(rows)
        values=self.numerical.transform(np.column_stack([nums,self._context(rows),codes]))[:,:2]
        # Drop the price and categorical predictor columns before returning numeric inputs.
        values=np.where(np.isfinite(nums),nums,values);values[:,1]=np.expm1(np.clip(values[:,1],0,np.log1p(2000000)))
        cats=[];inferred=[];positions={k:j for j,k in enumerate(self.active)}
        for i,row in enumerate(rows):
            rowcats=[];fields=[]
            for k,f in enumerate(FIELDS):
                if not missing(row.get(f)):value=str(row[f]).strip().lower()
                elif k in positions and self.base.levels[f]:
                    code=int(np.clip(round(codes[i,positions[k]]),0,len(self.base.levels[f])-1));value=self.base.levels[f][code];fields.append(f)
                else:value='__missing__'
                rowcats.append(value)
            fields += [f for j,f in enumerate(('year','odometer')) if not np.isfinite(nums[i,j])]
            cats.append(rowcats);inferred.append(fields)
        if not np.isfinite(values).all():raise ValueError('Incomplete price-assisted imputation')
        return cats,values.astype(np.float32),inferred
    def data(self):return dict(base=export(self.base),context=self.context,price_median=self.price_median,active=self.active.tolist(),medians=self.medians.tolist(),categorical=chain_data(self.categorical),numerical=chain_data(self.numerical))
    @classmethod
    def from_data(cls,d):
        obj=cls.__new__(cls);obj.base=restore(d['base']);obj.context=d['context'];obj.price_median=d['price_median'];obj.active=np.array(d['active']);obj.medians=np.array(d['medians']);obj.categorical=FrozenChain(d['categorical']) if d['categorical'] else None;obj.numerical=FrozenChain(d['numerical']);return obj

def load(root,side,rows):
    folder=Path(root)/'data/retrieval-imputation';meta=folder/(side+'-price-context.json');file=folder/(side+'-price-context.json.gz')
    if not meta.exists() or not file.exists():return None,None
    report=json.loads(meta.read_text())
    if report['reference_sha256']!=fingerprint(rows):return None,None
    if file.stat().st_size>8*1024*1024 or hashlib.sha256(file.read_bytes()).hexdigest()!=report['model_sha256']:raise ValueError('Invalid price-assisted imputer')
    return PriceContextImputer.from_data(json.load(gzip.open(file,'rt'))),report['price_basis_date']
