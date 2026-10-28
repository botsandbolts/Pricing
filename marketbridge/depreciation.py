"""Monotone additive depreciation with sparse pooled categorical age effects.

This is penalized MAP-style regression, not MCMC or an identified causal age
curve. It supplies price ratios; existing text/tree models remain the anchors.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder

AGE_KNOTS=np.array([0.,2.,5.,10.,15.,22.,30.])
MILE_KNOTS=np.log1p([0.,15000.,40000.,80000.,140000.,250000.,500000.])
CATS=['manufacturer','model','type','transmission','state','fuel','condition']
SLOPES=['manufacturer','model','type']


def basis(values,knots,kind):
    """Increasing integrated B splines or linear segments; negative coefs give monotonicity."""
    v=np.asarray(values,float);lo,hi=knots[[0,-1]];x=np.clip(v,lo,hi)
    if kind=='linear':return (v-lo)[:,None]/max(hi-lo,1)
    if kind=='piecewise':
        out=np.maximum(0,np.minimum(x[:,None]-knots[:-1],np.diff(knots)))
        out[:,0]+=np.minimum(v-lo,0);out[:,-1]+=np.maximum(v-hi,0)
        return out/(hi-lo)
    t=np.r_[np.repeat(lo,3),knots[1:-1],np.repeat(hi,3)]
    n=len(t)-3;b=BSpline(t,np.eye(n),2,extrapolate=False);integral=b.antiderivative()
    out=integral(x)-integral(lo)
    out+=np.minimum(v-lo,0)[:,None]*b(lo)+np.maximum(v-hi,0)[:,None]*b(hi)
    return out/(hi-lo)


def fields(frame):
    f=frame.copy();out=pd.DataFrame(index=f.index)
    for c in CATS:
        v=f[c] if c in f else pd.Series('unknown',index=f.index)
        out[c]=v.fillna('unknown').astype(str).str.strip().str.lower().replace('','unknown')
    out['model']=out.manufacturer+'::'+out.model
    out['age']=pd.to_numeric(f.age,errors='coerce')
    out['miles']=pd.to_numeric(f.odometer,errors='coerce')
    if not np.isfinite(out[['age','miles']]).all().all() or (out.age<0).any() or (out.miles<0).any():raise ValueError('Finite nonnegative age and mileage required')
    return out


class DepreciationCurve:
    def __init__(self,kind='spline',penalty=100.):self.kind=kind;self.penalty=float(penalty)
    def prepare(self,frame,fit=False):
        f=fields(frame)
        if fit:
            self.levels={c:sorted(f[c].value_counts().loc[lambda s:s>=50].nlargest(400 if c=='model' else 80).index.tolist()) for c in CATS}
        for c in CATS:f[c]=f[c].where(f[c].isin(self.levels[c]),'__pooled__')
        if fit:
            self.encoder=OneHotEncoder(handle_unknown='ignore',categories=[sorted(set(self.levels[c]+['__pooled__'])) for c in CATS],sparse_output=True)
            self.encoder.fit(f[CATS])
        one=sparse.csr_matrix(self.encoder.transform(f[CATS]))
        a=basis(f.age,AGE_KNOTS,self.kind);m=basis(np.log1p(f.miles),MILE_KNOTS,self.kind)
        parts=[sparse.csr_matrix(np.ones((len(f),1))),sparse.csr_matrix(a),sparse.csr_matrix(m),one]
        nshape=a.shape[1]+m.shape[1];nbase=1+nshape+one.shape[1]
        # Shared age/mileage shapes plus manufacturer/model/body extra age slopes.
        # All slope coefficients <= 0, so changing only age never increases price.
        start=0;indices=[]
        for c,levels in zip(CATS,self.encoder.categories_):
            if c in SLOPES:indices.extend(range(start,start+len(levels)))
            start+=len(levels)
        slope=one[:,indices].multiply(f.age.to_numpy()[:,None]/10)
        if self.kind!='linear':parts.append(slope)
        design=sparse.hstack(parts,format='csr')
        if fit:
            self.nshape=nshape;self.nbase=nbase;self.slope_indices=indices
            self.penalties=np.r_[1e-7,np.repeat(.5,nshape),np.repeat(self.penalty,one.shape[1]),np.repeat(self.penalty*4,design.shape[1]-nbase)]
            self.bounds=[(None,None)]+[(None,0.)]*nshape+[(None,None)]*one.shape[1]+[(None,0.)]*(design.shape[1]-nbase)
        return design
    def fit(self,frame,log_prices):
        X=self.prepare(frame,fit=True);y=np.asarray(log_prices,float)
        if len(y)!=len(frame) or not np.isfinite(y).all():raise ValueError('Finite log targets required')
        beta=np.zeros(X.shape[1]);beta[0]=np.median(y);weights=np.ones(len(y));iterations=[]
        for step in range(4):
            WX=X.multiply(weights[:,None]);H=(X.T@WX).toarray();H[np.diag_indices_from(H)]+=self.penalties
            rhs=np.asarray(X.T@(weights*y)).ravel();scale=len(y)
            def objective(b):return (.5*b@H@b-rhs@b)/scale,(H@b-rhs)/scale
            result=minimize(objective,beta,jac=True,method='L-BFGS-B',bounds=self.bounds,options={'maxiter':500,'ftol':1e-11,'gtol':2e-7})
            if not np.isfinite(result.x).all():raise ValueError('Nonfinite depreciation fit')
            beta=result.x;resid=y-X@beta;weights=np.minimum(1.,.2/np.maximum(abs(resid),1e-12))
            iterations.append({'iterations':int(result.nit),'converged':bool(result.success),'message':str(result.message)})
        self.beta=beta;f=fields(frame)
        self.metadata={'kind':self.kind,'penalty':self.penalty,'rows':len(frame),'age_range':[float(f.age.min()),float(f.age.max())],'miles_range':[float(f.miles.min()),float(f.miles.max())],'optimization':iterations,'role':'Huber-IRLS penalized monotone additive log-price shape; categorical partial pooling; not full Bayesian posterior or causal ageing'}
        return self
    def predict_log(self,frame):return np.asarray(self.prepare(frame)@self.beta).ravel()
    def ratio(self,frame,from_age,to_age,from_miles=None,to_miles=None):
        before=frame.copy();after=frame.copy();before['age']=from_age;after['age']=to_age
        if from_miles is not None:before['odometer']=from_miles
        if to_miles is not None:after['odometer']=to_miles
        return np.exp(self.predict_log(after)-self.predict_log(before))
    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        data={'kind':self.kind,'penalty':self.penalty,'levels':self.levels,'categories':[v.tolist() for v in self.encoder.categories_],'beta':self.beta.tolist(),'metadata':self.metadata}
        path.write_text(json.dumps(data,indent=2))
    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text());m=cls(d['kind'],d['penalty']);m.levels=d['levels'];m.metadata=d['metadata'];m.beta=np.array(d['beta'])
        m.encoder=OneHotEncoder(handle_unknown='ignore',categories=d['categories'],sparse_output=True)
        m.encoder.fit(pd.DataFrame([{c:vs[0] for c,vs in zip(CATS,d['categories'])}]))
        return m
