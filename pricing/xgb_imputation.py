"""Approximate stochastic XGBoost imputation for sensitivity, not identity recovery."""
import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.model_selection import train_test_split
class UncertainXGB(RegressorMixin,BaseEstimator):
    def __init__(self,random_state=42,trees=80,max_rows=20000):
        self.random_state=random_state;self.trees=trees;self.max_rows=max_rows
    def fit(self,X,y):
        from xgboost import XGBRegressor
        rng=np.random.default_rng(self.random_state)
        ix=rng.choice(len(y),min(len(y),self.max_rows),replace=False)
        x=np.asarray(X)[ix];y=np.asarray(y)[ix]
        self.models_=[]
        tr,te=train_test_split(np.arange(len(y)),test_size=.2,random_state=self.random_state)
        def model(seed):
            return XGBRegressor(n_estimators=self.trees,max_depth=4,reg_lambda=20,learning_rate=.08,n_jobs=2,tree_method='hist',random_state=seed)
        calibration=model(self.random_state).fit(x[tr],y[tr])
        errors=y[te]-calibration.predict(x[te])
        self.noise_=max(float(np.sqrt(np.mean(errors**2))),.001)
        for j in range(3):
            sampled=rng.integers(0,len(y),len(y))
            self.models_.append(model(self.random_state+j+1).fit(x[sampled],y[sampled]))
        return self
    def predict(self,X,return_std=False):
        p=np.array([m.predict(X) for m in self.models_])
        mean=p.mean(axis=0)
        return (mean,np.sqrt(p.var(axis=0)+self.noise_**2)) if return_std else mean

def completed_numeric(numeric,context,seed):
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    numeric=np.asarray(numeric,float);context=np.asarray(context,float)
    if np.any(np.all(~np.isfinite(numeric),axis=0)):
        raise ValueError('An entirely unobserved attribute cannot be recovered by this imputer.')
    x=np.column_stack((numeric,context))
    imp=IterativeImputer(estimator=UncertainXGB(random_state=seed),sample_posterior=True,
        max_iter=3,skip_complete=True,random_state=seed,initial_strategy='median')
    result=imp.fit_transform(x)[:,:numeric.shape[1]]
    observed=np.isfinite(numeric)
    assert np.array_equal(result[observed],numeric[observed])
    return result
