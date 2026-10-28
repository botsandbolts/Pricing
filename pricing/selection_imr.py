"""Optional pre-sale Probit selection feature, used ONLY by the bid cleaner.

No sales-price/outcome variables enter the design. Both sold and genuinely
unsold listings are essential; one-class sale data cannot identify selection.
This is a selection diagnostic, not a claim of identified Heckman correction
for flexible robust trees.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import log_ndtr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

NUMERIC=['age','log_miles','auction_grade']
CATEGORICAL=['manufacturer','type','transmission','state','fuel']


def inverse_mills(z):
    z=np.asarray(z,float)
    if not np.isfinite(z).all():raise ValueError('Finite Probit linear index required')
    return np.exp(-.5*z*z-.5*np.log(2*np.pi)-log_ndtr(z))


class ProbitSelection:
    def fit(self,x,sold):
        y=np.asarray(sold,float)
        if y.shape!=(len(x),) or set(np.unique(y))!={0.,1.}:
            raise ValueError('IMR requires observed sold AND unsold listings from the same listing population')
        self.transform=ColumnTransformer([
            ('numeric',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()),NUMERIC),
            ('categorical',OneHotEncoder(handle_unknown='ignore',min_frequency=20,max_categories=40),CATEGORICAL)],sparse_threshold=1.)
        design=sparse.csr_matrix(self.transform.fit_transform(x))
        design=sparse.hstack([sparse.csr_matrix(np.ones((len(x),1))),design],format='csr')
        n=len(y);penalty=1e-3
        def objective(beta):
            z=np.asarray(design@beta);signed=(2*y-1)*z
            value=-log_ndtr(signed).mean()+.5*penalty*np.dot(beta[1:],beta[1:])
            derivative=-(2*y-1)*inverse_mills(signed)/n
            gradient=np.asarray(design.T@derivative).ravel();gradient[1:]+=penalty*beta[1:]
            return value,gradient
        result=minimize(objective,np.zeros(design.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':300,'ftol':1e-9})
        if not result.success or not np.isfinite(result.x).all():raise ValueError('Probit failed to converge: '+str(result.message))
        self.beta=result.x;self.audit={'algorithm':'L2-regularized sparse Probit','rows':n,'sold':int(y.sum()),'unsold':int(n-y.sum()),'penalty':penalty,'iterations':int(result.nit),'converged':True,'design_features':NUMERIC+CATEGORICAL,'formula':'lambda_sold = phi(z) / Phi(z)','scope':'bid cleaning only; not the final price estimator','limitation':'No identified selection correction is claimed without defensible population coverage and exclusion restrictions.'}
        return self
    def predict(self,x):
        design=sparse.csr_matrix(self.transform.transform(x));z=self.beta[0]+np.asarray(design@self.beta[1:]).ravel()
        return inverse_mills(z)
