"""Small session-only pairwise utility model; whole-car cross validation."""
import json, sqlite3, zlib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

NUMERIC=('year','odometer','price')
CATEGORICAL=('manufacturer','model','type','fuel','transmission','drive','condition','title_status','cylinders')
def catalogue_features(folder):
    numeric=[]; categories=[]; blocks=[]
    hasher=FeatureHasher(n_features=512,input_type='dict',alternate_sign=False)
    with sqlite3.connect(str(folder/'cars.sqlite')) as db:
        for expected,(position,payload) in enumerate(db.execute('SELECT position,payload FROM cars ORDER BY position')):
            if position!=expected: raise ValueError('Catalogue position mismatch')
            car=json.loads(zlib.decompress(payload)); row=[]
            for field in NUMERIC:
                try: value=float(car.get(field))
                except (ValueError,TypeError): value=np.nan
                if field!='year': value=np.log1p(value) if value>=0 else np.nan
                row.append(value if np.isfinite(value) else np.nan)
            numeric.append(row)
            categories.append({field+'='+str(car[field]).strip().lower():1. for field in CATEGORICAL if car.get(field) is not None and str(car[field]).strip().lower() not in ('','unknown','nan','none')})
            if len(categories)>=2048:
                blocks.append(hasher.transform(categories).astype(np.float32));categories=[]
    if categories:blocks.append(hasher.transform(categories).astype(np.float32))
    # Stateless hashing avoids fitting a catalogue vocabulary or huge one-hot matrix.
    return np.asarray(numeric,dtype=np.float32), sparse.vstack(blocks,format='csr')

def design(semantic,numeric,categories,train):
    med=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in numeric[train].T])
    filled=np.where(np.isfinite(numeric),numeric,med)
    scaler=StandardScaler().fit(filled[train])
    values=np.clip(scaler.transform(filled),-5,5).astype(np.float32)
    return sparse.hstack((sparse.csr_matrix(semantic,dtype=np.float32),
                          sparse.csr_matrix(values/np.sqrt(3)),
                          sparse.csr_matrix((~np.isfinite(numeric)).astype(np.float32)/np.sqrt(3)),
                          categories/np.sqrt(len(CATEGORICAL))),format='csr')
def fit(x,ids,ratings):
    left,right=np.triu_indices(len(ids),1); delta=ratings[left]-ratings[right]; keep=delta!=0
    left,right,delta=left[keep],right[keep],delta[keep]
    if not len(delta): return None
    differences=x[ids[left]]-x[ids[right]]
    labels=(delta>0).astype(int)
    weights=np.abs(delta).astype(float);weights/=weights.sum()
    model=LogisticRegression(C=1.,fit_intercept=False,solver='lbfgs',max_iter=300)
    model.fit(sparse.vstack((differences,-differences)),np.r_[labels,1-labels],sample_weight=np.r_[weights,weights]*10)
    return model
def validate(semantic,numeric,categories,ids,ratings):
    wins=0.; pairs=0; folds=0
    if len(ids)<10: return {'accuracy':None,'pairs':0,'folds':0}
    for train,test in KFold(5,shuffle=True,random_state=71).split(ids):
        x=design(semantic,numeric,categories,ids[train])
        model=fit(x,ids[train],ratings[train])
        if model is None:continue
        score=model.decision_function(x[ids[test]])
        a,b=np.triu_indices(len(test),1); actual=ratings[test][a]-ratings[test][b]; valid=actual!=0
        predicted=score[a]-score[b];good=predicted[valid]*actual[valid]
        wins+=np.sum(good>0)+.5*np.sum(good==0);pairs+=len(good);folds+=1
    return {'accuracy':float(wins/pairs) if pairs else None,'pairs':int(pairs),'folds':folds}
def recommend_pairwise(semantic,numeric,categories,ratings,type_ids,prices,budget=None,k=25):
    ids=np.array(list(ratings),dtype=int); r=np.array(list(ratings.values()),dtype=float)
    if len(ids)<2 or len(np.unique(r))<2: raise ValueError('Rate cars differently to learn a preference.')
    evidence=validate(semantic,numeric,categories,ids,r)
    x=design(semantic,numeric,categories,ids); model=fit(x,ids,r)
    score=np.asarray(model.decision_function(x)); allowed=np.isfinite(score);allowed[ids]=False
    if budget is not None:allowed &= np.isfinite(prices)&(prices<=budget)
    pool=np.flatnonzero(allowed);pool=pool[np.argsort(-score[pool],kind='stable')]
    chosen=[];counts={}
    for i in pool:
        typ=str(type_ids[i])
        if counts.get(typ,0)>=2:continue
        chosen.append(int(i));counts[typ]=counts.get(typ,0)+1
        if len(chosen)==k:break
    return {'positions':chosen,'scores':score[chosen].tolist(),'validation':evidence,
            'method':'L2 pairwise logistic · MiniLM + vehicle features',
            'ordering':'learned utility descending'}
