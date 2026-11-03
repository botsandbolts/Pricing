"""Shrunk unsupervised vehicle/dealer SVD with bounded rating similarities."""
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from pricing.session_pairwise import catalogue_features,design
def build(folder):
    semantic=np.load(folder/'semantic.npy',mmap_mode='r')
    numeric,categories=catalogue_features(folder)
    # No text block here: MiniLM is retained as an independent similarity channel.
    matrix=design(np.zeros((len(numeric),0)),numeric,categories,np.arange(len(numeric)))
    svd=TruncatedSVD(n_components=64,n_iter=5,random_state=42)
    latent=svd.fit_transform(matrix).astype(np.float32)
    alpha=float(.25*svd.singular_values_[0]**2)
    shrink=svd.singular_values_**2/(svd.singular_values_**2+alpha)
    latent*=shrink
    vectors=normalize(np.hstack((normalize(latent)*np.sqrt(.5),np.asarray(semantic,dtype=np.float32)*np.sqrt(.5)))).astype(np.float16)
    np.save(folder/'shrunk-svd.npy',vectors)
    return {'dimensions':64,'spectral_alpha':alpha,'shrinkage':shrink.tolist(),'method':'Truncated SVD with ridge-style spectral shrinkage; 50% structured/dealer similarity, 50% MiniLM'}
def utility(vectors,ids,r):
    sim=np.asarray(vectors,dtype=np.float32)@np.asarray(vectors[ids],dtype=np.float32).T
    good=r>=3;bad=r<=2
    positive=np.max(sim[:,good]*((r[good]-2)/2),axis=1)
    negative=np.max(np.maximum(sim[:,bad],0)*((3-r[bad])/2),axis=1) if bad.any() else np.zeros(len(vectors))
    return positive-.35*negative
def recommend_svd(vectors,ratings,type_ids,prices,budget=None,k=25):
    ids=np.array(list(ratings),dtype=int);r=np.array(list(ratings.values()),dtype=float)
    if not np.any(r>=3):raise ValueError('Like at least one car.')
    scores=utility(vectors,ids,r)
    allowed=np.isfinite(scores);allowed[ids]=False
    if budget is not None:allowed&=np.isfinite(prices)&(prices<=budget)
    pool=np.flatnonzero(allowed);pool=pool[np.argsort(-scores[pool],kind='stable')[:2000]]
    chosen=[];counts={};redundancy=np.zeros(len(pool));remaining=np.ones(len(pool),bool)
    pv=np.asarray(vectors[pool],dtype=np.float32)
    for _ in range(min(k,len(pool))):
        adjusted=scores[pool]-.25*redundancy
        adjusted[~remaining]=-np.inf
        if not np.isfinite(adjusted).any():break
        j=int(np.argmax(adjusted));i=int(pool[j]);chosen.append(i)
        counts[str(type_ids[i])]=counts.get(str(type_ids[i]),0)+1
        remaining[j]=False
        if counts[str(type_ids[i])]>=2:remaining &= np.array([str(type_ids[t])!=str(type_ids[i]) for t in pool])
        redundancy=np.maximum(redundancy,np.maximum(pv@pv[j],0))
    chosen=sorted(chosen,key=lambda i:(-scores[i],i))
    return {'positions':chosen,'scores':scores[chosen].tolist(),'method':'Shrunk truncated SVD · MiniLM + vehicle features + dealer candidates','ordering':'preference descending after diversity selection'}
