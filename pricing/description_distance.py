"""Sparse three-block distance for UMAP; no dense TF-IDF or all-pairs matrix."""
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix,hstack
from sklearn.feature_extraction.text import TfidfVectorizer

def texts(frame):
    field='text' if 'text' in frame else 'description'
    if field not in frame:return ['']*len(frame)
    return frame[field].fillna('').astype(str).tolist()

def combine(frame,geometry,vectorizer,fit=False):
    t=vectorizer.fit_transform(texts(frame)) if fit else vectorizer.transform(texts(frame))
    return hstack([csr_matrix(geometry),t],format='csr',dtype=np.float32)

def make_distance(c,n,fields):
    @njit
    def distance(i1,v1,i2,v2):
        p=0;q=0;cat=0.;num=0.;dot=0.;norm1=0.;norm2=0.
        for j in range(len(i1)):
            if i1[j]>=c+n:norm1+=v1[j]*v1[j]
        for j in range(len(i2)):
            if i2[j]>=c+n:norm2+=v2[j]*v2[j]
        while p<len(i1) or q<len(i2):
            if q==len(i2) or (p<len(i1) and i1[p]<i2[q]):
                idx=i1[p];a=v1[p];b=0.;p+=1
            elif p==len(i1) or i2[q]<i1[p]:
                idx=i2[q];a=0.;b=v2[q];q+=1
            else:
                idx=i1[p];a=v1[p];b=v2[q];p+=1;q+=1
            if idx<c:cat+=abs(a-b)
            elif idx<c+n:num+=abs(a-b)
            else:dot+=a*b
        base=cat/(2*fields)+num/n
        if norm1>0 and norm2>0:
            cosine=max(0.,min(1.,dot/np.sqrt(norm1*norm2)))
            return (base+1-cosine)/3.
        return base/2.
    return distance
