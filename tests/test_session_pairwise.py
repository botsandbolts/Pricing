import numpy as np
from scipy import sparse
from pricing.session_pairwise import recommend_pairwise,design
def test_vehicle_features_and_validation():
    n=100
    semantic=np.zeros((n,4),dtype=np.float32)
    numeric=np.column_stack((np.ones(n)*2015,np.ones(n)*10,np.linspace(5,12,n)))
    categories=sparse.csr_matrix((n,512))
    ids=np.arange(0,100,4); ratings={int(i):int(1+i//25) for i in ids}
    result=recommend_pairwise(semantic,numeric,categories,ratings,list(range(n)),np.ones(n))
    assert len(set(result['positions']))==25
    assert not set(result['positions'])&set(ids)
    assert result['positions'][0]>80
    assert result['validation']['accuracy']>.9
    assert all(a>=b for a,b in zip(result['scores'],result['scores'][1:]))
def test_training_only_preprocessing():
    numeric=np.array([[2010,1,1],[2020,2,2],[9999,1000,1000]],dtype=float)
    x=design(np.zeros((3,2)),numeric,sparse.csr_matrix((3,512)),np.array([0,1]))
    assert np.allclose(x.toarray()[0,2:5],-np.ones(3)/np.sqrt(3))
