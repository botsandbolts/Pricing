import numpy as np
from pricing.dealer_signals import candidates,dealer_matrix,dealer_caption
def test_preserve_original_candidates_without_blocklists():
    rows=candidates([{'name':'EPA','score':.7},{'name':'Mint Motors','score':.9},{'name':'mint motors','score':.6}])
    assert len(rows)==2 and rows[0]['name']=='Mint Motors'
    assert rows[0]['score']==.9
    assert 'inferred, unverified' in dealer_caption(rows)
def test_confidence_and_ambiguity_weighting():
    x=dealer_matrix([[{'name':'A','score':.8}],[],[{'name':'A','score':.8},{'name':'B','score':.6}]])
    assert x.shape==(3,2) and x[0,0]==np.float32(.8)
    assert x[1].nnz==0
    assert np.isclose(x[2,0],.8/np.sqrt(2))

def test_dealer_features_can_change_recommendations():
    from scipy import sparse
    from pricing.session_pairwise import recommend_pairwise
    n=100
    rows=[[{'name':'Liked dealer' if i%2==0 else 'Disliked dealer','score':.9}] for i in range(n)]
    features=sparse.hstack((sparse.csr_matrix((n,512)),1.5*dealer_matrix(rows)),format='csr')
    result=recommend_pairwise(np.zeros((n,4)),np.ones((n,3)),features,{i:4 if i%2==0 else 1 for i in range(25)},list(range(n)),np.ones(n))
    assert len(result['positions'])==25
    assert all(i%2==0 for i in result['positions'])
    assert result['validation']['accuracy']==1
