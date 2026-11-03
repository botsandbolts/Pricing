import numpy as np
from pricing.svd_discovery import recommend_svd
def test_likes_exclusions_budget_distinct_and_order():
    rng=np.random.default_rng(1);x=rng.normal(size=(150,16));x/=np.linalg.norm(x,axis=1,keepdims=True)
    result=recommend_svd(x,{0:4,1:1},list(range(150)),np.arange(150)*100,budget=10000)
    assert len(set(result['positions']))==25
    assert not {0,1}&set(result['positions'])
    assert max(result['positions'])<=100
    assert all(a>=b for a,b in zip(result['scores'],result['scores'][1:]))
