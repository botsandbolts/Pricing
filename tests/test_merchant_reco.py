import numpy as np
from pricing.merchant_reco import merchant_evidence,recommend
def test_role_and_platform_evidence_does_not_create_merchants():
    assert merchant_evidence('https://facebook.com/a',['Great Auto Sales'])[0] is None
    assert merchant_evidence('https://bank.example.com',['Credit Finance Automotive'])[0] is None
    assert merchant_evidence('Visit https://abc-motors.example',['ABC Motors'])[0]=='abc-motors.example'
def test_recommendations_rank_preferences_not_farthest_distance():
    rng=np.random.default_rng(4);x=rng.normal(size=(100,8));x/=np.linalg.norm(x,axis=1,keepdims=True)
    result=recommend(x,np.zeros((100,3)),{0:4,1:1},list(range(100)),np.ones(100)*10000)
    assert len(set(result['positions']))==10 and not {0,1}.intersection(result['positions'])
    assert all(a>=b for a,b in zip(result['scores'],result['scores'][1:]))
def test_budget_is_explicit_and_no_dislike_veto():
    x=np.ones((30,4))*.5
    result=recommend(x,np.zeros((30,3)),{0:4,1:1},list(range(30)),np.arange(30)*1000,budget=15000)
    assert len(result['positions'])==10
    assert max(result['positions'])<=15
