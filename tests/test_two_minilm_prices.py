import numpy as np
from pricing.minilm_signal import two_price_crossfit,mean_price

def test_each_of_two_prices_excludes_the_entire_predicted_group():
    rng=np.random.default_rng(7);x=rng.normal(size=(120,8));y=np.exp(9+x[:,0]*.2);groups=np.repeat(np.arange(60),2)
    a,heads,individual=two_price_crossfit(x,y,groups)
    changed=y.copy();changed[groups==0]*=20
    b,_,other=two_price_crossfit(x,changed,groups)
    np.testing.assert_allclose(individual[groups==0],other[groups==0],rtol=0,atol=0)
    np.testing.assert_allclose(a,individual.mean(axis=1))
    assert np.any(np.abs(a[groups!=0]-b[groups!=0])>1e-6)
    assert len(heads)==2
    assert np.isfinite(mean_price(heads,x)).all()

def test_inference_averages_dollar_prices_arithmetically():
    heads=[{'coef':[0.],'intercept':np.log(100.)},{'coef':[0.],'intercept':np.log(400.)}]
    np.testing.assert_allclose(mean_price(heads,np.zeros((3,1))),250.)
