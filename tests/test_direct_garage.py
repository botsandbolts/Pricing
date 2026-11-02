import numpy as np
from pricing.mixed_garage import direct_hdbscan_geometry

def test_weighted_manhattan_matches_nominal_numeric_formula():
    # Two one-hot fields, each with two levels; two numeric features.
    state={'onehot_dimensions':4,'numeric_dimensions':2,'categorical':{'categories':{'make':['a','b'],'fuel':['gas','diesel']}}}
    x=np.array([[1,0,1,0,0.,0.],[0,1,1,0,1.,0.],[0,1,0,1,1.,1.]])
    y,w=direct_hdbscan_geometry(x,state)
    assert np.isclose(np.abs(y[0]-y[1]).sum(),.5*.5+.5*.5)
    assert np.isclose(np.abs(y[0]-y[2]).sum(),1.)
    np.testing.assert_allclose(y,x*w)
    assert y.shape==x.shape

def test_direct_hdbscan_supports_soft_memberships_and_new_points():
    import hdbscan
    from hdbscan.prediction import membership_vector
    rng=np.random.default_rng(42)
    x=np.r_[rng.normal(0,.01,(40,6)),rng.normal(2,.01,(40,6))]
    model=hdbscan.HDBSCAN(metric='manhattan',min_cluster_size=10,min_samples=5,prediction_data=True).fit(x)
    m=hdbscan.all_points_membership_vectors(model)
    assert m.shape==(80,2) and np.isfinite(m).all()
    new=membership_vector(model,x[:2]);assert new.shape==(2,2) and np.isfinite(new).all()
