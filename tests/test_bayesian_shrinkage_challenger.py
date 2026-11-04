import numpy as np
from scripts.bayesian_shrinkage_challenger import choose_weight,weights_for
def test_unhelpful_correction_shrinks_to_base():
    assert choose_weight(np.array([100.,100.]),np.array([100.,100.]),np.array([.5,.5]))==0
def test_helpful_correction_retained():
    assert choose_weight(np.array([200.,200.]),np.array([100.,100.]),np.log(np.array([2.,2.])))==1
def test_unseen_risk_stratum_uses_global():
    assert weights_for(['rare','unseen'],{'global':.5,'strata':{'rare':.25}}).tolist()==[.25,.5]
