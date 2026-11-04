import numpy as np
from scripts.bayesian_shrinkage_challenger import choose_weight,weights_for
def test_unhelpful_correction_shrinks_to_base():
    assert choose_weight(np.array([100.,100.]),np.array([100.,100.]),np.array([.5,.5]))==0
def test_helpful_correction_retained():
    assert choose_weight(np.array([200.,200.]),np.array([100.,100.]),np.log(np.array([2.,2.])))==1
def test_unseen_risk_stratum_uses_global():
    assert weights_for(['rare','unseen'],{'global':.5,'strata':{'rare':.25}}).tolist()==[.25,.5]

from scripts.bayesian_shrinkage_challenger import choose_calibration
def test_price_level_calibration_recovers_known_shift():
    result=choose_calibration(np.array([105.,210.]),np.array([100.,200.]),np.zeros(2))
    assert result['weight']==1
    assert np.isclose(result['multiplier'],1.05)
def test_price_level_shift_is_bounded():
    result=choose_calibration(np.array([1000.,2000.]),np.array([100.,200.]),np.zeros(2))
    assert np.isclose(result['multiplier'],np.exp(.1))
