import numpy as np,pandas as pd,pytest
from pricing.bayesian_calibration import apply,validate

def test_calibration_price_and_fallback_interval():
    f=pd.DataFrame({'age':[8,20],'odometer':[40000,180000]})
    p=pd.DataFrame({'base_price':[10000.,2000.],'delta_log':[.1,.1],'corrected_price':[11000.,2200.],'model_n':[100,0],'predictive_sd_log':[.2,.3]})
    c={'weights':{'global':{'weight':1.,'multiplier':1.05},'strata':{'common|5000plus|other':{'weight':2.,'multiplier':1.}}},
       'interval_scales':{'global':3.,'strata':{'common|5000plus|other':2.}},'interval_counts':{'common|5000plus|other':120},'interval_calibration_rows':500}
    result=apply(f,p,c)
    assert np.allclose(result.corrected_price,[10000*np.exp(.2),2000*np.exp(.1)*1.05])
    assert result.interval_multiplier.tolist()==[2,3]
    assert result.interval_calibration_n.tolist()==[120,500]
    assert np.allclose(result.predictive_sd_log,p.predictive_sd_log)
    assert np.allclose(result.delta_log,np.log(result.corrected_price/result.base_price))

def test_reject_wrong_release_or_unqualified():
    meta={'posterior_sha256':'a','base_evaluation_sha256':'b'}
    good={**meta,'eligible_for_promotion':True,'baseline_reproduced':True}
    validate(meta,good)
    with pytest.raises(ValueError):validate(meta,{**good,'posterior_sha256':'wrong'})
    with pytest.raises(ValueError):validate(meta,{**good,'eligible_for_promotion':False})
