"""Small validation calibration over a frozen Bayesian posterior."""
import numpy as np
from pricing.robust_hierarchy import interval_strata

def apply(frame,prediction,calibration):
    result=prediction.copy()
    strata=interval_strata(frame,result.model_n.to_numpy())
    settings=[calibration['weights']['strata'].get(k,calibration['weights']['global']) for k in strata]
    strength=np.array([v['weight'] for v in settings])
    multiplier=np.array([v['multiplier'] for v in settings])
    price=result.base_price.to_numpy()*np.exp(np.clip(strength*result.delta_log.to_numpy(),-5,5))*multiplier
    result['corrected_price']=price
    result['delta_log']=np.log(price/result.base_price.to_numpy())
    scale=calibration['interval_scales']
    result['interval_multiplier']=[scale['strata'].get(k,scale['global']) for k in strata]
    result['interval_calibration_group']=strata
    counts=calibration['interval_counts']
    result['interval_calibration_n']=[counts.get(k,calibration['interval_calibration_rows']) for k in strata]
    result['interval_calibration_source']=['validation risk stratum' if k in scale['strata'] else 'global validation fallback' for k in strata]
    return result

def validate(metadata,calibration):
    if not calibration.get('eligible_for_promotion') or not calibration.get('baseline_reproduced'):
        raise ValueError('Unqualified Bayesian calibration')
    for key in ['posterior_sha256','base_evaluation_sha256']:
        if calibration.get(key)!=metadata.get(key):
            raise ValueError('Bayesian calibration belongs to another model release')
