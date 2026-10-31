"""Compare identical held-out populations and select the displayed estimator."""
def mae_comparison(base,bayesian):
    if base<=0 or bayesian<0:raise ValueError('Invalid MAE')
    saved=base-bayesian
    return {'base':base,'bayesian':bayesian,'saved_usd':saved,
            'improvement_pct':100*saved/base,'improved':saved>0}

def selected_receipt(receipt,bayesian=False):
    result=dict(receipt)
    if 'error' in result:return result
    if not bayesian:
        result['price']=result['forest_price']
        result['historical_corrected_price']=result['historical_forest_price']
        result.pop('interval_low',None);result.pop('interval_high',None)
    return result
