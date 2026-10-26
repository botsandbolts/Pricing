"""Per-member attribution; avoid adding incompatible target-scale SHAP values."""
import numpy as np
from marketbridge.robust import engineer,invert

def explain_member(ensemble,frame,index=0):
    member,model=ensemble.members[index]
    x,_=engineer(frame,ensemble.schema,ensemble.structural,ensemble.feature_schema)
    if len(x)!=1:raise ValueError('Explain one vehicle at a time')
    values=np.asarray(model.predict(x,pred_contrib=True,num_threads=4),float)[0]
    contributions=values[:-1]*member['std'];base=float(values[-1]*member['std']+member['mean'])
    raw=float(model.predict(x,num_threads=4)[0]);scaled=raw*member['std']+member['mean']
    return {'member':member['file'],'weight':member['weight'],'target':member['target'],'base':base,
        'raw_prediction':scaled,'reconstructed':float(base+contributions.sum()),
        'additivity_error':float(abs(base+contributions.sum()-scaled)),
        'member_price_usd':float(invert([raw],member['mean'],member['std'],member['target'])[0]),
        'features':[{'feature':name,'contribution':float(value)} for name,value in zip(x.columns,contributions)],
        'method':'LightGBM native TreeSHAP, tree-path-dependent. Contributions are on this member’s target scale, not a cross-member causal decomposition.'}
