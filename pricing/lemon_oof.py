"""Reuse audited Lemon retained-cohort AFT OOF predictions by exact identity."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
def align_oof(cars,predictions,evaluation):
    if evaluation.get('objective')!='survival:aft' or evaluation.get('folds')!=5:
        raise ValueError('Expected five-fold asking AFT export')
    ids=pd.Index([str(c['row_id']) for c in cars])
    if not ids.is_unique or predictions.row_id.astype(str).duplicated().any():
        raise ValueError('Duplicate row IDs')
    p=predictions.copy();p['row_id']=p.row_id.astype(str)
    p=p[p.split.eq('calibration')].set_index('row_id')
    if set(p.index)!=set(ids):raise ValueError('Retained cohort identity mismatch')
    p=p.loc[ids]
    if not p.fold.isin(range(5)).all() or set(p.fold)!=set(range(5)):
        raise ValueError('Incomplete held-out fold coverage')
    observed=np.array([float(c['price']) for c in cars])
    predicted=p.predicted_price.to_numpy(float)
    if not np.isfinite(predicted).all() or (predicted<=0).any():raise ValueError('Invalid OOF predictions')
    if not np.allclose(observed,p.observed_price,rtol=1e-8,atol=.01):raise ValueError('Observed prices do not match')
    residual=np.log(observed/predicted)
    if not np.allclose(residual,p.residual_log,rtol=1e-7,atol=1e-8):raise ValueError('Residual units or values mismatch')
    return np.log(predicted),residual,p.vehicle_group.astype(str).to_numpy()
def load_oof(cars,folder='training-cache/lemon-hierarchy/ask'):
    folder=Path(folder);source=folder/'predictions.csv.gz'
    evaluation=json.loads((folder/'evaluation.json').read_text())
    installed=Path('models/xgboost-hierarchy/ask/evaluation.json')
    if installed.exists() and json.loads(installed.read_text())!=evaluation:
        raise ValueError('OOF export does not match the installed asking model evaluation')
    pred,residual,groups=align_oof(cars,pd.read_csv(source,dtype={'row_id':str,'vehicle_group':str}),evaluation)
    provenance={'source_run':41605651393,'artifact':'lemon-hierarchy-ask','objective':'survival:aft','folds':5,
      'rows':len(cars),'source_predictions_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'evaluation':evaluation,'selection_limit':'Frozen retained Lemon training cohort; not nested validation of original cleaning',
      'residual_unit':'log(observed_price/predicted_price)','description':'Two cross-fitted MiniLM dollar-price predictions averaged inside each outer training fold',
      'refitting':'No new vehicle-price models trained for merchant analysis'}
    return pred,residual,groups,provenance
