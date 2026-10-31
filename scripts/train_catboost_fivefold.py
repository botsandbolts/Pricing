"""CPU five-fold group OOF CatBoost assessment on original canonical cohorts.

Usage: python -m scripts.train_catboost_fivefold --ask INPUT --bid INPUT
Requires the locally prepared original cohorts; writes no fabricated metrics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold

from marketbridge.structural import IndexPanel, known_pricing_vins
from pricing.model_suite import load_cohort, predictors
from pricing.catboost_residuals import CATEGORICAL


def run_side(side, path, ask_path, panel, output, iterations):
    d=load_cohort(path)
    x,valid,dates=predictors(d,side)
    factor=np.ones(len(d))
    if side=='bid':
        ratios,_=panel.ratios(dates)
        overlap=d.vehicle_group.astype(str).str.removeprefix('vin:').isin(known_pricing_vins(ask_path))
        valid &= ratios.notna() & ~overlap
        factor=ratios.to_numpy(float)
    d=d.loc[valid].copy();x=x.loc[valid].copy()
    y=d.price.to_numpy(float)/factor[valid.to_numpy()]
    if not np.isfinite(y).all() or (y<=0).any():raise ValueError('Invalid target prices')
    x['log_miles']=np.log1p(pd.to_numeric(d.odometer,errors='coerce'))
    cols=['age','log_miles','auction_grade']+CATEGORICAL
    for c in CATEGORICAL:x[c]=x[c].fillna('unknown').astype(str).str.strip().str.lower()
    x=x[cols].reset_index(drop=True)
    d=d.reset_index(drop=True)
    phase=d.phase.map({'train':'calibration','validation':'validation','test':'test'})
    tr=np.flatnonzero(phase=='calibration')
    if len(tr)<100 or d.vehicle_group.iloc[tr].nunique()<5:raise ValueError('Insufficient distinct calibration vehicles')
    preds=np.full(len(d),np.nan);folds=np.full(len(d),-1)
    loss='SurvivalAft:dist=Normal;scale=1.0' if side=='ask' else 'Quantile:alpha=0.5'
    params=dict(loss_function=loss,iterations=iterations,depth=8,learning_rate=.05,l2_leaf_reg=10,
                random_seed=42,thread_count=4,task_type='CPU',allow_writing_files=False,verbose=False)
    def pool(idx):
        labels=np.column_stack((y[idx],y[idx])) if side=='ask' else np.log(y[idx])
        return Pool(x.iloc[idx],label=labels,cat_features=CATEGORICAL)
    def fit(idx):return CatBoostRegressor(**params).fit(pool(idx))
    def predict(model,idx):return np.exp(np.clip(model.predict(x.iloc[idx],prediction_type='RawFormulaVal'),-700,np.log(1e7)))
    folder=Path(output)/side;folder.mkdir(parents=True,exist_ok=True)
    for fold,(a,b) in enumerate(GroupKFold(5).split(tr,groups=d.vehicle_group.iloc[tr])):
        print(f'{side} group OOF {fold+1}/5',flush=True)
        model=fit(tr[a]);preds[tr[b]]=predict(model,tr[b]);folds[tr[b]]=fold
        model.save_model(str(folder/f'fold-{fold+1}.cbm'))
    print(f'{side} calibration-only refit for untouched validation/test',flush=True)
    final=fit(tr);held=np.flatnonzero(phase!='calibration');preds[held]=predict(final,held)
    final.save_model(str(folder/'calibration-refit.cbm'))
    if not np.isfinite(preds).all() or (preds<=0).any():raise ValueError('Incomplete OOF predictions')
    residual=np.log(y/preds)
    report={'market':side,'objective':loss,'folds':5,'iterations':iterations,'n':len(d),
            'residual_unit':'log(observed/reference prediction)', 'training':'calibration group OOF; calibration-only final refit',
            'scores':{}}
    for split in ('calibration','validation','test'):
        mask=(phase==split).to_numpy();report['scores'][split]={'n':int(mask.sum()),
            'mae_usd':float(np.abs(y[mask]-preds[mask]).mean()),
            'rmse_log':float(np.sqrt(np.mean(residual[mask]**2))),
            'median_signed_log_residual':float(np.median(residual[mask]))}
    out=pd.DataFrame({'row_id':d.row_id,'vehicle_group':d.vehicle_group,'split':phase,'fold':folds,
                      'observed_price':y,'predicted_price':preds,'residual_log':residual,
                      'residual_usd':y-preds})
    out.to_csv(folder/'predictions.csv.gz',index=False)
    (folder/'evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--ask',default='training-cache/full-cohort.jsonl')
    p.add_argument('--bid',default='training-cache/bid-cohort.jsonl')
    p.add_argument('--index',default='models/structural-transfer/used-car-index.csv')
    p.add_argument('--output',default='training-cache/catboost-fivefold')
    p.add_argument('--iterations',type=int,default=1000)
    args=p.parse_args()
    for side,path in [('ask',args.ask),('bid',args.bid)]:
        if not Path(path).exists():raise FileNotFoundError(f'{path} missing: prepare the original cohort first')
        run_side(side,path,args.ask,IndexPanel(args.index),args.output,args.iterations)
