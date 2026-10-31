"""CPU five-fold group OOF XGBoost assessment on original canonical cohorts.

Usage: python -m scripts.train_xgboost_fivefold --ask INPUT --bid INPUT
Requires the locally prepared original cohorts; writes no fabricated metrics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer

from marketbridge.structural import IndexPanel, known_pricing_vins
from pricing.model_suite import load_cohort, predictors
from pricing.catboost_residuals import CATEGORICAL


def run_side(side, path, ask_path, panel, output, iterations, residual_source):
    d=load_cohort(path)
    source=pd.read_csv(residual_source,dtype={'row_id':str})
    eligible=source.loc[source.split.eq('calibration')].copy()
    if eligible.row_id.duplicated().any() or eligible.fold.nunique()!=5:raise ValueError('Invalid original five-fold residuals')
    if not np.allclose(eligible.residual_log,np.log(eligible.observed_price/eligible.predicted_price)):raise ValueError('Incorrect residual unit')
    eligible['abs_residual']=eligible.residual_log.abs()
    ranked=eligible.sort_values(['abs_residual','row_id'],ascending=[False,True])
    number=int(np.ceil(.25*len(ranked)))
    removed=set(ranked.head(number).row_id)
    if not removed.issubset(set(d.row_id.astype(str))):raise ValueError('Residual IDs do not match original cohort')
    d=d.loc[~d.row_id.astype(str).isin(removed)].copy()
    audit=Path(output)/side;audit.mkdir(parents=True,exist_ok=True)
    ranked['retained']=~ranked.row_id.isin(removed)
    ranked.to_csv(audit/'cleaning-audit.csv.gz',index=False)
    (audit/'cleaning.json').write_text(json.dumps({'name':'Lemon Lite','source':'original five-fold group OOF log residuals','removed_training_rows':number,'original_training_rows':len(ranked),'fraction':.25,'validation_test_filtered':False,'selection_limit':'Fixed OOF selection precedes refitting. Later OOF scores are conditional diagnostics, not nested CV of selection. Full original validation/test remain unchanged.'},indent=2))
    if 'vehicle_group' not in d:
        if side!='ask':raise ValueError('Auction vehicle_group is required')
        # Craigslist preparation already deduplicates VINs and descriptions.
        d['vehicle_group']=d.row_id.astype(str)
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
    for c in CATEGORICAL:
        x[c]=x[c].fillna('unknown').astype(str).str.strip().str.lower() if c in x else 'unknown'
    x=x[cols].reset_index(drop=True)
    d=d.reset_index(drop=True)
    texts=None
    phase=d.phase.map({'train':'calibration','validation':'validation','test':'test'})
    tr=np.flatnonzero(phase=='calibration')
    if len(tr)<100 or d.vehicle_group.iloc[tr].nunique()<5:raise ValueError('Insufficient distinct calibration vehicles')
    preds=np.full(len(d),np.nan);folds=np.full(len(d),-1)
    loss='quantile'
    params=dict(objective='quantile',alpha=.5,n_estimators=300,num_leaves=15,
                max_depth=6,learning_rate=.05,min_child_samples=80,reg_lambda=10.,
                n_jobs=2,verbosity=-1,random_state=42)
    folder=Path(output)/side;folder.mkdir(parents=True,exist_ok=True)
    def categorical(frame,levels):
        result=frame.copy()
        for c in CATEGORICAL:result[c]=pd.Categorical(result[c],categories=levels[c])
        return result
    def fit(idx,features):
        levels={c:sorted(features[c].astype(str).unique().tolist()) for c in CATEGORICAL}
        prepared=categorical(features,levels)
        model=lgb.LGBMRegressor(**params)
        model.fit(prepared,np.log(y[idx]),categorical_feature=CATEGORICAL)
        return model.booster_,levels
    def predict(fitted,features):
        model,levels=fitted
        return np.exp(np.clip(model.predict(categorical(features,levels),num_threads=2),-700,np.log(1e7)))
    def augmented(fit_idx,pred_idx,prefix):
        return x.iloc[fit_idx].copy(),x.iloc[pred_idx].copy()
    for fold,(a,b) in enumerate(GroupKFold(5).split(tr,groups=d.vehicle_group.iloc[tr])):
        print(f'{side} group OOF {fold+1}/5',flush=True)
        train_x,held_x=augmented(tr[a],tr[b],f'fold-{fold+1}')
        model=fit(tr[a],train_x);preds[tr[b]]=predict(model,held_x);folds[tr[b]]=fold
        model[0].save_model(str(folder/f'fold-{fold+1}.txt'))
    print(f'{side} calibration-only refit for untouched validation/test',flush=True)
    held=np.flatnonzero(phase!='calibration')
    train_x,held_x=augmented(tr,held,'calibration-refit')
    final=fit(tr,train_x);preds[held]=predict(final,held_x)
    final[0].save_model(str(folder/'calibration-refit.txt'))
    (folder/'category-schema.json').write_text(json.dumps(final[1]))
    if not np.isfinite(preds).all() or (preds<=0).any():raise ValueError('Incomplete OOF predictions')
    residual=np.log(y/preds)
    report={'market':side,'objective':loss,'folds':5,'iterations':iterations,'n':len(d),
            'residual_unit':'log(observed/reference prediction)', 'training':'calibration group OOF; calibration-only final refit',
            'description_features':'none: compact structured-only Bayesian panel',
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
    p.add_argument('--output',default='training-cache/lightgbm-lite')
    p.add_argument('--market',choices=['ask','bid'])
    p.add_argument('--residual-source',required=True)
    p.add_argument('--iterations',type=int,default=1000)
    args=p.parse_args()
    for side,path in [('ask',args.ask),('bid',args.bid)]:
        if args.market and args.market!=side:continue
        if not Path(path).exists():raise FileNotFoundError(f'{path} missing: prepare the original cohort first')
        run_side(side,path,args.ask,IndexPanel(args.index),args.output,args.iterations,args.residual_source)
