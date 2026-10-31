"""CPU five-fold group OOF CatBoost assessment on original canonical cohorts.

Usage: python -m scripts.train_catboost_fivefold --ask INPUT --bid INPUT
Requires the locally prepared original cohorts; writes no fabricated metrics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer

from marketbridge.structural import IndexPanel, known_pricing_vins
from pricing.model_suite import load_cohort, predictors
from pricing.catboost_residuals import CATEGORICAL


def run_side(side, path, ask_path, panel, output, iterations):
    d=load_cohort(path)
    if side=='ask':
        # The shared cohort loader deliberately strips descriptions. Restore by
        # row identity, never by assumed row order after validity filtering.
        descriptions=pd.concat([c[['row_id','text']] for c in pd.read_json(path,lines=True,chunksize=5000)],ignore_index=True)
        if descriptions.row_id.duplicated().any():raise ValueError('Duplicate Craigslist row IDs')
        d=d.merge(descriptions,on='row_id',validate='one_to_one',sort=False)
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
    texts=d.text.fillna('').astype(str).tolist() if side=='ask' else None
    phase=d.phase.map({'train':'calibration','validation':'validation','test':'test'})
    tr=np.flatnonzero(phase=='calibration')
    if len(tr)<100 or d.vehicle_group.iloc[tr].nunique()<5:raise ValueError('Insufficient distinct calibration vehicles')
    preds=np.full(len(d),np.nan);folds=np.full(len(d),-1)
    loss='SurvivalAft:dist=Normal;scale=1.0' if side=='ask' else 'Quantile:alpha=0.5'
    params=dict(loss_function=loss,iterations=iterations,depth=8,learning_rate=.05,l2_leaf_reg=10,
                random_seed=42,thread_count=4,task_type='CPU',allow_writing_files=False,verbose=False)
    folder=Path(output)/side;folder.mkdir(parents=True,exist_ok=True)
    def text_coordinates(fit_idx,predict_idx,prefix):
        # No vocabulary, IDF statistic or manifold point from the held-out rows
        # participates in the fit. UMAP.transform only sees those rows afterward.
        import umap
        vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=10,max_df=.98,
                                   max_features=12000,sublinear_tf=True,dtype=np.float32)
        matrix=vectorizer.fit_transform([texts[i] for i in fit_idx])
        reducer=umap.UMAP(n_components=8,n_neighbors=25,min_dist=.1,metric='cosine',
                          random_state=42,transform_seed=42,low_memory=True,n_jobs=1)
        fitted=reducer.fit_transform(matrix)
        held=reducer.transform(vectorizer.transform([texts[i] for i in predict_idx]))
        joblib.dump((vectorizer,reducer),folder/f'{prefix}-tfidf-umap.joblib',compress=3)
        return fitted,held
    def pool(idx,features):
        labels=np.column_stack((y[idx],y[idx])) if side=='ask' else np.log(y[idx])
        return Pool(features,label=labels,cat_features=CATEGORICAL)
    def fit(idx,features):return CatBoostRegressor(**params).fit(pool(idx,features))
    def predict(model,features):return np.exp(np.clip(model.predict(features,prediction_type='RawFormulaVal'),-700,np.log(1e7)))
    def augmented(fit_idx,pred_idx,prefix):
        train=x.iloc[fit_idx].copy();held=x.iloc[pred_idx].copy()
        if side=='ask':
            a,b=text_coordinates(fit_idx,pred_idx,prefix)
            for j in range(a.shape[1]):
                train[f'description_umap_{j+1}']=a[:,j];held[f'description_umap_{j+1}']=b[:,j]
        return train,held
    for fold,(a,b) in enumerate(GroupKFold(5).split(tr,groups=d.vehicle_group.iloc[tr])):
        print(f'{side} group OOF {fold+1}/5',flush=True)
        train_x,held_x=augmented(tr[a],tr[b],f'fold-{fold+1}')
        model=fit(tr[a],train_x);preds[tr[b]]=predict(model,held_x);folds[tr[b]]=fold
        model.save_model(str(folder/f'fold-{fold+1}.cbm'))
    print(f'{side} calibration-only refit for untouched validation/test',flush=True)
    held=np.flatnonzero(phase!='calibration')
    train_x,held_x=augmented(tr,held,'calibration-refit')
    final=fit(tr,train_x);preds[held]=predict(final,held_x)
    final.save_model(str(folder/'calibration-refit.cbm'))
    if not np.isfinite(preds).all() or (preds<=0).any():raise ValueError('Incomplete OOF predictions')
    residual=np.log(y/preds)
    report={'market':side,'objective':loss,'folds':5,'iterations':iterations,'n':len(d),
            'residual_unit':'log(observed/reference prediction)', 'training':'calibration group OOF; calibration-only final refit',
            'description_features':'ask only: fold-fitted TF-IDF bigrams to 8 UMAP coordinates; no text on bid',
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
    p.add_argument('--market',choices=['ask','bid'])
    p.add_argument('--iterations',type=int,default=1000)
    args=p.parse_args()
    for side,path in [('ask',args.ask),('bid',args.bid)]:
        if args.market and args.market!=side:continue
        if not Path(path).exists():raise FileNotFoundError(f'{path} missing: prepare the original cohort first')
        run_side(side,path,args.ask,IndexPanel(args.index),args.output,args.iterations)
