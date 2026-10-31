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
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer

from marketbridge.structural import IndexPanel, known_pricing_vins
from pricing.model_suite import load_cohort, predictors
from pricing.catboost_residuals import CATEGORICAL


def run_side(side, path, ask_path, panel, output, iterations, residual_source, minilm_features=None):
    d=load_cohort(path)
    if residual_source:
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
        (audit/'cleaning.json').write_text(json.dumps({'name':'The Sparse Lemon Squeezer','source':'fresh five-fold sparse TF-IDF XGBoost group OOF log residuals','removed_training_rows':number,'original_training_rows':len(ranked),'fraction':.25,'validation_test_filtered':False,'selection_limit':'Fixed OOF selection precedes refitting. Later OOF scores are conditional diagnostics, not nested CV of selection. Full original validation/test remain unchanged.'},indent=2))
    if 'vehicle_group' not in d:
        if side!='ask':raise ValueError('Auction vehicle_group is required')
        # Craigslist preparation already deduplicates VINs and descriptions.
        d['vehicle_group']=d.row_id.astype(str)
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
    embeddings=None
    if side=='ask' and minilm_features:
        import hashlib
        mf=Path(minilm_features)
        provenance=json.loads((mf/'provenance.json').read_text())
        if provenance['cohort_sha256']!=hashlib.file_digest(open(path,'rb'),'sha256').hexdigest():raise ValueError('MiniLM cohort mismatch')
        row_ids=json.loads((mf/'row-ids.json').read_text())
        lookup=pd.Index(row_ids).get_indexer(d.row_id.astype(str))
        if (lookup<0).any():raise ValueError('Missing MiniLM row')
        embeddings=np.load(mf/'embeddings.npy',mmap_mode='r')[lookup]
    texts=d.text.fillna('').astype(str).tolist() if side=='ask' else None
    phase=d.phase.map({'train':'calibration','validation':'validation','test':'test'})
    tr=np.flatnonzero(phase=='calibration')
    if len(tr)<100 or d.vehicle_group.iloc[tr].nunique()<5:raise ValueError('Insufficient distinct calibration vehicles')
    preds=np.full(len(d),np.nan);folds=np.full(len(d),-1)
    loss='survival:aft' if side=='ask' else 'reg:quantileerror'
    params=dict(objective=loss,tree_method='hist',max_depth=5,eta=.08,reg_lambda=10.,
                subsample=.85,colsample_bytree=.85,max_cat_to_onehot=4,min_child_weight=20,
                seed=42,nthread=2,enable_categorical=True)
    if side=='ask':params.update(aft_loss_distribution='normal',aft_loss_distribution_scale=1.)
    else:params['quantile_alpha']=.5
    folder=Path(output)/side;folder.mkdir(parents=True,exist_ok=True)
    def matrix(frame,levels,vectorizer,text):
        from scipy.sparse import csr_matrix,hstack
        encoded=frame.copy()
        for c in CATEGORICAL:
            codes=pd.Categorical(frame[c],categories=levels[c]).codes.astype(float)
            codes[codes<0]=np.nan
            encoded[c]=codes
        values=encoded.to_numpy(dtype=np.float32)
        n,k=values.shape
        # Keep explicit zeros: a zero age/category is a value, not sparse missing.
        structured=csr_matrix((values.ravel(),np.tile(np.arange(k),n),np.arange(0,(n+1)*k,k)),shape=(n,k))
        names=list(frame);types=['c' if c in CATEGORICAL else 'q' for c in frame]
        if vectorizer is not None:
            sparse_text=vectorizer.transform(text)
            structured=hstack([structured,sparse_text],format='csr')
            names += [f'tfidf_{j}' for j in range(sparse_text.shape[1])]
            types += ['q']*sparse_text.shape[1]
        return xgb.DMatrix(structured,feature_names=names,feature_types=types,enable_categorical=True,nthread=2)
    def fit(idx,features):
        frame,text=features
        levels={c:sorted(frame[c].astype(str).unique().tolist()) for c in CATEGORICAL}
        vectorizer=None
        if side=='ask':
            vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=10,max_df=.98,
                                      max_features=2000,sublinear_tf=True,dtype=np.float32)
            vectorizer.fit(text)
        dm=matrix(frame,levels,vectorizer,text)
        if side=='ask':
            dm.set_float_info('label_lower_bound',y[idx]);dm.set_float_info('label_upper_bound',y[idx])
        else:dm.set_label(np.log(y[idx]))
        return xgb.train(params,dm,num_boost_round=iterations),levels,vectorizer
    def predict(fitted,features):
        model,levels,vectorizer=fitted
        frame,text=features
        raw=model.predict(matrix(frame,levels,vectorizer,text))
        return np.asarray(raw,float) if side=='ask' else np.exp(np.clip(raw,-700,np.log(1e7)))
    def augmented(fit_idx,pred_idx,prefix):
        train=x.iloc[fit_idx].copy();held=x.iloc[pred_idx].copy()
        if embeddings is not None:
            from pricing.minilm_signal import crossfit,price
            oof,head=crossfit(embeddings[fit_idx],y[fit_idx],d.vehicle_group.iloc[fit_idx].to_numpy())
            train['predicted_text_price']=oof
            held['predicted_text_price']=price(head,embeddings[pred_idx])
            if prefix=='calibration-refit':
                (folder/'minilm-head.json').write_text(json.dumps(head))
        return (train,[texts[i] for i in fit_idx] if texts else None),(held,[texts[i] for i in pred_idx] if texts else None)
    for fold,(a,b) in enumerate(GroupKFold(5).split(tr,groups=d.vehicle_group.iloc[tr])):
        print(f'{side} group OOF {fold+1}/5',flush=True)
        train_x,held_x=augmented(tr[a],tr[b],f'fold-{fold+1}')
        model=fit(tr[a],train_x);preds[tr[b]]=predict(model,held_x);folds[tr[b]]=fold
        model[0].save_model(str(folder/f'fold-{fold+1}.ubj'))
    print(f'{side} calibration-only refit for untouched validation/test',flush=True)
    held=np.flatnonzero(phase!='calibration')
    train_x,held_x=augmented(tr,held,'calibration-refit')
    final=fit(tr,train_x);preds[held]=predict(final,held_x)
    final[0].save_model(str(folder/'calibration-refit.ubj'))
    if side=='ask':joblib.dump(final[2],folder/'tfidf.joblib',compress=3)
    (folder/'category-schema.json').write_text(json.dumps(final[1]))
    if not np.isfinite(preds).all() or (preds<=0).any():raise ValueError('Incomplete OOF predictions')
    residual=np.log(y/preds)
    report={'market':side,'objective':loss,'folds':5,'iterations':iterations,'n':len(d),
            'residual_unit':'log(observed/reference prediction)', 'training':'calibration group OOF; calibration-only final refit',
            'description_features':'ask: fold-fitted sparse TF-IDF, 2000 features, no UMAP; bid: structured only',
            'transformer_feature':bool(embeddings is not None),'scores':{}}
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
    p.add_argument('--output',default='training-cache/xgboost-fivefold')
    p.add_argument('--market',choices=['ask','bid'])
    p.add_argument('--residual-source',default=None)
    p.add_argument('--minilm-features',default=None)
    p.add_argument('--iterations',type=int,default=300)
    args=p.parse_args()
    for side,path in [('ask',args.ask),('bid',args.bid)]:
        if args.market and args.market!=side:continue
        if not Path(path).exists():raise FileNotFoundError(f'{path} missing: prepare the original cohort first')
        run_side(side,path,args.ask,IndexPanel(args.index),args.output,args.iterations,args.residual_source,args.minilm_features)
