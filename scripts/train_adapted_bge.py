"""CPU five-fold group OOF XGBoost assessment on original canonical cohorts.

Usage: BGE_ARM=bge UMAP_DIM=32 UMAP_NEIGHBORS=30 UMAP_MIN_DIST=.1 python -m scripts.train_adapted_bge --market ask
Requires the locally prepared original cohorts; writes no fabricated metrics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from pricing.lemon_folds import fivefold


from marketbridge.structural import IndexPanel, known_pricing_vins
from pricing.model_suite import load_cohort, predictors
from pricing.catboost_residuals import CATEGORICAL


def run_side(side, path, ask_path, panel, output, iterations, residual_source, minilm_features=None):
    d=pd.read_csv(path,dtype={'row_id':str,'vehicle_group':str})
    if d.row_id.duplicated().any():raise ValueError('Duplicate Lemon row IDs')
    d['price']=d.observed_price
    d['phase']=d.split.map({'calibration':'train','validation':'validation','test':'test'})
    d['text']=''
    if d.phase.isna().any():raise ValueError('Unknown phase')
    if d.groupby('vehicle_group').phase.nunique().max()>1:raise ValueError('Vehicle crosses train/validation/test')
    if side=='bid' and 'auction_condition' in d:d['condition']=d.auction_condition
    if 'vehicle_group' not in d:
        if side!='ask':raise ValueError('Auction vehicle_group is required')
        # Craigslist preparation already deduplicates VINs and descriptions.
        d['vehicle_group']=d.row_id.astype(str)
    x,valid,dates=predictors(d,side)
    factor=np.ones(len(d))
    if not valid.all():raise ValueError('Previously eligible Lemon rows became invalid')
    d=d.loc[valid].copy();x=x.loc[valid].copy()
    y=d.price.to_numpy(float)/factor[valid.to_numpy()]
    if not np.isfinite(y).all() or (y<=0).any():raise ValueError('Invalid target prices')
    x['log_miles']=np.log1p(pd.to_numeric(d.odometer,errors='coerce'))
    cols=['age','log_miles','auction_grade']+CATEGORICAL
    for c in CATEGORICAL:
        x[c]=x[c].fillna('unknown').astype(str).str.strip().str.lower() if c in x else 'unknown'
    x=x[cols].reset_index(drop=True)
    d=d.reset_index(drop=True)
    if side!='ask':raise ValueError('Adapted BGE release is for asking descriptions')
    parts=[np.load(f'adapted-vectors/{i:02d}.npz') for i in range(25)]
    receipts=[json.loads(Path(f'adapted-vectors/{i:02d}.json').read_text()) for i in range(25)]
    if len({r['encoder_identity'] for r in receipts})!=1 or len({r['source_sha256'] for r in receipts})!=1:raise ValueError('Mixed embedding spaces')
    if receipts[0]['source_sha256']!='746bc9f8e02e1e708ea5e1d9479408effb131eb6b1d1ea7710380701e62dcf12':raise ValueError('Wrong source cohort')
    if receipts[0]['encoder_identity']!=json.loads(Path('adapted-bge/bundle.json').read_text())['identity']:raise ValueError('Wrong adapted encoder')
    for i,(part,receipt) in enumerate(zip(parts,receipts)):
        if receipt['shard']!=i or receipt['rows']!=len(part['ids']) or receipt['rows']!=len(range(i,receipt['source_rows'],25)):raise ValueError('Wrong shard partition')
    ids=np.concatenate([p['ids'].astype(str) for p in parts])
    if len(set(ids))!=len(ids) or len(ids)!=receipts[0]['source_rows']:raise ValueError('Incomplete adapted embeddings')
    values=np.concatenate([p['embeddings'] for p in parts])
    mini_values=np.concatenate([p['minilm'] for p in parts])
    for v in (values,mini_values):
        if v.shape!=(len(ids),384) or not np.isfinite(v).all():raise ValueError('Invalid description embeddings')
        if not np.allclose(np.linalg.norm(v,axis=1),1,atol=1e-3):raise ValueError('Unnormalized embeddings')
    lookup=pd.Index(ids).get_indexer(d.row_id.astype(str))
    if (lookup<0).any():raise ValueError('Missing description row')
    extra=values[lookup];embeddings=mini_values[lookup]
    provenance={'revision':'installed ONNX MiniLM; GUI-identical cleanup; source cohort encoded with fixed checkpoint 0'}
    texts=d.text.fillna('').astype(str).tolist() if side=='ask' else None
    phase=d.phase.map({'train':'calibration','validation':'validation','test':'test'})
    tr=np.flatnonzero(phase=='calibration')
    if len(tr)<100 or d.vehicle_group.iloc[tr].nunique()<5:raise ValueError('Insufficient distinct calibration vehicles')
    preds=np.full(len(d),np.nan);folds=np.full(len(d),-1)
    loss='survival:aft' if side=='ask' else 'reg:quantileerror'
    params=dict(objective=loss,tree_method='hist',max_depth=5,eta=.08,reg_lambda=10.,
                subsample=.85,colsample_bytree=.85,max_cat_to_onehot=4,min_child_weight=20,
                seed=42,nthread=2)
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
        return xgb.DMatrix(structured,feature_names=names,feature_types=types,enable_categorical=True,nthread=2)
    def fit(idx,features):
        frame,text=features
        levels={c:sorted(frame[c].astype(str).unique().tolist()) for c in CATEGORICAL}
        vectorizer=None
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
            from pricing.minilm_signal import two_price_crossfit,mean_price
            oof,heads,individual=two_price_crossfit(embeddings[fit_idx],y[fit_idx],d.vehicle_group.iloc[fit_idx].to_numpy())
            train['predicted_text_price']=oof
            held['predicted_text_price']=mean_price(heads,embeddings[pred_idx])
            if prefix=='calibration-refit':
                (folder/'minilm-heads.json').write_text(json.dumps({'heads':heads,'aggregation':'arithmetic mean of two dollar predictions','training_feature':'two repetitions of group-held-out two-fold prediction, averaged per row','outer_validation':'price heads refit within each outer XGBoost training fold','seeds':[42,1729]},indent=2))
                pd.DataFrame({'row_id':d.row_id.iloc[fit_idx].to_numpy(),'oof_price_1':individual[:,0],'oof_price_2':individual[:,1],'predicted_text_price':oof}).to_csv(folder/'text-price-oof.csv.gz',index=False)
        if extra is not None:
            from umap import UMAP
            dimensions=int(__import__('os').environ['UMAP_DIM'])
            reducer=UMAP(n_components=dimensions,n_neighbors=int(__import__('os').environ['UMAP_NEIGHBORS']),min_dist=float(__import__('os').environ['UMAP_MIN_DIST']),
                         metric='cosine',n_epochs=500,random_state=42,
                         transform_seed=42,n_jobs=1,low_memory=True)
            print(f'{prefix}: fitting training-only cosine UMAP',flush=True)
            a=reducer.fit_transform(extra[fit_idx]);b=reducer.transform(extra[pred_idx])
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('Non-finite UMAP features')
            for j in range(dimensions):
                train[f'bge_umap_{j}']=a[:,j];held[f'bge_umap_{j}']=b[:,j]
            if prefix=='calibration-refit':joblib.dump(reducer,folder/(prefix+'-bge-umap.joblib'))
        return (train,None),(held,None)
    fold_audit=[]
    for fold,(a,b) in enumerate(fivefold(d.vehicle_group.iloc[tr].to_numpy())):
        print(f'{side} group OOF {fold+1}/5',flush=True)
        train_x,held_x=augmented(tr[a],tr[b],f'fold-{fold+1}')
        assert not set(d.vehicle_group.iloc[tr[a]]) & set(d.vehicle_group.iloc[tr[b]])
        fold_audit.append({'fold':fold,'training_rows':len(a),'heldout_rows':len(b),'training_fraction':len(a)/len(tr)})
        model=fit(tr[a],train_x);preds[tr[b]]=predict(model,held_x);folds[tr[b]]=fold
        model[0].save_model(str(folder/f'fold-{fold+1}.ubj'))
    print(f'{side} calibration-only refit for untouched validation/test',flush=True)
    held=np.flatnonzero(phase!='calibration')
    train_x,held_x=augmented(tr,held,'calibration-refit')
    final=fit(tr,train_x);preds[held]=predict(final,held_x)
    final[0].save_model(str(folder/'calibration-refit.ubj'))
    if side=='ask':
        (folder/'features.json').write_text(json.dumps({'kind':'minilm-price','dimensions':1,'embedding_dimensions_internal':384,'frozen':True,'max_tokens':128,'encoder_source':'sentence-transformers/all-MiniLM-L6-v2','encoder_revision':provenance['revision']},indent=2))
    spec={'kind':'minilm-price-adapted-bge-umap','dimensions':1+int(__import__('os').environ['UMAP_DIM']),'umap_dimensions':int(__import__('os').environ['UMAP_DIM']),'n_neighbors':int(__import__('os').environ['UMAP_NEIGHBORS']),'min_dist':float(__import__('os').environ['UMAP_MIN_DIST']),'n_epochs':500,'metric':'cosine','encoder_bundle':json.loads(Path('adapted-bge/bundle.json').read_text())}
    (folder/'features.json').write_text(json.dumps(spec,indent=2))
    (folder/'category-schema.json').write_text(json.dumps(final[1]))
    import importlib.metadata as packages
    (folder/'runtime-versions.json').write_text(json.dumps({name:packages.version(name) for name in ('umap-learn','scikit-learn','joblib','numba')},indent=2))
    if not np.isfinite(preds).all() or (preds<=0).any():raise ValueError('Incomplete OOF predictions')
    residual=np.log(y/preds)
    report={'dataset':'retained Lemon Squeeze training; unchanged original validation/test','fold_audit':fold_audit,'market':side,'objective':loss,'folds':5,'iterations':iterations,'n':len(d),
            'residual_unit':'log(observed/reference prediction)', 'training':'calibration group OOF; calibration-only final refit',
            'description_features':'Cross-fitted MiniLM dollar-price feature plus adapted BGE training-fold-only cosine UMAP coordinates in XGBoost. No BGE coordinates enter the hierarchy.',
            'bge_reduction':{**spec,'n_components':spec['umap_dimensions'],'random_state':42,'transform_seed':42,'fit_scope':'outer training fold only; final fit on calibration only'},
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
    facts=d.drop(columns=['split','fold','observed_price','predicted_price','residual_log','residual_usd'],errors='ignore')
    merged=out.merge(facts,on=['row_id','vehicle_group'],validate='one_to_one')
    merged.to_csv(folder/'hierarchy-input.csv.gz',index=False)
    (folder/'cleaning.json').write_text(json.dumps({'name':'Lemon Squeeze retained cohort','fraction':.25,'validation_test_filtered':False,'selection_limit':'Retained rows inherited from frozen prior training-only trim. New OOF is conditional on that selection, not nested CV of cleaning.'}))
    (folder/'evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--market',required=True,choices=['ask','bid']);a=p.parse_args()
    side=a.market
    run_side(side,'evidence/'+side+'-minilm-cleaned.csv.gz',None,IndexPanel('models/structural-transfer/used-car-index.csv'),'training-cache/lemon-hierarchy',300,None,'training-cache/minilm-embeddings' if side=='ask' else None)
