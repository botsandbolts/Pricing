"""User-requested residual trimming, separately labelled filtered/original assessments."""
import argparse,gzip,hashlib,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor,Pool
from sklearn.model_selection import KFold
from pricing.model_suite import load_cohort,predictors,encode,CATS
from marketbridge.structural import IndexPanel,known_pricing_vins
from scripts.train_catboost_aft import metrics
from scripts.audit_outliertree import serial

PARAMS={'objective':'survival:aft','aft_loss_distribution':'normal','aft_loss_distribution_scale':1.,'eval_metric':'aft-nloglik','tree_method':'hist','learning_rate':.03,'max_depth':8,'min_child_weight':20.,'reg_lambda':20.,'reg_alpha':.1,'subsample':.8,'colsample_bytree':.8,'max_bin':256,'max_cat_to_onehot':4,'max_cat_threshold':32,'seed':42,'nthread':4}


def trim_mask(ids,phase,residual,ot_ids,fraction=.25):
    ids=np.asarray(ids).astype(str);phase=np.asarray(phase);r=np.asarray(residual,float)
    if len(set(ids))!=len(ids) or not np.isfinite(r).all():raise ValueError('Unique ids and finite residuals required')
    keep=np.ones(len(ids),bool);stats={}
    for name in ['train','validation','test']:
        at=np.flatnonzero(phase==name);order=sorted(at,key=lambda i:(-abs(r[i]),ids[i]));n=math.ceil(len(at)*fraction)
        ranked=np.asarray(order[:n],int);keep[ranked]=False
        ot=np.array([i for i in at if ids[i] in ot_ids],int);keep[ot]=False
        stats[name]={'original_rows':len(at),'top_residual_removed':n,'outliertree_flags':len(ot),'additional_outliertree_removed':int(sum(i not in set(ranked) for i in ot)),'removed_rows':int((~keep[at]).sum()),'retained_rows':int(keep[at].sum()),'retained_fraction':float(keep[at].mean()),'residual_abs_cutoff':float(abs(r[ranked[-1]]))}
    return keep,stats


def matrix(x,y):
    dm=xgb.DMatrix(x,enable_categorical=True,nthread=4);dm.set_float_info('label_lower_bound',y);dm.set_float_info('label_upper_bound',y);return dm


def save(model,path):
    with gzip.open(path,'wb',compresslevel=6) as f:f.write(model.save_raw(raw_format='ubj'))


def side_run(d,x,y,factor,side,out,a,source_sha):
    out.mkdir(parents=True,exist_ok=True);ids=d.row_id.astype(str).to_numpy();ph=d.phase.to_numpy();tr=np.flatnonzero(ph=='train')
    cats=[c for c in x if not pd.api.types.is_numeric_dtype(x[c])]
    for c in cats:x[c]=x[c].fillna('unknown').astype(str)
    residual=np.full(len(d),np.nan);fold_ids=np.full(len(d),-1)
    cache=out/'filter-predictions';cache.mkdir(exist_ok=True)
    fingerprint=hashlib.sha256((source_sha+side+str(a.oof_iterations)+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()).encode()).hexdigest()
    for fold,(fit,held) in enumerate(KFold(3,shuffle=True,random_state=42).split(tr)):
        fit=tr[fit];held=tr[held];file=cache/f'fold-{fold}.npz'
        if file.exists():
            saved=np.load(file);assert str(saved['fingerprint'])==fingerprint and np.array_equal(saved['ids'],ids[held]);raw=saved['raw']
            print(side,'OOF cache',fold,flush=True)
        else:
            print(side,'OOF TRAIN',fold,'fit',len(fit),'held',len(held),flush=True)
            model=CatBoostRegressor(loss_function='SurvivalAft:dist=Normal;scale=1.0',iterations=a.oof_iterations,depth=6,learning_rate=.05,l2_leaf_reg=3.,random_seed=42+fold,thread_count=4,allow_writing_files=False)
            model.fit(Pool(x.iloc[fit],label=np.column_stack([y[fit],y[fit]]),cat_features=cats),verbose=200)
            raw=np.asarray(model.predict(x.iloc[held],prediction_type='RawFormulaVal')).reshape(-1)
            np.savez_compressed(file,ids=ids[held],raw=raw,fingerprint=fingerprint)
            del model
        residual[held]=np.log(y[held])-raw;fold_ids[held]=fold
    # Existing CatBoost assessment predictions for original validation/test; never in-sample.
    for phase in ['validation','test']:
        at=np.flatnonzero(ph==phase);pred=pd.read_csv(Path(a.catboost)/side/f'{phase}-residuals.csv.gz',dtype={'row_id':str}).set_index('row_id')
        residual[at]=pred.loc[ids[at],'residual_log'].to_numpy(float)
    ot=pd.read_csv(Path(a.outliertree)/side/'full_cohort/flagged-cars.csv.gz',dtype={'row_id':str})
    keep,filter_stats=trim_mask(ids,ph,residual,set(ot.row_id))
    audit=pd.DataFrame({'row_id':ids,'phase':ph,'residual_log':residual,'oof_fold':fold_ids,'outliertree_flag':np.isin(ids,ot.row_id),'retained':keep})
    audit.to_csv(out/'filter-rows.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    (out/'filter-report.json').write_text(json.dumps(serial({'partitions':filter_stats,'rule':'Remove ceil(25% of each original phase) ranked by abs(log observed/predicted), tie break row_id, union all OutlierTree 0.099 full-cohort flags. Training residuals from 3-fold OOF CatBoost. Validation/test residuals from saved train-fit CatBoost assessment. No original row is overwritten.','warning':'Validation/test labels select easy cases; filtered metrics are selection-biased and are not generalization estimates. Unfiltered original test also reported, but filtering-rule development has already used its labels. A genuinely new external test is required.','source_sha256':source_sha}),indent=2))
    tr_keep=(ph=='train')&keep;va_keep=(ph=='validation')&keep;te_keep=(ph=='test')&keep
    schema={c:sorted(x.loc[tr_keep,c].unique().tolist()) for c in CATS};xf=encode(x,schema)
    print(side,'XGBOOST ASSESSMENT',filter_stats,flush=True)
    model=xgb.train(PARAMS,matrix(xf.loc[tr_keep],y[tr_keep]),num_boost_round=a.rounds,evals=[(matrix(xf.loc[va_keep],y[va_keep]),'filtered_validation')],early_stopping_rounds=150,verbose_eval=200)
    rounds=model.best_iteration+1;assessment=model[:rounds];save(assessment,out/'assessment.ubj.gz')
    records={}
    for name,mask in [('filtered_validation',va_keep),('filtered_test',te_keep),('original_unfiltered_test',ph=='test'),('excluded_test',(ph=='test')&~keep)]:
        p=assessment.predict(xgb.DMatrix(xf.loc[mask],enable_categorical=True,nthread=4))*factor[mask]
        observed=d.price.to_numpy(float)[mask];records[name]={'rows':int(mask.sum()),**metrics(observed,p)}
        if name=='original_unfiltered_test':pd.DataFrame({'row_id':ids[mask],'observed':observed,'predicted':p,'retained':keep[mask]}).to_csv(out/'original-test-predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    evaluation={'side':side,'models':{'xgb_aft_normal':{'test':records['filtered_test'],'validation':records['filtered_validation']}},'rows':filter_stats,'assessment':records,'selected_boosting_rounds':rounds,'parameters':PARAMS,'evaluation':'Filtered test/validation exclude highest-residual 25% plus OutlierTree flags using their labels. These are optimistic selected-cohort results. Original unfiltered-test errors also reported; a new external test is needed for clean model selection.','text_feature':False,'outlier_filter':{'algorithm':'OOF residual top25% + OutlierTree','removed_rows':int((~keep).sum()),'retained_rows':int(keep.sum()),'input_rows':len(d)},'source_sha256':source_sha}
    (out/'evaluation.json').write_text(json.dumps(serial(evaluation),indent=2,allow_nan=False))
    print(side,'ASSESSMENT COMPLETE',json.dumps(records),flush=True)
    # Deliberately separate deployment refit from untouched assessment artifacts.
    final_schema={c:sorted(x.loc[keep,c].unique().tolist()) for c in CATS};finalx=encode(x,final_schema)
    print(side,'FULL RETAINED REFIT',int(keep.sum()),'rounds',rounds,flush=True)
    final=xgb.train(PARAMS,matrix(finalx.loc[keep],y[keep]),num_boost_round=rounds,verbose_eval=False);save(final,out/'xgb_aft_normal.ubj.gz')
    (out/'schema.json').write_text(json.dumps(final_schema))
    manifest={'status':'ready','pipeline':'modest-model-suite','training_regime':'residual-trimmed-robust-xgb','side':side,'feature_names':list(x),'text_feature':False,'model_names':['xgb_aft_normal'],'trained_rows':int(keep.sum()),'training_candidates':len(d),'selected_boosting_rounds':rounds,'parameters':PARAMS,'outlier_filter':evaluation['outlier_filter'],'deployment_scope':'all-retained-phases-refit-after-separate-assessment','warning':evaluation['evaluation'],'source_sha256':source_sha}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(side,'READY',flush=True)
    return evaluation


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);known=known_pricing_vins(a.ask);panel=IndexPanel(a.index);prior=json.loads(Path(a.prior).read_text());results={}
    for side,path in [('ask',a.ask),('bid',a.bid)]:
        if (out/side/'manifest.json').exists():results[side]=json.loads((out/side/'evaluation.json').read_text());continue
        sha=hashlib.file_digest(open(path,'rb'),'sha256').hexdigest();assert sha==prior['markets'][side]['source_sha256']
        d=load_cohort(path)
        if side=='bid':d=d.loc[~d.vehicle_group.astype(str).str.removeprefix('vin:').isin(known)].reset_index(drop=True)
        x,valid,dates=predictors(d,side);ratios,_=panel.ratios(dates)
        if side=='bid':valid&=ratios.notna()
        d=d.loc[valid].copy();x=x.loc[valid].copy();assert len(d)==prior['markets'][side]['eligible_rows']
        factor=ratios.loc[d.index].to_numpy(float) if side=='bid' else np.ones(len(d));y=d.price.to_numpy(float)/factor
        results[side]=side_run(d,x,y,factor,side,out/side,a,sha);del d,x
    (out/'summary.json').write_text(json.dumps(serial({'status':'complete','markets':results}),indent=2,allow_nan=False));print('BOTH ROBUST XGBOOST MODELS READY',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ask',required=True);p.add_argument('--bid',required=True);p.add_argument('--output',default='models/residual-trimmed-xgb');p.add_argument('--catboost',default='reports/catboost-aft-residuals');p.add_argument('--outliertree',default='reports/outliertree-pct0099');p.add_argument('--index',default='models/structural-transfer/used-car-index.csv');p.add_argument('--prior',default='reports/cleanlab-rerun-20261027/summary.json');p.add_argument('--oof-iterations',type=int,default=1000);p.add_argument('--rounds',type=int,default=2500);run(p.parse_args())
