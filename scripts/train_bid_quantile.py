"""Bid-only log-median CatBoost residual cleaning and independent final refit."""
import argparse,gzip,hashlib,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor,Pool
from pricing.model_suite import load_cohort,predictors,apply_category_pooling
from pricing.oof import cross_fit
from pricing.selection_imr import ProbitSelection
from scripts.train_trimmed_catboost_aft import metrics,pack
from marketbridge.structural import IndexPanel
from pricing.dataset_sources import BID

COMMON=dict(loss_function='Quantile:alpha=0.5',learning_rate=.05,l2_leaf_reg=10.,random_strength=1.,rsm=.8,bootstrap_type='Bayesian',bagging_temperature=1.,random_seed=42,thread_count=4,allow_writing_files=False)
WARNING='Requested top-25% absolute log-residual filtering also uses validation/test outcomes. Filtered metrics are optimistic selected-cohort results, not generalization estimates. Original unfiltered-test diagnostics use a previously examined benchmark; a fresh external test is needed. Assessment weights fit retained training only; deployment is a separate refit on all retained phases.'


def trim_quarter(ids,phases,residuals):
    ids=np.asarray(ids,str);phases=np.asarray(phases);r=np.asarray(residuals,float)
    if len(set(ids))!=len(ids) or r.shape!=ids.shape or not np.isfinite(r).all():raise ValueError('Unique IDs and finite residuals required')
    keep=np.ones(len(ids),bool);stats={}
    for phase in ['train','validation','test']:
        at=np.flatnonzero(phases==phase)
        if not len(at):raise ValueError('All original phases are required')
        rank=np.lexsort((ids[at],-np.abs(r[at])));count=math.ceil(.25*len(at));keep[at[rank[:count]]]=False
        stats[phase]={'input_rows':len(at),'removed_rows':count,'retained_rows':len(at)-count,'removed_fraction':count/len(at),'smallest_removed_abs_log_residual':float(np.abs(r[at[rank[count-1]]]))}
    return keep,stats


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise ValueError('Choose a fresh output; do not overwrite a deployed model')
    source_sha=hashlib.file_digest(open(a.bid,'rb'),'sha256').hexdigest()
    old=json.loads((Path(a.prior)/'evaluation.json').read_text())
    if source_sha!=old['source_sha256']:raise ValueError('Auction cohort SHA mismatch')
    eligible=pd.read_csv(Path(a.prior)/'filter-rows.csv.gz',dtype={'row_id':str})[['row_id','phase']]
    d=load_cohort(a.bid);d=d.loc[d.row_id.isin(set(eligible.row_id))].reset_index(drop=True)
    if d.row_id.astype(str).tolist()!=eligible.row_id.tolist() or d.phase.tolist()!=eligible.phase.tolist():raise ValueError('Prior eligibility row identity/order mismatch')
    x,valid,dates=predictors(d,'bid');ratios,_=IndexPanel(a.index).ratios(dates)
    if not (valid&ratios.notna()).all():raise ValueError('Previously eligible cohort changed')
    factor=ratios.to_numpy(float);y=d.price.to_numpy(float)/factor;logy=np.log(y);ids=d.row_id.to_numpy(str);phase=d.phase.to_numpy();tr=np.flatnonzero(phase=='train')
    # Pool rare trims using ORIGINAL TRAINING ONLY, before any residual filtering.
    counts=x.iloc[tr].model.value_counts();levels=[v for v in sorted(counts.index,key=lambda v:(-counts[v],str(v))) if counts[v]>=20][:1000]
    pooling={'model':{'retained_levels':levels,'replacement':'__quantile_rare_model__','min_count':20,'max_levels':1000,'learned_on':'original training only'}}
    x=apply_category_pooling(x,pooling);cats=list(x.select_dtypes('object'))
    for c in cats:x[c]=x[c].fillna('unknown').astype(str)
    cleaner_params={**COMMON,'iterations':a.cleaner_iterations,'depth':7};final_params={**COMMON,'iterations':a.iterations,'depth':8}
    selection={'status':'disabled','reason':'The supplied wholesale dataset contains only reported sold vehicles. No same-market sold/unsold total listing population is available. Craigslist asking listings are not fabricated auction non-sales.','scope':'cleaning model only; never a feature of the final quantile model'}
    pop=px=None
    if a.selection_population:
        pop=load_cohort(a.selection_population)
        if not {'row_id','phase','sold'}.issubset(pop) or pop.row_id.duplicated().any():raise ValueError('Selection population needs unique row_id, phase, sold and pre-sale vehicle/date fields')
        if not set(pop.sold.unique()).issubset({0,1,False,True}) or set(pop.sold.unique())!={0,1}:raise ValueError('Selection population must include observed sales and non-sales')
        sold_rows=pop.loc[pop.sold.eq(1)].set_index('row_id')
        if not set(ids).issubset(sold_rows.index) or not np.array_equal(sold_rows.loc[ids,'phase'].to_numpy(),phase):raise ValueError('Selection population must cover every eligible sold row with the same original phase')
        px,pvalid,_=predictors(pop,'bid')
        if not pvalid.all():raise ValueError('Selection population must have valid pre-sale fields and dates')
        selection={'status':'enabled','population_sha256':hashlib.file_digest(open(a.selection_population,'rb'),'sha256').hexdigest(),'scope':'cleaning model only; omitted from final price features','algorithm':'Probit','formula':'lambda=phi(z)/Phi(z)','guard':'Population validation/test phases excluded from selection fitting; OOF held sold rows excluded too. No sale-price variable enters Probit.'}
    def cleaner_fit(indices,prices):
        xx=x.iloc[indices].copy();selector=None
        if pop is not None:
            held_ids=set(ids[tr])-set(ids[indices]);allowed=pop.phase.eq('train')&~pop.row_id.isin(held_ids)
            # Exclude every relisting of an OOF-held VIN where group IDs exist.
            if 'vehicle_group' in pop and 'vehicle_group' in d:
                held_groups=set(d.loc[d.row_id.isin(held_ids),'vehicle_group']);allowed&=~pop.vehicle_group.isin(held_groups)
            selector=ProbitSelection().fit(px.loc[allowed],pop.sold.loc[allowed]);xx['selection_imr']=selector.predict(xx)
        model=CatBoostRegressor(**cleaner_params);model.fit(Pool(xx,label=np.log(prices),cat_features=cats),verbose=200)
        return model,selector
    def cleaner_predict(bundle,indices):
        model,selector=bundle;xx=x.iloc[indices].copy()
        if selector is not None:xx['selection_imr']=selector.predict(xx)
        return np.exp(np.clip(model.predict(xx,thread_count=4),-700,np.log(1e7)))
    fingerprint=json.dumps({'source_sha':source_sha,'params':cleaner_params,'selection':selection,'row_order_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest()},sort_keys=True)
    print('BID CLEANER: 3-fold training OOF, log Quantile alpha=.5',len(d),flush=True)
    oof,folds=cross_fit(ids[tr],y[tr],lambda idx,prices:cleaner_fit(tr[idx],prices),lambda bundle,idx:cleaner_predict(bundle,tr[idx]),cache=out/'cleaning-oof',fingerprint=fingerprint)
    clean_prediction=np.full(len(d),np.nan);clean_prediction[tr]=oof
    print('BID CLEANER: training-only fit for original validation/test',flush=True)
    bundle=cleaner_fit(tr,y[tr]);held=np.flatnonzero(phase!='train');clean_prediction[held]=cleaner_predict(bundle,held)
    pack(bundle[0],out/'cleaning-assessment.cbm.gz')
    if bundle[1] is not None:
        import joblib;joblib.dump(bundle[1],out/'cleaning-probit.joblib');selection['fit']=bundle[1].audit
    residual=logy-np.log(clean_prediction);keep,counts=trim_quarter(ids,phase,residual)
    audit=pd.DataFrame({'row_id':ids,'phase':phase,'residual_log':residual,'cleaner_predicted_reference_usd':clean_prediction,'retained':keep,'oof_fold':-1});audit.loc[tr,'oof_fold']=folds
    audit.to_csv(out/'filter-rows.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    report={'algorithm':'CatBoost log Quantile alpha=0.5 residual top25% per phase','partitions':counts,'input_rows':len(d),'removed_rows':int((~keep).sum()),'retained_rows':int(keep.sum()),'rule':'Exactly ceil(25% of each original phase), ranked by absolute log observed/predicted residual; deterministic row_id tie break. Original training uses 3-fold OOF; validation/test use a training-only cleaner. No OutlierTree/IsolationForest union.','selection_imr':selection,'warning':WARNING,'source_sha256':source_sha,'cleaner_parameters':cleaner_params}
    (out/'filter-report.json').write_text(json.dumps(report,indent=2));print('BID MASKS',json.dumps(counts),flush=True)
    def pool(mask):return Pool(x.loc[mask],label=logy[mask],cat_features=cats)
    model=CatBoostRegressor(**final_params);model.fit(pool(keep&(phase=='train')),eval_set=pool(keep&(phase=='validation')),early_stopping_rounds=125,verbose=200);rounds=model.tree_count_;pack(model,out/'assessment.cbm.gz')
    assessment={}
    for name,mask in [('filtered_validation',keep&(phase=='validation')),('filtered_test',keep&(phase=='test')),('original_unfiltered_test',phase=='test'),('excluded_test',~keep&(phase=='test'))]:
        price=np.exp(model.predict(x.loc[mask],thread_count=4))*factor[mask];observed=d.price.to_numpy(float)[mask];assessment[name]=metrics(observed,price)
        if name=='original_unfiltered_test':pd.DataFrame({'row_id':ids[mask],'observed':observed,'predicted':price,'retained':keep[mask]}).to_csv(out/'original-test-predictions.csv.gz',index=False)
    evaluation={'side':'bid','assessment':assessment,'rows':counts,'parameters':final_params,'selected_boosting_rounds':rounds,'category_pooling':pooling,'outlier_filter':report,'selection_imr':selection,'source_sha256':source_sha,'text_feature':False,'evaluation':WARNING}
    (out/'evaluation.json').write_text(json.dumps(evaluation,indent=2));print('BID ASSESSMENT COMPLETE',json.dumps(assessment),flush=True)
    final=CatBoostRegressor(**{**final_params,'iterations':rounds});final.fit(pool(keep),verbose=200);pack(final,out/'catboost_quantile.cbm.gz')
    manifest={'status':'ready','pipeline':'catboost-log-quantile','training_regime':'bid-quantile-residual-trimmed','side':'bid','model_file':'catboost_quantile.cbm.gz','feature_names':list(x),'categorical_columns':cats,'text_feature':False,'target_transform':'log','quantile_alpha':.5,'trained_rows':int(keep.sum()),'training_candidates':len(d),'selected_boosting_rounds':rounds,'category_pooling':pooling,'parameters':final_params,'outlier_filter':report,'selection_imr':selection,'deployment_scope':'all-retained-phases-refit-after-separate-assessment','warning':WARNING,'source_sha256':source_sha,'library_version':__import__('catboost').__version__,'dataset':BID}
    assert 'selection_imr' not in manifest['feature_names']
    (out/'schema.json').write_text(json.dumps({c:sorted(x.loc[keep,c].unique().tolist()) for c in cats}));(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('BID QUANTILE DEFAULT READY',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bid',required=True);p.add_argument('--prior',default='training-cache/prior-bid');p.add_argument('--output',default='models/bid-quantile-cleaned');p.add_argument('--index',default='models/structural-transfer/used-car-index.csv');p.add_argument('--selection-population');p.add_argument('--cleaner-iterations',type=int,default=800);p.add_argument('--iterations',type=int,default=1500);run(p.parse_args())
