"""Five 5%-of-survivors residual rounds, then independent CatBoost assessments/refits.

The cleaner is tabular log-median CatBoost. Training residuals are cross-fitted;
validation/test residuals use a training-only cleaner. Label-selected scores are
explicitly biased. The original test population is also scored without trimming.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from pricing.model_suite import load_cohort, predictors, apply_category_pooling
from marketbridge.structural import IndexPanel, known_pricing_vins
from scripts.train_trimmed_catboost_aft import metrics, pack
from pricing.dataset_sources import ASK, BID

WARNING = ('Filtered validation/test use their outcomes to select easier cars and are optimistic. '
           'Original unfiltered-test scores are diagnostic: this historical test has informed development. '
           'A new external cohort is required for an unbiased improvement claim. Sequential selection is adaptive; '
           'cleaner cross-fitting is not nested cross-validation of the entire selection pipeline. '
           'Large residuals are not verified data errors; rare genuine cars can be excluded.')
LOSSES = {'quantile':'Quantile:alpha=0.5', 'aft':'SurvivalAft:dist=Normal;scale=1.0'}


def trim_stage(active, phase, residual, row_ids, fraction=.05):
    """Exactly ceil(fraction * survivors) per phase, deterministic row-ID ties."""
    active=np.asarray(active,bool); phase=np.asarray(phase); residual=np.asarray(residual,float)
    removed=np.zeros(len(active),bool)
    for split in ['train','validation','test']:
        idx=np.flatnonzero(active & (phase==split))
        if len(idx)<2:raise ValueError('Too few survivors in '+split)
        if not np.isfinite(residual[idx]).all():raise ValueError('Missing cross-fitted residuals')
        number=math.ceil(len(idx)*fraction)
        if number>=len(idx):raise ValueError('Cleaning would empty '+split)
        order=np.lexsort((np.asarray(row_ids,str)[idx],-np.abs(residual[idx])))
        removed[idx[order[:number]]]=True
    return active & ~removed,removed


def parameters(a,kind,iterations,cleaner=False):
    gpu=a.device=='GPU' and kind!='aft'
    p=dict(loss_function=LOSSES[kind],iterations=iterations,depth=a.cleaner_depth if cleaner else a.depth,
           learning_rate=.05,l2_leaf_reg=10.,random_strength=1.,bootstrap_type='Bayesian',
           bagging_temperature=1.,random_seed=42,thread_count=a.threads,allow_writing_files=False,
           task_type='GPU' if gpu else 'CPU')
    if gpu:p['devices']='0'
    else:p['rsm']=.8
    return p


def iterative_clean(x,y,phase,ids,cats,out,a):
    out.mkdir(parents=True,exist_ok=True)
    active=np.ones(len(x),bool); stages=[]; removed_stage=np.zeros(len(x),int)
    recipe={'row_ids_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
            'price_sha256':hashlib.sha256(np.asarray(y,np.float64).tobytes()).hexdigest(),
            'parameters':parameters(a,'quantile',a.cleaner_iterations,True),'stages':5,'fraction':.05}
    recipe_file=out/'recipe.json'
    if recipe_file.exists() and json.loads(recipe_file.read_text())!=recipe:
        raise ValueError('Cleaner cache differs; use a fresh output directory')
    recipe_file.write_text(json.dumps(recipe,indent=2))
    logy=np.log(y)
    for stage in range(1,6):
        checkpoint=out/f'stage-{stage}.csv.gz'; stats=out/f'stage-{stage}.json'
        if checkpoint.exists() and stats.exists():
            audit=pd.read_csv(checkpoint,dtype={'row_id':str})
            if audit.row_id.tolist()!=list(ids):raise ValueError('Cleaner checkpoint identity mismatch')
            new_active=audit.retained.to_numpy(bool)
            if np.any(new_active & ~active):raise ValueError('Checkpoint resurrects removed rows')
            removed_stage[active & ~new_active]=stage;active=new_active;stages.append(json.loads(stats.read_text()));continue
        print('CLEANING',out.name,'STAGE',stage,'SURVIVORS',int(active.sum()),flush=True)
        residual=np.full(len(y),np.nan); train=np.flatnonzero(active & (phase=='train'))
        for fold,(fit,held) in enumerate(KFold(3,shuffle=True,random_state=42).split(train)):
            model=CatBoostRegressor(**recipe['parameters'])
            model.fit(Pool(x.iloc[train[fit]],logy[train[fit]],cat_features=cats),verbose=False)
            residual[train[held]]=logy[train[held]]-model.predict(x.iloc[train[held]])
            print('STAGE',stage,'OOF FOLD',fold+1,'DONE',flush=True)
        model=CatBoostRegressor(**recipe['parameters'])
        model.fit(Pool(x.iloc[train],logy[train],cat_features=cats),verbose=False)
        held=np.flatnonzero(active & (phase!='train'))
        residual[held]=logy[held]-model.predict(x.iloc[held])
        next_active,removed=trim_stage(active,phase,residual,ids)
        rows=[]
        for split in ['train','validation','test']:
            before=active & (phase==split);cut=removed & (phase==split)
            rows.append({'phase':split,'before':int(before.sum()),'removed':int(cut.sum()),
                         'retained':int((next_active & (phase==split)).sum()),
                         'minimum_removed_absolute_log_residual':float(np.abs(residual[cut]).min())})
        report={'stage':stage,'fraction_of_remaining':.05,'counts':rows}
        pd.DataFrame({'row_id':ids,'phase':phase,'absolute_log_residual':np.abs(residual),
                      'removed_this_stage':removed,'retained':next_active}).to_csv(checkpoint,index=False)
        stats.write_text(json.dumps(report,indent=2));stages.append(report)
        removed_stage[removed]=stage;active=next_active
        print('STAGE COMPLETE',json.dumps(report),flush=True)
    audit=pd.DataFrame({'row_id':ids,'phase':phase,'retained':active,'removed_stage':removed_stage})
    audit.to_csv(out/'filter-rows.csv.gz',index=False)
    report={'algorithm':'Iterative CatBoost residual cleaning','stages':stages,'stage_count':5,
            'fraction_per_stage':.05,'remaining_rule':'ceil(5% of current survivors) independently per original phase',
            'original_rows':len(y),'retained_rows':int(active.sum()),'removed_rows':int((~active).sum()),
            'removed_fraction':float((~active).mean()),'cleaner_parameters':recipe['parameters'],
            'cleaner_features':list(x),'text_feature_in_cleaner':False,'warning':WARNING}
    (out/'filter-report.json').write_text(json.dumps(report,indent=2))
    return active,report


def run(a):
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    text=None;text_meta=None
    if a.side in [None,'ask']:
        if not a.text_features:raise ValueError('Asking refits require verified MiniLM OOF features')
        from pricing.suite_text import validated_features
        text=validated_features(a.ask,a.text_features)
        text_meta=json.loads((Path(a.text_model)/'provenance.json').read_text())
        if text_meta!=json.loads((Path(a.text_features)/'full-text-provenance.json').read_text()):
            raise ValueError('MiniLM deployment model does not match full OOF provenance')
    known=known_pricing_vins(a.ask);panel=IndexPanel(a.index)
    for side,path in [('ask',a.ask),('bid',a.bid)]:
        if a.side and side!=a.side:continue
        d=load_cohort(path);sha=hashlib.file_digest(open(path,'rb'),'sha256').hexdigest()
        excluded=0
        if side=='bid':
            overlap=d.vehicle_group.astype(str).str.removeprefix('vin:').isin(known)
            excluded=int(overlap.sum());d=d.loc[~overlap].reset_index(drop=True)
        x,valid,dates=predictors(d,side);ratios,_=panel.ratios(dates)
        if side=='bid':valid &= ratios.notna()
        factor=ratios.loc[valid].to_numpy(float) if side=='bid' else np.ones(int(valid.sum()))
        d=d.loc[valid].reset_index(drop=True);x=x.loc[valid].reset_index(drop=True)
        if d.row_id.duplicated().any():raise ValueError('Duplicate cohort identities')
        phase=d.phase.to_numpy();y=d.price.to_numpy(float)/factor
        if not np.isfinite(y).all() or (y<=0).any():raise ValueError('Invalid price targets')
        counts=x.loc[phase=='train','model'].value_counts()
        pool={'model':{'levels':sorted([str(k) for k,v in counts.items() if v>=20],key=lambda k:(-counts[k],k))[:1000],
                       'replacement':'__rare_model__'}}
        # Match apply_category_pooling's serialized contract.
        pool={'model':{'retained_levels':pool['model']['levels'],'replacement':'__rare_model__','min_count':20}}
        x=apply_category_pooling(x,pool)
        cats=[c for c in x if not pd.api.types.is_numeric_dtype(x[c])]
        for c in cats:x[c]=x[c].fillna('unknown').astype(str)
        keep,filter_report=iterative_clean(x,y,phase,d.row_id.astype(str).tolist(),cats,root/(side+'-cleaning'),a)
        if side=='ask':x['predicted_text_price']=text.loc[d.row_id].to_numpy(float)
        for kind in ['aft','quantile']:
            folder=side if (side,kind) in [('ask','aft'),('bid','quantile')] else side+'-'+kind
            out=root/folder;out.mkdir(exist_ok=True)
            if (out/'manifest.json').exists():
                manifest=json.loads((out/'manifest.json').read_text())
                if manifest.get('source_sha256')!=sha:raise ValueError('Existing output has different source')
                continue
            p=parameters(a,kind,a.iterations)
            def make_pool(mask,features=x):
                label=np.column_stack([y[mask],y[mask]]) if kind=='aft' else np.log(y[mask])
                return Pool(features.loc[mask],label=label,cat_features=cats)
            def prediction(model,mask):
                margin=np.asarray(model.predict(x.loc[mask],prediction_type='RawFormulaVal'),float)
                if not np.isfinite(margin).all():raise ValueError('Nonfinite prediction')
                return np.exp(np.clip(margin,-700,np.log(1e7)))*factor[mask]
            test=phase=='test';original_predictions={};evaluations={}
            for regime in ['unfiltered_baseline','iterative_cleaned']:
                assessment_file=out/(regime+'-assessment.cbm.gz'); evaluation_file=out/(regime+'-evaluation.json')
                fitmask=(phase=='train') & (keep if regime=='iterative_cleaned' else True)
                valmask=(phase=='validation') & (keep if regime=='iterative_cleaned' else True)
                if assessment_file.exists() and evaluation_file.exists():
                    import gzip
                    model=CatBoostRegressor();model.load_model(blob=gzip.decompress(assessment_file.read_bytes()))
                    evaluation=json.loads(evaluation_file.read_text())
                else:
                    print('FIT',side,kind,regime,int(fitmask.sum()),flush=True)
                    model=CatBoostRegressor(**p);model.fit(make_pool(fitmask),eval_set=make_pool(valmask),early_stopping_rounds=100,verbose=200)
                    evaluation={'selected_boosting_rounds':model.tree_count_,'assessment':{}}
                    for name,mask in [('original_validation',phase=='validation'),('filtered_validation',(phase=='validation')&keep),
                                      ('original_unfiltered_test',test),('filtered_test',test&keep),('excluded_test',test&~keep)]:
                        evaluation['assessment'][name]=metrics(d.price.to_numpy(float)[mask],prediction(model,mask))
                    pack(model,assessment_file);evaluation_file.write_text(json.dumps(evaluation,indent=2))
                pred=prediction(model,test);original_predictions[regime]=pred;evaluations[regime]=evaluation
                pd.DataFrame({'row_id':d.row_id.loc[test],'observed':d.price.loc[test],'predicted':pred,'retained':keep[test]}).to_csv(out/(regime+'-original-test-predictions.csv.gz'),index=False)
                print('ASSESSMENT',side,kind,regime,json.dumps(evaluation),flush=True)
            rounds=evaluations['iterative_cleaned']['selected_boosting_rounds']
            final_x=x.copy()
            if side=='ask':final_x['predicted_text_price']=text.attrs['full_oof'].loc[d.row_id].to_numpy(float)
            final=CatBoostRegressor(**{**p,'iterations':rounds})
            print('DEPLOYMENT REFIT',side,kind,int(keep.sum()),flush=True)
            final.fit(make_pool(keep,final_x),verbose=200)
            model_file='catboost_aft.cbm.gz' if kind=='aft' else 'catboost_quantile.cbm.gz'
            pack(final,out/model_file)
            baseline=evaluations['unfiltered_baseline']['assessment']['original_unfiltered_test']['mae_usd']
            cleaned=evaluations['iterative_cleaned']['assessment']['original_unfiltered_test']['mae_usd']
            evaluation={**evaluations['iterative_cleaned'],'side':side,'parameters':p,'source_sha256':sha,
                        'category_pooling':pool,'outlier_filter':filter_report,'evaluation':WARNING,
                        'text_feature':side=='ask','unfiltered_baseline':evaluations['unfiltered_baseline'],
                        'original_test_mae_change_usd':cleaned-baseline,
                        'original_test_mae_improvement_pct':100*(baseline-cleaned)/baseline,
                        'baseline_cleaned_prediction_spearman':float(spearmanr(original_predictions['unfiltered_baseline'],original_predictions['iterative_cleaned']).statistic)}
            manifest={'status':'ready','pipeline':'catboost-aft' if kind=='aft' else 'catboost-log-quantile',
                      'training_regime':'five-stage-catboost-residual','side':side,'feature_names':list(x),
                      'categorical_columns':cats,'model_file':model_file,'text_feature':side=='ask',
                      'trained_rows':int(keep.sum()),'training_candidates':len(d),'selected_boosting_rounds':rounds,
                      'category_pooling':pool,'parameters':p,'outlier_filter':filter_report,'warning':WARNING,
                      'source_sha256':sha,'auction_vin_overlap_exclusions':excluded,
                      'deployment_scope':'all-retained-phases-refit-after-separate-assessment',
                      'library_version':__import__('catboost').__version__,'dataset':ASK if side=='ask' else BID}
            if side=='ask':manifest.update(text_model_path=str(Path(a.text_model).resolve().relative_to(root.resolve().parent)),
                text_model_fingerprint=text_meta['fingerprint'],text_provenance=text_meta,
                text_stacking='Assessment: training-only OOF plus train-only held-out head; deployment: full OOF and matching full-cohort head')
            (out/'evaluation.json').write_text(json.dumps(evaluation,indent=2));(out/'schema.json').write_text(json.dumps({c:sorted(x.loc[keep,c].unique().tolist()) for c in cats}))
            (out/'filter-report.json').write_text(json.dumps(filter_report,indent=2));(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
            print('READY',side,kind,flush=True)
    folders=['ask','ask-quantile','bid','bid-aft']
    if all((root/f/'manifest.json').exists() for f in folders):
        summary={'status':'complete','method':'five-stage-catboost-residual','default_models':{'ask':'aft','bid':'quantile'},
                 'models':{f:json.loads((root/f/'evaluation.json').read_text()) for f in folders},'warning':WARNING}
        (root/'summary.json').write_text(json.dumps(summary,indent=2));print('ALL FOUR MODELS READY',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ask',required=True);p.add_argument('--bid',required=True)
    p.add_argument('--text-features');p.add_argument('--text-model',default='models/minilm-cleanlab/text-model')
    p.add_argument('--output',default='models/iterative-catboost');p.add_argument('--side',choices=['ask','bid'])
    p.add_argument('--index',default='models/structural-transfer/used-car-index.csv')
    p.add_argument('--iterations',type=int,default=1500);p.add_argument('--cleaner-iterations',type=int,default=400)
    p.add_argument('--depth',type=int,default=8);p.add_argument('--cleaner-depth',type=int,default=6)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--device',choices=['CPU','GPU'],default='CPU')
    run(p.parse_args())
