"""Configurable predeclared LightGBM experiments using cross-fitted fine-tuned E5 text prices.

The Isolation Forest filters only the LightGBM training rows. The shared E5 stage
is fit once on all training rows, with held-out fold predictions for stacking.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
from pricing.tabular import fit_schema,features,NUMERIC,CATEGORICAL
from pricing.train import validate_cohort
from pricing.text import MODEL_ID
from scripts.common import metrics

RATES=[0.,.05,.10,.25,.50]
LOSSES=[('rmse','regression',None),('mae','regression_l1',None),('huber','huber',.9),
        ('q10','quantile',.1),('q50','quantile',.5),('q90','quantile',.9)]
TARGETS=['dollars','log1p']


def inverse(z,mean,std,target):
    raw=np.asarray(z)*std+mean
    return np.expm1(np.clip(raw,0,np.log1p(1e7))) if target=='log1p' else np.clip(raw,0,1e7)


def pinball(y,p,q):
    r=np.asarray(y)-np.asarray(p)
    return float(np.mean(np.maximum(q*r,(q-1)*r)))


def score(y,p,q=None):
    result=metrics(y,p)
    result['mae_log']=float(np.mean(abs(np.log1p(y)-np.log1p(p))))
    if q is not None:result['pinball_usd']=pinball(y,p,q)
    return result


def retained_indices(scores,rate):
    if rate<0 or rate>=1:raise ValueError('Removal rate must be in [0,1)')
    n=len(scores);remove=int(np.floor(n*rate))
    # Stable tie handling; all requested subsets are nested.
    return np.argsort(-np.asarray(scores),kind='stable')[:n-remove]


def isolation_scores(x,seed,threads=4):
    # No ground-truth price, validation or test information enters this detector.
    numeric=NUMERIC+(['predicted_text_price'] if 'predicted_text_price' in x else [])
    categorical=x[CATEGORICAL].astype('object').where(x[CATEGORICAL].notna(),'__missing__')
    temp=x.copy()
    temp[CATEGORICAL]=categorical
    transform=ColumnTransformer([
        ('numeric',SimpleImputer(strategy='median',keep_empty_features=True),numeric),
        ('categorical',OneHotEncoder(handle_unknown='ignore',dtype=np.float32),CATEGORICAL)],
        sparse_threshold=1.)
    z=transform.fit_transform(temp)
    detector=IsolationForest(n_estimators=300,max_samples=min(256,len(x)),
                            random_state=seed,n_jobs=threads,contamination='auto').fit(z)
    return detector.score_samples(z)


def experiment_grid(losses=None,huber_deltas=None):
    if losses is None:return list(itertools.product(RATES,TARGETS,LOSSES))
    names=set(losses)
    if not names or not names.issubset({x[0] for x in LOSSES}):raise ValueError('Unknown or empty loss list')
    selected=[x for x in LOSSES if x[0] in names and x[0]!='huber']
    if 'huber' in names:
        for delta in huber_deltas or [.9]:
            if delta<=0:raise ValueError('Huber deltas must be positive')
            selected.append((f'huber_d{delta:g}','huber',delta))
    return list(itertools.product(RATES,TARGETS,selected))


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cache=out/'text-features.npz'
    tabular_only=getattr(a,'tabular_only',False)
    if not tabular_only and not cache.exists():
        raise RuntimeError('Fine-tuned OOF E5 feature cache missing. First run pricing.train --text-only with the same --data and --output. Frozen embeddings or in-sample predictions are not accepted.')
    d=pd.read_json(a.data,lines=True);validate_cohort(d)
    phases={p:d[d.phase==p].reset_index(drop=True) for p in ['train','validation','test']}
    train,val,test=[phases[p] for p in ['train','validation','test']]
    f=None
    if not tabular_only:
        f=np.load(cache)
        config=json.loads((out/'training-config.json').read_text())
        original={k:v for k,v in config.items() if k!='fingerprint'}
        fingerprint=hashlib.sha256(Path(a.data).read_bytes()+json.dumps(original,sort_keys=True).encode()).hexdigest()
        if fingerprint!=str(f['fingerprint']) or fingerprint!=config['fingerprint']:
            raise ValueError('The text feature cache belongs to different data/configuration')
        for p,key in [('train','train_ids'),('validation','validation_ids'),('test','test_ids')]:
            if not np.array_equal(f[key],phases[p].row_id.astype(str).to_numpy()):
                raise ValueError('Text feature row order mismatch')
    else:
        fingerprint=hashlib.sha256(Path(a.data).read_bytes()).hexdigest()
    schema=fit_schema(train)
    x=features(train,schema,None if tabular_only else f['oof'])
    xv=features(val,schema,None if tabular_only else f['validation'])
    xt=features(test,schema,None if tabular_only else f['test'])
    grid=experiment_grid(getattr(a,'losses',None),getattr(a,'huber_deltas',None))
    threads=getattr(a,'threads',4)
    anomaly=isolation_scores(x,a.seed,threads)
    audit=pd.DataFrame({'row_id':train.row_id,'score':anomaly})
    for rate in RATES:
        keep=retained_indices(anomaly,rate)
        audit[f'kept_{int(rate*100)}pct_removed']=np.isin(np.arange(len(train)),keep)
    audit.to_csv(out/'isolation-audit.csv',index=False)
    # Predeclare the sweep and selection rules before running any experiment.
    plan={'stage':'tabular-only baseline' if tabular_only else 'fine-tuned E5 + tabular stack','training_candidates':len(train),'validation':len(val),'test':len(test),
          'rates':RATES,'targets':TARGETS,'losses':list(dict.fromkeys(x[2] for x in grid)),'configurations':len(grid),
          'filter_scope':'LightGBM training rows only; shared E5 stage unchanged across sweep if enabled',
          'filter_inputs':list(x),'filter_true_prices':False,
          'selection':'Validation USD MAE and validation USD RMSE; q10/q90 excluded from point-model selection',
          'target_standardization':'Retained training rows only; each Huber alpha is in standardized target units',
          'fingerprint':fingerprint,'lightgbm_estimators':a.trees,'seed':a.seed}
    plan_path=out/'experiment-plan.json'
    previous={}
    if plan_path.exists() and (out/'validation-results.json').exists():
        old_plan=json.loads(plan_path.read_text())
        if json.dumps(old_plan,sort_keys=True)!=json.dumps(plan,sort_keys=True):
            raise ValueError('Experiment settings changed; choose a new output directory')
        previous={r['id']:r for r in json.loads((out/'validation-results.json').read_text())}
    plan_path.write_text(json.dumps(plan,indent=2))
    records=[];pending=[];models=out/'grid-models';models.mkdir(exist_ok=True)
    # Freeze every fitted configuration and validation score before accessing test targets for scoring.
    for rate,target,(loss,objective,alpha) in grid:
        keep=retained_indices(anomaly,rate)
        raw=np.log1p(train.price.to_numpy()) if target=='log1p' else train.price.to_numpy(float)
        vr=np.log1p(val.price.to_numpy()) if target=='log1p' else val.price.to_numpy(float)
        mean=float(raw[keep].mean());std=float(max(raw[keep].std(),1e-6))
        key=f'remove{int(rate*100):02d}_{target}_{loss}'
        if key in previous and (models/f'{key}.txt').exists():
            record=previous[key]
            restored=lgb.Booster(model_file=str(models/f'{key}.txt'))
            records.append(record)
            pending.append(inverse(restored.predict(xt,num_threads=threads),record['target_mean'],record['target_std'],target))
            print(f'{key}: resumed completed fit',flush=True)
            continue
        params=dict(objective=objective,n_estimators=a.trees,learning_rate=.03,num_leaves=31,
                    min_child_samples=30,reg_lambda=5.,colsample_bytree=.9,
                    random_state=a.seed,n_jobs=threads,verbosity=-1,metric='None')
        if alpha is not None:params['alpha']=alpha
        fit=lgb.LGBMRegressor(**params)
        metric='l1' if objective=='huber' else {'rmse':'rmse','mae':'l1','huber':'l1','q10':'quantile','q50':'quantile','q90':'quantile'}[loss]
        fit.fit(x.iloc[keep],(raw[keep]-mean)/std,
                eval_set=[(xv,(vr-mean)/std)],eval_metric=metric,
                categorical_feature=CATEGORICAL,
                callbacks=[lgb.early_stopping(min(75,max(5,a.trees//5)),first_metric_only=True,verbose=False)])
        vp=inverse(fit.predict(xv),mean,std,target)
        q=alpha if objective=='quantile' else None
        record={'id':key,'removed_fraction':rate,'retained_rows':len(keep),
                'removed_rows':len(train)-len(keep),'target':target,'loss':loss,'alpha':alpha,
                'best_iteration':fit.best_iteration_,'validation':score(val.price,vp,q),
                'target_mean':mean,'target_std':std,
                'band80_log_radius':float(np.quantile(abs(np.log1p(val.price)-np.log1p(vp)),.8))}
        fit.booster_.save_model(str(models/f'{key}.txt'))
        records.append(record)
        # Calculate test predictions without reading test targets; evaluate after all selection is fixed.
        pending.append(inverse(fit.predict(xt),mean,std,target))
        (out/'validation-results.json').write_text(json.dumps(records,indent=2))
        print(f'{key}: validation MAE=${record["validation"]["mae_usd"]:,.0f}, RMSE=${record["validation"]["rmse_usd"]:,.0f}',flush=True)
    points=[r for r in records if r['loss'] not in ['q10','q90']]
    selected={metric:min(points,key=lambda r:r['validation'][metric])['id'] for metric in ['mae_usd','rmse_usd']}
    (out/'selection.json').write_text(json.dumps({'selected_on':'validation only',**selected},indent=2))
    # Report the entire predeclared test grid as exploratory diagnostics; do not reselect on it.
    predictions={}
    for r,p in zip(records,pending):
        q=r['alpha'] if r['loss'].startswith('q') else None
        r['test']=score(test.price,p,q)
        r['band80_test_coverage']=float(np.mean(abs(np.log1p(test.price)-np.log1p(p))<=r['band80_log_radius']))
        predictions[r['id']]=p
    intervals=[]
    for rate,target in itertools.product(RATES,TARGETS):
        prefix=f'remove{int(rate*100):02d}_{target}_'
        if prefix+'q10' not in predictions or prefix+'q90' not in predictions:continue
        lo,hi=predictions[prefix+'q10'],predictions[prefix+'q90']
        intervals.append({'removed_fraction':rate,'target':target,
            'test_coverage_raw_q10_q90':float(np.mean((test.price>=lo)&(test.price<=hi))),
            'test_crossing_fraction':float(np.mean(lo>hi)),
            'mean_signed_width_usd':float(np.mean(hi-lo))})
    report={'status':'evaluated','plan':plan,'selected_on_validation':selected,'experiments':records,
            'quantile_intervals':intervals,'text_only_test':None if tabular_only else score(test.price,f['test']),
            'test_use':'One predeclared grid; test diagnostics must not drive model selection.'}
    (out/'experiments.json').write_text(json.dumps(report,indent=2))
    flat=[]
    for r in records:
        row={k:v for k,v in r.items() if k not in ['validation','test']}
        for phase in ['validation','test']:
            row.update({phase+'_'+k:v for k,v in r[phase].items()})
        flat.append(row)
    pd.DataFrame(flat).to_csv(out/'experiment-results.csv',index=False)
    if tabular_only:
        print(json.dumps({'status':'tabular baseline only, not the E5 stack','experiments':len(records),'selected_on_validation':selected},indent=2))
        return
    # Publish the validation-MAE champion to the live Python GUI bundle.
    champion=next(r for r in records if r['id']==selected['mae_usd'])
    (out/'lightgbm.txt').write_bytes((models/(champion['id']+'.txt')).read_bytes())
    booster=lgb.Booster(model_file=str(out/'lightgbm.txt'))
    (out/'schema.json').write_text(json.dumps(schema,indent=2))
    evaluation={'status':'evaluated','scope':'Fixed sampled database cohort, historical 2021 asking prices',
        'rows':{'train':champion['retained_rows'],'train_candidates':len(train),'removed':champion['removed_rows'],
                'validation':len(val),'test':len(test)},'stacked':champion['test'],
        'text_only':report['text_only_test'],'selected_experiment':champion['id'],
        'selected_on_validation':selected,'validation_stacked':champion['validation'],
        'band80_test_coverage':champion['band80_test_coverage'],'experiments':len(records)}
    (out/'evaluation.json').write_text(json.dumps(evaluation,indent=2))
    manifest={'status':'ready','pipeline':'fine-tuned-e5-base+lightgbm','model_id':MODEL_ID,
        'fingerprint':fingerprint,'feature_names':booster.feature_name(),
        'band80_log_radius':champion['band80_log_radius'],
        'target_transform':champion['target'],'target_mean':champion['target_mean'],
        'target_std':champion['target_std'],'selected_experiment':champion['id']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'experiments':len(records),'selected_on_validation':selected},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='training-cache/twenty-five-percent.jsonl')
    p.add_argument('--output',default='artifacts/e5-base-lightgbm-25')
    p.add_argument('--trees',type=int,default=1500)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--losses',nargs='+',choices=[x[0] for x in LOSSES])
    p.add_argument('--huber-deltas',nargs='+',type=float,default=[.5,.9,1.5])
    p.add_argument('--tabular-only',action='store_true',help='Run a clearly labeled CPU baseline without E5; never export an inference bundle')
    run(p.parse_args())
