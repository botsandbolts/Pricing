"""Full-data tabular auction-sale baseline, honest holdout assessment then refit."""
import argparse,json,gzip,hashlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
from carbids.data import load_cohort,TARGET
from carbids.tabular import features,fit_schema,CATEGORICAL
from carbids.common import metrics

def fit(x,y,target,alpha,rounds,threads,xv=None,yv=None):
    z=np.log1p(y) if target=='log1p' else np.asarray(y,float)
    mean=float(np.mean(z));std=float(max(np.std(z),1e-6))
    model=lgb.LGBMRegressor(objective='huber',alpha=alpha,n_estimators=rounds,learning_rate=.05,
        num_leaves=31,min_child_samples=40,reg_lambda=5,colsample_bytree=.9,n_jobs=threads,random_state=42,verbosity=-1)
    kwargs={'categorical_feature':CATEGORICAL}
    if xv is not None:
        v=np.log1p(yv) if target=='log1p' else np.asarray(yv,float)
        kwargs.update(eval_set=[(xv,(v-mean)/std)],eval_metric='l1',callbacks=[lgb.early_stopping(50,verbose=False)])
    model.fit(x,(z-mean)/std,**kwargs)
    return model,mean,std

def invert(raw,mean,std,target):
    z=np.asarray(raw)*std+mean
    return np.expm1(np.clip(z,0,np.log1p(1e7))) if target=='log1p' else np.clip(z,0,1e7)

def run(data='data/cohort.jsonl',output='models/tabular',trees=700,threads=4):
    d=load_cohort(data);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    dates=np.array(d.date);tr=d.phase.eq('train');va=d.phase.eq('validation');te=d.phase.eq('test')
    if set(d.phase)!={'train','validation','test'} or dates[tr].max()>dates[va].min() or dates[va].max()>dates[te].min():raise ValueError('Invalid chronological split')
    schema=fit_schema(d[tr]);x=features(d,schema);choices=[]
    for target in ['dollars','log1p']:
        for alpha in [.5,.9,1.5]:
            model,mean,std=fit(x[tr],d.price[tr],target,alpha,trees,threads,x[va],d.price[va])
            prediction=invert(model.predict(x[va]),mean,std,target)
            row={'target':target,'alpha':alpha,'rounds':int(model.best_iteration_ or trees),'validation':metrics(d.price[va],prediction)}
            print(json.dumps(row),flush=True);choices.append((row,model,mean,std,prediction))
    chosen,assessment,mean,std,vp=min(choices,key=lambda item:item[0]['validation']['mae_usd'])
    tp=invert(assessment.predict(x[te]),mean,std,chosen['target'])
    radius=float(np.quantile(abs(np.log1p(d.price[va])-np.log1p(vp)),.8))
    report={'mode':'tabular_sale_baseline','target_kind':TARGET,'settlement_verified':False,'currency':'USD',
        'selected':chosen,'grid':[r[0] for r in choices],'test':metrics(d.price[te],tp),
        'rows':{'train':int(tr.sum()),'validation':int(va.sum()),'test':int(te.sum()),'eligible':len(d)},
        'deployment_refit_rows':len(d),'band80_test_coverage':float(np.mean(abs(np.log1p(d.price[te])-np.log1p(tp))<=radius)),
        'note':'Chronological vehicle-disjoint assessment before all-row deployment refit. The final refit has no independent holdout. Public provider-reported sale prices; payment settlement is not verified. No text model or Isolation Forest in baseline.'}
    (out/'evaluation.json').write_text(json.dumps(report,indent=2))
    assessment.booster_.save_model(str(out/'assessment-lightgbm.txt'))
    schema=fit_schema(d);x=features(d,schema)
    model,mean,std=fit(x,d.price,chosen['target'],chosen['alpha'],chosen['rounds'],threads)
    with gzip.open(out/'lightgbm.txt.gz','wt') as f:f.write(model.booster_.model_to_string())
    (out/'schema.json').write_text(json.dumps(schema))
    manifest={'status':'ready','pipeline':'tabular-lightgbm','target_kind':TARGET,'currency':'USD',
        'target_transform':chosen['target'],'target_mean':mean,'target_std':std,'feature_names':model.booster_.feature_name(),
        'trained_rows':len(d),'training_candidates':len(d),'training_cohort_sha256':hashlib.sha256(Path(data).read_bytes()).hexdigest(),'band80_log_radius':radius,
        'sources':sorted(d.source.unique().tolist()),'sale_dates':[str(d.date.min()),str(d.date.max())],
        'model_year_range':[int(d.year.min()),int(d.year.max())],
        'assessment_note':report['note'],'outlier_filter':None}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(report,indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/cohort.jsonl');p.add_argument('--output',default='models/tabular');p.add_argument('--trees',type=int,default=700);p.add_argument('--threads',type=int,default=4);a=p.parse_args();run(a.data,a.output,a.trees,a.threads)
