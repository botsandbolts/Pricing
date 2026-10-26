"""Train a clearly labeled tabular-only studio baseline; E5 mode trains separately."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from pricing.train import lgb_fit,inverse,validate_cohort
from pricing.tabular import features,fit_schema
from scripts.common import metrics


def flatten(tree):
    nodes=[]
    def visit(t):
        at=len(nodes);nodes.append(None)
        if 'leaf_value' in t:nodes[at]={'v':t['leaf_value']};return at
        threshold=t['threshold'];kind=t['decision_type']
        n={'f':t['split_feature'],'t':list(map(int,threshold.split('||'))) if kind=='==' else threshold,
            'c':kind=='==','d':t['default_left'],'m':t['missing_type']}
        nodes[at]=n;n['l']=visit(t['left_child']);n['r']=visit(t['right_child']);return at
    visit(tree);return nodes


def run(a):
    # Descriptions are irrelevant to this baseline; discard them chunk by chunk.
    d=pd.concat([c.drop(columns=['text','description'],errors='ignore') for c in pd.read_json(a.data,lines=True,chunksize=5000)],ignore_index=True)
    validate_cohort(d,require_text=False)
    tr=d[d.phase=='train'];va=d[d.phase=='validation'];te=d[d.phase=='test']
    schema=fit_schema(tr);x=features(tr,schema);xv=features(va,schema);xt=features(te,schema)
    fits=[]
    for target in ['dollars','log1p']:
        for alpha in [.5,.9,1.5]:
            m,mean,std=lgb_fit(x,tr.price,(target,alpha,a.trees),a,xv,va.price)
            p=inverse(m.predict(xv),mean,std,target)
            row={'target':target,'alpha':alpha,'mean':mean,'std':std,'best_iteration':m.best_iteration_,
                'validation':metrics(va.price,p)}
            fits.append((row,m,p));print(target,alpha,row['validation']['mae_usd'],flush=True)
    row,m,vp=min(fits,key=lambda x:x[0]['validation']['mae_usd'])
    pred=inverse(m.predict(xt),row['mean'],row['std'],row['target'])
    band=float(np.quantile(abs(np.log1p(va.price)-np.log1p(vp)),.8))
    metadata=json.loads(Path(a.data).with_suffix('.meta.json').read_text()) if Path(a.data).with_suffix('.meta.json').exists() else {}
    report={'status':'evaluated','mode':'tabular_baseline','label':'Trained tabular-only baseline; not E5 enhanced',
        'sample_fraction':metadata.get('fraction',1.),'raw_sample_rows':metadata.get('sampled_raw_rows',len(d)),'eligible_rows':len(d),'train_rows':len(tr),'validation_rows':len(va),'test_rows':len(te),
        'filter':None,'filter_note':'No predicted_text_price exists in the baseline, so the E5-price filter does not apply.',
        'test':metrics(te.price,pred),'selected':row,'grid':[v[0] for v in fits],
        'band80_test_coverage':float(np.mean(abs(np.log1p(te.price)-np.log1p(pred))<=band)),
        'deployment_refit_rows':len(d),'assessment_note':'Metrics evaluate the chronological assessment model before deployment refit on all eligible rows.',
        'scope':'Historical April–May 2021 US asking prices, not transaction prices or current valuations.'}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    assessment=out/'assessment';assessment.mkdir(exist_ok=True)
    m.booster_.save_model(str(assessment/'lightgbm.txt'));(assessment/'schema.json').write_text(json.dumps(schema))
    print('Held-out assessment complete; refitting deployment on all',len(d),'eligible rows',flush=True)
    schema=fit_schema(d);final_x=features(d,schema)
    m,final_mean,final_std=lgb_fit(final_x,d.price,(row['target'],row['alpha'],max(1,row['best_iteration'])),a)
    m.booster_.save_model(str(out/'lightgbm.txt'));(out/'schema.json').write_text(json.dumps(schema))
    (out/'evaluation.json').write_text(json.dumps(report,indent=2))
    manifest={'status':'ready','pipeline':'tabular-lightgbm','feature_names':m.booster_.feature_name(),
        'target_transform':row['target'],'target_mean':final_mean,'target_std':final_std,'trained_rows':len(d),'sample_fraction':report['sample_fraction'],'band80_log_radius':band}
    (out/'manifest.json').write_text(json.dumps(manifest))
    dump=m.booster_.dump_model()
    catalog={str(make):sorted(set(frame.model.dropna().astype(str).str.lower().str.strip())) for make,frame in d.groupby('manufacturer')}
    bundle={'meta':report,'schema':schema,'catalog':catalog,'features':m.booster_.feature_name(),
        'trees':[flatten(t['tree_structure']) for t in dump['tree_info']], 'target':row['target'],
        'mean':final_mean,'std':final_std,'band':band}
    public=Path(a.public);public.mkdir(parents=True,exist_ok=True)
    (public/'baseline-model.json').write_text(json.dumps(bundle,separators=(',',':')))
    idx=np.linspace(0,len(final_x)-1,30,dtype=int);fixture=final_x.iloc[idx].copy()
    for c in schema:fixture[c]=fixture[c].cat.codes
    fixture=fixture.astype(float).replace({np.nan:None})
    (public/'baseline-fixtures.json').write_text(json.dumps({'rows':fixture.values.tolist(),
        'expected':inverse(m.predict(final_x.iloc[idx]),final_mean,final_std,row['target']).tolist()}))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='training-cache/full-cohort.jsonl');p.add_argument('--output',default='artifacts/studio-baseline-full');p.add_argument('--public',default='dist');p.add_argument('--trees',type=int,default=1000);p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=42)
    run(p.parse_args())
