"""Re-evaluate installed Studio models, including the live description regressor.

New partitions subdivide the existing assessment holdout; this is not new data.
Refitted all-row deployment bundles are labelled in-sample, never held-out scores.
Run: python -m scripts.evaluate_studio training-cache/full-cohort.jsonl
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.common import metrics
from pricing.inference import PricePipeline
from pricing.tabular import features
from pricing.text import predict_text


def make_partitions(frame, per_phase=1000):
    if frame.row_id.duplicated().any():
        raise ValueError('Duplicate listing identities')
    pool = frame[frame.phase == 'test'].copy().sort_values(['date', 'row_id'])
    dates = pd.to_datetime(pool.date, utc=True)
    cutoff = dates.iloc[len(pool) // 2]
    pool['audit_phase'] = np.where(dates < cutoff, 'validation', 'test')
    if pool.audit_phase.nunique() != 2:
        raise ValueError('Need two nonempty chronological date groups')
    if per_phase:
        pool['_selection']=pool.row_id.map(lambda identity:hashlib.sha256(('20261027:'+identity).encode()).hexdigest())
        pool=pool.sort_values('_selection').groupby('audit_phase',group_keys=False).head(per_phase).drop(columns='_selection').sort_values(['date','row_id'])
    return pool.reset_index(drop=True), str(cutoff)


def run(data, output, batch_size=8, per_phase=1000):
    import torch
    torch.set_num_threads(4)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    counts=Counter();seen=set();held=[]
    for chunk in pd.read_json(data,lines=True,chunksize=5000):
        identities=set(chunk.row_id)
        if chunk.row_id.duplicated().any() or seen.intersection(identities):raise ValueError('Duplicate listing identities')
        seen.update(identities);counts.update(chunk.phase)
        if (chunk.phase=='test').any():held.append(chunk[chunk.phase=='test'].copy())
    if counts != {'train':148290,'validation':31777,'test':31777}:
        raise ValueError('Cohort differs from the installed assessment; establish training membership first')
    frame=pd.concat(held,ignore_index=True)
    del held,seen,chunk
    pool, cutoff = make_partitions(frame, per_phase)
    del frame
    assignments = pool[['row_id', 'date', 'audit_phase']]
    assignments.to_csv(out/'audit-split.csv', index=False)
    report = {
        'status':'running', 'seed':20261027, 'cutoff_utc':cutoff, 'per_phase_limit':per_phase, 'original_test_pool_rows':31777,
        'data_sha256':hashlib.sha256(Path(data).read_bytes()).hexdigest(),
        'counts':pool.audit_phase.value_counts().to_dict(),
        'split_note':'New chronological subdivisions of the ORIGINAL test pool. Not newly collected data, not an untouched new benchmark. Original training and tuning-validation rows excluded. No refitting or tuning in this audit.',
        'model_manifests':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('models').glob('*/manifest.json')},
        'models':{},
    }
    prior=out/'studio-audit.json'
    if prior.exists():
        previous=json.loads(prior.read_text())
        for key in ['data_sha256','model_manifests','cutoff_utc','per_phase_limit']:
            if previous.get(key)!=report[key]:raise ValueError(f'Audit cache belongs to different {key}; choose a new output directory')
    def save(): (out/'studio-audit.json').write_text(json.dumps(report, indent=2))
    save()
    pipeline = PricePipeline('models/minilm-assessment')
    cache=out/'audit-text-predictions.csv'
    if cache.exists():
        saved=pd.read_csv(cache)
        if saved.row_id.tolist()!=pool.row_id.tolist():raise ValueError('Text cache does not match split identities')
        text=saved.predicted_text_price.to_numpy()
    else:
        values=[]
        progress=out/'text-progress.csv'
        if progress.exists():
            previous=pd.read_csv(progress)
            if previous.row_id.tolist()!=pool.row_id.iloc[:len(previous)].tolist():raise ValueError('Progress cache has different row identities')
            values=previous.predicted_text_price.tolist()
        for start in range(len(values),len(pool),128):
            values.extend(predict_text(pipeline.text_model,pipeline.tokenizer,pool.text.iloc[start:start+128].tolist(),'cpu',batch_size))
            print(f'Text inference {len(values)}/{len(pool)}',flush=True)
            # Resume inference without ever truncating the evaluated cohort silently.
            pd.DataFrame({'row_id':pool.row_id.iloc[:len(values)],'predicted_text_price':values}).to_csv(out/'text-progress.csv',index=False)
        text=np.asarray(values)
        pd.DataFrame({'row_id':pool.row_id,'predicted_text_price':text}).to_csv(cache,index=False)
    x=features(pool,pipeline.schema,text)
    raw=pipeline.lgb.predict(x,num_threads=4)
    prediction=np.clip(raw*pipeline.manifest['target_std']+pipeline.manifest['target_mean'],0,1e7)
    predictions=pool[['row_id','audit_phase','price']].copy()
    def record(name, values, status):
        predictions[name]=values
        report['models'][name]={'evaluation_scope':status, **{phase:metrics(pool.loc[pool.audit_phase==phase,'price'],np.asarray(values)[pool.audit_phase==phase]) for phase in ['validation','test']},'combined':metrics(pool.price,values)}
        rng=np.random.default_rng(20261027)
        for phase in ['validation','test']:
            error=np.abs(pool.loc[pool.audit_phase==phase,'price'].to_numpy()-np.asarray(values)[pool.audit_phase==phase])
            boot=np.mean(error[rng.integers(0,len(error),(2000,len(error)))],axis=1)
            report['models'][name][phase]['mae_bootstrap_95pct_usd']=np.quantile(boot,[.025,.975]).tolist()
        save()
    record('MiniLM + LightGBM',prediction,'Out of original fitting and tuning data; reused assessment holdout')
    record('MiniLM text only',text,'Out of original fitting and tuning data; reused assessment holdout')
    original=json.loads(Path('models/minilm-assessment/evaluation.json').read_text())
    report['reference']={'original_full_test_mae_usd':original['stacked']['mae_usd'],'note':'The new audit uses sampled partitions; its MAE need not equal the original full-test MAE.'}
    save()
    for folder in ['studio-baseline','robust-asking','robust-asking-isolation']:
        if not (Path('models')/folder/'manifest.json').exists():continue
        p=PricePipeline(Path('models')/folder)
        if hasattr(p,'_ensemble'):
            values=p._ensemble.ensemble.predict_frame(pool.assign(as_of='2021-05-01'))
        else:
            raw=p.lgb.predict(features(pool,p.schema),num_threads=4)
            values=np.expm1(np.clip(raw*p.manifest['target_std']+p.manifest['target_mean'],0,np.log1p(1e7)))
        record(folder,values,'IN-SAMPLE / training overlap: deployed bundle refitted on the full eligible cohort. Not a generalization MAE; consult original assessment.')
    predictions.to_csv(out/'audit-predictions.csv',index=False)
    report['status']='complete';save()
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data');p.add_argument('--output',default='reports/studio-audit');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--per-phase',type=int,default=1000,help='Rows per partition; 0 evaluates the entire original test pool')
    a=p.parse_args();run(a.data,a.output,a.batch_size,a.per_phase)
