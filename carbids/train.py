"""100% MiniLM description OOF + Isolation Forest + categorical/tabular Huber LightGBM."""
import argparse,gc,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from transformers import AutoTokenizer
from sklearn.ensemble import IsolationForest
from carbids.oof import cross_fit
from carbids.text import MODEL_ID,fit_text,predict_text,save_text,load_text
from carbids.tabular import features,fit_schema,CATEGORICAL
from carbids.common import clean,metrics


def validate_cohort(d,require_text=True):
    columns={'row_id','price','date','phase','year','odometer'}|set(CATEGORICAL)
    if require_text:columns.add('text')
    if not columns.issubset(d):raise ValueError(f'Missing columns: {columns-set(d)}')
    if d.row_id.duplicated().any() or not d.price.between(1,10000000).all():raise ValueError('Invalid targets or duplicate row IDs')
    if set(d.phase)!={'train','validation','test'}:raise ValueError('Need chronological train/validation/test partitions')
    dates=pd.to_datetime(d.date,utc=True,errors='coerce')
    if dates.isna().any() or dates[d.phase=='train'].max()>dates[d.phase=='validation'].min() or dates[d.phase=='validation'].max()>dates[d.phase=='test'].min():
        raise ValueError('Chronological partitions required')


def inverse(z,mean,std,target):
    p=np.asarray(z)*std+mean
    return np.expm1(np.clip(p,0,np.log1p(1e7))) if target=='log1p' else np.clip(p,0,1e7)


def lgb_fit(x,y,settings,a,xv=None,yv=None):
    target,alpha,rounds=settings;y=np.asarray(y,float);raw=np.log1p(y) if target=='log1p' else y
    mean=float(raw.mean());std=float(max(raw.std(),1e-6))
    m=lgb.LGBMRegressor(objective='huber',alpha=alpha,metric='None',n_estimators=rounds,
        learning_rate=.03,num_leaves=31,min_child_samples=30,reg_lambda=5.,colsample_bytree=.9,
        random_state=a.seed,n_jobs=a.threads,verbosity=-1)
    kwargs={'categorical_feature':CATEGORICAL}
    if xv is not None:
        v=np.log1p(yv) if target=='log1p' else np.asarray(yv,float)
        kwargs.update(eval_set=[(xv,(v-mean)/std)],eval_metric='l1',callbacks=[lgb.early_stopping(min(75,max(5,rounds//5)),verbose=False)])
    m.fit(x,(raw-mean)/std,**kwargs);return m,mean,std


def filter_prices(prices,path,row_ids,seed=42,threads=4):
    prices=np.asarray(prices,float)
    if prices.ndim!=1 or not np.isfinite(prices).all() or len(prices)<3:raise ValueError('Finite OOF description prices required')
    detector=IsolationForest(n_estimators=300,max_samples=min(256,len(prices)),contamination='auto',random_state=seed,n_jobs=threads)
    labels=detector.fit_predict(prices.reshape(-1,1));keep=np.flatnonzero(labels==1)
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'row_id':np.asarray(row_ids),'predicted_text_price':prices,'score':detector.score_samples(prices.reshape(-1,1)),'kept':labels==1}).to_csv(path/'isolation-audit.csv',index=False)
    if len(keep)<2:raise ValueError('Isolation Forest retained too few rows; no forced fallback is applied')
    return keep


def run(a):
    from carbids.data import load_cohort
    d=load_cohort(a.data, require_text=True)
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable. Run on a GPU, or explicitly use --device cpu --allow-cpu for slow MiniLM fine-tuning.')
    if a.device=='cpu' and not a.allow_cpu:raise RuntimeError('Explicit --allow-cpu required for slow full MiniLM training')
    torch.set_num_threads(a.threads)
    if 'text' not in d and 'description' in d:d['text']=d.description
    d['text']=d.text.fillna('').map(clean);validate_cohort(d)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    config=vars(a).copy();config.pop('output',None)
    config.update(description_model=MODEL_ID,description_revision=a.revision)
    source=b''.join(Path(__file__).with_name(p).read_bytes() for p in ['train.py','oof.py','text.py','tabular.py','common.py','data.py'])
    fingerprint=hashlib.sha256(Path(a.data).read_bytes()+json.dumps(config,sort_keys=True).encode()+source).hexdigest()
    cfg=out/'training-config.json'
    if cfg.exists() and json.loads(cfg.read_text())['fingerprint']!=fingerprint:raise ValueError('Output contains a different run; choose a new output directory')
    cfg.write_text(json.dumps({'fingerprint':fingerprint,**config},indent=2));(out/'manifest.json').unlink(missing_ok=True)
    tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=a.revision)
    text_kwargs=dict(tokenizer=tok,source=MODEL_ID,revision=a.revision,epochs=a.epochs,
        batch_size=a.batch_size,accumulation=a.accumulation,lr=a.lr,device=a.device,
        checkpoint_dir=out/'optimizer-checkpoints',checkpoint_steps=10)
    def release(model):
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    def stage(indices,path,held=None):
        path.mkdir(parents=True,exist_ok=True);local=d.iloc[indices].reset_index(drop=True)
        text_oof,fold_ids=cross_fit(local.row_id,local.price,
            lambda fit_idx,y:fit_text(local.text.iloc[fit_idx],y,seed=a.seed,**text_kwargs),
            lambda model,idx:predict_text(model,tok,local.text.iloc[idx].tolist(),a.device,a.batch_size),
            folds=a.folds,seed=a.seed,cache=path/'description-oof',fingerprint=fingerprint,release=release)
        # Fresh all-fitting-row regressors predict validation/test and future requests.
        desc_dir=path/'text-model'
        if (desc_dir/'provenance.json').exists():desc,_=load_text(desc_dir,a.device)
        else:
            desc=fit_text(local.text,local.price,seed=a.seed,**text_kwargs)
            save_text(desc,tok,desc_dir,{'source':MODEL_ID,'revision':a.revision,'fitting_rows':len(indices),'fingerprint':fingerprint})
        pd.DataFrame({'row_id':local.row_id,'fold':fold_ids,'predicted_text_price':text_oof}).to_csv(path/'oof-predictions.csv',index=False)
        held_text=None
        if held is not None:
            held_text=predict_text(desc,tok,d.text.iloc[held].tolist(),a.device,a.batch_size)
        del desc;release(None)
        return text_oof,held_text
    tr=np.flatnonzero(d.phase=='train');va=np.flatnonzero(d.phase=='validation');te=np.flatnonzero(d.phase=='test')
    held=np.r_[va,te]
    tprice,ht=stage(tr,out/'assessment',held)
    keep=filter_prices(tprice,out/'assessment',d.row_id.iloc[tr],a.seed,a.threads)
    schema=fit_schema(d.iloc[tr]);x=features(d.iloc[tr],schema,tprice)
    xv=features(d.iloc[va],schema,ht[:len(va)]);xt=features(d.iloc[te],schema,ht[len(va):])
    choices=[]
    for target in ['dollars','log1p']:
        for alpha in [.5,.9,1.5]:
            m,mean,std=lgb_fit(x.iloc[keep],d.price.iloc[tr[keep]],(target,alpha,a.trees),a,xv,d.price.iloc[va])
            vp=inverse(m.predict(xv),mean,std,target)
            row={'target':target,'alpha':alpha,'best_iteration':m.best_iteration_,'validation':metrics(d.price.iloc[va],vp),'mean':mean,'std':std}
            choices.append((row,m,vp))
    choice,model,vp=min(choices,key=lambda v:v[0]['validation']['mae_usd'])
    tp=inverse(model.predict(xt),choice['mean'],choice['std'],choice['target'])
    band=float(np.quantile(abs(np.log1p(d.price.iloc[va])-np.log1p(vp)),.8))
    evaluation={'status':'evaluated','rows':{'train':len(tr),'validation':len(va),'test':len(te),'eligible':len(d),'outlier_removed':len(tr)-len(keep),'retained_train':len(keep)},
        'selected':choice,'grid':[c[0] for c in choices],'stacked':metrics(d.price.iloc[te],tp),
        'description_only':metrics(d.price.iloc[te],ht[len(va):]),
        'band80_test_coverage':float(np.mean(abs(np.log1p(d.price.iloc[te])-np.log1p(tp))<=band)),
        'note':'Platform/provider-reported sale prices; not independently verified settlement. Chronological assessment before all-row refit.'}
    (out/'evaluation.json').write_text(json.dumps(evaluation,indent=2));model.booster_.save_model(str(out/'assessment/lightgbm.txt'))
    print(json.dumps(evaluation,indent=2),flush=True)
    del choices,model;gc.collect()
    all_idx=np.arange(len(d));tprice,_=stage(all_idx,out)
    keep=filter_prices(tprice,out,d.row_id,a.seed,a.threads)
    schema=fit_schema(d);x=features(d,schema,tprice)
    model,mean,std=lgb_fit(x.iloc[keep],d.price.iloc[keep],(choice['target'],choice['alpha'],max(1,choice['best_iteration'])),a)
    model.booster_.save_model(str(out/'lightgbm.txt'));(out/'schema.json').write_text(json.dumps(schema,indent=2))
    manifest={'status':'ready','pipeline':'fine-tuned-minilm-l6+lightgbm','trained_rows':len(keep),'target_kind':'platform_reported_sale_price','currency':'USD','sources':sorted(d.source.unique().tolist()),'sale_dates':[str(d.date.min()),str(d.date.max())],'model_year_range':[int(d.year.min()),int(d.year.max())],'training_candidates':len(d),'training_cohort_sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),'outlier_filter':{'algorithm':'IsolationForest','contamination':'auto','input':'predicted_text_price','removed':len(d)-len(keep)},
        'feature_names':model.booster_.feature_name(),'target_transform':choice['target'],'target_mean':mean,'target_std':std,
        'band80_log_radius':band,'fingerprint':fingerprint,'assessment_note':evaluation['note']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('100% MiniLM + Isolation Forest stack ready',flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',default='data/cohort.jsonl');p.add_argument('--output',default='artifacts/enhanced')
    p.add_argument('--device',default='cuda');p.add_argument('--allow-cpu',action='store_true');p.add_argument('--revision',default='1110a243fdf4706b3f48f1d95db1a4f5529b4d41')
    p.add_argument('--folds',type=int,default=3);p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--accumulation',type=int,default=8);p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--trees',type=int,default=1000);p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=42)
    return p


if __name__=='__main__':run(parser().parse_args())
