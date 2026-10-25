"""Out-of-fold E5 fine-tuning followed by LightGBM on text price + car fields."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import KFold
import lightgbm as lgb
from pricing.text import MODEL_ID, fit_text, predict_text, save_text
from pricing.tabular import fit_schema, features, CATEGORICAL
from scripts.common import metrics
from pricing.identity import read_features


def folds(n, count, seed):
    if count<2 or count>n:
        raise ValueError('Need 2..n cross-fitting folds')
    return list(KFold(count,shuffle=True,random_state=seed).split(np.arange(n)))


def validate_cohort(d):
    required={'text','price','phase','date','row_id'}|set(CATEGORICAL)|{'year','odometer'}
    if not required.issubset(d):
        raise ValueError(f'Missing columns: {required-set(d)}')
    if not set(d.phase).issubset({'train','validation','test'}):
        raise ValueError('Unknown phase')
    for p in ['train','validation','test']:
        if not (d.phase==p).any():
            raise ValueError(f'Missing {p} rows')
    if not d.price.between(1000,150000).all() or d.row_id.duplicated().any():
        raise ValueError('Invalid targets or repeated canonical listings')
    dates=pd.to_datetime(d.date,utc=True)
    if dates[d.phase=='train'].max()>dates[d.phase=='validation'].min() or dates[d.phase=='validation'].max()>dates[d.phase=='test'].min():
        raise ValueError('Holdouts must be chronological')


def fit_lgb(x,y,xv,yv,seed):
    model=lgb.LGBMRegressor(objective='regression',n_estimators=1500,learning_rate=.025,
        num_leaves=15,max_depth=-1,min_child_samples=30,reg_lambda=5.,colsample_bytree=.9,
        random_state=seed,n_jobs=8,verbosity=-1)
    model.fit(x,np.log1p(y),eval_set=[(xv,np.log1p(yv))],eval_metric='l1',
        categorical_feature=CATEGORICAL,callbacks=[lgb.early_stopping(75,verbose=False)])
    return model


def run(a):
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU. Use a GPU runtime, or explicitly --device cpu --allow-cpu for slow full fine-tuning.')
    if a.device=='cpu' and not a.allow_cpu:
        raise RuntimeError('CPU full-transformer fine-tuning is slow; explicitly opt in with --allow-cpu.')
    if a.epochs<1 or a.batch_size<1 or a.accumulation<1:
        raise ValueError('epochs, batch size and accumulation must be positive')
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    d=pd.read_json(a.data,lines=True)
    validate_cohort(d)
    train=d[d.phase=='train'].reset_index(drop=True)
    val=d[d.phase=='validation'].reset_index(drop=True)
    test=d[d.phase=='test'].reset_index(drop=True)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    # Resolve one immutable source revision for every fold and the inference model.
    revision=a.revision or 'ff629add2ff80256e42173ec84299eed1c1b6393'
    config={k:v for k,v in vars(a).items() if k not in ['output','allow_cpu','text_only']}
    config.update(source=MODEL_ID,revision=revision,code_sha256=hashlib.sha256(
        (Path(__file__).read_bytes()+Path(__file__).with_name('text.py').read_bytes()+
         Path(__file__).with_name('tabular.py').read_bytes()+Path('scripts/common.py').read_bytes())).hexdigest())
    fingerprint=hashlib.sha256(Path(a.data).read_bytes()+json.dumps(config,sort_keys=True).encode()).hexdigest()
    progress=out/'training-config.json'
    if progress.exists() and json.loads(progress.read_text())['fingerprint']!=fingerprint:
        raise ValueError('Output directory contains a different run. Choose a new --output directory.')
    progress.write_text(json.dumps({'fingerprint':fingerprint,**config},indent=2))
    # A partial/failed rerun must not advertise a ready bundle.
    if not getattr(a,'text_only',False):
        (out/'manifest.json').unlink(missing_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=revision)
    kwargs=dict(tokenizer=tokenizer,source=MODEL_ID,revision=revision,epochs=a.epochs,
                batch_size=a.batch_size,accumulation=a.accumulation,lr=a.lr,device=a.device)
    oof=np.full(len(train),np.nan)
    fold_ids=np.full(len(train),-1)
    for k,(fit_idx,held_idx) in enumerate(folds(len(train),a.folds,a.seed)):
        cache=out/f'oof-fold-{k}.npz'
        if cache.exists():
            saved=np.load(cache)
            if not np.array_equal(saved['held_idx'],held_idx):
                raise ValueError('Cached fold row mismatch')
            p=saved['price']
        else:
            print(f'Fine-tuning fold {k+1}/{a.folds}: fit {len(fit_idx)}, predict {len(held_idx)}',flush=True)
            model=fit_text(train.text.iloc[fit_idx],train.price.iloc[fit_idx],seed=a.seed,**kwargs)
            p=predict_text(model,tokenizer,train.text.iloc[held_idx].tolist(),a.device,a.batch_size)
            np.savez(cache,held_idx=held_idx,price=p)
            del model;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        oof[held_idx]=p;fold_ids[held_idx]=k
    if not np.isfinite(oof).all():raise ValueError('Incomplete OOF predictions')
    train[['row_id']].assign(fold=fold_ids,predicted_text_price=oof).to_csv(out/'oof-audit.csv',index=False)
    if getattr(a,'text_only',False) and (out/'text-features.npz').exists():
        print('Completed text model and feature cache already exist',flush=True)
        return
    # Fixed epochs, no validation/test target gradients. All training rows retained.
    print('Fine-tuning final text model on all training rows',flush=True)
    model=fit_text(train.text,train.price,seed=a.seed,**kwargs)
    provenance={'source':MODEL_ID,'revision':revision,'dimensions':768,'max_tokens':512,
                'full_transformer_finetuning':True,'target':'standardized log1p(asking_price)',
                'epochs':a.epochs,'train_rows':len(train),'fingerprint':fingerprint}
    save_text(model,tokenizer,out/'text-model',provenance)
    vp=predict_text(model,tokenizer,val.text.tolist(),a.device,a.batch_size)
    if getattr(a,'text_only',False):
        tp=predict_text(model,tokenizer,test.text.tolist(),a.device,a.batch_size)
        np.savez(out/'text-features.npz',oof=oof,validation=vp,test=tp,
            train_ids=train.row_id.to_numpy(str),validation_ids=val.row_id.to_numpy(str),test_ids=test.row_id.to_numpy(str),fingerprint=fingerprint)
        print('Saved OOF, validation and test text features; no test target metrics calculated.',flush=True)
        return
    identity=read_features(out,a.data,{'train':train,'validation':val,'test':test})
    schema=fit_schema(train)
    x=features(train,schema,oof,identity['oof']);xv=features(val,schema,vp,identity['validation'])
    stack=fit_lgb(x,train.price,xv,val.price,a.seed)
    baseline=fit_lgb(features(train,schema),train.price,features(val,schema),val.price,a.seed)
    # Only now evaluate the untouched test set, once selection is complete.
    tp=predict_text(model,tokenizer,test.text.tolist(),a.device,a.batch_size)
    xt=features(test,schema,tp,identity['test'])
    stack_v=np.expm1(np.clip(stack.predict(xv),0,np.log1p(1e7)))
    stack_t=np.expm1(np.clip(stack.predict(xt),0,np.log1p(1e7)))
    tab_t=np.expm1(np.clip(baseline.predict(features(test,schema)),0,np.log1p(1e7)))
    band=float(np.quantile(abs(np.log1p(val.price)-np.log1p(stack_v)),.8))
    scores={'status':'evaluated','scope':'historical US asking prices, 2021',
        'rows':{'train':len(train),'validation':len(val),'test':len(test),'removed':0},
        'text_only':metrics(test.price,tp),'tabular_only':metrics(test.price,tab_t),
        'stacked':metrics(test.price,stack_t),'validation_stacked':metrics(val.price,stack_v),
        'band80_log_radius':band,'band80_test_coverage':float(np.mean(abs(np.log1p(test.price)-np.log1p(stack_t))<=band)),
        'best_iteration':stack.best_iteration_,'fingerprint':fingerprint}
    stack.booster_.save_model(str(out/'lightgbm.txt'))
    baseline.booster_.save_model(str(out/'tabular-baseline.txt'))
    (out/'schema.json').write_text(json.dumps(schema,indent=2))
    test[['row_id','price']].assign(predicted_text_price=tp,predicted_price=stack_t).to_csv(out/'test-predictions.csv',index=False)
    (out/'evaluation.json').write_text(json.dumps(scores,indent=2))
    (out/'manifest.json').write_text(json.dumps({'status':'ready','model_id':MODEL_ID,
        'pipeline':'fine-tuned-e5-base+lightgbm','identity_price_feature':True,'fingerprint':fingerprint,'band80_log_radius':band,
        'feature_names':stack.booster_.feature_name(),'source_revision':revision},indent=2))
    print(json.dumps(scores,indent=2),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='training-cache/ten-percent.jsonl')
    p.add_argument('--output',default='artifacts/e5-base-two-prices-10')
    p.add_argument('--revision',help='Exact embaas source revision; resolves once if omitted')
    p.add_argument('--folds',type=int,default=3)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--accumulation',type=int,default=8)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    p.add_argument('--allow-cpu',action='store_true')
    p.add_argument('--text-only',action='store_true',help='Fine-tune and cache text-price features for the experiment grid')
    return p


if __name__=='__main__':
    run(parser().parse_args())
