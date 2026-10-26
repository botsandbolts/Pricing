"""Larger regularized LightGBM search and validation-selected USD ensemble.
Targets remain historical asking prices. No model is selected using test loss.
"""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from pricing.tabular import CATEGORICAL,fit_schema
from marketbridge.asking import inputs
from marketbridge.structural import StructuralTransfer,dates

COMPOSITES=['make_model','make_body']

def engineer(frame,schema,structural,feature_schema=None,signals=None):
    f=frame.copy()
    for c in ['year','odometer']+CATEGORICAL:
        if c not in f:f[c]=None
    x=inputs(f,schema,signals=signals) if signals is not None else inputs(f,schema,structural)
    year=pd.to_numeric(f.year,errors='coerce');miles=pd.to_numeric(f.odometer,errors='coerce').where(lambda v:v>=0)
    d=dates(f);age=(d.dt.year+(d.dt.dayofyear-1)/365.25-year).clip(lower=0)
    x['log_odometer']=np.log1p(miles)
    x['miles_per_year']=miles/(age+1)
    x['odometer_missing']=miles.isna().astype(int)
    x['year_missing']=year.isna().astype(int)
    make=f.manufacturer.astype('string').str.lower().str.strip().fillna('unknown')
    model=f.model.astype('string').str.lower().str.strip().fillna('unknown')
    body=f['type'].astype('string').str.lower().str.strip().fillna('unknown')
    composites={'make_model':make+'::'+model,'make_body':make+'::'+body}
    if feature_schema is None:
        feature_schema={c:{'categories':sorted(v.unique().tolist()),'counts':v.value_counts().astype(int).to_dict()} for c,v in composites.items()}
    for c,v in composites.items():
        x[c]=pd.Categorical(v,categories=feature_schema[c]['categories'])
        x[c+'_log_count']=np.log1p(v.map(feature_schema[c]['counts']).fillna(0).astype(float))
    return x.replace([np.inf,-np.inf],np.nan),feature_schema

CONFIGS=[
 {'name':'huber_log_63','target':'log1p','objective':'huber','alpha':.9,'num_leaves':63,'max_depth':10,'min_data_in_leaf':50,'lambda_l2':8.,'learning_rate':.025},
 {'name':'huber_log_127','target':'log1p','objective':'huber','alpha':.9,'num_leaves':127,'max_depth':12,'min_data_in_leaf':70,'lambda_l2':12.,'learning_rate':.025},
 {'name':'huber_log_255','target':'log1p','objective':'huber','alpha':.9,'num_leaves':255,'max_depth':14,'min_data_in_leaf':100,'lambda_l2':20.,'learning_rate':.025},
 {'name':'huber_log_127_strong','target':'log1p','objective':'huber','alpha':.7,'num_leaves':127,'max_depth':12,'min_data_in_leaf':140,'lambda_l2':30.,'learning_rate':.025},
 {'name':'mae_dollars_63','target':'dollars','objective':'regression_l1','num_leaves':63,'max_depth':10,'min_data_in_leaf':50,'lambda_l2':8.,'learning_rate':.03},
 {'name':'mae_dollars_127','target':'dollars','objective':'regression_l1','num_leaves':127,'max_depth':12,'min_data_in_leaf':80,'lambda_l2':12.,'learning_rate':.03},
 {'name':'huber_dollars_127','target':'dollars','objective':'huber','alpha':.9,'num_leaves':127,'max_depth':12,'min_data_in_leaf':80,'lambda_l2':12.,'learning_rate':.03},
 {'name':'mae_log_127','target':'log1p','objective':'regression_l1','num_leaves':127,'max_depth':12,'min_data_in_leaf':80,'lambda_l2':12.,'learning_rate':.03},
]

def invert(values,mean,std,target):
    v=np.asarray(values)*std+mean
    return np.expm1(np.clip(v,0,np.log1p(1e7))) if target=='log1p' else np.clip(v,0,1e7)

def fit_one(x,y,config,rounds=3500,valid=None,seed=42):
    raw=np.log1p(np.asarray(y,float)) if config['target']=='log1p' else np.asarray(y,float)
    mean=float(raw.mean());std=float(max(raw.std(),1e-6));params={k:v for k,v in config.items() if k not in ['name','target']}
    params.update(metric='None',verbosity=-1,num_threads=4,seed=seed,feature_fraction=.85,bagging_fraction=.85,bagging_freq=1,
        lambda_l1=.5,cat_l2=15.,cat_smooth=30.,min_data_per_group=50,max_cat_threshold=64,path_smooth=1.,max_bin=255,deterministic=True,force_col_wise=True)
    data=lgb.Dataset(x,label=(raw-mean)/std,categorical_feature=CATEGORICAL+COMPOSITES,free_raw_data=False)
    kwargs={}
    if valid is not None:
        xv,yv=valid;rv=np.log1p(np.asarray(yv,float)) if config['target']=='log1p' else np.asarray(yv,float)
        dv=lgb.Dataset(xv,label=(rv-mean)/std,reference=data,categorical_feature=CATEGORICAL+COMPOSITES,free_raw_data=False)
        def usd_mae(pred,dataset):return 'mae_usd',float(np.mean(abs(invert(pred,mean,std,config['target'])-np.asarray(yv,float)))),False
        kwargs.update(valid_sets=[dv],feval=usd_mae,callbacks=[lgb.early_stopping(150,first_metric_only=True,verbose=False)])
    model=lgb.train(params,data,num_boost_round=rounds,**kwargs)
    return model,mean,std

class RobustAsking:
    def __init__(self,path,structural):
        path=Path(path);self.manifest=json.loads((path/'manifest.json').read_text());self.structural=structural
        if self.manifest['structural_model_id']!=structural.model_id:raise ValueError('Robust ensemble upstream version mismatch')
        self.schema=json.loads((path/'schema.json').read_text());self.feature_schema=json.loads((path/'feature-schema.json').read_text());self.members=[]
        from marketbridge.model_storage import read_model_text
        for member in self.manifest['members']:
            model=lgb.Booster(model_str=read_model_text(path,member))
            if model.feature_name()!=self.manifest['feature_names']:raise ValueError('Robust ensemble feature mismatch')
            self.members.append((member,model))
    def predict_frame(self,frame):
        x,_=engineer(frame,self.schema,self.structural,self.feature_schema)
        return sum(m['weight']*invert(model.predict(x,num_threads=4),m['mean'],m['std'],m['target']) for m,model in self.members)
    def member_predictions(self,frame):
        x,_=engineer(frame,self.schema,self.structural,self.feature_schema)
        return np.column_stack([invert(model.predict(x,num_threads=4),m['mean'],m['std'],m['target']) for m,model in self.members])

def train(cohort,structural_path,output):
    from carbids.common import metrics
    out=Path(output);out.mkdir(parents=True,exist_ok=True);cache=out/'search-cache';cache.mkdir(exist_ok=True)
    d=pd.concat([c.drop(columns=['text','description'],errors='ignore') for c in pd.read_json(cohort,lines=True,chunksize=5000)],ignore_index=True)
    if d.row_id.duplicated().any() or not d.price.between(1000,150000).all():raise ValueError('Invalid cohort')
    tr=d.phase.eq('train');va=d.phase.eq('validation');te=d.phase.eq('test');time=pd.to_datetime(d.date,utc=True)
    if not (time[tr].max()<=time[va].min()<=time[va].max()<=time[te].min()):raise ValueError('Invalid chronological split')
    structural=StructuralTransfer(structural_path)
    if pd.to_datetime(structural.manifest['sale_dates'][1],utc=True)>=time.min():raise ValueError('Upstream labels must predate target cohort')
    signals=structural.predict_frame(d);schema=fit_schema(d.loc[tr]);xt,fs=engineer(d.loc[tr],schema,structural,signals=signals.loc[tr]);x,_=engineer(d,schema,structural,fs,signals=signals)
    fingerprint=hashlib.sha256(Path(cohort).read_bytes()+Path(__file__).read_bytes()+structural.model_id.encode()).hexdigest()
    cfgfile=cache/'fingerprint.txt'
    if cfgfile.exists() and cfgfile.read_text()!=fingerprint:raise ValueError('Search cache belongs to different source/cohort')
    cfgfile.write_text(fingerprint);candidates=[];vp=[]
    for config in CONFIGS:
        prefix=cache/config['name'];record=prefix.with_suffix('.json');model_path=prefix.with_suffix('.txt')
        if record.exists() and model_path.exists():
            row=json.loads(record.read_text());model=lgb.Booster(model_file=str(model_path));mean=row['mean'];std=row['std']
        else:
            model,mean,std=fit_one(x.loc[tr],d.loc[tr,'price'],config,valid=(x.loc[va],d.loc[va,'price']))
            p=invert(model.predict(x.loc[va],num_threads=4),mean,std,config['target']);score=metrics(d.loc[va,'price'].to_numpy(),p)
            row={'config':config,'mean':mean,'std':std,'best_iteration':model.best_iteration,'validation':score}
            model.save_model(str(model_path));record.write_text(json.dumps(row,indent=2))
        candidates.append(row);vp.append(invert(model.predict(x.loc[va],num_threads=4),mean,std,config['target']))
        print(config['name'],row['best_iteration'],row['validation'],flush=True)
    # Choose best three configurations using validation only. Include a single-model solution in the simplex search.
    top=np.argsort([r['validation']['mae_usd'] for r in candidates])[:3];pval=np.column_stack([vp[i] for i in top]);yv=d.loc[va,'price'].to_numpy()
    best=(float('inf'),None)
    for a in range(21):
        for b in range(21-a):
            w=np.array([a,b,20-a-b])/20.;p=pval@w;loss=float(np.mean(abs(yv-p)))
            if loss<best[0]:best=(loss,w)
    weights=best[1];selected=[(int(i),float(w)) for i,w in zip(top,weights) if w>0]
    # Seed repetitions are fixed in advance, not selected using the test set.
    assessment=[];pva=[];pte=[]
    for i,w in selected:
        row=candidates[i]
        for seed in [42,137,777]:
            model,mean,std=fit_one(x.loc[tr],d.loc[tr,'price'],row['config'],rounds=row['best_iteration'],seed=seed)
            assessment.append((row,seed,w/3,model,mean,std))
            pva.append((w/3)*invert(model.predict(x.loc[va],num_threads=4),mean,std,row['config']['target']))
            pte.append((w/3)*invert(model.predict(x.loc[te],num_threads=4),mean,std,row['config']['target']))
    pv=sum(pva);pt=sum(pte);test=metrics(d.loc[te,'price'].to_numpy(),pt);validation=metrics(yv,pv)
    baseline=json.loads((Path(structural_path).parent/'transfer-asking/evaluation.json').read_text())['models']['tabular_only']
    # Eligibility fixed on validation; test comparison is disclosed as a reused benchmark, not fresh discovery.
    promoted=validation['mae_usd']<baseline['validation']['mae_usd'] and test['mae_usd']<baseline['test']['mae_usd'] and test['rmse_usd']<=baseline['test']['rmse_usd']
    e={'status':'evaluated','target_kind':'asking_price','rows':len(d),'train_rows':int(tr.sum()),'validation_rows':int(va.sum()),'test_rows':int(te.sum()),
        'baseline':baseline,'validation':validation,'test':test,'grid':candidates,
        'selected_configs':[{'config':candidates[i]['config'],'rounds':candidates[i]['best_iteration'],'weight':w} for i,w in selected],
        'seed_repetitions':[42,137,777],'ensemble_members':len(assessment),'promoted':bool(promoted),
        'test_note':'The historical test benchmark has been inspected in earlier project experiments. This comparison is not a fresh untouched external replication. No hyperparameters or weights are chosen from its losses.',
        'robustness_note':'Robust losses, bagging, regularized categories, early stopping and seed averaging reduce sensitivity; they do not guarantee performance under market drift.',
        'structural_model_id':structural.model_id,'training_fingerprint':fingerprint}
    (out/'evaluation.json').write_text(json.dumps(e,indent=2));print('Assessment',test,'promoted',promoted,flush=True)
    # All eligible rows are refitted only after assessment. Fixed selected tree counts and weights.
    schema=fit_schema(d);full,fs=engineer(d,schema,structural,signals=signals);members=[]
    for j,(row,seed,weight,_,_,_) in enumerate(assessment):
        model,mean,std=fit_one(full,d.price,row['config'],rounds=row['best_iteration'],seed=seed)
        filename=f'member-{j}.txt.gz'
        with gzip.open(out/filename,'wt') as f:f.write(model.model_to_string())
        members.append({'file':filename,'weight':weight,'seed':seed,'target':row['config']['target'],'mean':mean,'std':std,'trees':model.num_trees(),'config':row['config']})
    (out/'schema.json').write_text(json.dumps(schema));(out/'feature-schema.json').write_text(json.dumps(fs))
    manifest={'status':'ready','pipeline':'regularized-lightgbm-ensemble','target_kind':'asking_price','trained_rows':len(d),'promoted':bool(promoted),
        'structural_model_id':structural.model_id,'feature_names':list(full.columns),'members':members,
        'band80_log_radius':float(np.quantile(abs(np.log1p(yv)-np.log1p(pv)),.8)),
        'scope':'Historical April–May 2021 asking prices, not verified sales or current valuations','training_fingerprint':fingerprint}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));return e

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',required=True);p.add_argument('--structural',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(train(a.cohort,a.structural,a.output),indent=2))
