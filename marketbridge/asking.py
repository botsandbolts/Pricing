"""LightGBM asking-price experiment using independent older auction structure."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from pricing.tabular import features,fit_schema,CATEGORICAL
from marketbridge.structural import StructuralTransfer

AUX=['vehicle_age_years','auction_reference_2015_usd','transported_auction_price_usd',
     'relative_model_value_factor','model_training_count','mileage_vs_age_typical',
     'one_year_age_effect_pct','market_index_factor']

def inputs(frame,schema,structural=None,signals=None):
    frame=frame.copy()
    for c in ['year','odometer']+CATEGORICAL:
        if c not in frame:frame[c]=None
    x=features(frame,schema)
    if structural is not None or signals is not None:
        signals=structural.predict_frame(frame) if signals is None else signals
        for c in AUX:x[c]=pd.to_numeric(signals[c],errors='coerce').to_numpy()
    return x.replace([np.inf,-np.inf],np.nan)

def fit(x,y,rounds):
    y=np.log1p(np.asarray(y,float));mean=float(y.mean());std=float(y.std())
    m=lgb.LGBMRegressor(objective='huber',alpha=.9,n_estimators=rounds,learning_rate=.03,num_leaves=31,
        min_child_samples=30,reg_lambda=5,colsample_bytree=.9,random_state=42,n_jobs=4,verbosity=-1)
    m.fit(x,(y-mean)/std,categorical_feature=CATEGORICAL);return m.booster_,mean,std

def predict(model,mean,std,x):return np.expm1(np.clip(model.predict(x,num_threads=4)*std+mean,0,np.log1p(1e7)))

class TransferredAsking:
    def __init__(self,path,structural):
        path=Path(path);self.manifest=json.loads((path/'manifest.json').read_text());self.schema=json.loads((path/'schema.json').read_text());self.structural=structural
        if self.manifest['structural_model_id']!=structural.model_id:raise ValueError('Asking model and upstream structural model versions differ')
        with gzip.open(path/'lightgbm.txt.gz','rt') as f:self.model=lgb.Booster(model_str=f.read())
    def predict_frame(self,frame):
        x=inputs(frame,self.schema,self.structural);p=predict(self.model,self.manifest['target_mean'],self.manifest['target_std'],x)
        return p
    def explain(self,frame):
        x=inputs(frame,self.schema,self.structural)
        values=np.asarray(self.model.predict(x,pred_contrib=True,num_threads=4))[0]*self.manifest['target_std']
        return {'base':float(values[-1]+self.manifest['target_mean']),'contributions':dict(zip(x.columns,map(float,values[:-1]))),
                'target':'log1p asking price','method':'native TreeSHAP; structural inputs are correlated, not causal effects'}

def train(cohort,structural_path,out):
    from carbids.common import metrics
    d=pd.concat([c.drop(columns=['text','description'],errors='ignore') for c in pd.read_json(cohort,lines=True,chunksize=5000)],ignore_index=True)
    if d.row_id.duplicated().any():raise ValueError('Duplicate rows')
    date=pd.to_datetime(d.date,utc=True)
    tr=d.phase.eq('train');va=d.phase.eq('validation');te=d.phase.eq('test')
    if not (date[tr].max()<=date[va].min()<=date[va].max()<=date[te].min()):raise ValueError('Chronological partitions required')
    structural=StructuralTransfer(structural_path)
    if pd.to_datetime(structural.manifest['sale_dates'][1],utc=True)>=date.min():raise ValueError('Upstream source labels must predate target cohort')
    signals=structural.predict_frame(d);grids={};selected={};predictions={};band=None
    for role in ['tabular_only','with_auction_structure']:
        schema=fit_schema(d.loc[tr]);x=inputs(d,schema,signals=signals if role=='with_auction_structure' else None)
        candidates=[]
        for rounds in [500,1000]:
            m,mean,std=fit(x.loc[tr],d.loc[tr,'price'],rounds);pv=predict(m,mean,std,x.loc[va]);score=metrics(d.loc[va,'price'].to_numpy(),pv)
            candidates.append((score,m,mean,std,pv,rounds));print(role,rounds,score,flush=True)
        best=min(candidates,key=lambda v:v[0]['mae_usd']);score,m,mean,std,pv,rounds=best
        pt=predict(m,mean,std,x.loc[te]);predictions[role]=pt
        selected[role]={'rounds':rounds,'validation':score,'test':metrics(d.loc[te,'price'].to_numpy(),pt)}
        grids[role]=[{'rounds':v[-1],'validation':v[0]} for v in candidates]
        if role=='with_auction_structure':band=float(np.quantile(abs(np.log1p(d.loc[va,'price'])-np.log1p(pv)),.8))
    delta=abs(d.loc[te,'price'].to_numpy()-predictions['with_auction_structure'])-abs(d.loc[te,'price'].to_numpy()-predictions['tabular_only'])
    # Paired day-cluster bootstrap respects within-day dependence rather than independent row resampling.
    temp=pd.DataFrame({'day':date[te].dt.strftime('%Y-%m-%d'),'delta':delta});clusters=temp.groupby('day').delta.agg(['sum','count'])
    rng=np.random.default_rng(42);boot=[]
    for _ in range(1000):
        idx=rng.integers(0,len(clusters),len(clusters));sample=clusters.iloc[idx];boot.append(float(sample['sum'].sum()/sample['count'].sum()))
    report={'target_kind':'asking_price','rows':len(d),'train_rows':int(tr.sum()),'validation_rows':int(va.sum()),'test_rows':int(te.sum()),
        'models':selected,'grid':grids,'paired_test_mae_delta_usd':float(delta.mean()),'paired_day_cluster_bootstrap_95pct':np.quantile(boot,[.025,.975]).tolist(),
        'bootstrap_day_clusters':len(clusters),'uncertainty_note':'Few test days constrain the bootstrap; this is not a retail-sale evaluation.',
        'structural_model_id':structural.model_id,'upstream_target_labels':'2014–2015 reported auction sales, never 2021 asking targets',
        'known_upstream_vin_exclusions':json.loads((Path(structural_path)/'evaluation.json').read_text())['known_target_vin_overlaps_excluded'],
        'source_to_target_scope':'2021 asking-price enhancement only. No target retail sale prices or validated transported interval coverage.',
        'deployment_choice':'Auction-structure variant is available as an experiment; original baseline remains unchanged.'}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    schema=fit_schema(d);x=inputs(d,schema,signals=signals)
    m,mean,std=fit(x,d.price,selected['with_auction_structure']['rounds'])
    with gzip.open(out/'lightgbm.txt.gz','wt') as f:f.write(m.model_to_string())
    (out/'schema.json').write_text(json.dumps(schema));(out/'evaluation.json').write_text(json.dumps(report,indent=2))
    manifest={'status':'ready','target_kind':'asking_price','pipeline':'structural-auction-transfer+lightgbm',
        'target_transform':'log1p','target_mean':mean,'target_std':std,'feature_names':list(x.columns),
        'trained_rows':len(d),'structural_model_id':structural.model_id,'band80_log_radius':band,
        'cohort_sha256':hashlib.sha256(Path(cohort).read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',required=True);p.add_argument('--structural',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(train(a.cohort,a.structural,a.output),indent=2))
