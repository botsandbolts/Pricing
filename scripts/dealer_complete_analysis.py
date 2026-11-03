"""End-to-end, model-level merchant comparison with explicit support gates."""
import hashlib,json,sqlite3,zlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import hstack,csr_matrix
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBRegressor
from pricing.xgb_imputation import completed_numeric

def shared_support(frame):
    fields=['model','state','month','year_band','mileage_band']
    clean=frame.dropna(subset=fields).copy()
    if clean.empty:return clean
    counts=clean.groupby(fields+['dealer']).size().rename('n').reset_index()
    support=counts[counts.n>=3].groupby(fields).dealer.nunique()
    keys=set(support[support>=2].index)
    return clean.loc[clean[fields].apply(tuple,axis=1).isin(keys)]

def estimator():
    return XGBRegressor(n_estimators=150,max_depth=4,learning_rate=.06,reg_lambda=30,
                        tree_method='hist',n_jobs=2,random_state=73)

def evaluate(frame,x,y,groups,seed):
    shared=shared_support(frame)
    evidence={'input_adverts':len(frame),'shared_adverts':len(shared),'retained_fraction':len(shared)/max(len(frame),1),'effects':[]}
    ids=shared.index.to_numpy(int)
    if len(ids)<100 or len(np.unique(groups[ids]))<20:
        evidence['status']='Unsupported: fewer than 100 shared adverts or 20 vehicle groups'
        return evidence
    pred=np.zeros(len(ids))
    for tr,te in GroupKFold(5).split(ids,groups=groups[ids]):
        model=estimator().fit(x[ids[tr]],y[ids[tr]])
        pred[te]=model.predict(x[ids[te]])
    residual=y[ids]-pred
    evidence['oof_mae_usd']=float(np.mean(abs(np.exp(y[ids])-np.exp(pred))))
    shared=shared.copy();shared['residual']=residual;shared['vehicle_group']=groups[ids]
    # Compare residuals within the same joint inventory cell.
    fields=['model','state','month','year_band','mileage_band']
    shared['cell']=shared[fields].apply(tuple,axis=1)
    center=shared.groupby('cell').residual.transform('mean')
    shared['contrast']=shared.residual-center
    rng=np.random.default_rng(seed)
    for dealer,g in shared.groupby('dealer'):
        # Each vehicle group contributes once, not one vote per duplicate listing.
        values=g.groupby('vehicle_group').contrast.mean().to_numpy()
        if len(values)<50:continue
        mean=float(values.mean())
        boot=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(300)])
        se=max(float(boot.std(ddof=1)),.001)
        variance=1/(1/.1**2+1/se**2);pooled=variance*mean/se**2
        evidence['effects'].append({'dealer':dealer,'vehicle_groups':len(values),'mean_log':pooled,'variance_log':variance})
    dates=pd.to_datetime(shared.date,utc=True)
    unique=sorted(dates.unique())
    evidence['time_holdout']={'status':'Unsupported: insufficient distinct dates or date provenance unverified'}
    if len(unique)>=10:
        cutoff=unique[int(.8*(len(unique)-1))]
        test=dates>cutoff;testgroups=set(shared.loc[test,'vehicle_group'])
        train=(dates<=cutoff)&~shared.vehicle_group.isin(testgroups)
        if train.sum()>=100 and test.sum()>=50:
            model=estimator().fit(x[shared.index[train]],y[shared.index[train]])
            temporal=y[shared.index[test]]-model.predict(x[shared.index[test]])
            temp=shared.loc[test].copy();temp['temporal_residual']=temporal
            effects=temp.groupby('dealer').temporal_residual.agg(['mean','count'])
            effects['mean']-=float(np.mean(temporal))
            evidence['time_holdout']={'status':'Completed; canonical date provenance still requires confirmation',
              'cutoff':str(cutoff),'train_adverts':int(train.sum()),'test_adverts':int(test.sum()),
              'mae_usd':float(np.mean(abs(np.exp(y[shared.index[test]])-np.exp(y[shared.index[test]]-temporal)))),
              'dealer_residuals':effects.reset_index().to_dict('records')}
    evidence['status']='Model-level support evaluated; trim remains uncontrolled'
    return evidence

def main():
    root=Path('data/merchant-discovery-minilm')
    with sqlite3.connect(str(root/'cars.sqlite')) as db:
        cars=[json.loads(zlib.decompress(p)) for p, in db.execute('SELECT payload FROM cars ORDER BY position')]
    names=json.loads((root/'dealers.json').read_text())['rows'];d=pd.DataFrame(cars)
    numeric=np.column_stack((pd.to_numeric(d.year,errors='coerce'),np.log1p(pd.to_numeric(d.odometer,errors='coerce').clip(lower=0))))
    numeric[~np.isfinite(numeric)]=np.nan
    records=[{c+'='+str(r.get(c,'unknown')).lower():1 for c in ['manufacturer','model','type','fuel','transmission','condition','drive','state','title_status']} for r in cars]
    cats=FeatureHasher(n_features=4096,input_type='dict',alternate_sign=False).transform(records).astype(np.float32)
    context=TruncatedSVD(n_components=16,random_state=42).fit_transform(cats)
    groups=[]
    for c in cars:
        vin=str(c.get('vin') or c.get('VIN') or '')
        key=vin if len(vin)==17 else '|'.join(str(c.get(k,'')) for k in ['manufacturer','model','year','odometer'])
        groups.append(hashlib.sha256(key.encode()).hexdigest())
    groups=np.array(groups);y=np.log(pd.to_numeric(d.price).to_numpy(float))
    rows=[]
    for i,(car,candidates) in enumerate(zip(cars,names)):
        if not candidates:continue
        date=car.get('posting_date') or car.get('listing_date') or car.get('date') or car.get('observed_date')
        dt=pd.to_datetime(date,errors='coerce',utc=True)
        rows.append({'id':i,'dealer':candidates[0]['name'].casefold(),'score':candidates[0]['score'],'candidates':len(candidates),
          'model':str(car.get('manufacturer'))+'|'+str(car.get('model')),'state':car.get('state'),
          'date':dt,'month':None if pd.isna(dt) else dt.strftime('%Y-%m')})
    base=pd.DataFrame(rows).set_index('id')
    all_runs=[]
    for seed in [71,83,97,109,127]:
        completed=completed_numeric(numeric,context,seed)
        x=hstack((csr_matrix(completed.astype(np.float32)),cats),format='csr')
        frame=base.copy();frame['year_band']=np.floor(completed[frame.index,0]/5)
        frame['mileage_band']=np.floor(np.expm1(np.clip(completed[frame.index,1],0,np.log1p(2000000)))/25000)
        primary=evaluate(frame[(frame.candidates==1)&(frame.score>=.9)],x,y,groups,seed)
        alternate=evaluate(frame[frame.score>=.8],x,y,groups,seed)
        all_runs.append({'seed':seed,'primary':primary,'alternative_top_candidate':alternate})
        print('Completed imputation, supported refit and temporal checks',seed,flush=True)
    # Rubin-style combination of conditional normal dealer posteriors.
    tables=[]
    keys=set.intersection(*[{r['dealer'] for r in run['primary']['effects']} for run in all_runs])
    for dealer in sorted(keys):
        effects=[next(r for r in run['primary']['effects'] if r['dealer']==dealer) for run in all_runs]
        means=np.array([r['mean_log'] for r in effects]);within=np.mean([r['variance_log'] for r in effects])
        variance=within+(1+1/5)*means.var(ddof=1);mean=float(means.mean())
        alt=[next((r for r in run['alternative_top_candidate']['effects'] if r['dealer']==dealer),None) for run in all_runs]
        temporal=[next((r for r in run['primary']['time_holdout'].get('dealer_residuals',[]) if r['dealer']==dealer and r['count']>=20),None) for run in all_runs]
        stable=all(r is not None for r in alt+temporal) and all(np.sign(r['mean_log'])==np.sign(mean) for r in alt) and all(np.sign(r['mean'])==np.sign(mean) for r in temporal)
        tables.append({'dealer':dealer,'minimum_vehicle_groups':min(r['vehicle_groups'] for r in effects),
          'association_pct':100*np.expm1(mean) if stable else None,
          'conditional_low_pct':100*np.expm1(mean-1.96*np.sqrt(variance)) if stable else None,
          'conditional_high_pct':100*np.expm1(mean+1.96*np.sqrt(variance)) if stable else None,
          'status':'Sensitivity-stable exploratory model-level association; trim uncontrolled, date provenance unverified' if stable else 'Suppressed: temporal or merchant-match sensitivity insufficient/inconsistent'})
    report={'version':1,'status':'Completed analysis; unsupported results explicitly suppressed','rows':len(cars),'imputations':5,'runs':all_runs,'table':tables,
      'limitations':'Historical asking associations, not causal dealer effects or evidence of misconduct. Trim is not controlled. Canonical dates are used, not confirmed original listing timestamps. Multiple imputation is approximate under MAR and Gaussian residual assumptions. Conditional normal random-effect prior SD .10. Vehicle-cluster bootstrap conditions on fitted OOF baselines; intervals omit full baseline-refit and identity uncertainty and are not coverage validated. Rubin-style combination is approximate. Alternative identity sensitivity uses the highest candidate at score >=.80, including multi-candidate adverts; not verified entity resolution. Positive and negative values can reflect inventory, services, condition, warranties and other unmeasured differences. Temporal estimates use later shared-support adverts with vehicle-group purging; overlap itself is transductive and this is not a prospective external benchmark.',
      'support_definition':'Exact make/model/state/month; five-year and 25000-mile bands; >=3 adverts per merchant and >=2 merchants per joint cell; >=50 unique vehicle groups per merchant in each imputation.',
      'method':'Five numeric imputations; five-fold vehicle-group XGBoost baseline refits within shared support; cell-centered log residual contrasts; 300 vehicle bootstrap samples per dealer; partial pooling; chronological held-out sensitivity; alternate top-candidate identity assignment; trim-level comparison remains unsupported.'}
    out=Path('reports/dealer-overlap');out.mkdir(parents=True,exist_ok=True)
    (out/'completed-analysis.json').write_text(json.dumps(report,default=str,allow_nan=False))
    print(json.dumps({'status':report['status'],'table_rows':len(tables),'displayable':sum(r['association_pct'] is not None for r in tables)},indent=2))
if __name__=='__main__':main()
