"""Training-only empirical-Bayes partial pooling of out-of-fold log-price errors.

Sequential conjugate approximations, not a joint MCMC posterior. Never filters rows.
"""
import hashlib
import json
import numpy as np
import pandas as pd

REQUIRED = ['row_id','vehicle_group','split','prediction_kind','base_fold',
            'manufacturer','model','age','observed_price','base_price','base_model_id','price_basis']


def validate(frame):
    missing=set(REQUIRED)-set(frame)
    if missing:raise ValueError('Missing columns: '+', '.join(sorted(missing)))
    d=frame[REQUIRED].copy()
    if d.isna().any().any():raise ValueError('Required fields cannot be missing.')
    if len(d)>100000:raise ValueError('Use at most 100,000 rows per experiment.')
    if d.row_id.astype(str).duplicated().any():raise ValueError('Duplicate row identities.')
    if not set(d.split).issubset({'calibration','validation','test'}):raise ValueError('Unknown split.')
    for split,minimum in [('calibration',30),('validation',10),('test',10)]:
        if (d.split==split).sum()<minimum:raise ValueError(f'Need at least {minimum} {split} rows.')
    if d.groupby('vehicle_group').split.nunique().max()>1:raise ValueError('Vehicle groups cross split boundaries.')
    cal=d.split=='calibration'
    if not d.loc[cal,'prediction_kind'].eq('oof').all():raise ValueError('Calibration requires out-of-fold base predictions.')
    if not d.loc[~cal,'prediction_kind'].eq('heldout').all():raise ValueError('Validation/test require held-out base predictions.')
    if d.loc[cal,'base_fold'].nunique()<2:raise ValueError('At least two calibration folds are required.')
    if d.loc[cal].groupby('vehicle_group').base_fold.nunique().max()>1:raise ValueError('Vehicle groups cross OOF folds.')
    for col in ['age','observed_price','base_price']:
        d[col]=pd.to_numeric(d[col],errors='raise')
        if not np.isfinite(d[col]).all():raise ValueError('Nonfinite numeric data.')
    if not d.age.between(0,40).all():raise ValueError('Age must be 0–40 years.')
    if not d[['observed_price','base_price']].gt(0).all().all():raise ValueError('Prices must be positive.')
    if not d[['observed_price','base_price']].le(1e7).all().all():raise ValueError('Unsupported price magnitude.')
    for col in ['base_model_id','price_basis']:
        if d[col].astype(str).nunique()!=1:raise ValueError('Use a single base-model identity and price basis per file.')
    for col in ['manufacturer','model']:
        d[col]=d[col].astype(str).str.strip().str.lower().str.slice(0,120)
    return d


def _pool(keys,residual,sigma2):
    t=pd.DataFrame({'key':keys,'r':residual}).groupby('key').r.agg(['mean','count'])
    tau2=max(0.,float(t['mean'].var(ddof=0)-np.mean(sigma2/t['count'])))
    if tau2<1e-10:return {str(k):{'effect':0.,'sd':0.,'n':int(v['count']),'weight':0.} for k,v in t.iterrows()},0.
    t['weight']=t['count']*tau2/(sigma2+t['count']*tau2)
    return {str(k):{'effect':float(v['weight']*v['mean']),
            'sd':float(np.sqrt(tau2*(1-v['weight']))),'n':int(v['count']),
            'weight':float(v['weight'])} for k,v in t.iterrows()},tau2


def fit(frame):
    d=frame.reset_index(drop=True)
    r=np.log(d.observed_price.to_numpy()/d.base_price.to_numpy())
    sigma2=max(float(np.var(r)),1e-8);global_mean=float(np.mean(r));center=float(d.age.mean())
    makes,make_tau=_pool(d.manufacturer,r-global_mean,sigma2)
    after=r-global_mean-np.array([makes[k]['effect'] for k in d.manufacturer])
    keys=d.manufacturer+'::'+d.model
    models,model_tau=_pool(keys,after,sigma2)
    after-=np.array([models[k]['effect'] for k in keys])
    # A strongly regularized common residual-age slope; model slopes require support.
    z=d.age.to_numpy()-center
    slope=float(z@after/(z@z+100.));after-=slope*z
    slopes={}
    for key,g in d.assign(key=keys).groupby('key'):
        idx=g.index.to_numpy();x=z[idx]
        if len(g)<40 or np.ptp(g.age)<3:continue
        denom=float(x@x+1000.)
        slopes[key]={'effect':float(x@after[idx]/denom),'variance':sigma2/denom,'n':len(g)}
    return {'method':'sequential_empirical_bayes_log_residuals','global':global_mean,
            'global_variance':sigma2/len(d),'sigma2':sigma2,'age_center':center,
            'age_slope':slope,'age_variance':sigma2/(z@z+100.),'makes':makes,
            'models':models,'slopes':slopes,'make_tau2':make_tau,'model_tau2':model_tau,
            'rows':len(d),'base_model_id':str(d.base_model_id.iloc[0]),
            'price_basis':str(d.price_basis.iloc[0])}


def predict(bundle,frame):
    rows=[]
    for _,r in frame.iterrows():
        make=str(r.manufacturer).strip().lower()[:120];key=make+'::'+str(r.model).strip().lower()[:120]
        m=bundle['makes'].get(make,{'effect':0.,'sd':np.sqrt(bundle['make_tau2']),'n':0,'weight':0.})
        v=bundle['models'].get(key,{'effect':0.,'sd':np.sqrt(bundle['model_tau2']),'n':0,'weight':0.})
        slope=bundle['slopes'].get(key,{'effect':0.,'variance':0.})
        z=float(r.age)-bundle['age_center'];age=(bundle['age_slope']+slope['effect'])*z
        delta=bundle['global']+m['effect']+v['effect']+age
        sd=float(np.sqrt(bundle['global_variance']+m['sd']**2+v['sd']**2+z*z*(bundle['age_variance']+slope['variance'])))
        base=float(r.base_price);corrected=base*np.exp(delta)
        if not np.isfinite(corrected) or not 0<corrected<=1e7:raise ValueError('Unsupported corrected price.')
        rows.append(dict(base_price=base,corrected_price=corrected,delta_log=delta,
            correction_sd_log=sd,make_n=m['n'],model_n=v['n'],make_weight=m['weight'],
            model_weight=v['weight'],global_effect=bundle['global'],make_effect=m['effect'],
            model_effect=v['effect'],age_effect=age))
    return pd.DataFrame(rows,index=frame.index)


def scores(y,p):
    y=np.asarray(y);p=np.asarray(p)
    return {'mae_usd':float(np.mean(np.abs(y-p))),
            'rmse_usd':float(np.sqrt(np.mean((y-p)**2))),
            'mae_log':float(np.mean(np.abs(np.log(y/p)))),
            'bias_usd':float(np.mean(p-y)),'rows':len(y)}


def experiment(frame,side):
    if side not in {'ask','bid'}:raise ValueError('Choose ask or bid.')
    d=validate(frame);cal=d[d.split=='calibration'];bundle=fit(cal)
    bundle['side']=side
    bundle['fingerprint']=hashlib.sha256(json.dumps(bundle,sort_keys=True).encode()).hexdigest()
    report={'side':side,'method':bundle['method'],'calibration_rows':len(cal),'cohorts':{}}
    test_predictions=None
    for split in ['validation','test']:
        held=d[d.split==split];p=predict(bundle,held)
        difference=np.abs(held.observed_price-p.corrected_price)-np.abs(held.observed_price-p.base_price)
        rng=np.random.default_rng(42)
        # Group-level bootstrap preserves repeated observations of the same vehicle.
        grouped=pd.DataFrame({'group':held.vehicle_group,'difference':difference}).groupby('group').difference.agg(['sum','count'])
        means=[]
        for _ in range(300):
            sample=grouped.iloc[rng.integers(0,len(grouped),len(grouped))]
            means.append(float(sample['sum'].sum()/sample['count'].sum()))
        report['cohorts'][split]={'base':scores(held.observed_price,p.base_price),
            'hierarchical':scores(held.observed_price,p.corrected_price),
            'mae_change_usd':float(difference.mean()),
            'mae_change_95_group_bootstrap':np.quantile(means,[.025,.975]).tolist()}
        if split=='test':test_predictions=pd.concat([held.reset_index(drop=True),p.drop(columns='base_price').reset_index(drop=True)],axis=1)
    return bundle,report,test_predictions
