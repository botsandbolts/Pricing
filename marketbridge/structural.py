"""Age-based, CPI-anchored wholesale transport. No raw model-year feature.
Empirical partial pooling + monotone Huber LightGBM, not a retail sale calibration.
"""
import argparse,gzip,hashlib,json,re
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

CATS=['segment','transmission','state']
NUMS=['age','log_miles','auction_grade']
VIN=re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b',re.I)


def norm(values):return values.astype('string').str.lower().str.strip().replace({'':pd.NA,'unknown':pd.NA})
def segment(values):
    s=norm(values).fillna('unknown')
    return s.map(lambda x:'truck' if any(w in x for w in ['cab','pickup','truck','crew']) else 'suv' if x=='suv' else 'van' if 'van' in x else 'coupe' if 'coupe' in x else 'convertible' if 'convertible' in x else 'wagon' if 'wagon' in x else 'hatchback' if 'hatchback' in x else 'sedan' if 'sedan' in x else 'unknown')

def dates(frame):
    for name in ['as_of','date','sale_date']:
        if name in frame:
            s=pd.to_datetime(frame[name],format='mixed',utc=True,errors='coerce')
            if name=='as_of' and 'date' in frame:s=s.fillna(pd.to_datetime(frame.date,format='mixed',utc=True,errors='coerce'))
            return s
    return pd.Series(pd.NaT,index=frame.index,dtype='datetime64[ns, UTC]')

def prepare(frame,training=False):
    f=frame.copy()
    for old,new in {'make':'manufacturer','mileage_miles':'odometer','body_type':'type'}.items():
        if new not in f and old in f:f[new]=f[old]
    for col in ['year','odometer','manufacturer','model','type','transmission','state']: 
        if col not in f:f[col]=None
    d=dates(f);year=pd.to_numeric(f.year,errors='coerce');miles=pd.to_numeric(f.odometer,errors='coerce')
    fractional=d.dt.year+(d.dt.dayofyear-1)/365.25
    age=(fractional-year).clip(lower=0)  # Model-year age proxy; production date is unavailable.
    grade=f['condition'] if training else (f['auction_condition'] if 'auction_condition' in f else pd.Series(np.nan,index=f.index))
    grade=pd.to_numeric(grade,errors='coerce');grade=grade.where(grade.between(1,5))
    x=pd.DataFrame({'age':age,'log_miles':np.log1p(miles.where(miles.ge(0))),'auction_grade':grade,
        'segment':segment(f.type),'transmission':norm(f.transmission),'state':norm(f.state)},index=f.index)
    make=norm(f.manufacturer).fillna('unknown');model=norm(f.get('auction_model',f.model)).fillna(norm(f.model)).fillna('unknown')
    keys=pd.DataFrame({'make':make,'model':make+'::'+model},index=f.index)
    valid=d.notna()&year.notna()&year.between(1886,2027)&(year<=d.dt.year+1)&(miles.isna()|miles.between(0,2_000_000))
    return x,keys,valid,d

class IndexPanel:
    def __init__(self,path):
        self.path=Path(path);self.data=pd.read_csv(path,dtype={'month':str}).set_index('month')['value']
        if self.data.index.duplicated().any() or not np.isfinite(self.data).all() or (self.data<=0).any():raise ValueError('Invalid monthly used-car index')
        self.base=float(self.data.loc['2015-01'])
    def ratios(self,date_values):
        # Two full months lag avoids use of a month not yet released. Snapshot revisions remain a limitation.
        months=date_values.dt.tz_localize(None).dt.to_period('M')-2
        values=months.astype(str).map(self.data)
        return values/self.base,months.astype(str)


def hierarchy(keys,target,shrink=40.):
    global_mean=float(np.median(target))
    temp=keys.copy();temp['resid']=np.clip(np.asarray(target)-global_mean,-2,2)
    make=temp.groupby('make').resid.agg(['mean','count']);make['offset']=make['mean']*make['count']/(make['count']+shrink)
    temp['resid']=np.clip(np.asarray(target)-global_mean-temp.make.map(make.offset).fillna(0),-2,2)
    model=temp.groupby('model').resid.agg(['mean','count']);model['offset']=model['mean']*model['count']/(model['count']+shrink)
    return {'global':global_mean,'make':make.offset.to_dict(),'model':model.offset.to_dict(),'make_counts':make['count'].astype(int).to_dict(),'model_counts':model['count'].astype(int).to_dict(),'shrink_pseudo_count':shrink}

def offsets(keys,h):return h['global']+keys.make.map(h['make']).fillna(0).to_numpy(float)+keys.model.map(h['model']).fillna(0).to_numpy(float)
def encode(x,schema):
    x=x.copy()
    for c in CATS:x[c]=pd.Categorical(x[c],categories=schema[c])
    return x[NUMS+CATS]

class StructuralTransfer:
    def __init__(self,path):
        self.path=Path(path);self.manifest=json.loads((self.path/'manifest.json').read_text());self.h=json.loads((self.path/'hierarchy.json').read_text())
        self.schema=json.loads((self.path/'schema.json').read_text());self.index=IndexPanel(self.path/'used-car-index.csv')
        with gzip.open(self.path/'lightgbm.txt.gz','rt') as f:self.model=lgb.Booster(model_str=f.read())
        self.model_id=hashlib.sha256((self.path/'manifest.json').read_bytes()+(self.path/'lightgbm.txt.gz').read_bytes()+(self.path/'hierarchy.json').read_bytes()+(self.path/'used-car-index.csv').read_bytes()).hexdigest()
    def log_reference(self,x,keys):return offsets(keys,self.h)+self.model.predict(encode(x,self.schema),num_threads=4)
    def predict_frame(self,frame):
        x,keys,valid,d=prepare(frame);ratio,months=self.index.ratios(d);valid=valid&ratio.notna()
        log=self.log_reference(x,keys);ref=np.exp(np.clip(log,np.log(1),np.log(1e7)))
        output=pd.DataFrame(index=frame.index)
        if 'row_id' in frame:output['row_id']=frame.row_id
        output['status']=np.where(valid,'estimated_under_transfer_assumptions',np.where(d.isna(),'missing_valuation_date',np.where(ratio.isna(),'market_index_unavailable','invalid_vehicle_fields')))
        output['vehicle_age_years']=x.age
        output['auction_reference_2015_usd']=np.where(valid,ref,np.nan)
        output['market_index_factor']=ratio;output['index_month_used']=months
        output['transported_auction_price_usd']=np.where(valid,ref*ratio,np.nan)
        output['relative_model_value_factor']=np.exp(keys.make.map(self.h['make']).fillna(0)+keys.model.map(self.h['model']).fillna(0))
        output['model_training_count']=keys.model.map(self.h['model_counts']).fillna(0).astype(int)
        output['manufacturer_known']=keys.make.isin(self.h['make'])
        output['model_known']=keys.model.isin(self.h['model'])
        output['age_supported']=x.age.between(*self.manifest['age_range'])
        output['mileage_supported']=np.expm1(x.log_miles).between(*self.manifest['mileage_range'])
        output['generation_supported']=pd.to_numeric(frame.get('year',pd.Series(np.nan,index=frame.index)),errors='coerce').between(*self.manifest['model_year_range'])
        output['auction_condition_supplied']=x.auction_grade.notna()
        # Age-conditioned descriptive mileage benchmark, separate from constrained predictor.
        agebin=x.age.fillna(0).astype(int).clip(0,40).astype(str)
        typical=agebin.map(self.manifest['typical_miles_by_age']).fillna(self.manifest['typical_miles_global'])
        output['mileage_vs_age_typical']=np.expm1(x.log_miles)-typical
        older=x.copy();older['age']=x.age+1
        output['one_year_age_effect_pct']=100*np.expm1(self.log_reference(older,keys)-log)
        # Source residual band is *not* transported coverage. Show source-scale sensitivity only.
        b=self.manifest['source_residual_log_radius80']
        output['source_residual_sensitivity_low_usd']=output.transported_auction_price_usd*np.exp(-b)
        output['source_residual_sensitivity_high_usd']=output.transported_auction_price_usd*np.exp(b)
        output['target_coverage_validated']=False;output['retail_sale_calibrated']=False
        output['structural_model_id']=self.model_id
        return output
    def decompose(self,fields):
        f=pd.DataFrame([fields]);x,keys,valid,d=prepare(f);r,months=self.index.ratios(d)
        if not valid.iloc[0] or pd.isna(r.iloc[0]):raise ValueError('Valid year, valuation date and lagged index required')
        ref=x.copy();ref['age']=3.;ref['log_miles']=np.log1p(36000);ref['auction_grade']=3.
        base=float(self.log_reference(ref,keys)[0]);steps=[{'component':'Reference vehicle: age 3, 36,000 miles, grade 3','log_effect':0.,'price_usd':float(np.exp(base))}]
        previous=base
        for label,col in [('Age at requested date','age'),('Actual mileage','log_miles'),('Actual auction grade (missing if unknown)','auction_grade')]:
            ref[col]=x[col];now=float(self.log_reference(ref,keys)[0]);steps.append({'component':label,'log_effect':now-previous,'price_usd':float(np.exp(now))});previous=now
        steps.append({'component':f'Used-car market index, lagged month {months.iloc[0]}','log_effect':float(np.log(r.iloc[0])),'price_usd':float(np.exp(previous)*r.iloc[0])})
        return steps


def fit_model(d,rounds,alpha):
    x,keys,valid,date=prepare(d,training=True);schema={c:sorted(x[c].dropna().unique().tolist()) for c in CATS}
    target=d.log_real_price.to_numpy(float)
    # Two alternating robust partial-pooling/shape updates, then final shape fit.
    h=hierarchy(keys,target);shape=np.zeros(len(d))
    for iteration in range(3):
        if iteration:h=hierarchy(keys,target-shape)
        residual=target-offsets(keys,h)
        model=lgb.LGBMRegressor(objective='huber',alpha=alpha,n_estimators=rounds,learning_rate=.04,num_leaves=31,
            min_child_samples=100,reg_lambda=8,n_jobs=4,random_state=42,verbosity=-1,
            monotone_constraints=[-1,-1,1,0,0,0],monotone_constraints_method='advanced')
        model.fit(encode(x,schema),residual,categorical_feature=CATS)
        shape=model.predict(encode(x,schema))
    return model.booster_,h,schema

def known_pricing_vins(path):
    result=set()
    for chunk in pd.read_json(path,lines=True,chunksize=5000):
        for text in chunk.get('text',pd.Series('',index=chunk.index)).fillna(''):
            result.update(v.upper() for v in VIN.findall(str(text)))
    return result

def train(cohort,index_file,out,exclude_pricing=None):
    from carbids.common import metrics
    d=pd.concat([c.drop(columns=['text','price_evidence','source_url'],errors='ignore') for c in pd.read_json(cohort,lines=True,chunksize=5000)],ignore_index=True)
    panel=IndexPanel(index_file);x,k,valid,date=prepare(d,training=True);ratios,_=panel.ratios(date)
    original=len(d);excluded=0;known=0
    if exclude_pricing:
        vins=known_pricing_vins(exclude_pricing);known=len(vins)
        overlap=d.vehicle_group.astype(str).str.removeprefix('vin:').isin(vins);excluded=int(overlap.sum());valid&=~overlap
    d=d[valid&ratios.notna()].copy();d['log_real_price']=np.log(d.price.to_numpy()/ratios.loc[d.index].to_numpy())
    tr=d[d.phase=='train'];va=d[d.phase=='validation'];te=d[d.phase=='test'];grid=[];fits=[]
    for rounds,alpha in [(250,.9),(450,.9)]:
        booster,h,schema=fit_model(tr,rounds,alpha)
        xv,kv,_,dv=prepare(va,training=True);rv,_=panel.ratios(dv)
        pv=np.exp(offsets(kv,h)+booster.predict(encode(xv,schema),num_threads=4))*rv
        score=metrics(va.price.to_numpy(),pv.to_numpy());grid.append({'rounds':rounds,'alpha':alpha,'validation':score});fits.append((booster,h,schema,pv))
        print('Structural grid',rounds,score,flush=True)
    best=int(np.argmin([r['validation']['mae_usd'] for r in grid]));booster,h,schema,pv=fits[best]
    xt,kt,_,dt=prepare(te,training=True);rt,_=panel.ratios(dt)
    pred=np.exp(offsets(kt,h)+booster.predict(encode(xt,schema),num_threads=4))*rt
    band=float(np.quantile(abs(np.log(va.price)-np.log(pv)),.8));assessment=metrics(te.price.to_numpy(),pred.to_numpy())
    report={'source':'Reported wholesale auction sales, 2014–2015','test':assessment,'grid':grid,'selected':grid[best],
        'train_rows':len(tr),'validation_rows':len(va),'test_rows':len(te),'eligible_rows':len(d),'source_rows':original,
        'known_pricing_vins':known,'known_target_vin_overlaps_excluded':excluded,
        'unknown_target_vins_note':'Unobserved VINs cannot establish vehicle disjointness; upstream auction labels predate 2021 Pricing outcomes.',
        'source_band80_test_coverage':float(np.mean(abs(np.log(te.price)-np.log(pred))<=band)),
        'evaluation_scope':'Source-domain chronological assessment before deployment refit; no retail-sale or 2021 target-sale accuracy established. Current-vintage lagged CPI snapshot, not a real-time vintage replay.'}
    print('Refitting structural on',len(d),flush=True)
    booster,h,schema=fit_model(d,grid[best]['rounds'],grid[best]['alpha'])
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with gzip.open(out/'lightgbm.txt.gz','wt') as f:f.write(booster.model_to_string())
    (out/'hierarchy.json').write_text(json.dumps(h));(out/'schema.json').write_text(json.dumps(schema))
    import shutil
    shutil.copyfile(index_file,out/'used-car-index.csv')
    xp,_,_,dates_=prepare(d,training=True)
    temp=pd.DataFrame({'age':xp.age.astype(int).clip(0,40),'miles':d.odometer});typ=temp.groupby('age').miles.median()
    manifest={'status':'ready','pipeline':'partial-pooled-monotone-huber-age-transfer','target_kind':'auction_equivalent_under_transfer_assumptions',
        'trained_rows':len(d),'feature_names':NUMS+CATS,'raw_model_year_feature':False,'reference_index_month':'2015-01','index_lag_months':2,
        'index_series':'CUUR0000SETA02','index_source':'https://api.bls.gov/publicAPI/v2/timeseries/data/CUUR0000SETA02',
        'index_snapshot_note':'Retrieved 2026-10-26; current vintage. Aggregate retail index applied to wholesale is an explicit transfer assumption.',
        'age_range':[float(xp.age.min()),float(xp.age.max())],'model_year_range':[int(d.year.min()),int(d.year.max())],
        'mileage_range':[float(d.odometer.min()),float(d.odometer.max())],
        'typical_miles_by_age':{str(k):float(v) for k,v in typ.items()},'typical_miles_global':float(d.odometer.median()),
        'source_residual_log_radius80':band,'sale_dates':[str(d.date.min()),str(d.date.max())],
        'cohort_sha256':hashlib.sha256(Path(cohort).read_bytes()).hexdigest(),
        'limitations':['Age curves are cross-sectional associations, not identified causal ageing trajectories.','Generation and quality changes after 2015 are unobserved.','No calibrated retail-sale outcome labels.','No target-domain interval coverage guarantee.']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'evaluation.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',required=True);p.add_argument('--index',required=True);p.add_argument('--output',required=True);p.add_argument('--exclude-pricing');a=p.parse_args()
    print(json.dumps(train(a.cohort,a.index,a.output,a.exclude_pricing),indent=2))
