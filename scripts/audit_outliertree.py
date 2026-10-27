"""Native categorical OutlierTree audit; never deletes cars or refits pricing models."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from outliertree import OutlierTree
from pricing.model_suite import load_cohort, predictors
from marketbridge.structural import IndexPanel, known_pricing_vins


def serial(v):
    if isinstance(v, dict): return {str(k):serial(w) for k,w in v.items()}
    if isinstance(v, (list,tuple,np.ndarray)): return [serial(w) for w in v]
    if isinstance(v,np.generic): return serial(v.item())
    if isinstance(v,float) and not np.isfinite(v): return None
    return v


def stage(d,x,out,pct,threads):
    out.mkdir(parents=True,exist_ok=True)
    x=x.dropna(axis=1,how='all').copy();cats=[];pool={}
    for c in x:
        if pd.api.types.is_numeric_dtype(x[c]): x[c]=x[c].astype(float)
        else: x[c]=x[c].astype(object);cats.append(c)
    if 'model' in cats:
        counts=x.model.value_counts();levels=sorted(counts.index,key=lambda k:(-counts[k],str(k)))
        retained=[k for k in levels if counts[k]>=20][:1000]
        replacement='__outliertree_rare_model__'
        while replacement in counts: replacement+='_' 
        rare=~x.model.isin(retained)
        pool={'original_levels':len(counts),'retained_named_levels':len(retained),'pooled_rows':int(rare.sum()),'min_count':20,'max_retained_levels':1000,'replacement':replacement}
        x.loc[rare,'model']=replacement
    print('FIT',out,'rows',len(x),flush=True);started=time.monotonic()
    settings=dict(max_depth=4,min_gain=.01,z_norm=2.67,z_outlier=8.,pct_outliers=pct,min_size_numeric=25,min_size_categ=50,categ_split='binarize',categ_outliers='tail',numeric_split='raw',follow_all=False,gain_as_pct=True,nthreads=threads)
    model=OutlierTree(**settings);flags=model.fit(x,outliers_print=0,return_outliers=True)
    actual=set(model.cols_cat_)|set(model.cols_ord_)|set(model.cols_bool_)
    if not set(cats)<=actual: raise AssertionError('Native categorical inputs omitted')
    if not flags.index.equals(x.index): raise AssertionError('Flag identity mismatch')
    mask=flags.suspicious_value.map(bool);selected=flags.loc[mask].sort_values('outlier_score')
    fields=selected.suspicious_value.map(lambda r:r['column']).value_counts().to_dict()
    meta=[c for c in ['row_id','phase','price','manufacturer','model','year','odometer'] if c in d]
    rows=d.loc[selected.index,meta].copy()
    for c in selected: rows[c]=selected[c].map(lambda v:json.dumps(serial(v),ensure_ascii=False,allow_nan=False))
    rows.to_csv(out/'flagged-cars.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    examples=[]
    for i,row in selected.head(15).iterrows():examples.append({'car':serial(d.loc[i,meta].to_dict()),'explanation':serial(row.to_dict())})
    report={'input_rows':len(x),'flagged_rows':len(selected),'flagged_fraction':len(selected)/len(x),'primary_flagged_fields':fields,'primary_price_flags':int(fields.get('log_reference_price',0)), 'settings':settings,'category_pooling':pool,'native_categorical_columns':cats,'phase_counts':d.phase.value_counts().to_dict(),'examples':examples,'elapsed_seconds':time.monotonic()-started,'cars_deleted':0,'weights_refitted':False,'input_columns':list(x)}
    (out/'report.json').write_text(json.dumps(serial(report),indent=2,allow_nan=False))
    print('COMPLETE',out,'flagged',len(selected),'/',len(x),flush=True)
    return {k:v for k,v in report.items() if k!='examples'}


def run(a):
    if not 0 < a.pct_outliers < .1:
        raise ValueError('OutlierTree requires 0 < pct_outliers < 0.10; 0.25 is unsupported')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Use a new output directory')
    known=known_pricing_vins(a.ask);panel=IndexPanel(a.index);prior=json.loads(Path(a.prior).read_text());results={}
    for side,path in [('ask',a.ask),('bid',a.bid)]:
        sha=hashlib.file_digest(open(path,'rb'),'sha256').hexdigest()
        assert sha==prior['markets'][side]['source_sha256'], 'Cohort hash mismatch'
        d=load_cohort(path);excluded=0
        if side=='bid':
            overlap=d.vehicle_group.astype(str).str.removeprefix('vin:').isin(known);excluded=int(overlap.sum());d=d.loc[~overlap].reset_index(drop=True)
        x,valid,dates=predictors(d,side);ratios,_=panel.ratios(dates)
        if side=='bid':valid&=ratios.notna()
        d=d.loc[valid].copy();x=x.loc[valid].copy();assert len(d)==prior['markets'][side]['eligible_rows']
        if side=='ask' and 'condition' in d:x['seller_condition']=d.condition.fillna('unknown').astype(object)
        y=d.price.to_numpy(float)
        if side=='bid':y=y/ratios.loc[d.index].to_numpy(float)
        x['log_reference_price']=np.log(y)
        stages={}
        for name,mask in [('assessment',d.phase=='train'),('full_cohort',np.ones(len(d),bool))]:
            stages[name]=stage(d.loc[mask],x.loc[mask],out/side/name,a.pct_outliers,a.threads)
        results[side]={'source_sha256':sha,'auction_vin_overlap_exclusions':excluded,'stages':stages}
        del d,x
    summary={'status':'complete','algorithm':'OutlierTree','pct_outliers':a.pct_outliers,'markets':results,'cars_deleted':0,'weights_refitted':False,'scope':'Assessment fits training only; full cohort audit is descriptive and includes validation/test. It must not be used to filter held-out evaluation. Price is an audit input, never a pricing predictor. No descriptions, text predictions, VIN, MMR or identifiers enter detector. Native object/string categorical columns; rare model names pooled to control memory. Flags are review candidates, not confirmed errors. pct_outliers is not a deletion quota.'}
    (out/'summary.json').write_text(json.dumps(serial(summary),indent=2,allow_nan=False));print('ALL OUTLIERTREE AUDITS COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ask',required=True);p.add_argument('--bid',required=True);p.add_argument('--output',required=True);p.add_argument('--pct-outliers',type=float,default=.01);p.add_argument('--threads',type=int,default=4);p.add_argument('--index',default='models/structural-transfer/used-car-index.csv');p.add_argument('--prior',default='reports/cleanlab-rerun-20261027/summary.json');run(p.parse_args())
