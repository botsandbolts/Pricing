"""Descriptive full-cohort Isolation Forest with sklearn one-hot categoricals."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from scipy import sparse
from pricing.model_suite import load_cohort,predictors
from marketbridge.structural import IndexPanel,known_pricing_vins
from scripts.audit_outliertree import serial


def fit_audit(d,x,out,ot_path):
    out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    x=x.dropna(axis=1,how='all');cats=[c for c in x if not pd.api.types.is_numeric_dtype(x[c])];nums=[c for c in x if c not in cats]
    pre=ColumnTransformer([('numeric',SimpleImputer(strategy='median'),nums),('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=True,dtype=np.float32),cats)],sparse_threshold=1.)
    encoded=pre.fit_transform(x);encoded=sparse.csc_matrix(encoded,dtype=np.float32)
    settings=dict(n_estimators=100,max_samples=256,contamination='auto',max_features=1.,bootstrap=False,n_jobs=4,random_state=42)
    print('FIT',out,'shape',encoded.shape,'nnz',encoded.nnz,flush=True)
    model=IsolationForest(**settings).fit(encoded)
    # score_samples: smaller is more anomalous. decision = score_samples - offset.
    scores=model.score_samples(encoded);decision=scores-model.offset_;flag=decision<0
    ids=d.row_id.astype(str);ot=pd.read_csv(ot_path,dtype={'row_id':str});ot_ids=set(ot.row_id);if_ids=set(ids.loc[flag]);both=ot_ids&if_ids
    rows=d[[c for c in ['row_id','phase','price','manufacturer','model','year','odometer'] if c in d]].copy()
    rows['score_samples']=scores;rows['decision_function']=decision;rows['flagged']=flag
    rows.to_csv(out/'scores.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    q=[0,.01,.05,.25,.5,.75,.95,.99,1.]
    phase={}
    for p in sorted(d.phase.unique()):
        m=(d.phase==p).to_numpy();phase[p]={'rows':int(m.sum()),'flagged':int(flag[m].sum()),'flagged_fraction':float(flag[m].mean())}
    report={'input_rows':len(d),'flagged_rows':int(flag.sum()),'flagged_fraction':float(flag.mean()),'settings':settings,'offset':float(model.offset_),'score_samples_quantiles':dict(zip(map(str,q),map(float,np.quantile(scores,q)))),'decision_function_quantiles':dict(zip(map(str,q),map(float,np.quantile(decision,q)))),'numeric_columns':nums,'categorical_columns':cats,'one_hot_categories':{c:len(levels) for c,levels in zip(cats,pre.named_transformers_['categorical'].categories_)},'encoded_columns':encoded.shape[1],'encoded_nonzero':encoded.nnz,'phase_statistics':phase,'outliertree_comparison':{'outliertree_flags':len(ot_ids),'both_flags':len(both),'isolation_only':len(if_ids-ot_ids),'outliertree_only':len(ot_ids-if_ids),'jaccard':len(both)/len(ot_ids|if_ids) if ot_ids|if_ids else None},'most_anomalous_examples':rows.iloc[np.argsort(scores)[:10]].to_dict('records'),'elapsed_seconds':time.monotonic()-start,'cars_deleted':0,'weights_refitted':False,'interpretation':'Lower score_samples means more anomalous. Negative decision_function is flagged. Auto uses offset=-0.5; it is not a removal fraction or calibrated error probability. No labelled anomaly truth exists, so precision/recall or accuracy cannot be reported. Full-cohort descriptive audit, never a held-out filtering rule. No rare model pooling; each distinct category gets its own one-hot column.'}
    (out/'report.json').write_text(json.dumps(serial(report),indent=2,allow_nan=False));print('COMPLETE',out,'flags',int(flag.sum()),flush=True)
    return report


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Use fresh output')
    known=known_pricing_vins(a.ask);panel=IndexPanel(a.index);prior=json.loads(Path(a.prior).read_text());result={}
    for side,path in [('ask',a.ask),('bid',a.bid)]:
        sha=hashlib.file_digest(open(path,'rb'),'sha256').hexdigest();assert sha==prior['markets'][side]['source_sha256']
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
        r=fit_audit(d,x,out/side,Path(a.outliertree)/side/'full_cohort'/'flagged-cars.csv.gz');r['source_sha256']=sha;r['auction_vin_overlap_exclusions']=excluded
        result[side]=r;del d,x
    (out/'summary.json').write_text(json.dumps(serial({'status':'complete','algorithm':'sklearn OneHotEncoder + IsolationForest','markets':result,'cars_deleted':0,'weights_refitted':False}),indent=2,allow_nan=False))
    print('ALL ISOLATION AUDITS COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ask',required=True);p.add_argument('--bid',required=True);p.add_argument('--output',required=True);p.add_argument('--outliertree',default='reports/outliertree-pct0099');p.add_argument('--index',default='models/structural-transfer/used-car-index.csv');p.add_argument('--prior',default='reports/cleanlab-rerun-20261027/summary.json');run(p.parse_args())
