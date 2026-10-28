"""Forward-date CV, independent source tests and explicit age-tail stress tests."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from marketbridge.depreciation import DepreciationCurve
from marketbridge.structural import IndexPanel


def metrics(y,p):
    return {'rows':len(y),'mae_usd':float(np.mean(abs(y-p))),'rmse_usd':float(np.sqrt(np.mean((y-p)**2))),'log_mae':float(np.mean(abs(np.log(y/p)))),'bias_observed_minus_predicted_usd':float(np.mean(y-p)),'spearman':float(spearmanr(y,p).statistic)}


def load(path,index):
    pieces=[]
    for d in pd.read_json(path,lines=True,chunksize=10000):
        d=d.drop(columns=['text','price_evidence','source_url'],errors='ignore');d['date']=pd.to_datetime(d.date,utc=True)
        d['age']=d.date.dt.year+(d.date.dt.dayofyear-1)/365.25-pd.to_numeric(d.year,errors='coerce')
        miles=pd.to_numeric(d.odometer,errors='coerce');price=pd.to_numeric(d.price,errors='coerce')
        good=d.age.between(0,30)&miles.between(0,2_000_000)&price.gt(0)&d.date.notna()
        d=d.loc[good].copy();d['odometer']=miles.loc[good];ratio,_=index.ratios(d.date);d['index_factor']=ratio.to_numpy();d=d[d.index_factor.notna()];pieces.append(d)
    d=pd.concat(pieces,ignore_index=True).sort_values(['date','row_id']).reset_index(drop=True)
    d['log_reference']=np.log(d.price/d.index_factor);return d


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);index=IndexPanel(a.index)
    for side,path in [('ask',a.ask),('bid',a.bid)]:
        file=out/f'{side}-curve.json'
        if file.exists():print(side,'already fitted',flush=True);continue
        d=load(path,index);tr=d[d.phase=='train'].reset_index(drop=True);va=d[d.phase=='validation'];te=d[d.phase=='test']
        print(side,'COHORT',len(d),'TRAIN',len(tr),'VALIDATION',len(va),'TEST',len(te),flush=True)
        # Only ORIGINAL TRAINING creates CV folds; chronological boundaries are whole dates.
        days=np.array(sorted(tr.date.dt.strftime('%Y-%m-%d').unique()));folds=[]
        for lo,hi in [(0.50,.67),(.67,.83),(.83,1.)]:
            cutoff=days[min(int(len(days)*lo),len(days)-2)];end=days[min(int(len(days)*hi),len(days)-1)]
            train=tr[tr.date.dt.strftime('%Y-%m-%d')<cutoff];held=tr[(tr.date.dt.strftime('%Y-%m-%d')>=cutoff)&(tr.date.dt.strftime('%Y-%m-%d')<end)]
            if len(train)>1000 and len(held)>100:folds.append((train,held))
        candidates=[]
        for kind in ['linear','piecewise','spline']:
            scores=[]
            for k,(train,held) in enumerate(folds):
                m=DepreciationCurve(kind).fit(train,train.log_reference);pred=np.exp(m.predict_log(held))*held.index_factor.to_numpy();score=metrics(held.price.to_numpy(),pred);scores.append(score)
                print(side,kind,'CV',k,score,flush=True)
            candidates.append({'kind':kind,'cv':scores,'selection_log_mae':float(np.mean([r['log_mae'] for r in scores]))})
        selected=min(candidates,key=lambda r:r['selection_log_mae'])['kind'];print(side,'SELECTED',selected,flush=True)
        model=DepreciationCurve(selected).fit(tr,tr.log_reference);assessment={}
        for label,part in [('validation',va),('test',te)]:
            pred=np.exp(model.predict_log(part))*part.index_factor.to_numpy();assessment[label]=metrics(part.price.to_numpy(),pred)
        # Source age-tail stress: no source-test labels and no claim of longitudinal ageing.
        younger=tr[tr.age<=12];older=tr[tr.age.between(12,18,inclusive='right')]
        tail=None
        if len(older)>100:
            m=DepreciationCurve(selected).fit(younger,younger.log_reference);tail=metrics(older.price.to_numpy(),np.exp(m.predict_log(older))*older.index_factor.to_numpy())
        final=DepreciationCurve(selected).fit(d,d.log_reference);final.save(file)
        report={'side':side,'source_sha256':hashlib.file_digest(open(path,'rb'),'sha256').hexdigest(),'source_dates':[str(d.date.min()),str(d.date.max())],'source_rows_in_age_scope':len(d),'original_phases':{'train':len(tr),'validation':len(va),'test':len(te)},'candidates':candidates,'selected':selected,'assessment':assessment,'age_tail_stress':tail,'age_scope':[0,30],'outcome_residual_trimming':False,'censoring':'exact observed advertised prices for ask; exact reported sale prices for bid','warning':'CV is forward-date within original training. Assessment is held source-period prices. Age-tail is a cross-sectional stress test, not observed ageing. No 2015→2026 or 2021→2026 individual-car validation is available. Deployment curve refits all in-scope rows after assessment. Existing CatBoost/MiniLM weights are untouched.'}
        (out/f'{side}-evaluation.json').write_text(json.dumps(report,indent=2));print(side,'READY',json.dumps(assessment),flush=True)
    (out/'manifest.json').write_text(json.dumps({'status':'ready','pipeline':'separate-smooth-depreciation-and-market-clock','ask_reference_date':'2021-05-01','bid_reference_date':'2015-01-01','bid_anchor_age':5.,'max_model_year':2021,'age_scope':[0,30],'forecast_months_max':6,'role':'experimental date-transport layer, not independently validated present-day car prices'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ask',default='training-cache/ask-cohort.jsonl');p.add_argument('--bid',default='training-cache/auction-cohort.jsonl');p.add_argument('--index',default='models/structural-transfer/used-car-index.csv');p.add_argument('--output',default='models/time-transport');run(p.parse_args())
