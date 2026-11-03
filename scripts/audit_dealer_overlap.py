"""Inventory overlap audit; do not interpret residuals outside shared support."""
import json,sqlite3,zlib,collections
from pathlib import Path
import pandas as pd
root=Path('data/merchant-discovery-minilm')
names=json.loads((root/'dealers.json').read_text())['rows']
with sqlite3.connect(str(root/'cars.sqlite')) as db:
    cars=[json.loads(zlib.decompress(p)) for p, in db.execute('SELECT payload FROM cars ORDER BY position')]
assert len(cars)==len(names)
def audit(threshold):
    records=[]
    for car,candidates in zip(cars,names):
        if len(candidates)!=1 or candidates[0]['score']<threshold:continue
        def number(key):
            try:return float(car[key])
            except (ValueError,KeyError,TypeError):return float('nan')
        date=car.get('posting_date') or car.get('listing_date')
        # Never mistake a model year or synthetic reference date for a listing date.
        dt=pd.to_datetime(date,errors='coerce',utc=True) if date else pd.NaT
        records.append({'dealer':candidates[0]['name'].casefold(),'model':str(car.get('manufacturer',''))+'|'+str(car.get('model','')),
          'trim':car.get('trim'),'year':number('year'),'miles':number('odometer'),'state':car.get('state'),
          'listing_month':None if pd.isna(dt) else str(dt.to_period('M'))})
    d=pd.DataFrame(records);d['age_band']=d.year.floordiv(5);d['mileage_band']=d.miles.floordiv(25000)
    required=['model','trim','age_band','mileage_band','state','listing_month']
    complete=d.dropna(subset=required).copy()
    # Joint support, not merely separate univariate ranges.
    counts=complete.groupby(required+['dealer']).size().rename('n').reset_index()
    supported=counts[counts.n>=3].groupby(required).dealer.nunique()
    keys=set(supported[supported>=2].index)
    complete['shared']=complete[required].apply(tuple,axis=1).isin(keys)
    summary=[]
    for dealer,g in d.groupby('dealer'):
        shared=complete[(complete.dealer==dealer)&complete.shared]
        summary.append({'dealer':dealer,'input_adverts':len(g),'shared_support_adverts':len(shared),
          'retained_fraction':len(shared)/len(g),'status':'Eligible for subsequent within-support refit' if len(shared)>=50 else 'Suppressed: insufficient verified joint support'})
    distributions={c: d.groupby(['dealer',c],dropna=False).size().rename('n').reset_index().astype(str).to_dict('records') for c in ['model','trim','age_band','mileage_band','state','listing_month']}
    return {'threshold':threshold,'summary':summary,'distributions':distributions,
       'missing_counts':{c:int(d[c].isna().sum()) for c in required},
       'catalogue_fields':sorted(cars[0]),
       'total_selected_adverts':len(d),'total_shared_support_adverts':int(complete.shared.sum()),
       'refit_status':'Blocked: no eligible shared-support merchant groups' if not any(s['shared_support_adverts']>=50 for s in summary) else 'Requires within-support refit before estimates can be displayed',
       'time_holdout':'Not evaluated: requires date-ordered baseline retraining; random-fold residuals cannot substitute',
       'estimates':'Suppressed until within-support refit and uncertainty evaluation complete'}
out=Path('reports/dealer-overlap');out.mkdir(parents=True,exist_ok=True)
report={'method':'Exact model/trim/state/month with 5-year and 25000-mile bands; >=3 adverts per merchant and >=2 merchants per cell. Adverts are not verified unique vehicles.',
'primary':audit(.9),'match_sensitivity':audit(.8),
'limitations':'Strict diagnostic support definition, not causal identification. No fuzzy identity merges. Original candidate errors remain possible. No supported pricing effects are asserted by this audit.'}
(out/'audit.json').write_text(json.dumps(report,default=str))
print(json.dumps({k:v for k,v in report['primary'].items() if k!='distributions'},default=str))
