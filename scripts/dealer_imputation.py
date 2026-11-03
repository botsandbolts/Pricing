"""Five completed numeric datasets; never fabricate trim, identity, state or dates."""
import json,sqlite3,zlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import TruncatedSVD
from pricing.xgb_imputation import completed_numeric
root=Path('data/merchant-discovery-minilm')
with sqlite3.connect(str(root/'cars.sqlite')) as db:
    cars=[json.loads(zlib.decompress(p)) for p, in db.execute('SELECT payload FROM cars ORDER BY position')]
names=json.loads((root/'dealers.json').read_text())['rows']
d=pd.DataFrame(cars)
numeric=np.column_stack((pd.to_numeric(d.year,errors='coerce'),np.log1p(pd.to_numeric(d.odometer,errors='coerce').clip(lower=0))))
numeric[~np.isfinite(numeric)]=np.nan
records=[{c+'='+str(r.get(c,'unknown')):1 for c in ['manufacturer','model','type','fuel','transmission','condition','drive']} for r in cars]
x=FeatureHasher(n_features=2048,input_type='dict',alternate_sign=False).transform(records)
context=TruncatedSVD(n_components=16,random_state=42).fit_transform(x)
ids=[i for i,r in enumerate(names) if len(r)==1 and r[0]['score']>=.9]
base=pd.DataFrame([{'dealer':names[i][0]['name'].casefold(),'model':str(cars[i].get('manufacturer'))+'|'+str(cars[i].get('model')),
  'state':cars[i].get('state'),'date':cars[i].get('posting_date') or cars[i].get('listing_date') or cars[i].get('date') or cars[i].get('observed_date')} for i in ids])
base['month']=pd.to_datetime(base.date,errors='coerce',utc=True).dt.strftime('%Y-%m')
runs=[]
for seed in [71,83,97,109,127]:
    completed=completed_numeric(numeric,context,seed)
    frame=base.copy();frame['year_band']=np.floor(np.clip(completed[ids,0],1900,2026)/5)
    frame['mileage_band']=np.floor(np.expm1(np.clip(completed[ids,1],0,np.log1p(2000000)))/25000)
    fields=['model','state','month','year_band','mileage_band'];clean=frame.dropna(subset=fields)
    counts=clean.groupby(fields+['dealer']).size().rename('n').reset_index()
    support=counts[counts.n>=3].groupby(fields).dealer.nunique()
    keys=set(support[support>=2].index)
    shared=clean[clean[fields].apply(tuple,axis=1).isin(keys)]
    runs.append(shared.groupby('dealer').size().to_dict())
    print('Completed imputation',seed,'shared adverts',len(shared),flush=True)
rows=[]
for dealer,g in base.groupby('dealer'):
    n=[run.get(dealer,0) for run in runs]
    rows.append({'dealer':dealer,'selected_adverts':len(g),'shared_adverts_min':min(n),'shared_adverts_max':max(n),
      'shared_adverts_mean':float(np.mean(n)),'supported_in_all_five':min(n)>=50})
report={'imputations':5,'missing_year':int(np.isnan(numeric[:,0]).sum()),'missing_mileage':int(np.isnan(numeric[:,1]).sum()),
 'summary':rows,'method':'Three bootstrap XGBoost regressors plus held-out residual RMSE supply Gaussian draws in IterativeImputer; three iterations, five seeds; at most 20000 observed fitting rows per target.',
 'limitations':'Approximate multiple-imputation sensitivity, not calibrated Bayesian posterior draws. MAR and Gaussian homoscedastic residual assumptions may fail. Model-level overlap does not control trim. Price, merchant identities, locations and dates do not enter imputation. Observed values remain fixed. Missing trim is not generated. No Rubin-pooled dealer effect or temporal holdout is claimed.',
 'estimate_status':'Named effects remain suppressed pending within-support price refitting, uncertainty and temporal checks'}
out=Path('reports/dealer-overlap');out.mkdir(parents=True,exist_ok=True)
(out/'imputation.json').write_text(json.dumps(report))
print(json.dumps({k:v for k,v in report.items() if k!='summary'}))
