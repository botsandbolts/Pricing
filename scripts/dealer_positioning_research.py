"""Historical dealer associations: exploratory; never a production pricing override."""
import json,sqlite3,zlib,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction import FeatureHasher
from scipy.sparse import hstack,csr_matrix
from xgboost import XGBRegressor
root=Path('data/merchant-discovery-minilm')
with sqlite3.connect(str(root/'cars.sqlite')) as db:
    cars=[json.loads(zlib.decompress(p)) for (p,) in db.execute('SELECT payload FROM cars ORDER BY position')]
d=pd.DataFrame(cars);names=json.loads((root/'dealers.json').read_text())['rows']
assert len(names)==len(d)
price=pd.to_numeric(d.price,errors='coerce').to_numpy()
assert np.all(np.isfinite(price)&(price>0))
num=pd.DataFrame({c:pd.to_numeric(d.get(c,pd.Series(np.nan,index=d.index)),errors='coerce') for c in ['year','odometer']})
num['odometer']=np.log1p(num.odometer.clip(lower=0))
records=[]
for row in cars:
    records.append({c+'='+str(row.get(c,'unknown')).lower():1 for c in ['manufacturer','model','type','fuel','transmission','condition','drive','state','title_status']})
x=hstack((csr_matrix(num.to_numpy(np.float32)),FeatureHasher(n_features=8192,input_type='dict',alternate_sign=False).transform(records)),format='csr')
groups=[]
for row in cars:
    vin=str(row.get('vin') or row.get('VIN') or '').strip().upper()
    if vin in ('NAN','NONE','UNKNOWN'):vin=''
    signature=vin if len(vin)==17 else '|'.join(str(row.get(c,'')) for c in ['manufacturer','model','year','odometer'])
    groups.append(hashlib.sha256(signature.encode()).hexdigest())
y=np.log(price);pred=np.zeros(len(y));dealerpred=np.zeros(len(y))
dealer_records=[{e['name'].casefold():e['score']/np.sqrt(len(row)) for e in row} for row in names]
dealer_x=FeatureHasher(n_features=4096,input_type='dict',alternate_sign=False).transform(dealer_records)
for tr,te in GroupKFold(5).split(x,y,groups):
    for features,out in [(x,pred),(hstack((x,dealer_x),format='csr'),dealerpred)]:
        model=XGBRegressor(n_estimators=200,max_depth=6,learning_rate=.05,reg_lambda=20,n_jobs=2,tree_method='hist')
        model.fit(features[tr],y[tr]);out[te]=model.predict(features[te])
residual=y-pred
# Fixed normal prior SD=0.10 log units; conditional normal posterior.
# This is an exploratory partially pooled residual analysis, not the installed hierarchy.
global_mean=float(np.mean(residual));sigma=max(float(np.std(residual)),.05)
buckets={}
for i,row in enumerate(names):
    if len(row)==1 and row[0]['score']>=.9:
        buckets.setdefault(row[0]['name'].casefold(),[]).append(i)
table=[]
for name,ids in buckets.items():
    unique={}
    for i in ids:unique.setdefault(groups[i],i)
    ids=list(unique.values())
    if len(ids)<50:continue
    n=len(ids);v=1/(n/sigma**2+1/.1**2)
    mean=v*np.sum(residual[ids]-global_mean)/sigma**2;sd=np.sqrt(v)
    table.append({'inferred_dealer':names[ids[0]][0]['name'],'distinct_vehicle_groups':n,
       'adjusted_association_pct':100*np.expm1(mean),'conditional_95_low_pct':100*np.expm1(mean-1.96*sd),
       'conditional_95_high_pct':100*np.expm1(mean+1.96*sd)})
out=Path('reports/dealer-positioning');out.mkdir(parents=True,exist_ok=True)
pd.DataFrame(sorted(table,key=lambda r:r['inferred_dealer'].casefold())).to_csv(out/'historical-associations.csv',index=False)
np.savez_compressed(out/'oof-residuals.npz',log_price=y,oof_log_prediction=pred,residual=residual)
report={'rows':len(y),'base_oof_mae_usd':float(np.mean(abs(price-np.exp(pred)))),
'dealer_features_oof_mae_usd':float(np.mean(abs(price-np.exp(dealerpred)))),'eligible_named_groups':len(table),
'status':'Research only; no production promotion',
'limitations':'Historical asking prices, not sales. No date control where catalogue dates are unavailable. Hashed vehicle controls; omitted condition/service variables; catalogue selected by prior cleaning. NER names unverified. Absence of extraction is unknown, not private. Conditional normal intervals exclude identity, base-model and prior uncertainty; not calibrated coverage. Not a dealer quality or overcharging ranking.',
'services':'Differences may reflect warranties, preparation, inspection, delivery, financing or other services that vary considerably between merchants.',
'method':'Five-fold grouped vehicle CV; XGBoost log squared error research baseline, dealer-free residuals. Separate confidence-weighted dealer feature comparator. Fixed normal random-effect prior SD .10; residual SD plug-in; minimum 50 distinct vehicle groups and single NER candidate score >=.90. No causal claims. Not the existing AFT/hierarchy refit.'}
(out/'method-and-metrics.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
