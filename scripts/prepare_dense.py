"""Reproduce the canonical cohort, then choose a deterministic pilot within each date split."""
from pathlib import Path
import argparse,zipfile,json
import pandas as pd,numpy as np
from common import clean
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--full',action='store_true');a=p.parse_args();root=Path(__file__).resolve().parents[1];cache=root/'training-cache';cache.mkdir(exist_ok=True);rng=np.random.default_rng(42);parts=[]
with zipfile.ZipFile(a.source) as z:
 with z.open('vehicles.csv') as f:
  for c in pd.read_csv(f,chunksize=10000,usecols=['price','description','VIN','posting_date','manufacturer','model'],low_memory=False):
   c.price=pd.to_numeric(c.price,errors='coerce');c=c[c.price.between(1000,150000)&c.description.notna()].copy();c=c[c.description.astype(str).str.len()>=80];c['_priority']=rng.random(len(c));parts=[pd.concat(parts+[c]).nsmallest(12000,'_priority')]
d=parts[0].sort_values('_priority');d['canonical']=d.description.map(lambda s:clean(s,lower=True));d=d[d.canonical.str.len()>=80].drop_duplicates('canonical');v=d.VIN.fillna('').astype(str).str.upper().str.strip();d=d.loc[~(v.str.fullmatch('[A-HJ-NPR-Z0-9]{17}')&v.duplicated())];d['date']=pd.to_datetime(d.posting_date,utc=True,errors='coerce');d=d.dropna(subset=['date']).sort_values('date').reset_index(drop=True);assert len(d)==10892
n=len(d);i=int(n*.7);j=int(n*.85);groups=[]
for phase,frame,count in [('train',d.iloc[:i],1200),('validation',d.iloc[i:j],300),('test',d.iloc[j:],300)]:
 if not a.full:frame=frame.sample(n=min(count,len(frame)),random_state=217).sort_values('date')
 frame=frame.copy();frame['phase']=phase;groups.append(frame)
d=pd.concat(groups);d['text']=d.description.map(clean);d['make']=d.manufacturer.fillna('unknown').map(clean);d['model_text']=d.model.fillna('unknown').map(clean);d[['price','text','make','model_text','date','phase']].to_json(cache/'dense-cohort.jsonl',orient='records',lines=True,date_format='iso',force_ascii=False)
print(json.dumps({'pilot':not a.full,'full_cohort':n,'rows':len(d),'counts':d.phase.value_counts().to_dict()}),flush=True)
