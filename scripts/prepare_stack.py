"""Recover structured columns while preserving the prior canonical chronological cohort."""
import argparse
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.common import clean
from pricing.tabular import NUMERIC, CATEGORICAL


def prepare(source, output, full=False, fraction=None):
    if fraction is not None and not 0 < fraction <= 1:
        raise ValueError('fraction must be in (0,1]')
    rng = np.random.default_rng(42)
    raw_rows = 0
    raw_parts = []
    columns = list(dict.fromkeys(['price','description','VIN','posting_date']+NUMERIC+CATEGORICAL))
    sample = None
    with zipfile.ZipFile(source) as z, z.open('vehicles.csv') as f:
        for c in pd.read_csv(f, chunksize=10000, usecols=columns, low_memory=False):
            raw_rows += len(c)
            if fraction is not None:
                # Sample the original database before any eligibility/price filtering.
                c['_priority'] = rng.random(len(c))
                raw_parts.append(c)
            else:
                c.price = pd.to_numeric(c.price, errors='coerce')
                c = c[c.price.between(1000,150000)&c.description.notna()].copy()
                c = c[c.description.astype(str).str.len()>=80]
                c['_priority'] = rng.random(len(c))
                sample = pd.concat([sample,c]).nsmallest(12000,'_priority')
    sampled_raw = None
    if fraction is not None:
        sampled_raw = int(raw_rows*fraction)
        sample = pd.concat(raw_parts).nsmallest(sampled_raw,'_priority')
        sample.price = pd.to_numeric(sample.price,errors='coerce')
        sample = sample[sample.price.between(1000,150000)&sample.description.notna()].copy()
        sample = sample[sample.description.astype(str).str.len()>=80]
    d = sample.sort_values('_priority')
    d['canonical'] = d.description.map(lambda s:clean(s,lower=True)[:8000])
    d = d[d.canonical.str.len()>=80].drop_duplicates('canonical')
    v = d.VIN.fillna('').astype(str).str.upper().str.strip()
    # Only drop repeated valid VINs (missing values are not entity IDs).
    d = d.loc[~(v.str.fullmatch('[A-HJ-NPR-Z0-9]{17}')&v.duplicated())]
    d['date'] = pd.to_datetime(d.posting_date,utc=True,errors='coerce')
    d = d.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    n = len(d); i = int(n*.7); j = int(n*.85)
    groups=[]
    for phase, frame, count in [('train',d.iloc[:i],1200),('validation',d.iloc[i:j],300),('test',d.iloc[j:],300)]:
        if not full and fraction is None:
            frame=frame.sample(n=min(count,len(frame)),random_state=217).sort_values('date')
        frame=frame.copy();frame['phase']=phase;groups.append(frame)
    result=pd.concat(groups).reset_index(drop=True)
    result['text']=result.description.map(lambda t:clean(t)[:8000])
    # Canonical text/VIN hashes protect against accidental split leakage without publishing seller text.
    import hashlib
    result['row_id']=result.canonical.map(lambda t:hashlib.sha256(t.encode()).hexdigest())
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    result[['row_id','price','text','date','phase']+NUMERIC+CATEGORICAL].to_json(output,orient='records',lines=True,date_format='iso')
    report={'canonical_rows':n,'pilot':not full and fraction is None,'raw_rows':raw_rows,'fraction':fraction,'sampled_raw_rows':sampled_raw,'counts':result.phase.value_counts().to_dict()}
    output.with_suffix('.meta.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',default='training-cache/stack-cohort.jsonl');p.add_argument('--full',action='store_true');p.add_argument('--fraction',type=float,help='Fraction of all raw database rows, before eligibility filtering');a=p.parse_args();prepare(a.source,a.output,a.full,a.fraction)
