"""Join original year/date context to published RF examples; never retrain models."""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.common import clean


def enrich(ask_archive,bid_archive,root):
    root=Path(root)
    for side,archive in [('ask',ask_archive),('bid',bid_archive)]:
        folder=root/side;examples=pd.read_csv(folder/'examples.csv.gz');ids=set(examples.row_id.astype(str))
        expected=json.loads((folder/'provenance.json').read_text())['source_archive_sha256']
        if hashlib.sha256(Path(archive).read_bytes()).hexdigest()!=expected:raise ValueError('Original archive checksum mismatch')
        matches=[]
        with zipfile.ZipFile(archive) as z,z.open(next(n for n in z.namelist() if n.endswith('.csv'))) as f:
            if side=='ask':
                wanted=set(zip(examples.age.round(7),examples.odometer.fillna(-1)))
                scanned=0
                for chunk in pd.read_csv(f,usecols=['description','year','posting_date','odometer'],chunksize=5000):
                    scanned+=len(chunk)
                    dates=pd.to_datetime(chunk.posting_date,utc=True,format='mixed',errors='coerce')
                    ages=(dates.dt.year+(dates.dt.dayofyear-1)/365.25-pd.to_numeric(chunk.year,errors='coerce')).clip(lower=0).round(7)
                    candidate=pd.Series([(a,m) in wanted for a,m in zip(ages,chunk.odometer.fillna(-1))],index=chunk.index)
                    chunk=chunk[candidate]
                    if not len(chunk):continue
                    if scanned%50000==0:print('Scanned',scanned,flush=True)
                    keys=chunk.description.fillna('').map(lambda t:hashlib.sha256(clean(t)[:8000].lower().encode()).hexdigest())
                    part=chunk[keys.isin(ids)].copy();part['row_id']=keys.loc[part.index]
                    part=part.rename(columns={'posting_date':'source_date'})
                    if len(part):matches.append(part[['row_id','year','source_date','odometer']])
            else:
                malformed=[]
                raw=pd.read_csv(f,engine='python',on_bad_lines=lambda row:malformed.append(row))
                keys=pd.Series(np.arange(len(raw)).astype(str)).map(lambda s:hashlib.sha256(('wholesale_2015:'+s).encode()).hexdigest()[:24])
                part=raw[keys.isin(ids)].copy();part['row_id']=keys.loc[part.index]
                part['source_date']=pd.to_datetime(part.saledate.astype(str).str.replace(r' \([^)]*\)$','',regex=True),format='%a %b %d %Y %H:%M:%S GMT%z',utc=True,errors='raise').astype(str)
                matches.append(part[['row_id','year','source_date','odometer']])
        context=pd.concat(matches).drop_duplicates()
        # Duplicate asking text can appear on several dates. Match its exact recorded age/mileage.
        joined=examples.drop(columns=['year','source_date'],errors='ignore').merge(context,on='row_id',suffixes=('','_source'))
        dates=pd.to_datetime(joined.source_date,utc=True,format='mixed')
        computed_age=(dates.dt.year+(dates.dt.dayofyear-1)/365.25-pd.to_numeric(joined.year)).clip(lower=0)
        mileage_equal=np.isclose(joined.odometer,joined.odometer_source,equal_nan=True)
        joined=joined[np.isclose(computed_age,joined.age,atol=1e-7)&mileage_equal].drop_duplicates('row_id')
        if len(joined)!=len(examples):raise ValueError(f'{side}: original context matching incomplete: {len(joined)}/{len(examples)}')
        joined=joined.set_index('row_id').loc[examples.row_id].reset_index().drop(columns=['odometer_source'])
        joined.to_csv(folder/'examples.csv.gz',index=False,float_format='%.12g')
        print(side,len(joined),'exact original date/year/mileage matches',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ask',default='training-cache/ask.zip');p.add_argument('--bid',default='training-cache/bid.zip');p.add_argument('--root',default='models/forest-original');a=p.parse_args();enrich(a.ask,a.bid,a.root)
