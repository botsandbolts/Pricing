"""Fetch official BLS used-car CPI. No API key or credentials are persisted."""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import requests
SERIES='CUUR0000SETA02'
def fetch(output,start_year=2013,end_year=None):
    end_year=end_year or datetime.now(timezone.utc).year
    if start_year>end_year:raise ValueError('Invalid year range')
    rows=[];urls=[]
    for first in range(start_year,end_year+1,10):
        last=min(first+9,end_year)
        response=requests.get(f'https://api.bls.gov/publicAPI/v2/timeseries/data/{SERIES}',params={'startyear':first,'endyear':last},timeout=60)
        response.raise_for_status();data=response.json()
        if data.get('status')!='REQUEST_SUCCEEDED':raise ValueError('BLS request failed: '+str(data.get('message')))
        urls.append(response.url)
        for series in data['Results']['series']:
            if series['seriesID']!=SERIES:raise ValueError('Unexpected BLS series')
            for r in series['data']:
                if r['period']=='M13':continue
                value=float(r['value'])
                if value<=0:raise ValueError('Non-positive index')
                rows.append({'month':r['year']+'-'+r['period'][1:],'value':value})
    table=pd.DataFrame(rows).drop_duplicates('month').sort_values('month')
    if table.empty:raise ValueError('No index values returned')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);table.to_csv(output,index=False)
    meta={'series':SERIES,'source_urls':urls,'retrieved_at':datetime.now(timezone.utc).isoformat(),
        'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'rows':len(table),
        'vintage':'Current retrieved snapshot; not an ALFRED real-time vintage archive','lag_months_used_by_model':2,
        'scope':'Aggregate retail used-car CPI used as an assumed wholesale market anchor'}
    output.with_suffix('.meta.json').write_text(json.dumps(meta,indent=2));return meta
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='models/structural-transfer/used-car-index.csv');p.add_argument('--start-year',type=int,default=2013);p.add_argument('--end-year',type=int);a=p.parse_args()
    print(json.dumps(fetch(a.output,a.start_year,a.end_year),indent=2))
