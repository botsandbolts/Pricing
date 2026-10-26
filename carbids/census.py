"""US Census ACS5 2016–2022 vintages; conservative release-aware regional joins."""
import argparse,json,os
from pathlib import Path
import numpy as np
import pandas as pd
import requests

RELEASES={2016:'2017-12-07',2017:'2018-12-06',2018:'2019-12-19',2019:'2020-12-10',2020:'2022-03-17',2021:'2022-12-08',2022:'2023-12-07'}
STATE_FIPS=dict(zip('al ak az ar ca co ct de dc fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj nm ny nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy pr'.split(),
'01 02 04 05 06 08 09 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56 72'.split()))
VARIABLES={'B19013_001':'household_income','B01003_001':'population',
 'B23025_003':'civilian_labor_force','B23025_005':'unemployed',
 'B08201_001':'households','B08201_002':'households_no_vehicle',
 'B25003_001':'occupied_housing','B25003_002':'owner_occupied',
 'B08301_001':'workers','B08301_021':'workers_from_home'}
FEATURES=['acs_household_income','acs_household_income_moe','acs_population','acs_unemployment_rate',
 'acs_no_vehicle_share','acs_homeownership_share','acs_work_from_home_share','acs_context_age_years','acs_context_available']

def parse_payload(payload,year,geography):
    if not isinstance(payload,list) or len(payload)<2:raise ValueError('Census did not return a data table; a Census API key may be required')
    d=pd.DataFrame(payload[1:],columns=payload[0])
    columns=[name+suffix for name in VARIABLES for suffix in ['E','M']]
    if not set(columns+['NAME','state']).issubset(d):raise ValueError('Unexpected Census variable schema')
    out=pd.DataFrame({'acs_name':d.NAME,'acs_vintage':year,'acs_geography':geography})
    out['acs_geo_id']=d.state.astype(str).str.zfill(2)+(d['county'].astype(str).str.zfill(3) if geography=='county' else '')
    out['acs_released_on']=RELEASES[year]
    # Release times vary; do not use any release-day sale in the historical join.
    out['acs_available_at']=(pd.Timestamp(RELEASES[year],tz='UTC')+pd.Timedelta(days=1)).isoformat()
    for code,name in VARIABLES.items():
        for suffix,label in [('E',''),('M','_moe')]:
            values=pd.to_numeric(d[code+suffix],errors='coerce')
            out['acs_'+name+label]=values.where(values>=0,np.nan)
    for name,numerator,denominator in [('unemployment_rate','unemployed','civilian_labor_force'),('no_vehicle_share','households_no_vehicle','households'),('homeownership_share','owner_occupied','occupied_housing'),('work_from_home_share','workers_from_home','workers')]:
        rate=out['acs_'+numerator]/out['acs_'+denominator].where(out['acs_'+denominator]>0)
        out['acs_'+name]=rate.where(rate.between(0,1))
    out['acs_source_url']=f'https://api.census.gov/data/{year}/acs/acs5'
    return out

def fetch_vintage(year,geography='state',cache_dir='data/census',api_key=None):
    if year not in RELEASES or geography not in {'state','county'}:raise ValueError('Use ACS5 vintages 2016–2022 and state/county geography')
    cache=Path(cache_dir);cache.mkdir(parents=True,exist_ok=True)
    file=cache/f'acs5-{year}-{geography}.json'
    if file.exists():return parse_payload(json.loads(file.read_text()),year,geography)
    params={'get':','.join(['NAME']+[v+s for v in VARIABLES for s in ['E','M']]),'for':geography+':*'}
    key=api_key or os.environ.get('CENSUS_API_KEY')
    if key:params['key']=key
    try:response=requests.get(f'https://api.census.gov/data/{year}/acs/acs5',params=params,timeout=(15,90))
    except requests.RequestException:raise RuntimeError(f'Census request failed for vintage {year}; credentials are not logged') from None
    if response.status_code!=200:raise RuntimeError(f'Census API returned HTTP {response.status_code} for vintage {year}')
    try:payload=response.json()
    except ValueError:raise RuntimeError('Census returned a non-JSON page. Its API currently requires a key for this request; set CENSUS_API_KEY privately or supply an official saved JSON response. No placeholder data was substituted.') from None
    result=parse_payload(payload,year,geography)
    file.write_text(json.dumps(payload));return result

def fetch_panel(years=range(2016,2023),geography='state',cache_dir='data/census'):
    panel=pd.concat([fetch_vintage(y,geography,cache_dir) for y in years],ignore_index=True)
    destination=Path(cache_dir)/f'acs5-2016-2022-{geography}.csv';panel.to_csv(destination,index=False)
    return panel

def join_asof(frame,panel,date_column='date'):
    """Only a previously released local estimate; no future/backward filling."""
    if date_column not in frame:raise ValueError('An explicit sale/valuation as-of date is required for Census enrichment')
    p=panel.copy();p['acs_geo_id']=p.acs_geo_id.astype(str).str.zfill(5 if p.acs_geography.eq('county').all() else 2)
    p['acs_available_at']=pd.to_datetime(p.acs_available_at,utc=True,errors='raise')
    if p.duplicated(['acs_geography','acs_geo_id','acs_available_at']).any():raise ValueError('Duplicate Census vintages for one geography and release')
    dates=pd.to_datetime(frame[date_column],utc=True,errors='coerce',format='mixed');records=[]
    for pos,row in enumerate(frame.to_dict('records')):
        fips=STATE_FIPS.get(str(row.get('state','')).lower().strip())
        county=str(row.get('county_fips','')).removesuffix('.0').zfill(5)
        if len(county)==5 and county.isdigit() and county[:2]==fips and p.acs_geography.eq('county').any():level,geo='county',county
        else:level,geo='state',fips
        subset=p[(p.acs_geography==level)&(p.acs_geo_id==geo)&(p.acs_available_at<dates.iloc[pos])] if not pd.isna(dates.iloc[pos]) else p.iloc[:0]
        if subset.empty:
            values={f:np.nan for f in FEATURES};values.update(acs_context_available=0,acs_vintage=None,acs_geo_id=geo,acs_available_at=None,acs_unavailable_reason='no_previously_released_matching_geography')
        else:
            selected=subset.sort_values('acs_available_at').iloc[-1]
            values=selected.to_dict();values['acs_context_available']=1
            values['acs_context_age_years']=(dates.iloc[pos]-selected.acs_available_at).total_seconds()/(365.25*86400)
            values['acs_available_at']=selected.acs_available_at.isoformat();values['acs_unavailable_reason']=''
        records.append(values)
    out=frame.copy()
    extra=pd.DataFrame(records,index=frame.index)
    for col in extra:out[col]=extra[col]
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('fetch');a.add_argument('--geography',choices=['state','county'],default='state');a.add_argument('--cache',default='data/census')
    a=sub.add_parser('enrich');a.add_argument('--input',required=True);a.add_argument('--panel',required=True);a.add_argument('--output',required=True);a.add_argument('--date-column',default='date')
    args=p.parse_args()
    if args.command=='fetch':print('Fetched regional rows:',len(fetch_panel(geography=args.geography,cache_dir=args.cache)))
    else:
        data=pd.read_json(args.input,lines=True) if args.input.endswith('.jsonl') else pd.read_csv(args.input)
        panel=pd.read_csv(args.panel,dtype={'acs_geo_id':str})
        result=join_asof(data,panel,args.date_column)
        if args.output.endswith('.jsonl'):result.to_json(args.output,orient='records',lines=True,date_format='iso')
        else:result.to_csv(args.output,index=False)
        print('Rows:',len(result),'with historically available Census context:',int(result.acs_context_available.sum()))
