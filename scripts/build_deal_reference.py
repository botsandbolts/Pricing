"""Create a deterministic public historical gallery without VINs or seller names."""
import argparse,hashlib,json,zipfile,gzip
from pathlib import Path
import pandas as pd

def build(source,output):
 with zipfile.ZipFile(source) as z,z.open(next(n for n in z.namelist() if n.endswith('.csv'))) as f:
  raw=pd.read_csv(f,engine='python',on_bad_lines='skip')
 raw['source_row']=range(len(raw))
 raw['date']=pd.to_datetime(raw.saledate.astype(str).str.replace(r' \([^)]*\)$','',regex=True),format='%a %b %d %Y %H:%M:%S GMT%z',utc=True,errors='coerce')
 for c in ['year','odometer','sellingprice']:raw[c]=pd.to_numeric(raw[c],errors='coerce')
 raw=raw[raw.date.notna() & raw.year.between(1980,2015) & raw.odometer.between(0,300000) & raw.sellingprice.between(500,200000) & raw.make.notna() & raw.model.notna()].copy()
 raw=raw.sort_values('date').drop_duplicates('vin',keep='last')
 raw['rank']=raw.source_row.map(lambda n:hashlib.sha256(f'deal-reference-v1:{n}'.encode()).hexdigest())
 raw=raw.sort_values('rank').groupby(['make','model'],sort=True).head(8).sort_values('rank').head(4000)
 cases=[]
 for r in raw.to_dict('records'):
  cases.append({'id':'auction-'+str(r['source_row']),'year':int(r['year']),'manufacturer':str(r['make']).lower(),'model':str(r['model'])[:80], 'odometer':int(r['odometer']),'transmission':str(r.get('transmission') or 'unknown').lower(),'state':str(r.get('state') or '').lower(),'type':str(r.get('body') or 'unknown'),'price':float(r['sellingprice']),'sale_date':r['date'].date().isoformat(),'currency':'USD','price_kind':'provider-reported wholesale auction sale','settlement_verified':False})
 payload={'source':'https://www.kaggle.com/datasets/tunguz/used-car-auction-prices','source_license':'CC0: Public Domain (official Kaggle API metadata, checked 2026-10-29)','creator':'Bojan Tunguz / upstream public auction records','source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'selection':'Deterministic hashed sample, up to eight rows per make/model, last sale per source VIN. Plausibility limits: 1980–2015; 0–300k miles; $500–200k. May overlap training; not representative.','modifications':'VIN and seller names removed; only normalized vehicle and sale facts retained. No current-price conversion.','purpose':'Historical evidence and entertainment; not an independent accuracy benchmark or verified settlement record.','cases':cases}
 Path(output).write_bytes(gzip.compress(json.dumps(payload,separators=(",",":")).encode(),mtime=0));print('Historical reference cases:',len(cases))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',default='data/deal-reference.json.gz');a=p.parse_args();build(a.source,a.output)
