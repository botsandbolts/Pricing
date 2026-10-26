"""Full database preparation; retain free text and every eligible deduplicated listing."""
import argparse,hashlib,json,re,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from pricing.tabular import NUMERIC,CATEGORICAL
from scripts.common import clean


def prepare(source,output):
    columns=list(dict.fromkeys(['id','price','description','VIN','posting_date']+NUMERIC+CATEGORICAL))
    parts=[];texts=set();vins=set();raw=bad_price=bad_date=duplicates=0
    with zipfile.ZipFile(source) as z,z.open('vehicles.csv') as file:
        for c in pd.read_csv(file,usecols=columns,chunksize=10000,low_memory=False):
            raw+=len(c);c.price=pd.to_numeric(c.price,errors='coerce')
            valid=c.price.between(1000,150000);bad_price+=int((~valid).sum());c=c[valid].copy()
            c['date']=pd.to_datetime(c.posting_date,utc=True,errors='coerce');bad_date+=int(c.date.isna().sum());c=c[c.date.notna()].copy()
            c['text']=c.description.fillna('').map(lambda s:clean(s)[:8000])
            keep=[];ids=[]
            for idx,row in c.iterrows():
                key=hashlib.sha256(row.text.lower().encode()).hexdigest() if row.text else None
                vin='' if pd.isna(row.VIN) else str(row.VIN).upper().strip()
                valid_vin=bool(re.fullmatch('[A-HJ-NPR-Z0-9]{17}',vin))
                if (key and key in texts) or (valid_vin and vin in vins):duplicates+=1;continue
                if key:texts.add(key)
                if valid_vin:vins.add(vin)
                keep.append(idx);ids.append(hashlib.sha256(('listing:'+str(row.id)).encode()).hexdigest())
            part=c.loc[keep,['price','date','text']+NUMERIC+CATEGORICAL].copy();part['row_id']=ids;parts.append(part)
            print(f'Scanned {raw:,} source rows',flush=True)
    d=pd.concat(parts,ignore_index=True).sort_values('date').reset_index(drop=True)
    n=len(d);i=int(n*.7);j=int(n*.85);d['phase']=np.where(d.index<i,'train',np.where(d.index<j,'validation','test'))
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);d.to_json(p,orient='records',lines=True,date_format='iso')
    meta={'source_rows':raw,'sample_fraction':1.,'eligible_rows':n,'invalid_price_rows':bad_price,
        'invalid_date_rows':bad_date,'duplicate_rows':duplicates,'empty_descriptions_retained':int((d.text=='').sum()),
        'counts':d.phase.value_counts().to_dict(),'outlier_filter':None,'price_range_usd':[1000,150000]}
    p.with_suffix('.meta.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',default='training-cache/all-cohort.jsonl');a=p.parse_args();prepare(a.source,a.output)
