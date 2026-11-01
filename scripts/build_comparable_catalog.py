"""Small public training-reference catalogue, without asking targets or VINs."""
import gzip,hashlib,json,re
from pathlib import Path
import pandas as pd
from scripts.common import clean
from pricing.comparable_retrieval import valid

def build(source,output,count=5000):
    selected=[]
    for chunk in pd.read_json(source,lines=True,chunksize=5000):
        chunk=chunk.loc[chunk.phase.eq('train')].copy()
        chunk['_rank']=chunk.row_id.astype(str).map(lambda x:hashlib.sha256(('comparable-catalog-v1:'+x).encode()).hexdigest())
        selected.extend(chunk.sort_values('_rank').head(count).to_dict('records'))
        selected=sorted(selected,key=lambda x:x['_rank'])[:count]
    rows=[]
    for r in selected:
        row={k:r.get(k) for k in ['manufacturer','model','year','odometer','type','fuel','transmission','drive','title_status','state','condition','cylinders','paint_color']}
        row={k:None if pd.isna(v) else v for k,v in row.items()}
        if not valid(row,allow_missing_miles=True):continue
        text=clean(r.get('text',''))
        text=re.sub(r'(?i)\b[A-HJ-NPR-Z0-9]{17}\b','[VIN removed]',text)
        text=re.sub(r'(?<!\d)(?:\+?1[ .-]*)?(?:\(\d{3}\)|\d{3})[ .-]*\d{3}[ .-]*\d{4}','[phone removed]',text)
        row.update(id=r['row_id'],description=text[:1200],description_is_excerpt=len(text)>1200,price_kind='Historical asking advert · price omitted',reference_period='April–May 2021')
        rows.append(row)
    if len(rows)<5:raise ValueError('Insufficient catalogue support for comparable retrieval')
    payload={'rows':rows,'source':'https://zenodo.org/records/10457828','creator':'Tom','license':'CC-BY-4.0','selection':'Deterministic hash sample from the original canonical training partition, up to 5,000 vehicles; validity checks may reduce this. Asking prices omitted, common contact/VIN strings redacted, description excerpts capped at 1,200 characters. Historical seller claims, not verified facts.'}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(output,'wt') as f:json.dump(payload,f,ensure_ascii=False,allow_nan=False)
    print('Published',len(rows),'historical asking references')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',default='data/comparable-catalog/ask.json.gz');a=p.parse_args();build(a.source,a.output)
