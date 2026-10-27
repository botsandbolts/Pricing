"""Build a reproducible diverse UI library from real training listings, without targets."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import pandas as pd
from scripts.common import clean
from pricing.tabular import NUMERIC, CATEGORICAL


def build(source, output='data/description-examples.json', count=400):
    d=pd.concat(pd.read_json(source,lines=True,chunksize=5000),ignore_index=True)
    # UI examples are from training, so displaying them does not expose audit targets.
    d=d[d.phase=='train'].copy()
    d=d[d.manufacturer.notna() & d.model.notna() & d.year.between(1900,2027) & d.odometer.between(0,2000000)]
    d['_rank']=d.row_id.map(lambda s:hashlib.sha256(('studio-examples-v1'+s).encode()).hexdigest())
    d=d.sort_values('_rank')
    # Round-robin across manufacturers rather than 400 near-identical dealer ads.
    d['_make_rank']=d.groupby('manufacturer').cumcount()
    d=d.sort_values(['_make_rank','_rank'])
    examples=[]
    options={
        'condition':['unknown','excellent','good','fair','like new','new','salvage'],
        'fuel':['gas','diesel','hybrid','electric','other','unknown'],
        'transmission':['automatic','manual','other','unknown'],
        'drive':['unknown','fwd','rwd','4wd'],
        'type':['sedan','SUV','truck','pickup','hatchback','coupe','wagon','van','convertible','other','unknown'],
        'title_status':['clean','rebuilt','salvage','lien','missing','parts only','unknown'],
        'cylinders':['unknown','3 cylinders','4 cylinders','5 cylinders','6 cylinders','8 cylinders','10 cylinders','12 cylinders','other'],
        'paint_color':['unknown','white','black','silver','grey','blue','red','green','brown','yellow','orange','purple','custom'],
    }
    for row in d.to_dict('records'):
        text=clean(row['text'])
        text=re.sub(r'(?i)\b[A-HJ-NPR-Z0-9]{17}\b','[VIN removed]',text)
        text=re.sub(r'(?<!\d)(?:\+?1[ .-]*)?(?:\(\d{3}\)|\d{3})[ .-]*\d{3}[ .-]*\d{4}','[phone removed]',text)
        text=re.sub(r'(?i)\b(?:www\.)\S+|\b\S+@\S+\b','[contact removed]',text)
        # Only reasonably readable full descriptions; do not silently truncate examples.
        if not 150<=len(text)<=5000:continue
        fields={k:(None if pd.isna(row[k]) else row[k]) for k in NUMERIC+CATEGORICAL}
        for key,values in options.items():
            value=str(fields[key] or 'unknown').strip().lower()
            fields[key]='SUV' if key=='type' and value=='suv' else value if value in values else 'unknown'
        fields['year']=int(fields['year']);fields['odometer']=int(fields['odometer'])
        fields['state']=str(fields['state'] or '')[:2]
        label=f"{fields['year']} {fields['manufacturer'].title()} {fields['model']} · {fields['odometer']:,} mi · {text[:95]}…"
        examples.append({'id':row['row_id'],'label':label,'fields':fields,'text':text})
        if len(examples)==count:break
    if len(examples)!=count:raise ValueError('Too few eligible examples')
    payload={'creator':'Tom','license':'CC-BY-4.0','license_url':'https://creativecommons.org/licenses/by/4.0/','modifications':'Selected training listings; monetary and common contact/VIN expressions redacted; formatting normalized.', 'source':'https://zenodo.org/records/10457828','dataset':'Used Cars Dataset / vehicles.csv; April–May 2021 historical asking listings',
        'selection':'Deterministic manufacturer-balanced selection from original training partition only. Prices omitted, monetary expressions and common contact/VIN formats redacted. Texts are historical seller claims, not verified vehicle facts.',
        'examples':examples}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    print(f'{len(examples)} examples, {len(set(x["fields"]["manufacturer"] for x in examples))} manufacturers',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',default='data/description-examples.json');a=p.parse_args();build(a.source,a.output)
