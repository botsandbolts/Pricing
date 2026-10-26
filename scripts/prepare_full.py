"""Prepare 100% with bounded memory: streamed cleaning and disk-backed deduplication."""
import argparse,hashlib,json,re,sqlite3,zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from pricing.tabular import NUMERIC,CATEGORICAL
from scripts.common import clean


def clean_text(text):return clean(text)[:8000]


def prepare(source,output,workers=4):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    database=output.with_suffix('.preparation.sqlite');database.unlink(missing_ok=True)
    con=sqlite3.connect(database);con.execute('PRAGMA journal_mode=OFF');con.execute('PRAGMA synchronous=OFF');con.execute('PRAGMA cache_size=-16384');con.execute('PRAGMA temp_store=FILE')
    con.execute('CREATE TABLE candidate(priority REAL, text_hash TEXT, vin TEXT, date TEXT, record TEXT)')
    con.execute('CREATE TABLE accepted(date TEXT, record TEXT)')
    columns=list(dict.fromkeys(['price','description','VIN','posting_date']+NUMERIC+CATEGORICAL))
    rng=np.random.default_rng(42);raw=0
    pool=ProcessPoolExecutor(max_workers=workers) if workers>1 else None
    try:
        with zipfile.ZipFile(source) as z,z.open('vehicles.csv') as f:
            for c in pd.read_csv(f,usecols=columns,chunksize=5000,low_memory=False):
                raw+=len(c);c['_priority']=rng.random(len(c));c.price=pd.to_numeric(c.price,errors='coerce')
                c=c[c.price.between(1000,150000)&c.description.notna()].copy()
                c=c[c.description.astype(str).str.len()>=80]
                texts=list(pool.map(clean_text,c.description.tolist(),chunksize=100)) if pool else list(map(clean_text,c.description))
                c['text']=texts;c=c[c.text.str.lower().str.len()>=80]
                c['date']=pd.to_datetime(c.posting_date,utc=True,errors='coerce')
                records=[]
                for row in c.to_dict('records'):
                    key=hashlib.sha256(row['text'].lower().encode()).hexdigest()
                    vin='' if pd.isna(row['VIN']) else str(row['VIN']).upper().strip()
                    if not re.fullmatch('[A-HJ-NPR-Z0-9]{17}',vin):vin=''
                    date='' if pd.isna(row['date']) else row['date'].isoformat()
                    record={k:(None if pd.isna(row[k]) else row[k]) for k in ['price','text']+NUMERIC+CATEGORICAL}
                    record.update(row_id=key,date=date)
                    records.append((row['_priority'],key,vin,date,json.dumps(record,ensure_ascii=False)))
                con.executemany('INSERT INTO candidate VALUES (?,?,?,?,?)',records);con.commit()
                print(f'Scanned {raw:,} raw rows',flush=True)
        con.execute('CREATE INDEX candidate_priority ON candidate(priority)');con.commit()
        seen_text=set();seen_vin=set();retained=0;pending=[]
        for key,vin,date,record in con.execute('SELECT text_hash,vin,date,record FROM candidate ORDER BY priority'):
            if key in seen_text or (vin and vin in seen_vin):continue
            seen_text.add(key)
            if vin:seen_vin.add(vin)
            if not date:continue
            pending.append((date,record));retained+=1
            if len(pending)>=2000:con.executemany('INSERT INTO accepted VALUES (?,?)',pending);pending.clear()
        con.executemany('INSERT INTO accepted VALUES (?,?)',pending);con.commit()
        i=int(retained*.7);j=int(retained*.85)
        with output.open('w') as f:
            for at,(record,) in enumerate(con.execute('SELECT record FROM accepted ORDER BY date')):
                row=json.loads(record);row['phase']='train' if at<i else 'validation' if at<j else 'test'
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
        report={'raw_rows':raw,'fraction':1.,'sampled_raw_rows':raw,'canonical_rows':retained,'pilot':False,
            'counts':{'train':i,'validation':j-i,'test':retained-j},'preparation':'memory-bounded SQLite; deterministic random-priority text/VIN deduplication'}
        output.with_suffix('.meta.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    finally:
        if pool:pool.shutdown()
        con.close()
    database.unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',default='training-cache/full-cohort.jsonl');p.add_argument('--workers',type=int,default=4);a=p.parse_args();prepare(a.source,a.output,a.workers)
