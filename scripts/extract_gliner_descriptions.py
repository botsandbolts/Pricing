"""Fifty source-order partitions; grounded extraction stored as JSON and SQLite."""
import argparse,json,hashlib,sqlite3,datetime
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--shard',type=int);p.add_argument('--merge',action='store_true');a=p.parse_args()
schema={'vehicle':['make::str','model::str','trim::str','year::str','mileage::str','engine::str','transmission::str','fuel::str','drivetrain::str','body::str','colour::str','equipment','ownership_history','title_status::str','service_history','accident_history::str'],
'repair':['component::str','action::str','time::str','mileage::str','evidence::str'],
'fault':['component::str','condition::str','negation_or_uncertainty::str','evidence::str']}
out=Path('gliner-output');out.mkdir(exist_ok=True)
if a.merge:
    dest=sqlite3.connect(out/'descriptions.sqlite')
    dest.execute('CREATE TABLE descriptions(row_id TEXT PRIMARY KEY, description TEXT NOT NULL, extraction_json TEXT NOT NULL CHECK(json_valid(extraction_json)))')
    counts=[];sources=set()
    for i in range(50):
        root=Path('gliner-parts')/f'gliner-{i}'
        receipt=json.loads((root/f'{i:02d}.receipt.json').read_text());sources.add(receipt['source_sha256'])
        src=sqlite3.connect(root/f'{i:02d}.sqlite')
        rows=src.execute('SELECT row_id,description,extraction_json FROM descriptions')
        dest.executemany('INSERT INTO descriptions VALUES(?,?,?)',rows);counts.append(receipt['rows'])
    if len(sources)!=1:raise ValueError('Mixed cohorts')
    expected=sum(1 for _ in open('training-cache/full-cohort.jsonl'))
    if sum(counts)!=expected or dest.execute('SELECT count(*) FROM descriptions').fetchone()[0]!=expected:raise ValueError('Missing descriptions')
    dest.commit();dest.close()
    (out/'manifest.json').write_text(json.dumps({'rows':expected,'chunks':50,'source_sha256':next(iter(sources)),'schema':schema},indent=2))
else:
    if a.shard is None or not 0<=a.shard<50:raise ValueError('Shard 0..49 required')
    import torch
    from gliner2 import AutoExtractor
    from huggingface_hub import snapshot_download
    torch.set_num_threads(2)
    manifest=json.loads(Path('training-cache/gliner-model.json').read_text())
    model=AutoExtractor.from_pretrained(snapshot_download(manifest['model'],revision=manifest['revision']))
    db=sqlite3.connect(out/f'{a.shard:02d}.sqlite')
    db.execute('CREATE TABLE descriptions(row_id TEXT PRIMARY KEY, description TEXT NOT NULL, extraction_json TEXT NOT NULL CHECK(json_valid(extraction_json)))')
    count=0
    with open('training-cache/full-cohort.jsonl') as source,open(out/f'{a.shard:02d}.jsonl','w') as output:
        for pos,line in enumerate(source):
            if pos%50!=a.shard:continue
            row=json.loads(line);text=row.get('text') or ''
            if not isinstance(text,str):text=''
            result=model.extract_json(text,schema) if text.strip() else {}
            if not isinstance(result,dict):raise ValueError('Non-object extraction')
            payload={'schema_version':1,'model':manifest,'seller_claims_unverified':True,'extraction':result,
                     'extracted_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False)
            db.execute('INSERT INTO descriptions VALUES(?,?,?)',(str(row['row_id']),text,encoded))
            output.write(json.dumps({'row_id':str(row['row_id']),'description':text,**payload},ensure_ascii=False,allow_nan=False)+'\n')
            count+=1
            if count%100==0:db.commit();print(a.shard,count,flush=True)
    db.commit();db.close()
    with open('training-cache/full-cohort.jsonl','rb') as source:digest=hashlib.file_digest(source,'sha256').hexdigest()
    (out/f'{a.shard:02d}.receipt.json').write_text(json.dumps({'shard':a.shard,'rows':count,'source_sha256':digest,'model':manifest}))
