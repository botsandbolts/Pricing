"""Extract named seller businesses from full advert text, without website gating."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
p=argparse.ArgumentParser();p.add_argument('--part',type=int);p.add_argument('--merge',action='store_true');a=p.parse_args()
root=Path('training-cache/gliner-merchants');root.mkdir(parents=True,exist_ok=True)
if a.merge:
    rows=[]
    for i in range(10):rows.extend(json.loads((root/f'part-{i}.json').read_text()))
    ids=[r['row_id'] for r in rows]
    assert len(ids)==len(set(ids))==110832
    (root/'merged.json').write_text(json.dumps(rows))
    print(json.dumps({'rows':len(rows),'rows_with_candidates':sum(bool(r['merchant_candidates']) for r in rows),'identity_status':'extracted candidates, not verified seller identities'}))
else:
    from gliner import GLiNER
    torch.set_num_threads(2)
    frame=pd.read_json('training-cache/lemon-mixed/cohort.jsonl',lines=True)
    frame=frame.iloc[np.array_split(np.arange(len(frame)),10)[a.part]]
    model=GLiNER.from_pretrained('gliner-community/gliner_small-v2.5')
    labels=['car dealership','seller business','vehicle manufacturer','finance company']
    cache={};rows=[]
    for n,row in enumerate(frame.to_dict('records')):
        text=str(row.get('text') or '')
        key=hashlib.sha256(text.encode()).hexdigest()
        if key not in cache:
            found=[]
            # Overlapping windows include the full advert, including seller footers.
            for start in range(0,len(text),1200):
                chunk=text[start:start+1600]
                for e in model.predict_entities(chunk,labels,threshold=0.5):
                    found.append({'name':e['text'],'label':e['label'],'score':float(e['score']),'start':start+e['start'],'end':start+e['end']})
            unique={(e['start'],e['end'],e['label']):e for e in found}
            cache[key]=list(unique.values())
        entities=cache[key]
        rows.append({'row_id':str(row['row_id']),'entities':entities,'merchant_candidates':[e for e in entities if e['label'] in ['car dealership','seller business']]})
        if n%250==0:print('Processed',a.part,n,flush=True)
    (root/f'part-{a.part}.json').write_text(json.dumps(rows))
    print('Completed merchant-name shard',a.part,len(rows),flush=True)
