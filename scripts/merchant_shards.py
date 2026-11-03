import argparse,json,hashlib,re
from pathlib import Path
import numpy as np,pandas as pd,torch
p=argparse.ArgumentParser();p.add_argument('--part',type=int);p.add_argument('--merge',action='store_true');a=p.parse_args()
root=Path('training-cache/transformer-shards');root.mkdir(parents=True,exist_ok=True)
if a.merge:
    pieces=[np.load(root/f'part-{i}.npz') for i in range(10)]
    ids=np.concatenate([x['ids'] for x in pieces])
    assert len(set(ids.tolist()))==len(ids)==110832
    vectors=np.concatenate([x['vectors'] for x in pieces])
    assert vectors.shape==(len(ids),384) and np.isfinite(vectors).all()
    np.savez_compressed(root/'merged.npz',ids=ids,vectors=vectors)
    print('Validated all 110832 unique rows and 384D vectors')
else:
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer
    from pricing.merchant_reco import vehicle_text
    torch.set_num_threads(2)
    frame=pd.read_json('training-cache/lemon-mixed/cohort.jsonl',lines=True)
    frame=frame.iloc[np.array_split(np.arange(len(frame)),10)[a.part]]
    ner=pipeline('token-classification',model='dslim/bert-base-NER',aggregation_strategy='simple',device=-1)
    cache={};evidence=[]
    for row in frame.to_dict('records'):
        text=str(row.get('text') or '')
        snippets=[text[max(0,m.start()-300):m.end()+300] for m in re.finditer(r'https?://|www\\.',text)]
        passage=' '.join(snippets);key=hashlib.sha256(passage.encode()).hexdigest()
        if passage and key not in cache:cache[key]=[x for x in ner(passage) if x['entity_group']=='ORG']
        evidence.append({'row_id':str(row['row_id']),'organizations':cache.get(key,[])})
    (root/f'ner-{a.part}.json').write_text(json.dumps(evidence))
    print('NER shard complete',a.part,flush=True)
    del ner
    model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu');model.max_seq_length=256
    texts=[vehicle_text(str(r.get('text') or '')) for r in frame.to_dict('records')]
    vectors=model.encode(texts,batch_size=32,normalize_embeddings=True,show_progress_bar=True)
    np.savez_compressed(root/f'part-{a.part}.npz',ids=frame.row_id.astype(str).to_numpy(dtype=str),vectors=vectors)
    print('Embedding shard complete',a.part,flush=True)
