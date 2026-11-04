"""Twenty independent, row-ID aligned frozen BGE encoding shards."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sentence_transformers import SentenceTransformer
from pricing.advert_text import scrub
p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args()
if not 0<=a.shard<20:raise ValueError('Shard outside 0..19')
source=Path('training-cache/full-cohort.jsonl')
model=SentenceTransformer('BAAI/bge-small-en-v1.5',device='cpu')
model.max_seq_length=512
model.eval()
mini=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu');mini.max_seq_length=128
ids=[];vectors=[];minis=[];scanned=0;changed=0
for chunk in pd.read_json(source,lines=True,chunksize=1000):
    positions=np.arange(scanned,scanned+len(chunk));scanned+=len(chunk)
    selected=chunk.loc[positions%20==a.shard]
    if selected.empty:continue
    raw=selected.text.fillna('').astype(str).tolist();texts=[scrub(t) for t in raw]
    changed+=sum(x!=y for x,y in zip(raw,texts))
    vectors.append(model.encode(texts,batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False))
    minis.append(mini.encode(texts,batch_size=64,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False))
    ids.extend(selected.row_id.astype(str).tolist())
    print(a.shard,len(ids),flush=True)
out=Path('bge-shards');out.mkdir(exist_ok=True)
np.savez_compressed(out/f'{a.shard:02d}.npz',ids=np.asarray(ids),embeddings=np.concatenate(vectors).astype('float32'),minilm=np.concatenate(minis).astype('float32'))
(out/f'{a.shard:02d}.json').write_text(json.dumps({'shard':a.shard,'rows':len(ids),'money_redacted_descriptions':changed,'source_sha256':hashlib.file_digest(open(source,'rb'),'sha256').hexdigest(),'model':'BAAI/bge-small-en-v1.5','frozen':True}))
