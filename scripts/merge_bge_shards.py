import json
from pathlib import Path
import numpy as np,pandas as pd
root=Path('bge-shards');parts=[np.load(root/f'{i:02d}.npz') for i in range(20)]
ids=np.concatenate([p['ids'] for p in parts])
if len(set(ids))!=len(ids):raise ValueError('Duplicate encoding IDs')
receipts=[json.loads((root/f'{i:02d}.json').read_text()) for i in range(20)]
if len({r['source_sha256'] for r in receipts})!=1:raise ValueError('Different source cohorts')
bge=np.concatenate([p['embeddings'] for p in parts]);mini=np.concatenate([p['minilm'] for p in parts])
np.savez(root/'merged.npz',ids=ids,embeddings=bge)
out=Path('training-cache/minilm-embeddings');out.mkdir(parents=True,exist_ok=True)
np.save(out/'embeddings.npy',mini)
(out/'row-ids.json').write_text(json.dumps(ids.tolist()))
(out/'provenance.json').write_text(json.dumps({'frozen':True,'supervised_backbone_training':False,'revision':'sanitized-frozen-MiniLM-128','source_sha256':receipts[0]['source_sha256']}))
(root/'audit.json').write_text(json.dumps(receipts))
