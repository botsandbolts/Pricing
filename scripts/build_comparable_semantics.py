"""Offline-only MiniLM corpus encoding, using the installed frozen encoder."""
import hashlib,json
from pathlib import Path
import numpy as np
from pricing.comparable_retrieval import ComparableIndex,reference_rows
from pricing.comparable_semantics import fingerprint
from pricing.description_features import MiniLMFeatures
root=Path(__file__).resolve().parents[1]
from pricing.retrieval_imputation import load as load_imputer
rows=reference_rows(root,'ask')
index=ComparableIndex(rows,imputer=load_imputer(root,'ask',rows))
encoder=MiniLMFeatures(root/'models/xgboost-hierarchy/ask/minilm')
values=encoder.encoder.encode([r['description'] for r in index.rows],batch_size=16)
folder=root/'data/comparable-minilm';folder.mkdir(exist_ok=True)
np.savez_compressed(folder/'vectors.npz',vectors=values.astype(np.float16))
(folder/'manifest.json').write_text(json.dumps({'rows_sha256':fingerprint(index.rows),'vectors_sha256':hashlib.sha256((folder/'vectors.npz').read_bytes()).hexdigest(),'rows':len(values),'dimensions':values.shape[1],'model':'Frozen int8 all-MiniLM-L6-v2; max 128 tokens','scope':'Historical training-example descriptions only; no target prices or training updates'},indent=2))
print('Prepared',len(values),'reference descriptions')
