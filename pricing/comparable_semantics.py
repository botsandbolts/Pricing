"""Optional precomputed description vectors; never download models in the UI."""
import hashlib,json
from pathlib import Path
import numpy as np

def fingerprint(rows):
    return hashlib.sha256(json.dumps([(r.get('id'),r.get('description','')) for r in rows],ensure_ascii=False).encode()).hexdigest()

def attach(index,root):
    folder=Path(root)/'data/comparable-minilm'
    meta=folder/'manifest.json';vectors=folder/'vectors.npz'
    if not meta.exists() or not vectors.exists():return False
    info=json.loads(meta.read_text())
    if info['rows_sha256']!=fingerprint(index.rows):return False
    if hashlib.sha256(vectors.read_bytes()).hexdigest()!=info['vectors_sha256']:return False
    with np.load(vectors,allow_pickle=False) as data:index.attach_descriptions(data['vectors'])
    return True

def encode_query(root,text):
    from pricing.description_features import MiniLMFeatures
    encoder=MiniLMFeatures(Path(root)/'models/xgboost-hierarchy/ask/minilm')
    try:
        from scripts.common import clean
        return encoder.encoder.encode([clean(text)])[0]
    finally:del encoder
