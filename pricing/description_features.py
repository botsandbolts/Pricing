"""Release-declared description features; never relabel a TF-IDF bundle MiniLM."""
import json
from pathlib import Path
import numpy as np
import streamlit as st

class MiniLMFeatures:
    def __init__(self,path):
        from pricing.minilm_signal import Encoder
        self.encoder=Encoder(path)
    def transform(self,texts):
        from scipy.sparse import csr_matrix
        from scripts.common import clean
        cleaned=[clean(t) for t in texts]
        unique=list(dict.fromkeys(cleaned));lookup={t:i for i,t in enumerate(unique)}
        values=self.encoder.encode(unique)[[lookup[t] for t in cleaned]]
        n,k=values.shape
        return csr_matrix((values.ravel(),np.tile(np.arange(k),n),np.arange(0,(n+1)*k,k)),shape=(n,k))
    def get_feature_names_out(self):return np.array([f'MiniLM dimension {i+1}' for i in range(384)])

def kind(root):
    path=Path(root)/'models/xgboost-hierarchy/ask/features.json'
    return json.loads(path.read_text())['kind'] if path.exists() else 'tfidf'

def label(root):return 'MiniLM embeddings' if kind(root)=='minilm-embeddings' else 'TF-IDF'

def projector(folder):
    folder=Path(folder)
    if (folder/'features.json').exists():
        if json.loads((folder/'features.json').read_text())['kind']!='minilm-embeddings':raise ValueError('Unknown description-feature contract')
        return MiniLMFeatures(folder/'minilm')
    import joblib
    return joblib.load(folder/'tfidf.joblib')
