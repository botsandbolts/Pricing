"""Reuse the pinned compact label-free MiniLM encoder; encode the verified cohort."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from pricing.minilm_signal import Encoder
from scripts.common import clean

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',required=True);p.add_argument('--output',required=True);p.add_argument('--encoder',default='models/xgboost-sparse-minilm/ask/minilm');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    encoder=Encoder(a.encoder);ids=[];chunks=[]
    for d in pd.read_json(a.cohort,lines=True,chunksize=1024):
        chunks.append(encoder.encode([clean(t) for t in d.text.fillna('').astype(str).tolist()]));ids.extend(d.row_id.astype(str).tolist())
        print('MINILM',len(ids),'documents',flush=True)
    np.save(out/'embeddings.npy',np.concatenate(chunks))
    (out/'row-ids.json').write_text(json.dumps(ids))
    provenance=json.loads((Path(a.encoder)/'provenance.json').read_text())
    provenance.update(cohort_sha256=hashlib.file_digest(open(a.cohort,'rb'),'sha256').hexdigest(),feature_kind='minilm-embeddings',supervised_backbone_training=False,frozen=True,feature_dimension=384)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
if __name__=='__main__':main()
