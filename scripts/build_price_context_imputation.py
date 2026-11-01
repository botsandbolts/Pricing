"""Offline price-assisted Ask/Bid imputation, with no price output to SVD."""
import gzip,hashlib,json
from pathlib import Path
import numpy as np
from pricing.comparable_retrieval import reference_rows
from pricing.price_context_imputation import PriceContextImputer
from pricing.retrieval_imputation import fingerprint
from pricing.reference_prices import feature_prices
from marketbridge.time_engine import TimeTransport,today

def build(root):
    root=Path(root);transport=TimeTransport(root);basis=today()
    for side in ['ask','bid']:
        rows=reference_rows(root,side);prices=feature_prices(rows,side,basis,transport)
        model=PriceContextImputer().fit(rows,prices);payload=model.data();restored=PriceContextImputer.from_data(payload)
        probe=[dict(rows[0],fuel=None,transmission=None,odometer=None)]
        a=model.transform(probe);b=restored.transform(probe)
        assert a[0]==b[0] and np.allclose(a[1],b[1]) and b[1].shape==(1,2)
        folder=root/'data/retrieval-imputation';folder.mkdir(exist_ok=True);file=folder/(side+'-price-context.json.gz')
        with gzip.open(file,'wt') as f:json.dump(payload,f,allow_nan=False)
        if file.stat().st_size>8*1024*1024:raise ValueError('Price-assisted imputer exceeds size budget')
        report=dict(reference_sha256=fingerprint(rows),model_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),price_basis_date=str(basis),rows=len(rows),supported_prices=int(np.isfinite(prices).sum()),artifact_bytes=file.stat().st_size,method='IterativeImputer with native-categorical LightGBM. Adjusted price is auxiliary context only and is dropped before SVD. Observed source facts are preserved.')
        (folder/(side+'-price-context.json')).write_text(json.dumps(report,indent=2));print(side,report,flush=True)
if __name__=='__main__':build(Path(__file__).resolve().parents[1])
