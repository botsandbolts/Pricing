"""Observed ask text-price features + genuinely missing auction features, jointly imputed."""
import gzip,hashlib,json
from pathlib import Path
from datetime import date
import numpy as np
from pricing.comparable_retrieval import reference_rows
from pricing.comparable_semantics import fingerprint
from pricing.description_price_imputation import DescriptionPriceImputer,identity
from pricing.minilm_signal import mean_price
from pricing.reference_prices import feature_prices
from marketbridge.time_engine import TimeTransport

def build(root):
    root=Path(root);asks=reference_rows(root,'ask');bids=reference_rows(root,'bid');rows=asks+bids
    meta=json.loads((root/'data/comparable-minilm/manifest.json').read_text())
    if meta['rows_sha256']!=fingerprint(asks):raise ValueError('Description-vector row identity mismatch')
    with np.load(root/'data/comparable-minilm/vectors.npz',allow_pickle=False) as data:vectors=data['vectors'].astype(np.float32)
    heads_file=root/'models/xgboost-hierarchy/ask/minilm-heads.json';heads=json.loads(heads_file.read_text())['heads']
    description=np.r_[mean_price(heads,vectors),np.full(len(bids),np.nan)]
    transport=TimeTransport(root)
    prices=np.r_[feature_prices(asks,'ask',date(2021,5,1),transport),feature_prices(bids,'bid',date(2021,5,1),transport)]
    held=np.array([i for i,r in enumerate(asks) if int(hashlib.sha256(str(r['id']).encode()).hexdigest()[:8],16)%5==0])
    masked=description.copy();masked[held]=np.nan
    probe=DescriptionPriceImputer().fit(rows,prices,masked)
    prediction,_=probe.transform([rows[i] for i in held],prices[held])
    audit={'held_out_asking_records':len(held),'signal_reconstruction_mae_usd':float(np.mean(np.abs(prediction-description[held]))),
           'median_signal_baseline_mae_usd':float(np.mean(np.abs(np.nanmedian(masked)-description[held]))),
           'scope':'Reconstruction of a transformer-price feature withheld from asking records; not true vehicle-price accuracy or validation on auction descriptions, which do not exist.'}
    model=DescriptionPriceImputer().fit(rows,prices,description);values,imputed=model.transform(rows,prices)
    assert np.isfinite(values).all() and imputed[len(asks):].all()
    payload=model.data();restored=DescriptionPriceImputer.from_data(payload)
    probe_values,_=restored.transform(rows[-5:],prices[-5:])
    assert np.allclose(probe_values,values[-5:])
    folder=root/'data/retrieval-imputation';folder.mkdir(exist_ok=True)
    file=folder/'description-price.json.gz'
    with gzip.open(file,'wt') as f:json.dump(payload,f,allow_nan=False)
    report={'reference_sha256':identity(asks,bids),'model_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'heads_sha256':hashlib.sha256(heads_file.read_bytes()).hexdigest(),
            'asking_rows':len(asks),'auction_rows':len(bids),'auction_description_prices_observed':0,'artifact_bytes':file.stat().st_size,'audit':audit,
            'method':'Shared sklearn IterativeImputer with native-categorical LightGBM regression, structured vehicle predictors and reference-2021 price context. Missing auction description-price signals are NaN before imputation; no auction descriptions are invented.',
            'limits':'The auction text-value feature is a transferred statistical proxy from asking records. It is not an observed price, seller description, bid or validated auction text valuation. Used only in retrieval; serving pricing models are unchanged.'}
    (folder/'description-price.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(report,flush=True)
if __name__=='__main__':build(Path(__file__).resolve().parents[1])
