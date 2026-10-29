"""Lazy access to installed auction serving models; no training or downloads."""
from pathlib import Path
from pricing.deal_tools import safe_fields


def bid_estimate(fields, valuation_date):
    import pandas as pd
    from pricing.residual_ui import bid_folder,load,estimate
    from marketbridge.structural import IndexPanel
    root=Path(__file__).resolve().parents[1]
    folder=bid_folder(root)
    model,schema,record=load(str(folder),(folder/'manifest.json').stat().st_mtime_ns)
    clean=safe_fields(fields)
    if clean.get('year',2022)>2015:
        raise ValueError('Auction challenge/comparison requires model years through 2015.')
    raw=pd.DataFrame([{**clean,'as_of':str(valuation_date),'condition':None}])
    dates=pd.to_datetime(raw.as_of,utc=True)
    ratio,_=IndexPanel(root/'models/structural-transfer/used-car-index.csv').ratios(dates)
    if ratio.isna().any():raise ValueError('No market index for this historical date.')
    return float(estimate(model,schema,record,raw,'bid')[0]*ratio.iloc[0])
