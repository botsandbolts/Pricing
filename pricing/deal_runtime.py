"""Lazy access to installed auction serving models; no training or downloads."""
from pathlib import Path
from pricing.deal_tools import safe_fields


def bid_estimate(fields, valuation_date):
    import pandas as pd
    from pricing.residual_ui import bid_folder,load,estimate
    from pricing.current_value import current_transport
    from marketbridge.time_engine import today
    root=Path(__file__).resolve().parents[1]
    folder=bid_folder(root)
    model,schema,record=load(str(folder),(folder/'manifest.json').stat().st_mtime_ns)
    clean=safe_fields(fields)
    if clean.get('year',2022)>2015:
        raise ValueError('Auction challenge/comparison requires model years through 2015.')
    transport,_=current_transport(root)
    return float(transport.estimate(model,'',clean,valuation_date,'bid',False,as_of=today())['price'])
