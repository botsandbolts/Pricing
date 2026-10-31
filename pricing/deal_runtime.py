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


def current_callbacks(root):
    """Use the same installed estimator/date/mode for current shortlist comparisons."""
    import json
    import streamlit as st
    from marketbridge.time_engine import today
    from pricing.hierarchy_sparse_live import estimate,release_identity
    from pricing.pricing_comparison import selected_receipt
    saved=st.session_state.get('last_displayed_price')
    date=saved[1].get('valuation_date',str(today())) if saved else str(today())
    bayesian=st.session_state.get('_current_pricing_mode',False)
    allow=st.session_state.get('_forest_editor',{}).get('allow',True)
    memo={}
    def pair(text,fields,day):
        key=json.dumps([text,fields,str(day)],sort_keys=True,default=str)
        if key not in memo:
            values,_,_=estimate(root,{**fields,'text':text},day,allow)
            memo[key]={s:selected_receipt(r,bayesian) for s,r in values.items()}
        return memo[key]
    def ask(text,fields):
        value=pair(text,fields,date)['ask']
        if 'error' in value:raise ValueError(value['error'])
        return {'price':value['price']}
    def bid(fields,day):
        value=pair('',fields,day)['bid']
        if 'error' in value:raise ValueError(value['error'])
        return value['price']
    model='Current XGBoost'+(' + Bayesian' if bayesian else '')+' · '+release_identity(root)[:10]
    return ask,bid,model
