"""Native live inference for the verified calibration-only random forests."""
import hashlib,json,io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from pricing.suite_storage import read_model_file
from pricing.hierarchical import predict as correct
from pricing.original_residual_results import load as load_results
from pricing.forest_residuals import CATEGORICAL
from marketbridge.time_engine import age_on,today


@st.cache_resource(show_spinner=False,max_entries=2)
def load(root,side):
    from pricing.forest_compact import CompactForest
    folder=Path(root)/'models/forest-original'/side
    spec=json.loads((folder/'serving-compact.json').read_text());packed=read_model_file(folder/'serving-compact.npz')
    if hashlib.sha256(packed).hexdigest()!=spec['sha256']:raise ValueError('Forest checksum mismatch')
    with np.load(io.BytesIO(packed),allow_pickle=False) as data:model=CompactForest({k:data[k] for k in data.files})
    meta=json.loads((folder/'base-assessment.json').read_text())
    bundle,report,_=load_results(root,side)
    return model,meta,bundle,report


def predict(model,meta,bundle,fields,age):
    x=pd.DataFrame({'age':[age],'log_miles':[np.log1p(float(fields['odometer']))],'auction_grade':[fields.get('auction_grade')]})
    x['auction_grade']=pd.to_numeric(x.auction_grade,errors='coerce').astype(float)
    for c in CATEGORICAL:x[c]=pd.Categorical([str(fields.get(c) or 'unknown').strip().lower()],categories=meta['category_schema'][c])
    base=float(np.exp(np.clip(model.predict(x[meta['features']])[0],0,np.log(1e7))))
    row=pd.DataFrame([{**fields,'age':age,'base_price':base}]);result=correct(bundle,row).iloc[0].to_dict()
    unknown=[c for c in CATEGORICAL if str(fields.get(c) or 'unknown').lower().strip() not in meta['category_schema'][c]]
    return result,unknown


def estimate(root,fields,date,allow=False):
    from pricing.current_value import current_transport
    from pricing.forest_time import adjust
    transport,error=current_transport(root);transport.validate(fields,date)
    if not str(fields.get('manufacturer','')).strip() or not str(fields.get('model','')).strip():raise ValueError('Enter a make and model.')
    results={}
    for side in ['ask','bid']:
        try:
            if side=='bid' and int(fields['year'])>2015 and not allow:raise ValueError('Enable the explicit auction-generation analogue for 2016–2021 cars. This generation was not observed in auction training.')
            age=age_on(fields['year'],'2021-05-01') if side=='ask' else 5.
            if not 0<=age<=30:raise ValueError('The historical age anchor is outside the supported 0–30 year range.')
            model,meta,bundle,_=load(str(root),side)
            correction,unknown=predict(model,meta,bundle,fields,age)
            row={**fields,**correction,'age':age,'source_date':'2021-05-01' if side=='ask' else f"{int(fields['year'])+5}-01-01"}
            receipt=adjust(transport,row,side,date,fields['odometer'],as_of=today())
            receipt.update(unknown_categories=unknown,make_support=correction['make_n'],model_support=correction['model_n'],analogue=bool(side=='bid' and int(fields['year'])>2015))
            results[side]=receipt
        except (ValueError,OSError,KeyError) as exc:results[side]={'error':str(exc)}
    return results,transport,error


def render(root):
    from pricing.forest_form import render as form
    fields,notes,date,allow,submitted=form()
    signature=json.dumps([fields,notes,str(date),allow],sort_keys=True)
    if submitted:
        try:
            with st.spinner('Two forests. One car. Negotiations can wait a moment…'):
                results,_,_=estimate(root,fields,date,allow)
            st.session_state['forest_live_result']={'signature':signature,'results':results,'date':str(date)}
        except (ValueError,OSError,KeyError) as exc:st.error(str(exc))
    saved=st.session_state.get('forest_live_result')
    if not saved:
        st.info('Fill in your car, then press Estimate asking + auction prices. No spreadsheet required.');return
    if saved['signature']!=signature:
        st.info('Car details or date changed. Press Estimate asking + auction prices to refresh both estimates.');return
    st.markdown('### Two prices. Two different questions.')
    for col,side,title,meaning in zip(st.columns(2),['ask','bid'],["Seller’s asking price","Wholesale auction estimate"],["What might a seller put in the advert? Before negotiation; not a completed retail sale.","What might a comparable car sell for at wholesale auction? Not a live bid or dealer offer."]):
        with col:
            with st.container(border=True):
                st.markdown('#### '+title);r=saved['results'][side]
                if 'error' in r:st.warning(r['error']);continue
                st.metric('Date-adjusted scenario',f"${r['price']:,.0f}");st.caption(meaning)
                st.caption(f"{date:%d %b %Y} · XGBoost RF + hierarchical correction")
                if r['analogue']:st.warning('Unobserved newer generation: analogue assumption enabled.')
                if r['model_support']==0:st.caption('Unseen model/trim: the correction borrows from its make and broader market.')
                if r['unknown_categories']:st.caption('Unseen categorical inputs: '+', '.join(r['unknown_categories'])+'. Native missing-value routing applies.')
    st.info('The gap is not dealer profit or a promised discount. These models describe different markets and vehicle populations.')
    available=[r for r in saved['results'].values() if 'error' not in r]
    if available:
        r=available[0];market=r['market']
        st.caption(f"Market data observed through {market['last_known_month']}. Selected month {market['month']}: "+(f"{market['horizon']}-month {market['method'].replace('_',' ')} forecast." if market['horizon'] else 'observed monthly index.'))
        st.warning('Experimental date-adjusted estimates, not verified current sales. Historical test MAE does not validate current-car prices.')
        with st.expander('Follow the dollars · historical anchors, ageing and market movement'):
            from pricing.forest_time import adjustment_chart
            for side,r in saved['results'].items():
                if 'error' in r:continue
                st.markdown('**'+('Asking' if side=='ask' else 'Auction')+'**')
                st.plotly_chart(adjustment_chart(r),width='stretch',key='forest_live_waterfall_'+side)
                st.caption(f"Age/mileage ×{r['vehicle_factor']:.3f} · market ×{r['market_factor']:.3f}. Reference anchor is a model scenario, not an observed sale of your car.")
            st.json(saved['results'],expanded=False)
