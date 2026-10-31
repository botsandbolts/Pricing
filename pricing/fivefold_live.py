"""Verified XGBoost refits with matching OOF Bayesian residual posteriors."""
import io,json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from pricing.suite_storage import read_model_file
from pricing.bayesian_hierarchy import load as load_bayesian,predict as posterior_predict,interval as posterior_interval
from marketbridge.time_engine import age_on,today

@st.cache_resource(show_spinner=False,max_entries=2)
def load(root,side):
    import xgboost as xgb
    import joblib
    folder=Path(root)/'models/fivefold-live'/side
    model=xgb.Booster()
    model.load_model(bytearray(read_model_file(folder/'calibration-refit.ubj')))
    schema=json.loads((folder/'category-schema.json').read_text())
    posterior,draws=load_bayesian(folder/'bayesian')
    if posterior['base_model_id']!='xgboost-fivefold-'+side:raise ValueError('Mismatched Bayesian base model')
    projector=joblib.load(io.BytesIO(read_model_file(folder/'calibration-refit-tfidf-umap.joblib'))) if side=='ask' else None
    report=json.loads((folder/'bayesian/bayesian-evaluation.json').read_text())
    return model,schema,posterior,draws,projector,report

def base_predict(model,schema,projector,fields,age,side):
    import xgboost as xgb
    x=pd.DataFrame({'age':[age],'log_miles':[np.log1p(float(fields['odometer']))],
                    'auction_grade':[fields.get('auction_grade')]})
    x['auction_grade']=pd.to_numeric(x.auction_grade,errors='coerce').astype(float)
    unknown=[]
    for c,levels in schema.items():
        value=str(fields.get(c) or 'unknown').strip().lower()
        if value not in levels:unknown.append(c)
        x[c]=pd.Categorical([value],categories=levels)
    if projector:
        vectorizer,reducer=projector
        z=reducer.transform(vectorizer.transform([str(fields.get('text',''))]))
        for j in range(8):x[f'description_umap_{j+1}']=z[:,j]
    raw=float(model.predict(xgb.DMatrix(x[model.feature_names],enable_categorical=True,nthread=2))[0])
    price=raw if side=='ask' else float(np.exp(np.clip(raw,-700,np.log(1e7))))
    if not np.isfinite(price) or price<=0:raise ValueError('Invalid base price')
    return price,unknown

def estimate(root,fields,date,allow=False):
    from pricing.current_value import current_transport
    from pricing.forest_time import adjust
    transport,error=current_transport(root);transport.validate(fields,date)
    results={}
    for side in ['ask','bid']:
        try:
            if side=='bid' and int(fields['year'])>2015 and not allow:
                raise ValueError('Enable the auction-generation analogue for 2016–2021 cars; these generations were not observed.')
            age=age_on(fields['year'],'2021-05-01') if side=='ask' else 5.
            if not 0<=age<=30:raise ValueError('Historical age anchor is outside 0–30 years.')
            model,schema,posterior,draws,projector,report=load(str(root),side)
            base,unknown=base_predict(model,schema,projector,fields,age,side)
            case=pd.DataFrame([{**fields,'age':age,'base_price':base}])
            bp=posterior_predict(posterior,draws,case).iloc[0]
            lo,hi=posterior_interval(pd.DataFrame([bp]),posterior['interval']['multiplier'])
            row={**fields,**bp.to_dict(),'age':age,'source_date':'2021-05-01' if side=='ask' else f"{int(fields['year'])+5}-01-01"}
            receipt=adjust(transport,row,side,date,fields['odometer'],as_of=today())
            ratio=receipt['vehicle_factor']*receipt['market_factor']
            receipt.update(interval_low=float(lo[0]*ratio),interval_high=float(hi[0]*ratio),
                historical_coverage=float(report['test_coverage']),interval_level=posterior['interval']['level'],
                interval_multiplier=posterior['interval']['multiplier'],posterior_draws=len(draws['global']),
                predictive_sd_log=float(bp.predictive_sd_log),unknown_categories=unknown,
                make_support=int(bp.make_n),model_support=int(bp.model_n),
                analogue=bool(side=='bid' and int(fields['year'])>2015))
            results[side]=receipt
        except (ValueError,OSError,KeyError) as exc:results[side]={'error':str(exc)}
    return results,transport,error

def render(root):
    from pricing.forest_form import render as form
    fields,notes,date,allow,submitted=form(description_sensitive=True)
    signature=json.dumps([fields,notes,str(date),allow],sort_keys=True)
    if submitted:
        try:
            with st.spinner('Reading the advert and consulting the Bayesian family tree…'):
                results,_,_=estimate(root,{**fields,'text':notes},date,allow)
            st.session_state['fivefold_live_result']={'signature':signature,'results':results,'date':str(date)}
        except (ValueError,OSError,KeyError) as exc:st.error(str(exc))
    saved=st.session_state.get('fivefold_live_result')
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
                st.markdown(f"**Historical 90% range, date-adjusted · ${r['interval_low']:,.0f}–${r['interval_high']:,.0f}**")
                st.caption(f"{date:%d %b %Y} · XGBoost + Bayesian make/model correction")
                if r['analogue']:st.warning('Unobserved newer generation: analogue assumption enabled.')
                if r['model_support']==0:st.caption('Unseen model/trim: the correction borrows from its make and broader market.')
                if r['unknown_categories']:st.caption('Unseen categorical inputs: '+', '.join(r['unknown_categories'])+'. Native missing-value routing applies.')
    st.info('The gap is not dealer profit or a promised discount. These models describe different markets and vehicle populations.')
    available=[r for r in saved['results'].values() if 'error' not in r]
    if available:
        import plotly.graph_objects as go
        labels={'ask':'Seller asking','bid':'Wholesale auction'}
        colors={'ask':'#6752d4','bid':'#17846e'}
        fig=go.Figure()
        for side,r in saved['results'].items():
            if 'error' in r:continue
            fig.add_trace(go.Scatter(x=[r['price']],y=[labels[side]],mode='markers',name=labels[side],
                marker={'size':14,'color':colors[side]},
                error_x={'type':'data','symmetric':False,'array':[r['interval_high']-r['price']],
                         'arrayminus':[r['price']-r['interval_low']],'color':colors[side],
                         'thickness':3,'width':7},
                hovertemplate='%{y}: $%{x:,.0f}<extra></extra>'))
        fig.update_layout(height=190,margin={'l':4,'r':12,'t':14,'b':14},showlegend=False,
            xaxis_title='USD · dot = price estimate · bar = historical range',
            yaxis={'categoryorder':'array','categoryarray':['Wholesale auction','Seller asking']})
        fig.update_xaxes(tickprefix='$',tickformat=',.0f',rangemode='tozero')
        st.plotly_chart(fig,width='stretch',key='fivefold_bayesian_ranges')
        r=available[0];market=r['market']
        st.caption(f"Market data observed through {market['last_known_month']}. Selected month {market['month']}: "+(f"{market['horizon']}-month {market['method'].replace('_',' ')} forecast." if market['horizon'] else 'observed monthly index.'))
        st.warning('Ranges are 90% historical-outcome intervals calibrated on held-out historical cars, then transported to the selected date with the same ageing and market factors as the point estimate. Their 90% coverage is NOT validated for today, especially for forecast months or unobserved generations. They are not confidence intervals for the model mean or price guarantees.')
        with st.expander('Follow the dollars · historical anchors, ageing and market movement'):
            from pricing.forest_time import adjustment_chart
            for side,r in saved['results'].items():
                if 'error' in r:continue
                st.markdown('**'+('Asking' if side=='ask' else 'Auction')+'**')
                st.plotly_chart(adjustment_chart(r),width='stretch',key='fivefold_live_waterfall_'+side)
                st.caption(f"Age/mileage ×{r['vehicle_factor']:.3f} · market ×{r['market_factor']:.3f}. Reference anchor is a model scenario, not an observed sale of your car.")
                st.caption(f"Joint posterior {r['posterior_draws']} draws; predictive SD {r['predictive_sd_log']:.3f} log units. Validation scale {r['interval_multiplier']:.2f}; untouched historical test coverage {r['historical_coverage']:.1%} versus 90% nominal.")
            st.json(saved['results'],expanded=False)
