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
def _load(root,side,version):
    import xgboost as xgb
    import joblib
    folder=Path(root)/'models/xgboost-hierarchy'/side
    model=xgb.Booster()
    model.load_model(bytearray(read_model_file(folder/'calibration-refit.ubj')))
    model.set_param({'nthread':2})
    schema=json.loads((folder/'category-schema.json').read_text())
    posterior,draws=load_bayesian(folder/'bayesian')
    if posterior['base_model_id']!='xgboost-fivefold-'+side:raise ValueError('Mismatched Bayesian base model')
    from pricing.description_features import projector as load_projector
    projector=load_projector(folder) if side=='ask' else None
    report=json.loads((folder/'bayesian/bayesian-evaluation.json').read_text())
    return model,schema,posterior,draws,projector,report

def load(root,side):
    folder=Path(root)/'models/xgboost-hierarchy'/side
    paths=[folder/'evaluation.json',folder/'category-schema.json',folder/'features.json',folder.parent/'release.json']
    version=tuple((p.name,p.stat().st_mtime_ns,p.stat().st_size) for p in paths if p.exists())
    return _load(root,side,version)

def design_matrix(model,schema,projector,fields,age):
    import xgboost as xgb
    x=pd.DataFrame({'age':[age],'log_miles':[np.log1p(float(fields['odometer']))],
                    'auction_grade':[fields.get('auction_grade')]})
    x['auction_grade']=pd.to_numeric(x.auction_grade,errors='coerce').astype(float)
    unknown=[]
    for c,levels in schema.items():
        value=str(fields.get(c) or 'unknown').strip().lower()
        if value not in levels:unknown.append(c)
        x[c]=pd.Categorical([value],categories=levels)
    from scipy.sparse import csr_matrix,hstack
    frame=x.copy()
    for c,levels in schema.items():
        codes=pd.Categorical([str(fields.get(c) or 'unknown').strip().lower()],categories=levels).codes.astype(float)
        codes[codes<0]=np.nan
        frame[c]=codes
    values=frame.to_numpy(dtype=np.float32);n,k=values.shape
    data=csr_matrix((values.ravel(),np.tile(np.arange(k),n),np.arange(0,(n+1)*k,k)),shape=(n,k))
    types=['c' if c in schema else 'q' for c in frame]
    if projector is not None:
        t=projector.transform([str(fields.get('text',''))])
        data=hstack([data,t],format='csr');types+=['q']*t.shape[1]
    return xgb.DMatrix(data,feature_names=model.feature_names,feature_types=types,enable_categorical=True,nthread=2),unknown

def base_predict(model,schema,projector,fields,age,side):
    matrix,unknown=design_matrix(model,schema,projector,fields,age)
    raw=float(model.predict(matrix)[0])
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
        except (ValueError,OSError,KeyError,MemoryError) as exc:results[side]={'error':str(exc)}
    return results,transport,error

def render(root,*,bayesian=True):
    from pricing.forest_form import render as form
    from pricing.description_features import kind,label
    feature_kind=kind(root)
    fields,notes,date,allow,submitted=form(description_sensitive=feature_kind)
    signature=json.dumps([fields,notes,str(date),allow],sort_keys=True)
    if submitted:
        st.session_state.pop('hierarchy_sparse_result',None)
        st.session_state.pop('last_explained_price',None)
        st.session_state.pop('last_displayed_price',None)
        try:
            with st.spinner('Reading the advert and consulting the Bayesian family tree…'):
                results,_,_=estimate(root,{**fields,'text':notes},date,allow)
            st.session_state['hierarchy_sparse_result']={'signature':signature,'results':results,'date':str(date),'fields':fields,'notes':notes,'request_id':__import__('uuid').uuid4().hex}
        except (ValueError,OSError,KeyError,MemoryError) as exc:st.error(str(exc))
    saved=st.session_state.get('hierarchy_sparse_result')
    if not saved:
        st.info('Fill in your car, then press Estimate asking + auction prices. No spreadsheet required.');return
    if saved['signature']!=signature:
        st.info('Car details or date changed. Press Estimate asking + auction prices to refresh both estimates.');return
    from pricing.pricing_comparison import selected_receipt
    results={side:selected_receipt(r,bayesian) for side,r in saved['results'].items()}
    st.markdown('### '+('Bayesian estimates · with family evidence' if bayesian else 'Regular XGBoost estimates'))
    st.caption('Asking and wholesale auction are different market targets. '+('These estimates include the Bayesian correction.' if bayesian else 'These estimates use XGBoost directly.'))

    for col,side,title,meaning in zip(st.columns(2),['ask','bid'],["Seller’s asking price","Wholesale auction estimate"],["What might a seller put in the advert? Before negotiation; not a completed retail sale.","What might a comparable car sell for at wholesale auction? Not a live bid or dealer offer."]):
        with col:
            with st.container(border=True):
                st.markdown('#### '+title);r=results[side]
                if 'error' in r:st.warning(r['error']);continue
                st.metric(f"Estimated value · {date:%d %b %Y}",f"${r['price']:,.0f}");st.caption(meaning)
                if bayesian:st.markdown(f"**Historical 90% range, date-adjusted · USD {r['interval_low']:,.0f}–{r['interval_high']:,.0f}**")
                st.caption(f"{date:%d %b %Y} · "+("XGBoost + Bayesian correction" if bayesian else f"Regular XGBoost · {label(root)} asking AFT / auction median"))
                if r['analogue']:st.warning('Auction estimate uses a generation analogue: auction training ends at model year 2015. This 2016–2021 vehicle generation was not observed; the date adjustment does not validate that assumption.')
                if bayesian and r['model_support']==0:st.caption('Unseen model/trim: the correction borrows from its make and broader market.')
                if r['unknown_categories']:st.caption('Unseen categorical inputs: '+', '.join(r['unknown_categories'])+'. Native missing-value routing applies.')
    st.info('The gap is not dealer profit or a promised discount. These models describe different markets and vehicle populations.')
    available=[r for r in results.values() if 'error' not in r]
    if available:
        import plotly.graph_objects as go
        labels={'ask':'Seller asking','bid':'Wholesale auction'}
        colors={'ask':'#6752d4','bid':'#17846e'}
        fig=go.Figure()
        for side,r in results.items():
            if 'error' in r:continue
            fig.add_trace(go.Scatter(x=[r['price']],y=[labels[side]],mode='markers',name=labels[side],
                marker={'size':14,'color':colors[side]},
                error_x={'type':'data','symmetric':False,'array':[r['interval_high']-r['price']],
                         'arrayminus':[r['price']-r['interval_low']],'color':colors[side],
                         'thickness':3,'width':7} if bayesian else None,
                hovertemplate='%{y}: $%{x:,.0f}<extra></extra>'))
        fig.update_layout(height=190,margin={'l':4,'r':12,'t':14,'b':14},showlegend=False,
            xaxis_title='USD · dot = price estimate'+(' · bar = historical range' if bayesian else ''),
            yaxis={'categoryorder':'array','categoryarray':['Wholesale auction','Seller asking']})
        fig.update_xaxes(tickprefix='$',tickformat=',.0f',rangemode='tozero')
        st.plotly_chart(fig,width='stretch',key='separate_price_ranges_'+str(bayesian))
        r=available[0];market=r['market']
        st.caption(f"Market data observed through {market['last_known_month']}. Selected month {market['month']}: "+(f"{market['horizon']}-month {market['method'].replace('_',' ')} forecast." if market['horizon'] else 'observed monthly index.'))
        if bayesian:st.warning('Ranges are 90% historical-outcome intervals calibrated on held-out historical cars, then transported to the selected date with the same ageing and market factors as the point estimate. Their 90% coverage is NOT validated for today, especially for forecast months or unobserved generations. They are not confidence intervals for the model mean or price guarantees.')
        with st.expander('Follow the dollars · historical anchors, ageing and market movement'):
            from pricing.forest_time import adjustment_chart
            for side,r in results.items():
                if 'error' in r:continue
                st.markdown('**'+('Asking' if side=='ask' else 'Auction')+'**')
                chart=adjustment_chart(r)
                chart.data[0].x=['Historical XGBoost'+(' + Bayes' if bayesian else ''),'Age & mileage','Market movement',r['date']]
                st.plotly_chart(chart,width='stretch',key='separate_waterfall_'+side+str(bayesian))
                st.caption(f"Age/mileage ×{r['vehicle_factor']:.3f} · market ×{r['market_factor']:.3f}. Reference anchor is a model scenario, not an observed sale of your car.")
                if bayesian:st.caption(f"Market-specific posterior {r['posterior_draws']} draws; predictive SD {r['predictive_sd_log']:.3f} log units. Validation scale {r['interval_multiplier']:.2f}; untouched historical test coverage {r['historical_coverage']:.1%} versus 90% nominal.")
            st.json(results,expanded=False)

    from pricing.current_tools import publish_receipt, render_briefing
    publish_receipt(root,fields,notes,date,results,saved,bayesian)
    if 'ask' in results and 'error' not in results['ask']:
        from pricing.deal_detective import open_share
        st.button('📣 Share my price · image, link & group-chat fuel',on_click=open_share,key='current_share')
    render_briefing(root,fields,notes,date,results)
    from pricing.current_tools import render_drivers,render_minilm
    render_drivers(root,fields,notes,results,bayesian)
    from pricing.current_explanations import render as render_graphs
    render_graphs(root,fields,notes,date,results,bayesian)
    render_minilm(root,notes,fields,date,results,bayesian)
