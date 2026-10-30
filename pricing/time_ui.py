"""Explicit experimental present-date transport and value-over-time scenarios."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from marketbridge.market_clock import MarketClock,fetch_snapshot
from marketbridge.time_engine import TimeTransport,today


@st.cache_data(ttl=21600,show_spinner=False)
def live_index():
    try:return fetch_snapshot(),None
    except Exception as exc:return None,str(exc)


@st.cache_resource
def engine(root,stamp):return TimeTransport(root)


@st.fragment(run_every='1h')
def render_time_studio(root,saved,ask_pipeline):
    root=Path(root);folder=root/'models/time-transport'
    if not (folder/'manifest.json').exists():
        st.info('The date-aware transport experiment is being fitted. Historical prices remain available.');return
    with st.container(border=True,key='time_machine'):
        st.subheader('The time garage · what about today?')
        from pricing.price_meaning import render_market_guide,render_market_price
        render_market_guide()
        st.caption('Experimental present-date estimates. Historical accuracy scores do not validate these transported prices.')
        mode=st.radio('Valuation clock',['Today · follows the calendar','Choose a date'],horizontal=True,key='time_clock_mode')
        date=today() if mode.startswith('Today') else st.date_input('Valuation date',today(),key='time_valuation_date')
        allow=st.checkbox('Allow generation analogues for 2016–2021 auction cars',value=False,key='time_allow_analogue',help='Auction training contains model years through 2015. This is an explicit unsupported-generation assumption; never a measured bid for the newer generation.')
        if not saved:
            st.info('Estimate a car first. Then compare today’s Ask and Bid, and draw its value through time.');return
        _,result,car,text=saved
        with st.spinner('Checking the public market clock…'):
            live,error=live_index()
        current=TimeTransport(root,MarketClock(folder,live=live))
        try:current.validate(car,date)
        except ValueError as exc:st.warning(str(exc));return
        from pricing.residual_ui import bid_folder,load
        bid_path=bid_folder(root);bid,_,_=load(str(bid_path),(bid_path/'manifest.json').stat().st_mtime_ns)
        # The shared serving wrapper preserves the original quantile model and schema.
        pipelines={'ask':ask_pipeline,'bid':bid};receipts={}
        cols=st.columns(2)
        for col,(side,pipeline) in zip(cols,pipelines.items()):
            with col:
                try:
                    r=current.estimate(pipeline,text,car,date,side,allow,as_of=today(),text_price=result.get('predicted_text_price') if side=='ask' else None);receipts[side]=r
                    from pricing.analytics import record_price
                    request=st.session_state.get('_price_request_id','')
                    record_price(side,r['price'],car,'time-transport/'+side,'time-scenario',str(date),request+':'+str(date)+':'+str(allow))
                    render_market_price(side,r['price'],date,scenario=True)
                    st.caption(f"Age {r['age']:.2f} years · age factor ×{r['age_factor']:.3f} · market factor ×{r['market_factor']:.3f}")
                    if not r['model_known_to_curve']:st.caption('Sparse/unseen model: pooled manufacturer/body depreciation applies.')
                    if r['generation_analogue']:st.warning('Generation analogue enabled. Auction-era generation effects are unobserved.')
                except ValueError as exc:st.warning(str(exc))
        st.caption(f"Market source: BLS/FRED CUUR0000SETA02 · latest snapshot month {current.clock.meta['latest_observation']} · published {current.clock.meta['latest_release_date']}. {current.clock.live_status}. Monthly index, not a daily transaction feed.")
        if error:st.caption('Live refresh unavailable; using the dated saved snapshot. Forecast-horizon limits still apply.')
        st.info('Same selected date, different markets. These scenario prices still do not reveal dealer profit, a retail transaction price or a guaranteed discount.')
        st.caption('Ask uses a 2021 advertised-price anchor. Auction uses its 2015 reference-dollar model and an age-five anchor. Both use separately fitted depreciation; the common retail CPI is an explicit wholesale-transport assumption.')
        st.markdown('#### Same car. Different date. No time machine.')
        mileage_mode=st.radio('Mileage through time',['Hold mileage constant','Assume annual driving'],horizontal=True,key='time_mileage_mode')
        annual=st.slider('Assumed miles per year',0,30000,12000,1000,key='time_annual_miles') if mileage_mode=='Assume annual driving' else 0
        if st.button('Build value-over-time graph',type='primary',key='build_time_graph'):
            start=max(pd.Timestamp('2015-01-01'),pd.Timestamp(f"{int(car['year'])}-01-01"))
            end=pd.Timestamp(current.clock.meta['latest_observation'])+pd.DateOffset(months=6)
            months=pd.date_range(start,end,freq='MS').union(pd.DatetimeIndex([pd.Timestamp(today()),pd.Timestamp(date)]))
            with st.spinner('Following this car through the calendar…'):
                df=current.trajectory(pipelines,text,car,months,annual,allow,as_of=today())
            st.session_state['time_value_graph']={'frame':df,'signature':signature(car,text,annual,allow,current.clock.meta)}
        graph=st.session_state.get('time_value_graph')
        if graph and graph['signature']==signature(car,text,annual,allow,current.clock.meta):
            df=graph['frame']
            if df.empty:st.warning('No supported dates for this vehicle and market snapshot.')
            else:
                import plotly.graph_objects as go
                fig=go.Figure();colours={'Ask · advertised':'#6351e8','Bid · wholesale':'#309574'}
                for market,group in df.groupby('Market'):
                    for future,label,dash in [(False,'reconstructed / transported','solid'),(True,'future scenario','dash')]:
                        part=group[group.Date.gt(pd.Timestamp(today()))==future]
                        if part.empty:continue
                        fig.add_trace(go.Scatter(x=part.Date,y=part['Value USD'],mode='lines',name=market+' · '+label,line={'color':colours[market],'dash':dash,'width':3},customdata=part[['Mileage','Age','Market index']].to_numpy(),hovertemplate='%{x|%b %Y}<br>$%{y:,.0f}<br>%{customdata[0]:,.0f} miles · age %{customdata[1]:.1f}<br>%{customdata[2]}<extra>%{fullData.name}</extra>'))
                fig.add_vline(x=pd.Timestamp(today()).timestamp()*1000,line_dash='dot',line_color='#e57c57')
                fig.add_annotation(x=pd.Timestamp(today()),y=1,yref='paper',text='TODAY',showarrow=False,yshift=15,font={'color':'#b85f42'})
                fig.update_layout(height=440,margin={'t':40,'l':20,'r':20,'b':20},paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',yaxis_title='Scenario value · USD',xaxis_title='Valuation date',legend={'orientation':'h','y':-0.22},hovermode='x unified');fig.update_yaxes(tickprefix='$',tickformat=',.0f',rangemode='tozero')
                st.plotly_chart(fig,use_container_width=True,key='value_over_time')
                st.caption('Every car value is model-derived, including past dates. Solid lines are reconstructed/transported scenarios; dashed lines are future scenarios. Observed monthly CPI values do not make the individual-car curve observed. Annual driving is a user assumption; mileage is floored at zero. No validated whole-price confidence band is claimed.')
                st.download_button('Download this value timeline',df.to_csv(index=False),'car-value-over-time.csv','text/csv',key='download_time_graph')
        with st.expander('Time model · evidence, decomposition & limitations'):
            st.json(receipts)
            for side in ['ask','bid']:
                r=json.loads((folder/f'{side}-evaluation.json').read_text());st.write(side.upper()+' · depreciation component');st.write('Selected curve:',r['selected']);st.json(r['assessment']);st.caption(r['warning'])
            report=json.loads((folder/'market-backtest.json').read_text());st.write('Market forecast:',report['selected']);st.dataframe(pd.DataFrame(report['summary']),hide_index=True);st.caption(report['warning'])
            st.warning('An 11-year negative-age input is not a time correction. Vehicle age, model generation and calendar market level are separate. The source snapshots cannot identify unrestricted age/period/cohort effects or validate 2026 individual-car prices. These are conditional scenarios; unsold selection and retail-sale conversion remain unresolved.')
            st.markdown('[Full method, source research and reproduction](https://github.com/botsandbolts/Pricing/blob/main/TIME_TRANSPORT.md)')


def signature(car,text,annual,allow,meta):return json.dumps([car,text,annual,allow,meta['latest_observation'],meta['latest_release_date']],sort_keys=True,default=str)
