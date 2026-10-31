"""Connect current-model receipts to sharing, stories, telemetry and time scenarios."""
import json
from pathlib import Path
import streamlit as st
from pricing.pricing_comparison import selected_receipt

def publish_receipt(root,fields,notes,date,results,saved,bayesian):
    model='XGBoost Lemon Squeezer'+(' + Bayesian correction' if bayesian else '')
    request=saved['request_id']+':'+str(bayesian)
    from pricing.analytics import record_price
    for side,r in results.items():
        if 'error' not in r:
            record_price(side,r['price'],fields,model,'current_pricing',str(date),request+':'+side)
    ask=results.get('ask',{})
    if 'price' in ask:
        prediction={'price':ask['price'],'valuation_date':str(date),'prediction_kind':'date_adjusted_scenario','reference_date':str(date)}
        receipt=(model,prediction,fields,notes)
        st.session_state['last_explained_price']=receipt
        st.session_state['last_displayed_price']=receipt
    st.session_state['_current_pricing_mode']=bayesian

def build_report(root,fields,date,results):
    sections=[{'title':'Installed model','text':'Cleaned XGBoost: ask normal AFT with sparse TF-IDF descriptions; bid median quantile on log dollars. No MiniLM or UMAP. AFT uses exact observed price bounds; no verified censoring intervals are available.'},
              {'title':'Cleaning','text':'Five vehicle-group cross-validation models supply log residuals. One largest-quarter training trim precedes refitting. Validation and test remain untrimmed. Cleaning can discard valid unusual cars; improved accuracy from cleaning has not been demonstrated against a matched uncleaned model.'},
              {'title':'Other studio tools','text':'Sharing offers image cards, QR codes and public links containing vehicle inputs. Evidence shows historical auction comparables. The timeline reconstructs prices at entered mileage; these are not observed sales. No dealer-profit or guaranteed-discount claim is supported.'}]
    estimates={}
    for side,r in results.items():
        if 'error' in r:continue
        estimates[side]={'estimated_price_usd':r['price'],'historical_anchor_usd':r['historical_corrected_price'],
          'age_multiplier':r['vehicle_factor'],'market_multiplier':r['market_factor'],'vehicle_age_years':r['age'],
          'market_index_status':r['market'],'newer_generation_analogue':r['analogue']}
        report=json.loads((Path(root)/'models/xgboost-hierarchy'/side/'bayesian/bayesian-evaluation.json').read_text())
        sections.append({'title':side+' held-out evidence','text':f"Historical full-test regular MAE ${report['base_test_mae_usd']:,.2f}; Bayesian MAE ${report['point_scores']['mae_usd']:,.2f}. These are different from today's date-adjusted errors, which have not been measured. Bayesian ranges have historical calibration only; ask and bid are fitted separately. Gaussian/t/mixture LOO comparisons and genuine censoring models are not implemented."})
    return {'valuation_date':str(date),'market_estimates':estimates,'unavailable_markets':[s for s in ['ask','bid'] if 'error' in results.get(s,{})],
        'market_source':'BLS/FRED CUUR0000SETA02 monthly retail used-car CPI (also used as a wholesale proxy)',
        'latest_observed_month':next((r['market']['last_known_month'] for r in results.values() if 'error' not in r),None),'sections':sections}

def render_briefing(root,fields,notes,date,results):
    from pricing.bart_summary import build_valuation_story,render_story
    report=build_report(root,fields,date,results)
    st.markdown('### Your valuation briefing')
    st.caption('The source report follows the current estimates and their recorded evidence. AI prose is optional and never changes a price.')
    with st.expander('Read the complete report · facts, evidence and limitations'):
        st.text(build_valuation_story(fields,notes,report))
        st.download_button('Download valuation report',json.dumps(report,indent=2,default=str),'valuation-report.json','application/json')
    identity=json.dumps([fields,notes,report],sort_keys=True,default=str)
    if st.button('Generate BART overview',key='current_bart_requested'):
        st.session_state['_current_bart_report']=identity
    if st.session_state.get('_current_bart_report')==identity:
        render_story(fields,notes,valuation=report)

def timeline_frame(saved,annual,bayesian,transport):
    """Vectorize depreciation over months; no repeated model fits or neural calls."""
    import numpy as np
    import pandas as pd
    from marketbridge.time_engine import today,age_on
    fields=saved.get('fields')
    if not fields:
        try:fields=json.loads(saved['signature'])[0]
        except (KeyError,ValueError,TypeError):raise ValueError('Please estimate the car again to refresh this older saved receipt.')
    start=max(pd.Timestamp('2015-01-01'),pd.Timestamp(f"{int(fields['year'])+1}-01-01"))
    end=pd.Timestamp(transport.clock.meta['latest_observation'])+pd.DateOffset(months=6)
    dates=pd.date_range(start,end,freq='MS').union(pd.DatetimeIndex([pd.Timestamp(today()),pd.Timestamp(saved['date'])])).unique().sort_values()
    frames=[]
    for side,raw in saved['results'].items():
        if 'error' in raw:continue
        r=selected_receipt(raw,bayesian)
        accepted=[];levels=[]
        for date in dates:
            age=age_on(fields['year'],date)
            if not 0<=age<=30:continue
            try:level,meta=transport.clock.value(date,as_of=today())
            except ValueError:continue
            accepted.append(date);levels.append(level)
        if not accepted:continue
        ages=np.array([age_on(fields['year'],d) for d in accepted])
        miles=np.clip(fields['odometer']+annual*np.array([(d-pd.Timestamp(saved['date'])).days/365.25 for d in accepted]),0,2000000)
        cars=pd.DataFrame([fields]*len(accepted))
        factors=transport.curves[side].ratio(cars,r['source_age'],ages,from_miles=r['source_mileage'],to_miles=miles)
        reference=transport.clock.reference(r['source_date'])[0] if side=='ask' else float(transport.clock.data.loc['2015-01'])
        prices=r['historical_corrected_price']*factors*np.asarray(levels)/reference
        frames.append(pd.DataFrame({'Date':accepted,'USD':prices,'Market':'Seller asking' if side=='ask' else 'Wholesale auction','Mileage':miles,'Period':['Future scenario' if d>pd.Timestamp(today()) else 'Reconstructed past / present' for d in accepted]}))
    if not frames:raise ValueError('No supported dates for this car and the available market snapshot.')
    return pd.concat(frames,ignore_index=True)

def render_timeline(root):
    from marketbridge.time_engine import today
    from pricing.current_value import current_transport,render_market_clock
    import plotly.express as px
    st.header('📅 Value over time · same car, same models')
    saved=st.session_state.get('hierarchy_sparse_result')
    if not saved:
        st.info('Estimate your car in Price my car first. This timeline uses that exact estimate, not a different historical model.')
        from pricing.deal_detective import move_to
        st.button('Go to Price my car',on_click=move_to,args=('🚗 Price my car',))
        return
    fields=saved.get('fields')
    if not fields:
        try:fields=json.loads(saved['signature'])[0]
        except (KeyError,ValueError,TypeError):st.warning('Estimate the car again to refresh this older saved receipt.');return
    bayesian=st.session_state.get('_current_pricing_mode',False)
    st.write(f"{fields['year']} {fields['manufacturer']} {fields['model']} · {fields['odometer']:,} miles")
    st.caption(('Bayesian' if bayesian else 'Regular XGBoost')+' anchors. Historical values are reconstructed scenarios, not recorded sales. Zero annual miles holds mileage constant.')
    annual=st.slider('Assumed miles per year',0,30000,0,1000,key='current_timeline_annual')
    signature=json.dumps([saved,annual,bayesian,str(today())],sort_keys=True,default=str)
    try:
        transport,error=current_transport(root)
        signature+=json.dumps(transport.clock.meta,sort_keys=True,default=str)
        cached=st.session_state.get('_current_timeline',{})
        if cached.get('signature')!=signature:
            with st.spinner('Building both market timelines from your saved estimate…'):
                frame=timeline_frame(saved,annual,bayesian,transport)
            cached={'signature':signature,'frame':frame};st.session_state['_current_timeline']=cached
        frame=cached['frame']
        fig=px.line(frame,x='Date',y='USD',color='Market',line_dash='Period',hover_data=['Mileage'])
        fig.update_yaxes(title='Model scenario value · USD',tickprefix='$',tickformat=',.0f')
        st.plotly_chart(fig,width='stretch',key='current_timeline_chart')
        st.download_button('Download this value timeline',frame.to_csv(index=False),'car-value-over-time.csv','text/csv')
        render_market_clock(transport,today(),error)
        st.caption('Solid past values are reconstructions; dashed future values are scenarios. Mileage is floored at zero and capped at two million miles. A monthly index is not a daily transaction feed. Current individual-car accuracy and intervals are unvalidated.')
    except (ValueError,KeyError,OSError,TypeError) as exc:
        st.warning('Timeline unavailable: '+str(exc))


def render_drivers(root,fields,notes,results,bayesian):
    st.markdown('### Why this estimate?')
    st.caption('Explain the installed XGBoost model, not a substitute model. Feature associations are not causal effects or an inspection.')
    identity=json.dumps([fields,notes,results,bayesian],sort_keys=True,default=str)
    if st.button('Show current model price drivers',key='current_native_drivers'):
        st.session_state['_current_driver_report']=identity
    if st.session_state.get('_current_driver_report')!=identity:return
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from pricing.hierarchy_sparse_live import load,design_matrix
    for side,r in results.items():
        if 'error' in r:continue
        model,schema,posterior,draws,projector,report=load(str(root),side)
        matrix,_=design_matrix(model,schema,projector,{**fields,'text':notes},r['source_age'])
        terms=model.predict(matrix,pred_contribs=True)[0]
        names=list(model.feature_names)
        if projector is not None:
            n=len(names)-len(projector.get_feature_names_out())
            names=names[:n]+['Advert: '+t for t in projector.get_feature_names_out()]
        frame=pd.DataFrame({'Feature':names,'Log-dollar contribution':terms[:-1]})
        top=frame.reindex(frame['Log-dollar contribution'].abs().sort_values(ascending=False).index).head(15)
        st.subheader('Seller asking' if side=='ask' else 'Wholesale auction')
        st.plotly_chart(px.bar(top.sort_values('Log-dollar contribution'),x='Log-dollar contribution',y='Feature',orientation='h'),width='stretch')
        st.caption('Native TreeSHAP contributions add to the historical log-price prediction. Exponentiating a single contribution is a multiplicative model effect, not an independent dollar amount. Small contributions are grouped below.')
        # Dollar decomposition is an explicitly ordered transformation of log SHAP.
        chosen=top.index.tolist();rest=float(frame.drop(index=chosen)['Log-dollar contribution'].sum())
        ordered=[('Other features',rest)]+[(frame.loc[i,'Feature'],float(terms[i])) for i in reversed(chosen)]
        current=float(np.exp(terms[-1]));rows=[{'Step':'Model reference','Dollar change':current,'Result USD':current}]
        factor=r['vehicle_factor']*r['market_factor']
        for name,delta in ordered:
            next_value=current*np.exp(delta)
            rows.append({'Step':name,'Dollar change':next_value-current,'Result USD':next_value});current=next_value
        with st.expander(side.title()+' · ordered dollar explanation'):
            st.dataframe(rows,hide_index=True)
            st.write(f"Historical model ${current:,.0f} × date factors {factor:.3f} = regular date-adjusted ${current*factor:,.0f}.")
            if bayesian:st.write(f"Bayesian correction then changes the displayed value to ${r['price']:,.0f}; it is separate from the tree contributions.")
            st.caption('Dollar steps depend on this displayed order because the underlying model is in log dollars. The historical total is order-invariant; individual dollar steps are not.')
            st.download_button('Download '+side+' current feature contributions',frame.to_csv(index=False),side+'-current-shap.csv','text/csv')

def render_minilm(root,notes):
    st.markdown('### 🖍 MiniLM · which phrases move its description-price signal?')
    st.caption('Separate historical description-only experiment. This compact MiniLM/Ridge head is NOT an input to the current TF-IDF XGBoost estimate above. Its dollar effects must not be read as changes to today’s car valuation.')
    if not notes.strip():
        st.info('Add a seller description and estimate the car to investigate its wording.');return
    st.caption('Runs only on request, locally on CPU. The encoder reads up to 128 tokens; phrase removal can change truncation. Highlights allocate model response, not causal market premiums.')
    requested=st.button('Analyse description with MiniLM',key='current_minilm_phrases')
    cached=st.session_state.get('_current_minilm_analysis',{})
    if not requested and cached.get('notes')!=notes:return
    folder=Path(root)/'models/xgboost-sparse-minilm/ask'
    @st.cache_resource(show_spinner=False,max_entries=1)
    def encoder(path):
        from pricing.minilm_signal import Encoder
        return Encoder(path)
    try:
        from pricing.minilm_signal import price
        from pricing.text_attribution import explain_text,highlight_html,contribution_chart
        if requested:
            head=json.loads((folder/'minilm-head.json').read_text())
            model=encoder(str(folder/'minilm'))
            with st.spinner('Comparing phrase combinations with the compact MiniLM encoder…'):
                payload=explain_text(lambda text:float(price(head,model.encode([text]))[0]),notes,limit=6,pairs=2)
            st.session_state['_current_minilm_analysis']={'notes':notes,'payload':payload}
        else:payload=cached['payload']
        a,b,c=st.columns(3)
        a.metric('Blank-description historical signal',f"${payload['base']:,.0f}")
        b.metric('Full-description historical signal',f"${payload['prediction']:,.0f}")
        c.metric('Net description effect',f"${payload['difference']:+,.0f}")
        st.markdown(highlight_html(payload),unsafe_allow_html=True)
        chart=contribution_chart(payload,True)
        chart.update_xaxes(title='Allocated effect on historical description-only signal (USD)')
        st.plotly_chart(chart,width='stretch')
        st.caption('Grouped permutation Shapley attribution, holding the model fixed. Description-only historical asking-price associations; no vehicle-fact adjustment, no date transport and no sale-price guarantee. Current-estimate drivers are in the XGBoost section above.')
    except (ImportError,ValueError,OSError,RuntimeError):
        st.warning('The separate MiniLM analysis could not load. Your current estimates remain available.')
