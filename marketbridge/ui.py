"""Shared structural-transfer GUI: learned components versus explicit assumptions."""
import json
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
from marketbridge.engine import MarketComparison

@st.cache_resource
def _cached_comparison(root,version):return MarketComparison(root)

def comparison(root):
    # A newly pulled model must invalidate a warm Streamlit resource cache.
    manifests=sorted((Path(root)/'models').glob('**/manifest.json'))
    version=tuple((str(p.relative_to(root)),p.stat().st_mtime_ns,p.stat().st_size) for p in manifests)
    return _cached_comparison(str(root),version)

def show_prices(result):
    a,b=st.columns(2)
    a.metric('Pricing · 2021 asking estimate',f"${result['asking_price_usd']:,.0f}")
    price=result['transported_auction_price_usd']
    b.metric('CarBids · age- and market-adjusted auction equivalent',f'${price:,.0f}' if np.isfinite(price) else 'Unavailable')
    if result.get('asking_model_kind')=='regularized_lightgbm_ensemble':
        st.caption('Active Pricing model: the larger validation-selected, regularized LightGBM ensemble. Original baseline: '+f"${result['baseline_asking_price_usd']:,.0f}"+'.')
    elif result.get('robust_asking_price_usd') is not None:
        st.caption('Larger ensemble is available as an experiment; promotion criteria were not met.')
    st.caption('The asking price belongs to the April–May 2021 market. The auction equivalent uses the requested valuation date, its vehicle age and a lagged BLS used-car index. These are different targets; their difference is not dealer profit or a negotiated discount.')
    if result.get('asking_with_auction_structure_usd') is not None:
        st.metric('Experimental 2021 asking estimate · with auction structure',f"${result['asking_with_auction_structure_usd']:,.0f}")
        st.caption('A separately trained LightGBM uses transferred depreciation/relative-value features. It still predicts asking prices, not retail sales.')
    if not np.isfinite(price):st.warning('Auction equivalent unavailable: '+result['status']);return
    a,b,c=st.columns(3)
    a.metric('Age at valuation',f"{result['vehicle_age_years']:.1f} years")
    b.metric('Market index multiplier',f"{result['market_index_factor']:.3f}×")
    c.metric('One additional year · fixed mileage',f"{result['one_year_age_effect_pct']:.1f}%")
    st.caption(f"Index month used: {result['index_month_used']}. A two-month lag avoids unreleased current-month observations; the bundled values are a current-vintage snapshot, not a real-time historical archive.")
    if not result['model_known']:st.warning('This exact make/model was not observed. The model falls back to the make/market hierarchy; no fuzzy trim or generation match is invented.')
    if not result['generation_supported']:st.warning('This model year is outside 1982–2015 source coverage. Age can be computed, but newer generations/technology are unobserved.')
    if not result['age_supported'] or not result['mileage_supported']:st.warning('Age or mileage is outside source coverage. Treat this as extrapolation.')
    if not result['auction_condition_supplied']:st.caption('Auction grade unknown: learned missing-value route used. Seller adjectives are never converted into auction grades.')
    st.caption('Source residual sensitivity: '+f"${result['source_residual_sensitivity_low_usd']:,.0f}–${result['source_residual_sensitivity_high_usd']:,.0f}. This is NOT a validated target-date 80% interval. Transfer, generation and retail-channel uncertainty remain uncalibrated.")

def render_comparison(root):
    st.subheader('The car got older. The model should notice.')
    st.write('CarBids supplies depreciation and relative-value structure. Pricing supplies the asking-market model. A BLS index anchors market movement; it does not magically turn wholesale prices into retail transactions.')
    try:engine=comparison(str(root))
    except Exception as exc:st.error(f'Transfer models unavailable: {exc}');return
    single,batch,report,meaning=st.tabs(['Value and explain','Price every row','Transfer report card','Assumptions · two gears'])
    with single:
        with st.form('shared_market_vehicle'):
            a,b,c=st.columns(3)
            year=a.number_input('Model year',1886,2027,2014)
            make=b.text_input('Make','Toyota');model=c.text_input('Model','Camry')
            a,b,c=st.columns(3)
            as_of=a.date_input('Valuation date',value=date(2022,5,1),min_value=date(2013,3,1),max_value=date(2026,11,30))
            miles=b.number_input('Actual mileage · miles',0,2_000_000,95000,step=1000)
            state=c.text_input('US state','ca',max_chars=2)
            a,b,c=st.columns(3)
            condition=a.selectbox('Seller condition',['unknown','excellent','good','fair','like new','new','salvage'])
            transmission=b.selectbox('Transmission',['automatic','manual','unknown'])
            body=c.selectbox('Body class',['sedan','suv','coupe','hatchback','wagon','convertible','truck','van','unknown'])
            grade=st.selectbox('Actual auction grade · optional',['Not known']+[f'{i/10:.1f}' for i in range(10,51)])
            submitted=st.form_submit_button('Calculate age-adjusted prices')
        if submitted:
            fields=dict(year=year,manufacturer=make,model=model,odometer=miles,state=state,condition=condition,transmission=transmission,type=body,as_of=as_of.isoformat())
            if grade!='Not known':fields['auction_condition']=float(grade)
            result=engine.predict_frame(pd.DataFrame([fields])).iloc[0].to_dict()
            st.session_state['market_comparison']=(fields,result)
        saved=st.session_state.get('market_comparison')
        if saved:
            fields,result=saved
            st.caption(f"Submitted: {fields['year']} {fields['manufacturer']} {fields['model']} · {fields['odometer']:,} miles · valuation {fields['as_of']}. Submit again after edits.")
            show_prices(result)
            if np.isfinite(result['transported_auction_price_usd']):
                st.subheader('Follow the money, without inventing a time machine')
                steps=pd.DataFrame(engine.structural.decompose(fields));steps['multiplicative_effect']=np.exp(steps.log_effect)
                st.dataframe(steps,hide_index=True,width='stretch')
                st.caption('The first row holds this make/model, body class, transmission and state fixed at age 3, 36,000 miles, grade 3 in January-2015 index dollars. Subsequent effects are an ordered model decomposition, not causal estimates. No market-time effect is applied twice.')
                a,b=st.columns(2)
                a.write('**Same 2014 model, different dates**')
                pairs=pd.DataFrame([{**fields,'as_of':'2015-05-01'},{**fields,'as_of':'2022-05-01'}])
                demonstration=engine.structural.predict_frame(pairs)
                demonstration.insert(0,'valuation_date',['2015-05-01','2022-05-01'])
                a.dataframe(demonstration[['valuation_date','vehicle_age_years','auction_reference_2015_usd','market_index_factor','transported_auction_price_usd']],hide_index=True)
                b.caption('Mileage is held fixed to isolate ageing from market movement. For the same physical car driven over time, enter its later mileage too. Cross-sectional age patterns assume comparable generations; they are not longitudinal proof of depreciation.')
                with st.expander('Retail channel sensitivity · an assumption, not a fitted sale model'):
                    premium=st.slider('Assumed retail-versus-auction adjustment (%)',-30,60,0,step=1)
                    scenario=result['transported_auction_price_usd']*(1+premium/100)
                    st.metric('Assumption-based retail-sale scenario',f'${scenario:,.0f}')
                    st.warning('This adjustment is user-specified. It has not been learned from matched retail sale outcomes. At 0%, this is simply the auction-equivalent value; it is not a calibrated retail sale price.')
                st.download_button('Download valuation and decomposition',json.dumps({'inputs':fields,'estimates':result,'decomposition':steps.to_dict('records')},indent=2),file_name='structural-transfer-valuation.json')
            with st.expander('Highly technical · hierarchy and LightGBM attribution'):
                st.write('Robust clipped-residual empirical partial pooling shrinks sparse model effects toward make/global effects. Monotone Huber LightGBM models log-price residuals with non-increasing age/mileage and non-decreasing auction grade. Raw model year, source outcome text and asking targets are excluded from upstream features. Generation is unavailable, so post-2015 generations receive an explicit support warning.')
                if engine.robust:
                    st.write('**Active ensemble components**')
                    st.dataframe(pd.DataFrame(engine.robust.manifest['members'])[['file','weight','seed','target','trees']],hide_index=True)
                    st.caption('Members may use different target scales. Their dollar predictions are averaged; log/dollar SHAP values cannot be added across members.')
                    from marketbridge.robust_explanations import explain_member
                    member_index=st.selectbox('Explain an ensemble member',list(range(len(engine.robust.members))),format_func=lambda i:engine.robust.members[i][0]['file'])
                    native=explain_member(engine.robust,pd.DataFrame([{**fields,'as_of':'2021-05-01'}]),member_index)
                    st.write('Member target scale:',native['target']);st.write('SHAP reconstruction error:',native['additivity_error'])
                    st.dataframe(pd.DataFrame(native['features']).sort_values('contribution',key=abs,ascending=False).head(15),hide_index=True)
                    st.caption(native['method'])
                if engine.enhanced:
                    st.write('**Earlier single-model experiment attribution**')
                    reference=pd.DataFrame([{**fields,'as_of':'2021-05-01'}]);e=engine.enhanced.explain(reference)
                    st.dataframe(pd.DataFrame({'feature':list(e['contributions']),'log1p_contribution':list(e['contributions'].values())}).sort_values('log1p_contribution',key=abs,ascending=False),hide_index=True)
                    st.caption(e['method'])
    with batch:
        st.write('CSV fields: year, manufacturer/make, model, odometer/mileage_miles, and as_of (or date). Optional: state, transmission, type, auction_condition, row_id. The date is mandatory for structural transfer.')
        upload=st.file_uploader('Cars with valuation dates',type=['csv'],key='two_market_batch')
        if upload is not None and st.button('Attach structural sale equivalents to every row'):
            try:
                frame=pd.read_csv(upload)
                if frame.empty or len(frame)>20000:raise ValueError('Browser batches require 1–20,000 rows; use CLI for the complete database')
                output=engine.predict_frame(frame)
                st.session_state['market_batch']=pd.concat([frame.add_prefix('input_').reset_index(drop=True),output.reset_index(drop=True)],axis=1)
            except Exception as exc:st.error(str(exc))
        if 'market_batch' in st.session_state:
            output=st.session_state['market_batch'];st.dataframe(output.head(100),hide_index=True)
            st.download_button('Download every row with prices and support flags',output.to_csv(index=False),'pricing-with-structural-carbids.csv','text/csv')
        st.code('python -m marketbridge.engine --input training-cache/full-cohort.jsonl --output reports/pricing-with-carbids.csv',language='bash')
        st.caption('Rows are preserved. Missing dates/model years or invalid numeric fields receive a reason and no transported price. The CLI output excludes seller descriptions and contact information.')
    with report:
        source=json.loads((Path(root)/'models/structural-transfer/evaluation.json').read_text())
        st.write('**Structural source assessment · historical wholesale sales**');st.json(source,expanded=False)
        target=Path(root)/'models/transfer-asking/evaluation.json'
        if target.exists():
            e=json.loads(target.read_text());st.write('**Does auction structure improve 2021 asking predictions?**')
            st.dataframe(pd.DataFrame([{'model':name,**v['test']} for name,v in e['models'].items()]),hide_index=True)
            st.write('Paired test MAE change (negative is better):',round(e['paired_test_mae_delta_usd'],2))
            st.write('Day-cluster bootstrap 95% range:',e['paired_day_cluster_bootstrap_95pct'])
            st.caption(e['uncertainty_note']);st.json(e,expanded=False)
        robust_file=Path(root)/'models/robust-asking/evaluation.json'
        if robust_file.exists():
            r=json.loads(robust_file.read_text());st.write('**Larger regularized LightGBM ensemble**')
            st.dataframe(pd.DataFrame([{'model':'Original baseline',**r['baseline']['test']},{'model':'Robust ensemble',**r['test']}]),hide_index=True)
            st.write('Active model:', 'Robust ensemble' if r['promoted'] else 'Original baseline')
            st.write('Ensemble members:',r['ensemble_members'])
            if engine.robust:st.write('Total deployed trees:',sum(m['trees'] for m,_ in engine.robust.members))
            st.caption(r['test_note']);st.json(r,expanded=False)
            st.download_button('Download ensemble evaluation',json.dumps(r,indent=2),'robust-ensemble-evaluation.json','application/json')
        st.warning('Neither assessment validates retail-sale conversion or future transported interval coverage.')
    with meaning:
        st.write('**Plain English:** age is measured at your requested date, mileage has its own effect, and the market index adjusts the price level. The old auction model is not asked to pretend that a one-year-old car stayed one forever.')
        st.write('**Identification:** age = calendar time − model year. The short auction history cannot independently identify unrestricted age, period and generation effects. We anchor period using CPI, omit raw model year, partially pool make/model and assume age relationships transfer across comparable generations. That assumption may fail for redesigns, EVs, collector cars and changed condition mix.')
        st.write('**Market index:** official BLS CUUR0000SETA02, not seasonally adjusted, two-month lag. This is a retail-market index used as an explicit wholesale transport anchor, with a current-vintage snapshot. It is neither a segment-specific wholesale index nor a real-time archive. No future index fill is allowed.')
        st.write('**What is learned:** auction relative value and age/mileage/grade structure; an experimental LightGBM asking-price model using those features. **What remains assumed:** stability across eras/generations, wholesale index transport and any chosen retail premium. Retail calibration needs matched actual retail outcomes.')
        st.link_button('Pricing source','https://github.com/botsandbolts/Pricing');st.link_button('CarBids source','https://github.com/botsandbolts/Carbids')

def render_auction_companion(root,fields):
    try:
        fields={**fields,'as_of':'2021-05-01'}
        st.subheader('CarBids structural companion · evaluated for May 2021')
        show_prices(comparison(str(root)).predict_frame(pd.DataFrame([fields])).iloc[0].to_dict())
    except Exception as exc:st.warning(f'Structural companion unavailable: {exc}')

def render_asking_companion(root,fields):
    try:
        fields=dict(fields);fields['auction_condition']=fields.get('condition');fields['condition']=None
        st.subheader('Pricing + structural CarBids companion')
        show_prices(comparison(str(root)).predict_frame(pd.DataFrame([fields])).iloc[0].to_dict())
    except Exception as exc:st.warning(f'Structural companion unavailable: {exc}')
