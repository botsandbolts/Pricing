"""Same saved vehicle, date and mileage; no additional model loading."""
import math
DISCLAIMER=('Experimental comparison of modelled asking and wholesale prices for the same vehicle scenario. '
'Estimates describe different markets and are not verified acquisition or resale prices. '
'The difference is not dealer profit, a promised discount, or evidence concerning tax compliance, '
'undeclared work, legality or business conduct. Merchant services and costs—including preparation, '
'repairs, warranties, transport, overhead and financing—may vary considerably. Historical data, '
'missing vehicle information and date extrapolation limit comparability.')
def comparison(saved):
    fields=saved['fields'];results=saved['results'];rows=[];values={}
    for side,title in [('ask','Bayesian asking estimate'),('bid','Bayesian wholesale estimate')]:
        r=results.get(side,{})
        try:
            price=float(r['price'])
            if not math.isfinite(price) or price<=0:raise ValueError()
        except (KeyError,TypeError,ValueError):
            rows.append({'Market':title,'Estimate · USD':None,'Historical range lower · USD':None,
                         'Historical range upper · USD':None,'Range level':None,
                         'Coverage / assumptions':str(r.get('error','Estimate unavailable'))})
            continue
        values[side]=price
        notes=[]
        if r.get('analogue'):notes.append('Unobserved auction generation: explicit analogue assumption')
        market=r.get('market',{})
        if market.get('horizon'):notes.append(str(market['horizon'])+'-month market forecast')
        notes.append('Selected-date range coverage unvalidated')
        def finite(name):
            try:
                v=float(r[name]);return v if math.isfinite(v) else None
            except (KeyError,TypeError,ValueError):return None
        rows.append({'Market':title,'Estimate · USD':price,
                     'Historical range lower · USD':finite('interval_low'),
                     'Historical range upper · USD':finite('interval_high'),
                     'Range level':r.get('interval_level'),
                     'Coverage / assumptions':'; '.join(notes)})
    gap=None
    if len(values)==2:
        gap={'Modelled asking–wholesale gap · USD':values['ask']-values['bid'],
             'Gap / wholesale estimate · %':100*(values['ask']/values['bid']-1)}
    return rows,gap
def render(root):
    import streamlit as st
    import pandas as pd
    from pricing.release_contract import release_identity
    st.subheader('Same car · two Bayesian markets')
    st.write('Use the exact saved Price my car estimates. Same vehicle, mileage and valuation date; different market questions.')
    saved=st.session_state.get('hierarchy_sparse_result')
    if (not saved or saved.get('inference_mode')!='hierarchical-only-v1'
        or not saved.get('fields') or saved.get('model_version')!=release_identity(root)):
        st.info('Estimate your car in Price my car first, or refresh it after a model update. This table reuses the completed receipt without loading pricing models again.')
        st.caption(DISCLAIMER)
        if st.button('Open the full car form',key='merchant_same_car_open'):
            st.session_state['_navigate_to']='🚗 Price my car'
            st.rerun()
        return
    f=saved['fields']
    st.write(f"{f.get('year','')} {f.get('manufacturer','')} {f.get('model','')} · {f['odometer']:,.0f} miles · valuation date {saved['date']}")
    st.caption('Inferred merchant names in the research below do not identify who owned, purchased or sold this car. No observed advert price or dealer acquisition cost is supplied by this saved estimate.')
    rows,gap=comparison(saved)
    st.dataframe(pd.DataFrame(rows).round(2),hide_index=True,width='stretch')
    if gap is not None:
        st.table(pd.DataFrame([gap]).round(2))
    else:st.info('A gap requires both supported estimates. An unavailable wholesale estimate is not replaced with a fabricated value.')
    st.warning(DISCLAIMER)
    st.caption('Ranges belong to each market separately. No gap interval is shown: subtracting range endpoints would not produce a validated joint interval. A negative gap is possible and does not establish an arbitrage opportunity.')
    with st.expander('How this comparison works'):
        st.write('Reuses the installed dealer-neutral XGBoost + Bayesian corrections and their saved date adjustments. The asking model targets advertised prices; wholesale targets auction outcomes. The historical monthly retail index is also used as a wholesale proxy. No dealer-specific correction, new inference or refitting occurs in this panel. Range levels and coverage limitations come from the existing receipt.')
