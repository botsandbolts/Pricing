"""Explicit depreciation and release-aware market transport of historical RF prices."""
import math
import numpy as np
import pandas as pd
from marketbridge.time_engine import today


def adjust(transport,row,side,date,mileage,*,as_of=None):
    if side not in {'ask','bid'}:raise ValueError('Unknown market')
    fields=row.to_dict() if hasattr(row,'to_dict') else dict(row)
    fields['year']=int(fields['year']);fields['odometer']=float(mileage)
    _,age,miles=transport.validate(fields,date)
    source_age=float(fields['age']);source_miles=float(row['odometer'])
    if not np.isfinite(source_miles):raise ValueError('Recorded mileage is missing; choose another example.')
    if not 0<=source_age<=30:raise ValueError('Historical age exceeds the depreciation curve’s 0–30 year domain.')
    factor=float(transport.curves[side].ratio(pd.DataFrame([fields]),source_age,age,from_miles=source_miles,to_miles=miles)[0])
    level,market=transport.clock.value(date,as_of=as_of or today())
    if side=='ask':reference,reference_month=transport.clock.reference(fields['source_date'])
    else:reference=float(transport.clock.data.loc['2015-01']);reference_month='2015-01'
    market_factor=level/reference;base=float(row['base_price']);corrected=float(row['corrected_price'])
    result={'date':str(pd.Timestamp(date).date()),'source_date':str(pd.Timestamp(fields['source_date']).date()),'side':side,
        'forest_price':base*factor*market_factor,'price':corrected*factor*market_factor,
        'historical_forest_price':base,'historical_corrected_price':corrected,'source_age':source_age,'age':age,
        'source_mileage':source_miles,'mileage':miles,'vehicle_factor':factor,'market_factor':market_factor,
        'reference_index_month':reference_month,'market':market,'curve':transport.curves[side].kind,
        'information_date':str(as_of or today()),'status':'unvalidated_date_adjusted_scenario'}
    if not all(math.isfinite(result[k]) and 0<result[k]<=1e7 for k in ['price','forest_price']):raise ValueError('Unsupported transported price')
    return result


def render(root,row,side):
    import streamlit as st
    from pricing.current_value import current_transport,render_market_clock
    st.markdown('#### 📅 What about today?')
    mode=st.radio('Forest valuation date',['Today · follows the calendar','Choose a date'],horizontal=True,key='forest_date_mode')
    selected=today() if mode.startswith('Today') else st.date_input('Forest estimate for',today(),key='forest_custom_date')
    st.caption(f'Valuation date: {selected:%d %b %Y}. Today advances automatically on refresh; historical test scores remain unchanged.')
    miles=float(row.odometer)
    if not math.isfinite(miles):st.info('Choose an example with recorded mileage for date adjustment.');return
    mileage=st.number_input('Mileage at the selected date',min_value=0,max_value=2000000,value=int(miles),step=1000,key=f'forest_miles_{side}_{row.row_id}')
    st.caption(f'Recorded mileage: {miles:,.0f} on {str(row.source_date)[:10]}. The default holds mileage constant; enter your own assumption. No annual driving is silently invented.')
    try:
        with st.spinner('Checking the public market clock…'):
            transport,error=current_transport(root)
            r=adjust(transport,row,side,selected,mileage,as_of=today())
        a,b=st.columns(2)
        a.metric('Forest · date-adjusted scenario',f"${r['forest_price']:,.0f}")
        b.metric('Forest + hierarchy · date-adjusted scenario',f"${r['price']:,.0f}")
        st.caption(f"Age {r['source_age']:.2f} → {r['age']:.2f} years · ageing/mileage ×{r['vehicle_factor']:.3f} · market ×{r['market_factor']:.3f}")
        render_market_clock(transport,selected,error)
        st.plotly_chart(adjustment_chart(r),width='stretch',key='forest_date_waterfall')
        with st.expander('Date adjustment · evidence and assumptions'):
            st.write('Historical price × fitted age/mileage ratio × market-index ratio. The forest is not asked to extrapolate a negative year. The market clock forecasts beyond observed monthly data using the component-backtest-selected method, capped at six months.')
            st.write('Asking anchors use the original source month; auction anchors already use January 2015 CPI-reference dollars, which are applied once. Source dates and years are matched to exact original rows.')
            st.json(r)
            st.dataframe(pd.DataFrame(transport.clock.report.get('summary',[])),hide_index=True)
            st.caption('Component tests do not establish current-car MAE or an individual prediction interval. Cross-sectional depreciation is not identified causal ageing; retail CPI is a proxy for auction market movement.')
    except (ValueError,KeyError,OSError,TypeError) as exc:st.warning('Date-adjusted scenario unavailable: '+str(exc))


def adjustment_chart(r):
    import plotly.graph_objects as go
    base=r['historical_corrected_price'];aged=base*r['vehicle_factor']
    fig=go.Figure(go.Waterfall(x=['Historical RF + hierarchy','Age & mileage','Market movement',r['date']],measure=['absolute','relative','relative','total'],y=[base,aged-base,r['price']-aged,0],text=[f'${base:,.0f}',f'{aged-base:+,.0f}',f"{r['price']-aged:+,.0f}",f"${r['price']:,.0f}"],textposition='outside',totals={'marker_color':'#6351e8'},increasing={'marker_color':'#309574'},decreasing={'marker_color':'#ce6347'}))
    fig.update_layout(height=350,showlegend=False,margin={'l':15,'r':15,'t':30,'b':30},yaxis_title='Scenario USD');fig.update_yaxes(tickprefix='$',tickformat=',.0f',rangemode='tozero');fig.update_xaxes(type='category')
    return fig
