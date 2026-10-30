"""Shared current-date serving and UI; original trained estimators stay intact."""
from pathlib import Path

import streamlit as st

from marketbridge.market_clock import MarketClock, fetch_snapshot
from marketbridge.time_engine import TimeTransport, today


@st.cache_data(ttl=21600, show_spinner=False)
def live_index():
    try:
        return fetch_snapshot(), None
    except Exception:
        return None, 'Live index refresh unavailable; using the dated saved snapshot.'


def current_transport(root):
    live, error = live_index()
    folder = Path(root)/'models/time-transport'
    return TimeTransport(root, MarketClock(folder, live=live)), error


class DateAdjustedAsking:
    """Explain the same USD function displayed for the selected valuation date.

    Text perturbations must rerun MiniLM and the base tree before adjustment;
    copying a historical text attribution onto a current price is not allowed.
    """
    def __init__(self, pipeline, transport, valuation_date, as_of=None):
        self.base = pipeline
        self.transport = transport
        self.date = valuation_date
        self.as_of = as_of or today()
        self.text_model = getattr(pipeline, 'text_model', None)
        self.explanation_identity = ('date-adjusted', id(pipeline))
        self.manifest = {**pipeline.manifest, 'display_time_adjustment': {
            'date': str(valuation_date), 'information_date': str(self.as_of),
            'market_snapshot': transport.clock.meta,
            'curve': transport.manifest}}

    def adjust(self, historical, fields):
        receipt = self.transport.estimate(self.base, '', fields, self.date, 'ask',
            as_of=self.as_of, anchor_price=historical['price'])
        return {**historical, 'price': receipt['price'], 'range': None,
                'historical_price': historical['price'],
                'valuation_date': receipt['date'], 'reference_date': receipt['date'],
                'time_adjustment': receipt,
                'prediction_kind': 'date_adjusted_scenario'}

    def predict(self, text, fields, explain=False):
        return self.adjust(self.base.predict(text, fields, explain=explain), fields)


def clock_controls(prefix='main'):
    """Today follows Berlin's calendar on every app rerun, never a frozen widget."""
    day = today()
    with st.expander('Valuation date & auction coverage', expanded=False):
        mode = st.radio('Price date', ['Today · follows the calendar', 'Choose a date'],
                        horizontal=True, key=prefix+'_clock_mode')
        if mode.startswith('Today'):
            selected = day
        else:
            selected = st.date_input('Estimate for', day, key=prefix+'_clock_date')
        allow = st.checkbox('Allow auction generation analogues for 2016–2021 cars',
            value=False, key=prefix+'_allow_analogue', help='Auction training ends at model year 2015. This optional assumption cannot recover the unobserved newer generation.')
    st.markdown(f'**Valuation date: {selected:%d %b %Y}** · '+
                ('Today, automatically updated on refresh.' if selected == day else 'Selected-date estimate.'))
    st.caption('Enter the mileage and condition at that date. Ageing and market movement are adjusted separately; no extra mileage is silently added.')
    return selected, allow


def render_market_clock(transport, valuation_date, error=None):
    """Always visible evidence status, rather than hiding extrapolation in docs."""
    _, market = transport.clock.value(valuation_date, as_of=today())
    observed = market['last_known_month']
    if market['kind'] == 'forecast_monthly_index':
        st.info(f"Market clock: observed through {observed}; {market['month']} is extrapolated "
                f"{market['horizon']} month(s) using {market['method'].replace('_',' ')}. "
                'The index is monthly, not a live daily price feed.')
    else:
        st.info(f"Market clock: observed monthly index for {market['month']}. "
                'The individual car price is still model-estimated.')
    st.caption(f"BLS/FRED CUUR0000SETA02 · latest release {transport.clock.meta['latest_release_date']} · "
               f"{'refreshed public series' if transport.clock.live_status=='refreshed_public_index' else 'saved, dated snapshot'}. "
               'Retail index used as a wholesale proxy for auctions.')
    if error:
        st.caption(error+' Forecasts stop beyond six months of available observations.')
    st.warning('Date-adjusted estimates, not verified current sale prices. There is no measured current-car MAE or validated individual prediction interval for the full extrapolation.')


def render_adjustment(receipt):
    st.caption(f"Vehicle age {receipt['age']:.2f} years · ageing ×{receipt['age_factor']:.3f} · market ×{receipt['market_factor']:.3f}")
    if not receipt['model_known_to_curve']:
        st.caption('Sparse or unseen trim: pooled make/body depreciation is used.')
    if receipt['generation_analogue']:
        st.warning('Newer-generation auction analogue: this generation was not observed in auction training.')


def adjustment_chart(receipts):
    """Dollar changes telescope; this is a time decomposition, not SHAP."""
    import plotly.graph_objects as go
    fig = go.Figure()
    for side, r in receipts.items():
        base = r['anchor_price_reference_usd']
        aged = base*r['age_factor']
        fig.add_trace(go.Waterfall(name='Seller asking' if side=='ask' else 'Wholesale auction',
            x=['Historical anchor', 'Vehicle ageing', 'Market movement', 'Selected date'],
            y=[base, aged-base, r['price']-aged, 0],
            measure=['absolute', 'relative', 'relative', 'total'],
            hovertemplate='%{x}<br>%{y:,.2f} USD<extra>%{fullData.name}</extra>'))
    fig.update_layout(height=360, waterfallgroupgap=.3, yaxis_title='USD',
        margin=dict(l=10,r=10,t=15,b=25), legend=dict(orientation='h',y=-.2))
    return fig


def render_time_evidence(transport, receipts):
    st.markdown('#### From then to now · follow the dollars')
    st.write('Historical anchor → vehicle ageing → market movement → your selected date. The time evidence stays in plain sight; the equations can use the glovebox.')
    st.plotly_chart(adjustment_chart(receipts), width='stretch', key='current_adjustment_waterfall')
    st.caption('This is an arithmetic time adjustment, not causal feature attribution. Asking and auction anchors have different source periods and economic targets. For the full calendar curve, open 📅 Value over time.')
    with st.expander('How did we get from then to now? · evidence & backtests'):
        st.write('Historical model anchor × vehicle-age adjustment × market-index adjustment = selected-date estimate. The entered mileage stays fixed. No negative-age trick, inflation-only shortcut or hidden retraining.')
        import pandas as pd
        rows = []
        for side,r in receipts.items():
            rows.append({'Market':side, 'Historical anchor USD':r['anchor_price_reference_usd'],
                'Anchor age':r['anchor_age'], 'Age at valuation':r['age'],
                'Age multiplier':r['age_factor'], 'Market multiplier':r['market_factor'],
                'Adjusted USD':r['price']})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        report = transport.clock.report
        st.write('Market forecast selected by earlier rolling-origin tests:', report['selected'])
        if report.get('summary'):
            st.dataframe(pd.DataFrame(report['summary']), hide_index=True, width='stretch')
        st.caption(report.get('warning','Index tests do not validate individual-car prices.'))
        st.write('Depreciation uses separately fitted monotone curves with pooled make/model/body effects. Source-date cross-validation selected the curve family; age-tail tests stress extrapolation. These checks do not observe the same cars selling today.')
        st.json(receipts, expanded=False)
        st.link_button('Read the full time-model method and measured component tests',
            'https://github.com/botsandbolts/Pricing/blob/main/TIME_TRANSPORT.md')
