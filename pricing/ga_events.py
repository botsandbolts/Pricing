"""Bounded GA completion events. No prices, vehicle facts or text are transmitted."""
import uuid

SOURCES = {'current_pricing':'calculator','calculator':'calculator', 'calculator-companion':'calculator',
           'time-scenario':'value_over_time', 'robust_pair':'two_market',
           'suite':'model_lab', 'suite_robust':'model_lab', 'suite_pair':'model_lab',
           'integrated_new':'two_market', 'integrated':'two_market', 'integrated_robust':'two_market'}


def queue_estimate(side, source, event_key):
    import streamlit as st
    from pricing.analytics import valid_consent
    if side not in {'ask', 'bid'} or not valid_consent(st.session_state.get('_analytics_consent')):
        return
    feature = SOURCES.get(source)
    active = {'🚗 Price my car':'calculator', '↔ Two-market garage':'two_market',
              '🌳 Hierarchical model':'calculator', '📅 Value over time':'value_over_time', '🧪 Model lab':'model_lab'}.get(
                  st.session_state.get('studio_tabs', '🚗 Price my car'))
    if feature is None or feature != active:
        return
    # A stable explicit-request key prevents re-rendering receipts counting as new estimates.
    if not event_key:
        event_key = str(uuid.uuid4())
    marker = (side, str(source), str(event_key))
    seen = st.session_state.setdefault('_ga_estimate_seen', set())
    if marker in seen:
        return
    seen.add(marker)
    if len(seen) > 1000:
        seen.clear(); seen.add(marker)
    events = st.session_state.setdefault('_ga_events', [])
    events.append({'id': str(uuid.uuid4()), 'name':'estimate_completed',
                   'params':{'market_side':side,'feature_name':feature}})
    del events[:-50]


def render_events():
    """Mount after results so completed events are available in the same rerun."""
    import streamlit as st
    from pricing.analytics import valid_consent
    if not valid_consent(st.session_state.get('_analytics_consent')):
        st.session_state.pop('_ga_events', None)
        st.session_state.pop('_ga_estimate_seen', None)
        return
    try:
        import streamlit.components.v2 as components
        from pathlib import Path
        component = components.component('pricing_ga_events',
            js=(Path(__file__).parent/'consent'/'events.js').read_text())
        component(key='pricing_ga_events', data={'events':st.session_state.get('_ga_events', [])})
    except Exception:
        # Analytics must never break an estimate.
        pass
