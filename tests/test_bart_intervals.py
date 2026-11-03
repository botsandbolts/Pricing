"""Prices/ranges must reach BART and remain visible even during model failure."""
import json
from datetime import date
from pricing.bart_summary import build_valuation_story,qualitative_report_source,valuation_receipt
from pricing.current_tools import build_report

def valuation():
    return {'valuation_date':'2026-11-03','market_estimates':{
        'ask':{'estimated_price_usd':19876,'historical_anchor_usd':25000,'age_multiplier':.8,'market_multiplier':.9938,'vehicle_age_years':10,
               'interval_low_usd':14500,'interval_high_usd':27300,'interval_level':.9},
        'bid':{'estimated_price_usd':14765,'historical_anchor_usd':18000,'age_multiplier':.8,'market_multiplier':1.025,'vehicle_age_years':10,
               'interval_low_usd':11200,'interval_high_usd':18900,'interval_level':.9}}}

def test_both_prices_and_intervals_reach_both_internal_stories():
    for source in [build_valuation_story,qualitative_report_source]:
        story=source({'year':2015,'manufacturer':'Toyota','model':'Corolla'},'Needs new tires.',valuation())
        for text in ['19,876','14,500','27,300','14,765','11,200','18,900','90%','2026-11-03','Needs new tires.']:
            assert text in story
        assert 'coverage at the selected valuation date is not validated' in story

def test_missing_market_or_interval_is_not_invented():
    v=valuation();v['market_estimates'].pop('bid')
    v['market_estimates']['ask'].pop('interval_low_usd')
    text=valuation_receipt(v)
    assert 'Wholesale auction estimate' not in text and 'prediction range' not in text

def test_report_preserves_exact_unrounded_price_and_bounds(tmp_path):
    folder=tmp_path/'models/xgboost-hierarchy/ask/bayesian';folder.mkdir(parents=True)
    (folder/'bayesian-evaluation.json').write_text(json.dumps({'base_test_mae_usd':3706,'point_scores':{'mae_usd':3591}}))
    result={'price':19876.31,'historical_corrected_price':25000,'vehicle_factor':.8,'market_factor':.9938,'age':10,
            'market':{'last_known_month':'2026-09'},'analogue':False,'interval_low':14500.24,'interval_high':27300.78,'interval_level':.9}
    report=build_report(tmp_path,{},date(2026,11,3),{'ask':result})
    r=report['market_estimates']['ask']
    assert (r['estimated_price_usd'],r['interval_low_usd'],r['interval_high_usd'],r['interval_level'])==(19876.31,14500.24,27300.78,.9)

def test_exact_overview_visible_when_bart_fails():
    from streamlit.testing.v1 import AppTest
    source="""
from unittest.mock import patch
import streamlit as st
from pricing.bart_summary import render_story
valuation=st.session_state['_valuation_report']
with patch('pricing.bart_summary.SummaryEngine',side_effect=RuntimeError('controlled load failure')):
    render_story({'manufacturer':'Toyota','model':'Corolla'},'PRIVATE INTERNAL STORY',valuation)
"""
    app=AppTest.from_string(source)
    app.session_state['_valuation_report']=valuation()
    app.run(timeout=30)
    assert not app.exception
    text=' '.join(x.value for x in app.markdown)
    for number in ['19,876','14,500','27,300','14,765','11,200','18,900']:assert number in text
    assert 'PRIVATE INTERNAL STORY' not in text

def test_unsupported_numeric_draft_retries_without_changing_exact_receipt():
    from streamlit.testing.v1 import AppTest
    source="""
from unittest.mock import patch
import streamlit as st
from pricing.bart_summary import render_story
class Engine:
    def encode(self,text):return text.split()
    def decode(self,tokens):return ' '.join(tokens)
    def generate(self,text,forbid_numbers=False):
        st.session_state['calls']=st.session_state.get('calls',0)+1
        return 'A vehicle valuation based on historical data.' if forbid_numbers else 'The price is 999999999 dollars.'
with patch('pricing.bart_summary.SummaryEngine',Engine):
    render_story({'manufacturer':'Toyota','model':'Corolla'},'PRIVATE SOURCE',st.session_state['_valuation_report'])
"""
    app=AppTest.from_string(source)
    app.session_state['_valuation_report']=valuation()
    app.run(timeout=30)
    assert not app.exception
    assert app.session_state['calls']==2
    text=' '.join(x.value for x in app.markdown)
    assert '999999999' not in text and 'historical data' in text
    assert all(n in text for n in ['19,876','14,500','27,300','14,765','11,200','18,900'])
