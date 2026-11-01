"""Automatic overview UI tests use a fake engine: no model downloads or inference."""
from streamlit.testing.v1 import AppTest

SOURCE = """
import sys, types
from unittest.mock import patch
from datetime import date
import streamlit as st
from pricing.current_tools import render_briefing
from pricing.bart_summary import _cached_overview

model=st.text_input('Test vehicle model','Corolla',key='test_model')
class Engine:
    def encode(self,text): return text.split()
    def decode(self,tokens): return ' '.join(tokens)
    def generate(self,text,forbid_numbers=False):
        st.session_state['_fake_calls']=st.session_state.get('_fake_calls',0)+1
        if st.session_state.get('_fail_once',False):
            st.session_state['_fail_once']=False
            raise RuntimeError('Test-only temporary failure')
        return 'Automatic overview for '+model+'.'

stub=types.ModuleType('pricing.hierarchy_sparse_live')
stub._load=types.SimpleNamespace(clear=lambda:None)
report={'valuation_date':'2026-11-01','market_estimates':{'ask':{
'estimated_price_usd':8000,'historical_anchor_usd':10000,'age_multiplier':.8,
'market_multiplier':1.,'vehicle_age_years':11,'market_index_status':{'kind':'observed'}}},
'sections':[{'title':'Evaluation technique','text':'INTERNAL METHOD INPUT ONLY'}]}
with patch('pricing.current_tools.build_report',return_value=report), patch('pricing.bart_summary.SummaryEngine',Engine), patch.dict(sys.modules,{'pricing.hierarchy_sparse_live':stub}):
    render_briefing('.',{'manufacturer':'Toyota','model':model},'INTERNAL SELLER INPUT ONLY',date(2026,11,1),{'ask':{'price':8000}})
st.button('Unrelated action')
"""

def test_automatic_summary_is_visible_and_assembled_story_is_hidden():
    app=AppTest.from_string(SOURCE).run(timeout=20)
    assert not app.exception
    text='\n'.join(item.value for item in app.markdown)
    assert 'Automatic overview for Corolla.' in text
    assert 'INTERNAL SELLER INPUT ONLY' not in text
    assert 'INTERNAL METHOD INPUT ONLY' not in text
    assert not app.expander and not app.get('download_button')
    assert all('Generate BART' not in b.label for b in app.button)
    assert app.session_state['_fake_calls']==1

def test_reruns_and_revisiting_a_valuation_reuse_cached_overviews():
    app=AppTest.from_string(SOURCE).run(timeout=20)
    app.button[0].click().run()
    assert app.session_state['_fake_calls']==1
    app.text_input[0].set_value('Camry').run()
    assert app.session_state['_fake_calls']==2
    app.text_input[0].set_value('Corolla').run()
    assert app.session_state['_fake_calls']==2
    assert 'Automatic overview for Corolla.' in '\n'.join(item.value for item in app.markdown)

def test_temporary_failure_is_cached_and_retry_can_recover():
    app=AppTest.from_string(SOURCE)
    app.session_state['_fail_once']=True
    app.run(timeout=20)
    assert not app.exception and app.session_state['_fake_calls']==1
    next(b for b in app.button if b.label=='Unrelated action').click().run()
    assert app.session_state['_fake_calls']==1
    next(b for b in app.button if b.label=='Retry summary').click().run()
    assert not app.exception and app.session_state['_fake_calls']==2
    assert 'Automatic overview for Corolla.' in '\n'.join(item.value for item in app.markdown)

def test_session_cache_is_bounded():
    from pricing.bart_summary import _cached_overview
    state={}
    for i in range(7):_cached_overview(state,str(i),lambda i=i:{'status':'ready','summary':str(i)})
    assert len(state['_bart_overview_cache'])==4
    assert list(state['_bart_overview_cache'])==['3','4','5','6']
