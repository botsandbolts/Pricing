from unittest.mock import patch
from datetime import datetime, timezone
import uuid
import streamlit as st
from pricing.analytics import CONSENT_VERSION
from pricing.ga_events import queue_estimate


def test_completions_consent_deduplication_and_no_private_values():
    c=dict(choice='accepted',version=CONSENT_VERSION,consent_at=datetime.now(timezone.utc).isoformat(),
           visitor_id=str(uuid.uuid4()),session_id=str(uuid.uuid4()))
    with patch.dict(st.session_state,{},clear=True):
        queue_estimate('ask','calculator','request')
        assert '_ga_events' not in st.session_state
        st.session_state['_analytics_consent']=c
        queue_estimate('ask','calculator','request')
        queue_estimate('ask','calculator','request')
        queue_estimate('bid','calculator-companion','request')
        events=st.session_state['_ga_events']
        assert len(events)==2
        assert events[0]['params']=={'market_side':'ask','feature_name':'calculator'}
        assert events[1]['params']=={'market_side':'bid','feature_name':'calculator'}
        assert 'request' not in str(events)
        assert all(set(e)=={'id','name','params'} for e in events)
        queue_estimate('invalid','calculator','new')
        assert len(events)==2
