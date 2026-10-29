from streamlit.testing.v1 import AppTest


def test_summary_appears_after_estimate_and_is_session_cached(monkeypatch):
    import pricing.summary_ui as ui
    calls=[]
    monkeypatch.setattr(ui,'get_summary_engine',lambda:object())
    def summary(fields,description,engine):
        calls.append(description)
        return dict(status='ready',summary='Seller reports a serviced Ford Focus.',story='The make is Ford.')
    monkeypatch.setattr(ui,'summarize_vehicle',summary)
    app=AppTest.from_string('''
import streamlit as st
from pricing.summary_ui import render_vehicle_summary
text=st.text_area('Description',value='Service history')
if st.button('Estimate'):st.session_state['saved']=text
if 'saved' in st.session_state:render_vehicle_summary({'manufacturer':'Ford'},st.session_state['saved'])
''').run()
    assert not calls and not app.exception
    app.button[0].click().run()
    assert len(calls)==1 and not app.exception
    assert any('serviced Ford' in m.value for m in app.markdown)
    app.run();assert len(calls)==1
    app.text_area[0].set_value('New tires').run();app.button[0].click().run()
    assert len(calls)==2
