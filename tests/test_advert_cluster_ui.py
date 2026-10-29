from streamlit.testing.v1 import AppTest


def test_render_names_uncertainty_and_cache_invalidates(monkeypatch,tmp_path):
    import pricing.advert_cluster_ui as ui
    import pricing.advert_clusters as core
    calls = []
    (tmp_path/'manifest.json').write_text('{}')
    monkeypatch.setattr(ui, 'FOLDER', tmp_path)
    monkeypatch.setattr(ui, '_load', lambda *a: {})
    def classify(bundle, text, fields):
        calls.append(text)
        return dict(name='Family Sedan · C001', cluster_id=1, keywords=['family','sedan'], strength=.8, training_members=100)
    monkeypatch.setattr(core, 'classify', classify)
    app = AppTest.from_string('''
import streamlit as st
from pricing.advert_cluster_ui import render_advert_type
text=st.text_area('Description',value='An excellent clean family car with service history')
if st.button('Estimate'):
    st.session_state['estimated']=text
if 'estimated' in st.session_state:
    render_advert_type(st.session_state['estimated'], {'year':2015,'odometer':80000})
''').run()
    assert not app.exception and not calls
    app.button[0].click().run(timeout=30)
    assert not app.exception and len(calls) == 1
    assert any('Family Sedan' in x.value for x in app.markdown)
    app.run()
    assert len(calls) == 1
    app.text_area[0].set_value('Another family vehicle with a full service history').run()
    app.button[0].click().run()
    assert len(calls) == 2


def test_missing_bundle_is_graceful(monkeypatch,tmp_path):
    import pricing.advert_cluster_ui as ui
    monkeypatch.setattr(ui,'FOLDER',tmp_path)
    app = AppTest.from_string("from pricing.advert_cluster_ui import render_advert_type\nrender_advert_type('family sedan',{})").run()
    assert not app.exception
    assert any('unavailable' in x.value for x in app.caption)
