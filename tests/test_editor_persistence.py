"""Exercise widget cleanup without loading any pricing or neural models."""
from streamlit.testing.v1 import AppTest

def test_editor_values_remain_widget_state_while_panel_is_hidden():
    source='''
import streamlit as st
from pricing.forest_form import render
page=st.radio('Test page',['Editor','Other'],key='test_page')
if page=='Editor':
    render(description_sensitive='minilm-price')
else:
    st.write('Another panel')
'''
    app=AppTest.from_string(source).run()
    assert not app.exception
    next(w for w in app.text_input if w.label=='Make').set_value('Ford')
    next(w for w in app.text_input if w.label=='Model / trim').set_value('Focus')
    next(w for w in app.number_input if w.label=='Mileage (miles)').set_value(135000)
    app.text_area[0].set_value('Service history. Needs tires.')
    app.button[0].click().run()
    app.radio[0].set_value('Other').run()
    assert app.session_state['forest_car_manufacturer']=='Ford'
    assert app.session_state['forest_car_model']=='Focus'
    assert app.session_state['forest_car_odometer']==135000
    assert app.session_state['forest_description']=='Service history. Needs tires.'
    app.radio[0].set_value('Editor').run()
    assert not app.exception
    assert next(w.value for w in app.text_input if w.label=='Make')=='Ford'
    assert next(w.value for w in app.number_input if w.label=='Mileage (miles)')==135000
    assert app.text_area[0].value=='Service history. Needs tires.'
