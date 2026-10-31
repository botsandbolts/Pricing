from streamlit.testing.v1 import AppTest

APP='''
import streamlit as st
from pricing.deal_detective import render
render(lambda text,fields:dict(price=12000),lambda fields,day:9000,'test')
'''

def test_importer_without_estimate_and_review_callback():
    at=AppTest.from_string(APP).run(timeout=30)
    assert not at.exception
    at.text_area(key='deal_advert_text').set_value('2015 Toyota Corolla | 95,000 miles | $13,500')
    next(b for b in at.button if b.label=='Extract vehicle details').click().run()
    assert not at.exception
    assert at.number_input(key='review_year').value==2015
    next(b for b in at.button if b.label.startswith('Use reviewed')).click().run()
    assert not at.exception
    assert at.session_state['vehicle_manufacturer']=='toyota'
    assert at.session_state['vehicle_odometer']==95000

def test_complete_estimate_render_park_and_compare():
    at=AppTest.from_string(APP)
    at.session_state['last_explained_price']=('test',{'price':12000},{'year':2015,'manufacturer':'toyota','model':'Corolla','odometer':95000},'Service history')
    at.run(timeout=30)
    assert not at.exception
    next(b for b in at.button if b.label=='Park this estimate').click().run()
    assert not at.exception and len(at.session_state['detective_garage'])==1
    next(b for b in at.button if b.label=='Compare all with the selected model').click().run()
    assert not at.exception
    assert at.session_state['garage_comparison']['rows'][0]['Bid · 2021-05-01 USD']==9000
    assert any(d.label=='⬇ Download image card · PNG' for d in at.get('download_button'))
