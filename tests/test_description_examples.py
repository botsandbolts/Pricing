import json
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


def test_real_examples_load_matching_fields_remain_editable_and_clear_stale_price(monkeypatch):
    monkeypatch.chdir(ROOT)
    examples=json.loads((ROOT/'data/description-examples.json').read_text())['examples']
    assert len(examples)==400
    assert len({e['id'] for e in examples})==400
    assert all('price' not in e['fields'] for e in examples)
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
    assert not app.exception
    selector=app.selectbox(key='description_example')
    assert len(selector.options)==401
    selector.set_value(1).run()
    assert not app.exception
    assert app.text_area(key='seller_description').value==examples[0]['text']
    assert app.number_input(key='vehicle_year').value==examples[0]['fields']['year']
    assert app.text_input(key='vehicle_model').value==examples[0]['fields']['model']
    app.text_area(key='seller_description').set_value('My own editable description.')
    next(button for button in app.button if button.label=='Estimate asking price').click().run()
    assert app.text_area(key='seller_description').value=='My own editable description.'
    app.session_state['last_explained_price']=('obsolete',{}, {},'')
    app.session_state['pricing_visual_explanation']={'obsolete':True}
    app.selectbox(key='description_example').set_value(400).run()
    assert app.text_area(key='seller_description').value==examples[-1]['text']
    assert 'last_explained_price' not in app.session_state
    assert 'pricing_visual_explanation' not in app.session_state
    assert not app.exception
