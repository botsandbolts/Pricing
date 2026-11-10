"""Regression against the current public BGE/Granite calculator's real examples.

The former historical sidebar has moved to the canonical US pricing page.
Example selection pre-fills fields but must never launch a paid/slow inference.
"""
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
    selector=app.selectbox(key='experimental_example')
    assert len(selector.options)==401
    selector.set_value(1).run()
    assert not app.exception
    assert app.text_area(key='bge_ask_description').value==examples[0]['text']
    assert app.number_input(key='bge_ask_year').value==examples[0]['fields']['year']
    assert app.text_input(key='bge_ask_model').value==examples[0]['fields']['model']
    assert 'experimental_combined_receipt' not in app.session_state
    app.text_area(key='bge_ask_description').set_value('My own editable description.').run()
    assert app.text_area(key='bge_ask_description').value=='My own editable description.'
    app.session_state['experimental_combined_receipt']={'obsolete':True}
    app.selectbox(key='experimental_example').set_value(400).run()
    assert app.text_area(key='bge_ask_description').value==examples[-1]['text']
    assert 'experimental_combined_receipt' not in app.session_state
    assert not app.exception
