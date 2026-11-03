"""Verify the installed Doc2Vec route and a real five-rating shortlist."""
from pathlib import Path
from streamlit.testing.v1 import AppTest
import json
root=Path('.')
meta=json.loads((root/'data/merchant-discovery-doc2vec/manifest.json').read_text())
assert 'Doc2Vec' in meta['semantic_model']
source=(root/'pricing/merchant_discovery_ui.py').read_text()
assert 'merchant_text_method' not in source
assert "data/merchant-discovery-doc2vec" in source
app=AppTest.from_string("from pathlib import Path\nfrom pricing.merchant_discovery_ui import render\nrender(Path('.'))",default_timeout=120).run()
assert not app.exception,app.exception
assert not any(x.label=='Text matching model' for x in app.radio)
for i,widget in enumerate(app.pills):
    widget.set_value(4 if i==0 else 1)
app.run()
app.button(key='merchant_find').click().run(timeout=120)
assert not app.exception,app.exception
assert any('Your shortlist' in x.value for x in app.markdown)
results=[v for k,v in app.session_state.filtered_state.items() if k.startswith('merchant_result_')]
assert len(results)==1 and len(set(results[0]['positions']))==10
print('Doc2Vec default: five ratings submitted; ten distinct results rendered; no TF-IDF selector.')
