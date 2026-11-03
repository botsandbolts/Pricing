"""Verify the installed MiniLM route and a real 25-rating shortlist."""
from pathlib import Path
from streamlit.testing.v1 import AppTest
import json,hashlib
root=Path('.')
meta=json.loads((root/'data/merchant-discovery-minilm/manifest.json').read_text())
assert 'MiniLM' in meta['semantic_model']
source=(root/'pricing/merchant_discovery_ui.py').read_text()
assert 'merchant_text_method' not in source
assert "data/merchant-discovery-minilm" in source
app=AppTest.from_string("from pathlib import Path\nfrom pricing.merchant_discovery_ui import render\nrender(Path('.'))",default_timeout=120).run()
assert not app.exception,app.exception
assert not any(x.label=='Text matching model' for x in app.radio)
assert len(app.pills)==25
for i,widget in enumerate(app.pills):
    widget.set_value(4 if i==0 else 1)
app.run()
app.button(key='merchant_find').click().run(timeout=120)
assert not app.exception,app.exception
assert any('Your shortlist' in x.value for x in app.markdown)
version=hashlib.sha256((root/'data/merchant-discovery-minilm/manifest.json').read_bytes()).hexdigest()[:12]
result=app.session_state['merchant_pairwise_v1_'+version]
assert len(result['positions'])==len(set(result['positions']))==25
assert all(a>=b for a,b in zip(result['scores'],result['scores'][1:]))
assert any(x.value.startswith('### 25.') for x in app.markdown)
print('MiniLM default: 25 ratings submitted; 25 distinct results rendered; no TF-IDF selector.')

assert 'pairwise' in result['method']
assert result['validation']['pairs'] > 0
