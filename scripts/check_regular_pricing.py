from pathlib import Path
from datetime import date
from pricing.hierarchy_sparse_live import estimate,load
root=Path('.')
results,_,_=estimate(root,dict(year=2015,manufacturer='Toyota',model='Corolla',odometer=95000,state='ca',text='Full service history.'),date.today(),True)
for side,r in results.items():
    assert 'error' not in r,(side,r)
    assert r['posterior_draws']>0
    assert 0<r['interval_low']<r['price']<r['interval_high']
    assert load(str(root),side)[2] is not None
from streamlit.testing.v1 import AppTest
app=AppTest.from_string("from pathlib import Path\nfrom pricing.hierarchy_sparse_studio import render\nrender(Path('.'))",default_timeout=90).run()
assert not app.exception,app.exception
assert app.radio(key='hierarchical_only_pricing_view').value=='🌳 Hierarchical pricing'
assert not any('🚗 Regular pricing' in w.options for w in app.radio)
app.radio(key='hierarchical_only_pricing_view').set_value('📊 Accuracy comparison').run()
assert not app.exception,app.exception
assert len(app.metric)>=4
print('PASS: hierarchical estimates and ranges for both markets; no regular-only option.')
