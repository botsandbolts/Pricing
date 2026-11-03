"""Exercise installed XGBoost inference without loading any Bayesian posterior."""
from pathlib import Path
from datetime import date
import pricing.hierarchy_sparse_live as live
def forbidden(*args,**kwargs):raise AssertionError('Bayesian inference must not run')
live.load_bayesian=forbidden
live.posterior_predict=forbidden
root=Path('.')
results,_,_=live.estimate(root,dict(year=2015,manufacturer='Toyota',model='Corolla',odometer=95000,state='ca',text='Full service history.'),date.today(),True)
for side,r in results.items():
    assert 'error' not in r,(side,r)
    assert r['price']==r['forest_price'] and r['price']>0
    assert 'interval_low' not in r and 'posterior_draws' not in r
    assert live.load(str(root),side)[2] is None
from pricing.current_tools import build_report
report=build_report(root,{},date.today(),results)
assert not any('Bayesian MAE' in s['text'] for s in report['sections'])
from streamlit.testing.v1 import AppTest
app=AppTest.from_string("from pathlib import Path\nfrom pricing.hierarchy_sparse_studio import render\nrender(Path('.'))",default_timeout=90).run()
assert not app.exception,app.exception
assert not any('Bayesian' in str(w.options) for w in app.radio)
app.radio(key='xgboost_pricing_view').set_value('📊 Accuracy').run()
assert not app.exception,app.exception
assert len(app.metric)==2
print('PASS: both real XGBoost estimates; no posterior loading; single-model UI and correct MAE scorecard.')
