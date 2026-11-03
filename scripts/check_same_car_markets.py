"""Exercise same-car comparison from a saved Bayesian receipt."""
from streamlit.testing.v1 import AppTest
source="""from pathlib import Path
import streamlit as st
from pricing.release_contract import release_identity
from pricing.same_car_markets import render
st.session_state['hierarchy_sparse_result']={
'inference_mode':'hierarchical-only-v1','model_version':release_identity(Path('.')),
'fields':{'year':2014,'manufacturer':'Example','model':'Sedan','odometer':90000},
'date':'2026-11-03','results':{
'ask':{'price':15000,'interval_low':12000,'interval_high':18000,'interval_level':.9},
'bid':{'price':10000,'interval_low':8000,'interval_high':12000,'interval_level':.9}
}}
render(Path('.'))
"""
app=AppTest.from_string(source).run()
assert not app.exception,app.exception
assert len(app.dataframe)==1 and len(app.dataframe[0].value)==2
assert len(app.table)==1 and app.table[0].value.iloc[0,0]==5000
assert any('not dealer profit' in w.value for w in app.warning)
print('Saved same-car estimates, separate ranges, gap and caveats rendered.')
