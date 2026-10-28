"""End-to-end UI check with real saved models and the audited index snapshot."""
import json
from pathlib import Path
from streamlit.testing.v1 import AppTest
import pricing.time_ui as tui
# Avoid a network dependency in the GUI test; production refresh is tested separately.
tui.live_index=lambda:(None,'QA uses the audited snapshot')
root=Path(__file__).resolve().parents[1]
at=AppTest.from_file(str(root/'app.py')).run(timeout=180)
assert not at.exception,[e.value for e in at.exception]
assert at.selectbox(key='pricing_model').value=='MiniLM + CatBoost AFT · residual-trimmed (default)'
fields=dict(year=2015,manufacturer='toyota',model='corolla se',odometer=95000,condition='good',fuel='gas',transmission='automatic',drive='fwd',type='sedan',title_status='clean',cylinders='4 cylinders',paint_color='silver',state='ca')
text='Well maintained Toyota Corolla SE. Clean title, automatic transmission, regular oil changes, working air conditioning. Minor scratches and normal wear. Runs and drives well.'
for key in ['year','odometer']:at.number_input(key='vehicle_'+key).set_value(fields[key])
for key in ['manufacturer','model','state']:at.text_input(key='vehicle_'+key).set_value(fields[key])
for key in ['condition','fuel','transmission','drive','type','title_status','cylinders','paint_color']:at.selectbox(key='vehicle_'+key).set_value(fields[key])
at.text_area(key='seller_description').set_value(text)
next(b for b in at.button if b.label=='Estimate asking price').click().run(timeout=180)
assert not at.exception and not at.error,([x.value for x in at.exception],[x.value for x in at.error])
saved=at.session_state['last_explained_price'];assert abs(saved[1]['price']-10608.983456680162)<1e-7
assert at.selectbox(key='companion_bid_model').value=='CatBoost log-median · quantile 0.5 (default)'
metrics=[{'label':m.label,'value':m.value} for m in at.metric]
assert any('Current asking scenario' in r['label'] for r in metrics)
assert any('Current auction scenario' in r['label'] for r in metrics)
at.button(key='build_time_graph').click().run(timeout=180)
assert not at.exception and not at.error,([x.value for x in at.exception],[x.value for x in at.error])
frame=at.session_state['time_value_graph']['frame'];assert set(frame.Market)=={'Ask · advertised','Bid · wholesale'} and len(frame)>30
assert frame['Value USD'].gt(0).all() and len(at.get('plotly_chart'))>=1
at.radio(key='time_mileage_mode').set_value('Assume annual driving').run(timeout=180)
at.slider(key='time_annual_miles').set_value(10000)
at.button(key='build_time_graph').click().run(timeout=180)
assert not at.exception and not at.error
mileage_frame=at.session_state['time_value_graph']['frame'];assert mileage_frame.Mileage.max()>mileage_frame.Mileage.min()
# All scientific functions remain reachable after the design change.
assert any(b.label=='Build all explanation graphs' for b in at.button)
assert any(t.label=='📊 Evaluation' for t in at.tabs)
report={'status':'passed','historical_ask_price':saved[1]['price'],'historical_ask_unchanged':True,'bid_default':'quantile 0.5','metrics':metrics,'timeline_rows':len(frame),'timeline_markets':sorted(frame.Market.unique()),'annual_mileage_scenario':'passed','explanation_graph_controls_preserved':True,'newer_than_2021_guard':'unit tested','network':'UI QA uses saved audited index; live adapter verified separately'}
out=root/'models/time-transport/gui-smoke.json';out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
