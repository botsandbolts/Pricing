"""Exercise the installed defaults in Streamlit, plus old bid comparisons.

Run from repository root after installation. Optional --ask-before PATH verifies
an independently captured pre-installation asking prediction on identical input.
"""
import argparse,json
from pathlib import Path
from streamlit.testing.v1 import AppTest
from pricing.inference import PricePipeline
from pricing.residual_ui import bid_folder,model_root

p=argparse.ArgumentParser();p.add_argument('--ask-before');p.add_argument('--output',default='/tmp/bid-huber-gui-check.json');a=p.parse_args()
fields=dict(year=2015,manufacturer='toyota',model='corolla se',odometer=95000,condition='good',fuel='gas',transmission='automatic',drive='fwd',type='sedan',title_status='clean',cylinders='4 cylinders',paint_color='silver',state='ca')
text='Well maintained Toyota Corolla SE. Clean title, automatic transmission, regular oil changes, working air conditioning. Minor scratches and normal wear. Runs and drives well.'
root=Path('.')
assert model_root(root)==Path('models/minilm-catboost-aft')
assert bid_folder(root)==Path('models/bid-huber-cleaned')
ask=PricePipeline('models/minilm-catboost-aft/ask').predict(text,fields,explain=True)
if a.ask_before:
    before=json.loads(Path(a.ask_before).read_text())
    assert before['fields']==fields and before['text']==text
    assert ask['price']==before['price'] and ask['predicted_text_price']==before['text_price']
at=AppTest.from_file(str((root/'app.py').resolve())).run(timeout=180)
assert not at.exception,[x.value for x in at.exception]
assert at.selectbox[0].value=='MiniLM + CatBoost AFT · residual-trimmed (default)'
for key in ['year','odometer']:at.number_input(key='vehicle_'+key).set_value(fields[key])
for key in ['manufacturer','model','state']:at.text_input(key='vehicle_'+key).set_value(fields[key])
for key in ['condition','fuel','transmission','drive','type','title_status','cylinders','paint_color']:at.selectbox(key='vehicle_'+key).set_value(fields[key])
at.text_area(key='seller_description').set_value(text)
next(b for b in at.button if b.label=='Estimate asking price').click().run(timeout=180)
assert not at.exception and not at.error,([x.value for x in at.exception],[x.value for x in at.error])
label='CatBoost log-Huber · delta 0.2 (default)'
assert at.selectbox(key='companion_bid_model').value==label
metrics=[dict(label=m.label,value=m.value) for m in at.metric]
assert next(m['value'] for m in metrics if m['label']=='Estimated asking price')==f"${ask['price']:,.0f}"
new_bid=next(m['value'] for m in metrics if m['label']=='Transported wholesale auction estimate')
old_results={}
for previous in ['CatBoost log-median · quantile 0.5 · previous','CatBoost normal AFT · previous']:
    at.selectbox(key='companion_bid_model').set_value(previous).run(timeout=180)
    assert not at.exception and not at.error
    old_results[previous]=next(m.value for m in at.metric if m.label=='Transported wholesale auction estimate')
at.selectbox(key='companion_bid_model').set_value(label).run(timeout=180)
assert not at.exception and not at.error
# The integrated calculator independently resolves the active bid bundle.
at.text_input(key='integrated_new_make').set_value('Toyota')
at.text_input(key='integrated_new_model').set_value('Corolla SE')
at.text_input(key='integrated_new_state').set_value('ca')
at.text_area(key='integrated_new_description').set_value(text)
next(b for b in at.button if b.label=='Price it in both markets').click().run(timeout=180)
assert not at.exception and not at.error
frames=[d.value for d in at.dataframe]
assert any('Model' in f and f.Model.astype(str).str.contains('CatBoost log-Huber').any() for f in frames)
# Native log-SHAP restoration is checked on the actual deployed bid model.
bid=PricePipeline(str(bid_folder(root))).predict('',fields,explain=True)
assert bid['explanation']['additivity_error']<1e-5
result={'status':'passed','ask_default_unchanged':True,'ask_prediction':ask['price'],'ask_text_prediction':ask['predicted_text_price'],'ask_native_shap_error':ask['explanation']['additivity_error'],'pre_install_prediction_compared':bool(a.ask_before),'bid_default':label,'bid_native_shap_error':bid['explanation']['additivity_error'],'metrics':metrics,'previous_bid_comparisons':old_results,'integrated_calculator_uses_huber':True,'selection_imr':'disabled; genuine sold/unsold population unavailable'}
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
