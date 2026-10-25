"""Run with: python -m streamlit run app.py"""
import json
import os
from pathlib import Path
import streamlit as st
from pricing.text import MODEL_ID, MAX_CHARACTERS

st.set_page_config(page_title='pricing · E5 + LightGBM',page_icon='🚗',layout='wide')
st.title('pricing')
st.caption('Fine-tuned E5-base text regression + LightGBM · historical US asking prices')
path=Path(os.environ.get('PRICING_MODEL_DIR','artifacts/e5-base-isolation-text-10'))
ready=(path/'manifest.json').exists()
if not ready:
    st.info('The new model has not been trained yet. Complete the GPU training run to enable dollar estimates. The previous model is not used.')

@st.cache_resource
def get_pipeline(location):
    from pricing.inference import PricePipeline
    return PricePipeline(location)

value,how,results=st.tabs(['Value a car','How it works','Evaluation'])
with value:
    with st.form('vehicle'):
        a,b,c=st.columns(3)
        year=a.number_input('Year',min_value=1900,max_value=2027,value=2015)
        make=b.text_input('Make',value='Toyota')
        model=c.text_input('Model / trim',value='Corolla SE')
        a,b,c=st.columns(3)
        miles=a.number_input('Mileage (miles)',min_value=0,max_value=2000000,value=95000,step=1000)
        condition=b.selectbox('Condition',['unknown','excellent','good','fair','like new','new','salvage'])
        fuel=c.selectbox('Fuel',['gas','diesel','hybrid','electric','other','unknown'])
        a,b,c=st.columns(3)
        transmission=a.selectbox('Transmission',['automatic','manual','other','unknown'])
        drive=b.selectbox('Drive',['unknown','fwd','rwd','4wd'])
        body=c.selectbox('Body type',['sedan','SUV','truck','pickup','hatchback','coupe','wagon','van','convertible','other','unknown'])
        a,b,c=st.columns(3)
        title=a.selectbox('Title status',['clean','rebuilt','salvage','lien','missing','parts only','unknown'])
        cylinders=b.selectbox('Cylinders',['unknown','3 cylinders','4 cylinders','5 cylinders','6 cylinders','8 cylinders','10 cylinders','12 cylinders','other'])
        color=c.selectbox('Paint color',['unknown','white','black','silver','grey','blue','red','green','brown','yellow','orange','purple','custom'])
        state=st.text_input('US state abbreviation',value='ca',max_chars=2)
        notes=st.text_area('Seller description — free text',height=250,max_chars=MAX_CHARACTERS,
            placeholder='Describe maintenance, condition, equipment, defects and vehicle history. Long descriptions are processed in 512-token chunks.')
        submitted=st.form_submit_button('Estimate asking price',disabled=not ready)
    if submitted:
        if len(notes.strip())<80:
            st.warning('Please provide at least 80 characters of description, matching the training data requirement.')
        else:
            fields=dict(year=year,manufacturer=make,model=model,odometer=miles,condition=condition,
                fuel=fuel,transmission=transmission,drive=drive,type=body,title_status=title,
                cylinders=cylinders,paint_color=color,state=state)
            try:
                with st.spinner('Reading the description and calculating price…'):
                    p=get_pipeline(str(path)).predict(notes,fields)
                st.metric('Estimated asking price',f"${p['price']:,.0f}")
                st.caption(f"Text-only feature predicted_text_price: ${p['predicted_text_price']:,.0f}")
                st.write(f"Indicative 80% band: ${p['range'][0]:,.0f}–${p['range'][1]:,.0f}")
            except Exception as exc:
                st.error(f'Model unavailable: {exc}')
    st.caption('Snapshot: April–May 2021. Asking prices, not verified sale prices or present-day valuations. Newer cars are outside the training period.')
with how:
    st.write(f'**Text model:** `{MODEL_ID}`. XLM-RoBERTa, 768 dimensions, 512 tokens per chunk. Every transformer parameter is fine-tuned with a regression head against standardized log1p(price), using descriptions only. Longer descriptions up to 8,000 characters use pooled chunk representations.')
    st.write('**Dynamic feature:** reverse the target scaling and log transform to create one dollar-valued `predicted_text_price`. Descriptions and price-like phrases are cleaned consistently. Structured form fields enter LightGBM separately.')
    st.write('**Final model:** LightGBM combines the text-price scalar with year, mileage and categorical vehicle fields. The active 10% experiment compares dollar and log-price targets with Huber thresholds 0.5, 0.9 and 1.5. The validation-MAE winner supplies the displayed dollar estimate. Make/model use categorical values, with no separate embedding model.')
    st.write('**Leakage control:** three-fold cross-fitting creates each training row’s predicted_text_price using an E5 model that never saw its target. Isolation Forest fits that single dollar-valued scalar only, with 300 trees and automatic contamination. It removes only LightGBM training rows labeled outliers. Validation and test remain untouched. E5 training is shared across the Huber experiments.')
    st.write('For a log-target winner, the dollar point estimate is the back-transformed log prediction, not a bias-corrected arithmetic mean. Dollar-target winners use the inverse linear scaling. The 80% band is calibrated from validation log residuals; its actual test coverage is reported.')
with results:
    if (path/'evaluation.json').exists() and ready:
        e=json.loads((path/'evaluation.json').read_text())
        st.write(e['rows'])
        st.dataframe([{'Model':label,**e[key]} for key,label in [('text_only','Fine-tuned text only'),('tabular_only','Tabular LightGBM'),('stacked','Stacked E5 + LightGBM')] if key in e],hide_index=True)
        st.write(f"80% band test coverage: {e['band80_test_coverage']:.1%}")
    else:
        st.info('No evaluation exists for this architecture yet. Run training to produce held-out metrics; earlier model results do not apply.')

    grid=path/'experiments.json'
    baseline=Path('reports/isolation-text-10.json')
    result_path=grid if grid.exists() else baseline
    if result_path.exists():
        report=json.loads(result_path.read_text())
        st.subheader('6-configuration Huber experiment grid')
        st.write(report['plan']['stage'])
        if result_path==baseline:
            st.warning('These measured results are tabular-only baselines. The fine-tuned E5 stack has not been evaluated yet.')
        st.caption('Isolation Forest on out-of-fold text-price scalars × two target transforms × three Huber thresholds. Selection uses validation only. q10 and q90 estimate interval endpoints, not the central price.')
        records=report['experiments']
        st.dataframe([{'Experiment':r['id'],'Training rows':r['retained_rows'],
            'Validation MAE $':r['validation']['mae_usd'],'Validation RMSE $':r['validation']['rmse_usd'],
            'Test MAE $':r['test']['mae_usd'],'Test RMSE $':r['test']['rmse_usd']} for r in records],hide_index=True)
        st.write('Selected using validation:',report['selected_on_validation'])
        if report['quantile_intervals']:
            st.dataframe(report['quantile_intervals'],hide_index=True)
