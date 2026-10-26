"""Run with: python -m streamlit run app.py"""
import json
import os
from pathlib import Path
import streamlit as st
from pricing.text import MODEL_ID, MAX_CHARACTERS

st.set_page_config(page_title='Car Price Studio',page_icon='🚗',layout='wide')
st.title('Car Price Studio')
st.caption('100% database · historical US asking prices · transparent model status')
mode=st.selectbox('Pricing mode',['Trained tabular baseline','MiniLM + Isolation Forest (training required)'])
path=Path(os.environ.get('PRICING_MODEL_DIR','models/studio-baseline' if mode=='Trained tabular baseline' else 'artifacts/car-price-studio-full'))
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
        notes=st.text_area('Seller description — free text',height=250,max_chars=MAX_CHARACTERS,disabled=mode=='Trained tabular baseline',
            placeholder='The trained baseline does not use descriptions. Free text is used only after MiniLM training.')
        submitted=st.form_submit_button('Estimate asking price',disabled=not ready)
    if submitted:
        if len(notes)>8000:
            st.warning('Please keep the description within 8,000 characters.')
        else:
            fields=dict(year=year,manufacturer=make,model=model,odometer=miles,condition=condition,
                fuel=fuel,transmission=transmission,drive=drive,type=body,title_status=title,
                cylinders=cylinders,paint_color=color,state=state)
            try:
                with st.spinner('Reading the description and calculating price…'):
                    p=get_pipeline(str(path)).predict(notes,fields)
                st.metric('Estimated asking price',f"${p['price']:,.0f}")
                if p.get('predicted_text_price') is not None:
                    st.caption(f"Description-price feature: ${p['predicted_text_price']:,.0f}")
                st.write(f"Indicative 80% band: ${p['range'][0]:,.0f}–${p['range'][1]:,.0f}")
            except Exception as exc:
                st.error(f'Model unavailable: {exc}')
    st.caption('Snapshot: April–May 2021. Asking prices, not verified sale prices or present-day valuations. Newer cars are outside the training period.')
with how:
    st.write('Make and model enter LightGBM as ordinary categorical fields. Their transformer and separate neural price head have been removed.')
    st.write('The active baseline uses structured car fields only. Its scores are measured on a chronological test set. It does not use descriptions or the MiniLM-price filter.')
    st.write('The MiniLM workflow fine-tunes sentence-transformers/all-MiniLM-L6-v2 on descriptions and log prices. Three-fold predictions create predicted_text_price without using each predicted row’s target in its fitting model.')
    st.write('Isolation Forest uses contamination="auto" on training OOF description prices only. No fixed percentage is removed; validation and test are unfiltered. Huber LightGBM combines retained training prices with the structured fields. This enhanced mode still requires GPU training.')
    st.write('The complete raw database is scanned before eligibility checks and deduplication. The eligible sample is split chronologically 70/15/15. Historical 2021 asking prices are not completed-sale prices or current valuations.')
with results:
    if (path/'evaluation.json').exists() and ready:
        e=json.loads((path/'evaluation.json').read_text())
        if e.get('mode')=='tabular_baseline':
            st.info(e['label']);st.json(e['test'])
            st.write({k:e[k] for k in ['train_rows','validation_rows','test_rows','eligible_rows']})
            st.write('Final deployment refit rows:',e.get('deployment_refit_rows',e['train_rows']))
            st.caption(e.get('assessment_note','Chronological held-out assessment.'))
        else:
            st.write(e['rows']);st.json(e['stacked'])
        st.write(f"Indicative 80% band test coverage: {e['band80_test_coverage']:.1%}")
    else:st.info('This mode has no completed evaluation. Train MiniLM to enable it; baseline scores do not apply.')
