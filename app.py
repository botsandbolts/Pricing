"""Run with: python -m streamlit run app.py"""
import json
import os
from pathlib import Path
import streamlit as st
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHARACTERS = 8000

st.set_page_config(page_title='Car Price Studio',page_icon='🚗',layout='wide')
st.title('Car Price Studio + CarBids')
st.caption('Age-aware auction transfer and historical asking prices. Serious maths. Zero tyre-kicking.')
from marketbridge.engine import best_asking_bundle
choices=['Best trained ensemble','Unfiltered trained ensemble','Trained tabular baseline','MiniLM + Isolation Forest (training required)']
filtered=Path('models/robust-asking-isolation')
if (filtered/'manifest.json').exists():choices.insert(1,'Isolation Forest trained ensemble')
mode=st.selectbox('Pricing model',choices)
locations={'Best trained ensemble':best_asking_bundle(Path('.')),
           'Isolation Forest trained ensemble':filtered,
           'Unfiltered trained ensemble':Path('models/robust-asking'),
           'Trained tabular baseline':Path('models/studio-baseline'),
           'MiniLM + Isolation Forest (training required)':Path('artifacts/car-price-studio-full')}
path=Path(os.environ.get('PRICING_MODEL_DIR',str(locations[mode])))
ready=(path/'manifest.json').exists()
if not ready:
    st.info('This historical MiniLM mode has not been trained. The integrated Pricing + CarBids tab uses its own completed models and remains available.')

@st.cache_resource
def _get_pipeline(location,version):
    from pricing.inference import PricePipeline
    return PricePipeline(location)

def get_pipeline(location):
    file=Path(location)/'manifest.json'
    return _get_pipeline(location,file.stat().st_mtime_ns)

compare,value,how,results=st.tabs(['Pricing + CarBids','Value a car','How it works','Evaluation'])
with compare:
    from marketbridge.ui import render_comparison
    render_comparison(Path(__file__).parent)
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
        notes=st.text_area('Seller description — free text',height=250,max_chars=MAX_CHARACTERS,disabled=not mode.startswith('MiniLM'),
            placeholder='The trained ensemble/baseline uses structured fields. Free text is used only after MiniLM training.')
        submitted=st.form_submit_button('Estimate asking price',disabled=not ready)
    if submitted:
        if len(notes)>8000:
            st.warning('Please keep the description within 8,000 characters.')
        else:
            fields=dict(year=year,manufacturer=make,model=model,odometer=miles,condition=condition,
                fuel=fuel,transmission=transmission,drive=drive,type=body,title_status=title,
                cylinders=cylinders,paint_color=color,state=state)
            try:
                with st.spinner('Calculating the asking-price estimate…'):
                    p=get_pipeline(str(path)).predict(notes,fields,explain=True)
                st.session_state['last_explained_price']=(str(path),p,fields,notes)
            except Exception as exc:
                st.error(f'Model unavailable: {exc}')
    saved=st.session_state.get('last_explained_price')
    if saved and saved[0]==str(path):
        from pricing.explanations import render_explanation
        st.divider()
        result=saved[1];car=saved[2]
        st.caption(f"Explaining submitted car: {car['year']} {car['manufacturer']} {car['model']} · {car['odometer']:,} miles")
        st.metric('Estimated asking price',f"${result['price']:,.0f}")
        st.write(f"Indicative 80% band: ${result['range'][0]:,.0f}–${result['range'][1]:,.0f}")
        if result.get('predicted_text_price') is not None:
            st.caption(f"Description-price feature: ${result['predicted_text_price']:,.0f}")
        from marketbridge.ui import render_auction_companion
        render_auction_companion(Path(__file__).parent,car)
        render_explanation(saved[1],saved[2],saved[3],get_pipeline(str(path)))
    st.caption('Snapshot: April–May 2021. Asking prices, not verified sale prices or present-day valuations. Newer cars are outside the training period.')
with how:
    st.subheader('A pricing committee with thousands of small opinions')
    st.write('LightGBM learns lots of small decision trees, each correcting earlier mistakes. Huber loss softens the influence of large errors: one wildly ambitious seller does not get to chair the committee. The fitted tree count and exact drivers appear after an estimate.')
    st.write('Make and model enter LightGBM as ordinary categorical fields. Their transformer and separate neural price head have been removed.')
    st.write('The trained ensembles average regularized LightGBM models with age-adjusted CarBids features. The filtered experiment uses Isolation Forest with an automatic cutoff on training vehicle facts and frozen CarBids signals; asking-price labels are excluded. Validation and test cars are never filtered. The original unfiltered ensemble remains selectable. The best-model choice uses the filtered model only if its recorded assessment passed. Benchmark results are in Evaluation.')
    st.write('The MiniLM workflow fine-tunes sentence-transformers/all-MiniLM-L6-v2 on descriptions and log prices. Three-fold predictions create predicted_text_price without using each predicted row’s target in its fitting model.')
    st.write('Isolation Forest uses contamination="auto" on training OOF description prices only. No fixed percentage is removed; validation and test are unfiltered. Huber LightGBM combines retained training prices with the structured fields. This enhanced mode still requires GPU training.')
    st.write('The complete raw database is scanned before eligibility checks and deduplication. The eligible sample is split chronologically 70/15/15. Historical 2021 asking prices are not completed-sale prices or current valuations.')
with results:
    if (path/'evaluation.json').exists() and ready:
        e=json.loads((path/'evaluation.json').read_text())
        if e.get('ensemble_members'):
            st.info(f"Regularized ensemble: {e['ensemble_members']} members; promoted: {e['promoted']}");st.json(e['test'])
            st.write({k:e[k] for k in ['train_rows','validation_rows','test_rows']})
            st.caption(e['test_note'])
            if e.get('outlier_filter',{}).get('method')=='IsolationForest':
                f=e['outlier_filter']
                st.write(f"Isolation Forest retained {f['retained_rows']:,} of {f['input_rows']:,} training cars; removed {f['removed_fraction']:.1%}.")
                st.caption('The bouncer checked vehicle facts, not price tags. Unusual can still mean legitimate. Every validation and test car stayed in the queue.')
                st.dataframe([{'model':'Unfiltered reference',**e['baseline']['test']},{'model':'Isolation Forest ensemble',**e['test']}],hide_index=True)
                st.json(f,expanded=False)
                if not e['promoted']:st.info('This is the completed filtered experiment. The unfiltered ensemble remains the default because its benchmark errors are lower.')
            st.download_button('Download selected ensemble assessment',json.dumps(e,indent=2),'asking-ensemble-evaluation.json','application/json')
            with st.expander('All training candidates · technical details'):st.json(e['grid'])
        elif e.get('mode')=='tabular_baseline':
            st.info(e['label']);st.json(e['test'])
            st.write({k:e[k] for k in ['train_rows','validation_rows','test_rows','eligible_rows']})
            st.write('Final deployment refit rows:',e.get('deployment_refit_rows',e['train_rows']))
            st.caption(e.get('assessment_note','Chronological held-out assessment.'))
        else:
            st.write(e['rows']);st.json(e['stacked'])
        if 'band80_test_coverage' in e:st.write(f"Indicative 80% band test coverage: {e['band80_test_coverage']:.1%}")
        else:st.caption('The ensemble band is calibrated on historical validation residuals; no individual or future-market coverage guarantee is claimed.')
    else:st.info('This mode has no completed evaluation. Train MiniLM to enable it; baseline scores do not apply.')
