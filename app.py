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
# Prefer a completed full-data deployment once installed; serve the real assessment meanwhile.
installed_minilm=Path('models/minilm-assessment')
for candidate in [Path('models/minilm-cleanlab'),Path('artifacts/car-price-studio-cleanlab'),Path('models/minilm-full'),Path('artifacts/car-price-studio-full'),installed_minilm]:
    manifest_file=candidate/'manifest.json'
    if manifest_file.exists():
        candidate_manifest=json.loads(manifest_file.read_text())
        if (candidate_manifest.get('status')=='ready' and
            candidate_manifest.get('pipeline')=='fine-tuned-minilm-l6+lightgbm' and
            candidate_manifest.get('outlier_filter',{}).get('algorithm') in ['Cleanlab','IsolationForest']):
            installed_minilm=candidate
            break
minilm_record=json.loads((installed_minilm/'manifest.json').read_text()) if (installed_minilm/'manifest.json').exists() else {}
minilm_method=minilm_record.get('outlier_filter',{}).get('algorithm','none')
minilm_label='MiniLM + '+('Cleanlab' if minilm_method=='Cleanlab' else 'Isolation Forest (legacy)')+' + LightGBM'
choices=[minilm_label,'Best trained ensemble','Unfiltered trained ensemble','Trained tabular baseline']
filtered=Path('models/robust-asking-isolation')
if (filtered/'manifest.json').exists():choices.append('Isolation Forest trained ensemble')
cleanlab_ensemble=Path('models/robust-asking-cleanlab')
if (cleanlab_ensemble/'manifest.json').exists() and json.loads((cleanlab_ensemble/'manifest.json').read_text()).get('status')=='ready':choices.append('Cleanlab trained ensemble')
aft_path=Path('models/price-model-suite/ask')
aft_label='MiniLM + Cleanlab + XGBoost AFT (default)'
aft_ready=(aft_path/'manifest.json').exists() and json.loads((aft_path/'manifest.json').read_text()).get('status')=='ready'
if aft_ready:choices.insert(0,aft_label)
mode=st.selectbox('Pricing model',choices,index=0)
locations={aft_label:aft_path,'Cleanlab trained ensemble':cleanlab_ensemble,'Best trained ensemble':best_asking_bundle(Path('.')),
           minilm_label:installed_minilm,
           'Isolation Forest trained ensemble':filtered,
           'Unfiltered trained ensemble':Path('models/robust-asking'),
           'Trained tabular baseline':Path('models/studio-baseline')}
path=Path(os.environ.get('PRICING_MODEL_DIR',str(locations[mode])))
ready=(path/'manifest.json').exists()
selected_manifest=json.loads((path/'manifest.json').read_text()) if ready else {}
uses_text=selected_manifest.get('pipeline')=='fine-tuned-minilm-l6+lightgbm' or (selected_manifest.get('pipeline')=='modest-model-suite' and selected_manifest.get('text_feature',False))
if selected_manifest.get('pipeline')=='modest-model-suite':
    st.success('AFT is the preferred default: lowest recorded test RMSE on both datasets and lowest auction MAE. Huber remains better for asking-price MAE.')
    st.caption('Normal AFT predicts the log-normal median. This trained run used exact price bounds; no interval-censoring mechanism was verified.')
if selected_manifest.get('deployment_scope')=='completed-chronological-assessment':
    st.info('MiniLM is installed and ready to read descriptions. These are the completed assessment weights: 148,290 descriptions for MiniLM and 109,609 retained training rows for LightGBM. The separate full-data refit is still pending. The neural mechanic has passed this exam; it has not graduated from every garage yet.')
if uses_text:
    if selected_manifest.get('outlier_filter',{}).get('algorithm')=='Cleanlab':
        st.caption('MiniLM reads the description. Cleanlab audited suspected training price-label errors using vehicle facts, internal cross-validation and uncertainty estimation. It does not filter the car you enter. Validation and test rows remain intact.')
    else:
        st.info('Cleanlab has replaced Isolation Forest in the training code. This installed model still uses its original Isolation Forest weights; it will be replaced only after a Cleanlab bundle is trained and installed.')
        st.caption('MiniLM reads the description. Isolation Forest in auto mode filtered unusual training text-price signals; it does not reject your car at prediction time. LightGBM combines that signal with the form fields. The bouncer worked during training; the pricing desk is open.')
if not ready:
    st.info('This historical MiniLM mode has not been trained. The integrated Pricing + CarBids tab uses its own completed models and remains available.')

@st.cache_resource
def _get_pipeline(location,version):
    from pricing.inference import PricePipeline
    return PricePipeline(location)

def get_pipeline(location):
    file=Path(location)/'manifest.json'
    return _get_pipeline(location,file.stat().st_mtime_ns)

value,compare,how,results,laboratory=st.tabs(['Value a car','Pricing + CarBids','How it works','Evaluation','New model laboratory'])
with laboratory:
    from pricing.suite_ui import render
    render(Path(__file__).parent)
with compare:
    from marketbridge.ui import render_comparison
    render_comparison(Path(__file__).parent)
with value:
    for key,default in {'year':2015,'manufacturer':'Toyota','model':'Corolla SE','odometer':95000,'state':'ca'}.items():
        st.session_state.setdefault('vehicle_'+key,default)
    from pricing.description_examples import render_examples
    render_examples(uses_text)
    with st.form('vehicle'):
        a,b,c=st.columns(3)
        year=a.number_input('Year',key='vehicle_year',min_value=1900,max_value=2027,value=None)
        make=b.text_input('Make',key='vehicle_manufacturer',value=None)
        model=c.text_input('Model / trim',key='vehicle_model',value=None)
        a,b,c=st.columns(3)
        miles=a.number_input('Mileage (miles)',key='vehicle_odometer',min_value=0,max_value=2000000,value=None,step=1000)
        condition=b.selectbox('Condition',['unknown','excellent','good','fair','like new','new','salvage'],key='vehicle_condition')
        fuel=c.selectbox('Fuel',['gas','diesel','hybrid','electric','other','unknown'],key='vehicle_fuel')
        a,b,c=st.columns(3)
        transmission=a.selectbox('Transmission',['automatic','manual','other','unknown'],key='vehicle_transmission')
        drive=b.selectbox('Drive',['unknown','fwd','rwd','4wd'],key='vehicle_drive')
        body=c.selectbox('Body type',['sedan','SUV','truck','pickup','hatchback','coupe','wagon','van','convertible','other','unknown'],key='vehicle_type')
        a,b,c=st.columns(3)
        title=a.selectbox('Title status',['clean','rebuilt','salvage','lien','missing','parts only','unknown'],key='vehicle_title_status')
        cylinders=b.selectbox('Cylinders',['unknown','3 cylinders','4 cylinders','5 cylinders','6 cylinders','8 cylinders','10 cylinders','12 cylinders','other'],key='vehicle_cylinders')
        color=c.selectbox('Paint color',['unknown','white','black','silver','grey','blue','red','green','brown','yellow','orange','purple','custom'],key='vehicle_paint_color')
        state=st.text_input('US state abbreviation',key='vehicle_state',value=None,max_chars=2)
        notes=st.text_area('Seller description — free text',key='seller_description',height=250,max_chars=MAX_CHARACTERS,disabled=not uses_text,
            placeholder='Describe maintenance, equipment, condition and known problems. MiniLM reads this text and produces a price signal for LightGBM.' if uses_text else 'Select the installed MiniLM model to include your description.')
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
        if result.get('range') is not None:
            st.text(f"Indicative 80% band: ${result['range'][0]:,.0f}–${result['range'][1]:,.0f}")
        else:
            st.caption('Point estimate only: the saved assessment did not include its interval calibration radius. No crystal-ball interval has been invented.')
        if result.get('predicted_text_price') is not None:
            st.caption(f"Description-price feature: ${result['predicted_text_price']:,.0f}")
            st.caption('The companion below uses an independent structured ensemble for comparison. Your selected text-and-tree estimate is the price shown above.')
        from marketbridge.ui import render_auction_companion
        render_auction_companion(Path(__file__).parent,car)
        render_explanation(saved[1],saved[2],saved[3],get_pipeline(str(path)))
        from pricing.explanation_ui import render_visual_explanations
        render_visual_explanations(saved[1],saved[2],saved[3],get_pipeline(str(path)))
    st.caption('Snapshot: April–May 2021. Asking prices, not verified sale prices or present-day valuations. Newer cars are outside the training period.')
with how:
    st.write('The default asking and auction option is XGBoost normal AFT: a modest 100-round log-price likelihood model. It has the lowest recorded RMSE on both test datasets; Huber wins asking MAE. AFT supports genuine interval bounds, but the installed models used exact prices. Read TOBIT_AND_CENSORING.md for classical Tobit, censoring versus selection, and the evidence needed for interval-censored experiments in either dataset.')
    st.subheader('A pricing committee with thousands of small opinions')
    st.write('LightGBM learns lots of small decision trees, each correcting earlier mistakes. Huber loss softens the influence of large errors: one wildly ambitious seller does not get to chair the committee. The fitted tree count and exact drivers appear after an estimate.')
    st.write('Make and model enter LightGBM as ordinary categorical fields. Their transformer and separate neural price head have been removed.')
    st.write('The trained ensembles average regularized LightGBM models with age-adjusted CarBids features. New filtered training uses Cleanlab regression on vehicle facts and observed training log prices. Its internal cross-validation and uncertainty estimates identify suspected price-label errors. Historical Isolation Forest bundles keep their original names and scores. Validation and test cars are never filtered. The original unfiltered ensemble remains selectable. The best-model choice uses the filtered model only if its recorded assessment passed. Benchmark results are in Evaluation.')
    st.write('The MiniLM workflow fine-tunes sentence-transformers/all-MiniLM-L6-v2 on descriptions and log prices. Three-fold predictions create predicted_text_price without using each predicted row’s target in its fitting model.')
    st.write('Cleanlab audits training labels from structured vehicle fields, excluding the text-price stacking feature to avoid indirect target leakage. It chooses a suspected-noise fraction internally; flags are not verified errors. Validation and test are unfiltered. Huber LightGBM combines retained training prices with the structured fields. The installed MiniLM assessment model runs on CPU and dynamically reads your description; GPU training is not needed to use it.')
    if minilm_record.get('outlier_filter',{}).get('algorithm')=='Cleanlab':
        st.write('The current MiniLM + Cleanlab default uses a modest Huber LightGBM with library-default statistical parameters (100 trees) and a log1p price target. MiniLM produces a dynamic dollar-price feature. Separate training-only text predictions protect the assessment; the completed full-data text head serves deployment. Make and model remain categorical fields.')
    else:
        st.write('The legacy assessment used standardized dollar targets in its selected Huber LightGBM. MiniLM itself learned standardized log1p prices. Its inverse-transformed dollar estimate becomes predicted_text_price.')
    st.write('The complete raw database is scanned before eligibility checks and deduplication. The eligible sample is split chronologically 70/15/15. Historical 2021 asking prices are not completed-sale prices or current valuations.')
with results:
    from pricing.evaluation_ui import render_model_comparison
    render_model_comparison()
    if (path/'evaluation.json').exists() and ready:
        e=json.loads((path/'evaluation.json').read_text())
        if e.get('ensemble_members'):
            st.info(f"Regularized ensemble: {e['ensemble_members']} members; promoted: {e['promoted']}");st.json(e['test'])
            st.write({k:e[k] for k in ['train_rows','validation_rows','test_rows']})
            st.caption(e['test_note'])
            if e.get('outlier_filter',{}).get('method') in ['Cleanlab','IsolationForest']:
                f=e['outlier_filter']
                st.write(f"{f['method']} retained {f['retained_rows']:,} of {f['input_rows']:,} training cars; removed {f['removed_fraction']:.1%}.")
                st.caption('Cleanlab uses observed training prices as labels; legacy Isolation Forest used predictors only. Suspected issues can be legitimate cars. Validation and test remain unfiltered.')
                st.dataframe([{'model':'Unfiltered reference',**e['baseline']['test']},{'model':f["method"]+' ensemble',**e['test']}],hide_index=True)
                st.json(f,expanded=False)
                if not e['promoted']:st.info('This is the completed filtered experiment. The Best trained ensemble option selects the unfiltered ensemble because its benchmark errors are lower.')
            st.download_button('Download selected ensemble assessment',json.dumps(e,indent=2),'asking-ensemble-evaluation.json','application/json')
            with st.expander('All training candidates · technical details'):st.json(e['grid'])
        elif selected_manifest.get('pipeline')=='modest-model-suite':
            st.subheader('Default asking AFT · chronological held-out assessment')
            st.json(e['models']['xgb_aft_normal']['test'])
            st.write(e['rows'])
            st.caption(e['evaluation'])
            st.info('AFT is preferred by recorded RMSE on both market datasets. Compare all models, MAE trade-offs, Spearman and Ridge calibration in New model laboratory.')
        elif e.get('mode')=='tabular_baseline':
            st.info(e['label']);st.json(e['test'])
            st.write({k:e[k] for k in ['train_rows','validation_rows','test_rows','eligible_rows']})
            st.write('Final deployment refit rows:',e.get('deployment_refit_rows',e['train_rows']))
            st.caption(e.get('assessment_note','Chronological held-out assessment.'))
        else:
            st.write(e['rows']);st.json(e['stacked'])
            if selected_manifest.get('deployment_scope')=='completed-chronological-assessment':
                st.caption(selected_manifest['assessment_note'])
                st.text(f"Description-only test MAE: ${e['description_only']['mae_usd']:,.0f} · MiniLM + LightGBM test MAE: ${e['stacked']['mae_usd']:,.0f}.")
                st.info('The installed MiniLM model is the default studio model. Its filter is recorded in the model manifest. The other trained models remain selectable for comparison. The recorded assessment above belongs to these installed weights; the full-data refit has not finished yet.')
                st.download_button('Download MiniLM assessment',json.dumps(e,indent=2),'minilm-assessment.json','application/json')
        if selected_manifest.get('pipeline')=='modest-model-suite':
            st.caption('AFT point estimate only. Its assumed log-normal distribution is not a validated individual prediction interval.')
        elif selected_manifest.get('band80_log_radius') is None and uses_text:
            st.caption('The original evaluation recorded interval coverage, but its calibration radius was not persisted. This installation serves point estimates only.')
        elif 'band80_test_coverage' in e:st.write(f"Indicative 80% band test coverage: {e['band80_test_coverage']:.1%}")
        else:st.caption('The ensemble band is calibrated on historical validation residuals; no individual or future-market coverage guarantee is claimed.')
    else:st.info('This mode has no completed evaluation. Train MiniLM to enable it; baseline scores do not apply.')
