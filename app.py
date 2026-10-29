"""Run with: python -m streamlit run app.py"""
import json
import os
from pathlib import Path
import streamlit as st
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHARACTERS = 8000

st.set_page_config(page_title='Car Price Studio · Less haggle. More signal.',page_icon=str(Path(__file__).parent/'assets/pricing-icon.png'),layout='wide',initial_sidebar_state='collapsed')
# Keep troubleshooting reachable without importing/loading any pricing models.
if st.query_params.get('connection_check') == '1':
    from pricing.analytics import render_connection_check
    render_connection_check()
    st.stop()

from pricing.studio_design import install_style,render_hero,section,empty_receipt,sidebar_receipt,footer
install_style()
render_hero()
from pricing.analytics import render_privacy, record_price
render_privacy()
from pricing.dataset_sources import render_sources
from pricing.model_method_ui import render_methods
with st.sidebar:
    st.markdown('## Nerd mode, unlocked.')
    st.caption('Model controls, real scores and receipts. The science has its own parking space.')
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
aft_label='MiniLM + Cleanlab + XGBoost AFT · original'
aft_ready=(aft_path/'manifest.json').exists() and json.loads((aft_path/'manifest.json').read_text()).get('status')=='ready'
if aft_ready:choices.insert(0,aft_label)
from pricing.residual_ui import ready as robust_aft_ready,model_root
_default_root=model_root(Path('.'))
robust_aft_label=('MiniLM + CatBoost AFT' if _default_root.name=='minilm-catboost-aft' else 'CatBoost AFT' if _default_root.name=='catboost-aft' else 'XGBoost AFT')+' · residual-trimmed (default)'
if robust_aft_ready(Path('.')):choices.insert(0,robust_aft_label)
tabular_catboost_label='CatBoost AFT · tabular comparison'
if _default_root.name=='minilm-catboost-aft' and Path('models/catboost-aft/ask/manifest.json').exists():choices.append(tabular_catboost_label)
mode=st.sidebar.selectbox('Pricing model',choices,index=0,key='pricing_model')
locations={tabular_catboost_label:Path('models/catboost-aft/ask'),robust_aft_label:model_root(Path('.'))/'ask',aft_label:aft_path,'Cleanlab trained ensemble':cleanlab_ensemble,'Best trained ensemble':best_asking_bundle(Path('.')),
           minilm_label:installed_minilm,
           'Isolation Forest trained ensemble':filtered,
           'Unfiltered trained ensemble':Path('models/robust-asking'),
           'Trained tabular baseline':Path('models/studio-baseline')}
path=Path(os.environ.get('PRICING_MODEL_DIR',str(locations[mode])))
ready=(path/'manifest.json').exists()
selected_manifest=json.loads((path/'manifest.json').read_text()) if ready else {}
uses_text=selected_manifest.get('pipeline')=='fine-tuned-minilm-l6+lightgbm' or (selected_manifest.get('pipeline') in ['modest-model-suite','catboost-aft'] and selected_manifest.get('text_feature',False))
with st.sidebar.expander('Active model · technical receipt'):
    if selected_manifest.get('training_regime') in ['residual-trimmed-robust-xgb','residual-trimmed-catboost-aft','residual-trimmed-minilm-catboost-aft']:
        st.info('MiniLM reads the seller description, then regularized CatBoost normal AFT combines its predicted_text_price with vehicle facts.' if uses_text else 'CatBoost normal AFT · tabular-only comparison; this model does not use the description.')
        st.warning('Filtered validation/test scores are optimistic because their labels select retained cars. See Evaluation for original unfiltered-test errors too.')
    elif selected_manifest.get('pipeline')=='modest-model-suite':
        st.success('AFT is the preferred default: lowest recorded test RMSE on both datasets and lowest auction MAE. Huber remains better for asking-price MAE.')
        st.caption('Normal AFT predicts the log-normal median. This trained run used exact price bounds; no interval-censoring mechanism was verified.')
    if selected_manifest.get('deployment_scope')=='completed-chronological-assessment':
        st.info('MiniLM is installed and ready to read descriptions. These are the completed assessment weights: 148,290 descriptions for MiniLM and 109,609 retained training rows for LightGBM. The separate full-data refit is still pending. The neural mechanic has passed this exam; it has not graduated from every garage yet.')
    if uses_text:
        if selected_manifest.get('pipeline')=='catboost-aft':
            st.caption('MiniLM supplies one dynamic description-price feature. CatBoost learned this feature from out-of-fold text predictions. Make and model remain native categorical fields. Highest-residual 25% per split plus OutlierTree flags were excluded; filtered scores are optimistic. The neural mechanic reads the advert; the tree committee checks the paperwork.')
        elif selected_manifest.get('outlier_filter',{}).get('algorithm')=='Cleanlab':
            st.caption('MiniLM reads the description. Cleanlab audited suspected training price-label errors using vehicle facts, internal cross-validation and uncertainty estimation. It does not filter the car you enter. Validation and test rows remain intact.')
        else:
            st.info('Cleanlab has replaced Isolation Forest in the training code. This installed model still uses its original Isolation Forest weights; it will be replaced only after a Cleanlab bundle is trained and installed.')
            st.caption('MiniLM reads the description. Isolation Forest in auto mode filtered unusual training text-price signals; it does not reject your car at prediction time. LightGBM combines that signal with the form fields. The bouncer worked during training; the pricing desk is open.')
    if not ready:
        st.info('This historical MiniLM mode has not been trained. The integrated Pricing + CarBids tab uses its own completed models and remains available.')
sidebar_receipt(Path(__file__).parent)
with st.sidebar:
    with st.expander('Datasets & provenance'):
        render_sources(stacked=True)
    render_methods(Path(__file__).parent)
if not ready:
    st.warning('The selected historical model is unavailable. Choose an installed model in the sidebar.')

@st.cache_resource
def _get_pipeline(location,version):
    from pricing.inference import PricePipeline
    return PricePipeline(location)

def get_pipeline(location):
    file=Path(location)/'manifest.json'
    return _get_pipeline(location,file.stat().st_mtime_ns)

st.markdown('<div id="studio-calculator"></div>',unsafe_allow_html=True)
value,compare,time_panel,how,results,laboratory,detective=st.tabs(['🚗 Price my car','↔ Two-market garage','📅 Value over time','🧠 The science','📊 Evaluation','🧪 Model lab','🕵️ Deal Detective'],key='studio_tabs',default='🕵️ Deal Detective' if st.query_params.get('garage') else '🚗 Price my car',on_change='rerun')
with laboratory:
    from pricing.suite_ui import render
    render(Path(__file__).parent)
with compare:
    from marketbridge.ui import render_comparison
    render_comparison(Path(__file__).parent)
with value:
    for key,default in {'year':2015,'manufacturer':'Toyota','model':'Corolla SE','odometer':95000,'state':'ca'}.items():
        st.session_state.setdefault('vehicle_'+key,default)
    editor,receipt=st.columns([1.65,1],gap='large')
    with editor:
        with st.container(key='vehicle_editor'):
            section('01 / The test drive','Meet your next number.')
            st.caption('Try a real advert or bring your own car. Every detail is editable.')
            from pricing.description_examples import render_examples
            render_examples(uses_text)
            with st.form('vehicle'):
                a,b,c=st.columns(3)
                year=a.number_input('Year',key='vehicle_year',min_value=1900,max_value=2021,value=None)
                make=b.text_input('Make',key='vehicle_manufacturer',value=None)
                model=c.text_input('Model / trim',key='vehicle_model',value=None)
                a,b,c=st.columns(3)
                miles=a.number_input('Mileage (miles)',key='vehicle_odometer',min_value=0,max_value=2000000,value=None,step=1000)
                state=b.text_input('US state abbreviation',key='vehicle_state',value=None,max_chars=2)
                body=c.selectbox('Body type',['sedan','SUV','truck','pickup','hatchback','coupe','wagon','van','convertible','other','unknown'],key='vehicle_type')
                with st.expander('Fine-tune the details · condition, engine & more'):
                    a,b,c=st.columns(3)
                    condition=a.selectbox('Condition',['unknown','excellent','good','fair','like new','new','salvage'],key='vehicle_condition')
                    fuel=b.selectbox('Fuel',['gas','diesel','hybrid','electric','other','unknown'],key='vehicle_fuel')
                    transmission=c.selectbox('Transmission',['automatic','manual','other','unknown'],key='vehicle_transmission')
                    a,b=st.columns(2)
                    drive=a.selectbox('Drive',['unknown','fwd','rwd','4wd'],key='vehicle_drive')
                    title=b.selectbox('Title status',['clean','rebuilt','salvage','lien','missing','parts only','unknown'],key='vehicle_title_status')
                    cylinders=a.selectbox('Cylinders',['unknown','3 cylinders','4 cylinders','5 cylinders','6 cylinders','8 cylinders','10 cylinders','12 cylinders','other'],key='vehicle_cylinders')
                    color=b.selectbox('Paint color',['unknown','white','black','silver','grey','blue','red','green','brown','yellow','orange','purple','custom'],key='vehicle_paint_color')
                st.markdown('**02 / The backstory**')
                notes=st.text_area('Seller description — free text',key='seller_description',height=165,max_chars=MAX_CHARACTERS,disabled=not uses_text,
                    placeholder='Service history? Fresh tyres? That mysterious rattle? Tell the model the whole story.' if uses_text else 'Choose a MiniLM model in the sidebar to include your description.')
                submitted=st.form_submit_button('Estimate asking price',disabled=not ready,type='primary',use_container_width=True)
            st.caption('Historical data, not today’s market. 2021 asking listings + age/CPI-transported auction records. Estimates, not guaranteed transactions.')
    if submitted:
        if len(notes)>8000:
            st.warning('Please keep the description within 8,000 characters.')
        else:
            fields=dict(year=year,manufacturer=make,model=model,odometer=miles,condition=condition,
                fuel=fuel,transmission=transmission,drive=drive,type=body,title_status=title,
                cylinders=cylinders,paint_color=color,state=state)
            try:
                with st.spinner('Interviewing the trees. They have opinions…'):
                    p=get_pipeline(str(path)).predict(notes,fields,explain=True)
                st.session_state['last_explained_price']=(str(path),p,fields,notes)
                import uuid
                st.session_state['_price_request_id']=str(uuid.uuid4())
                record_price('ask',p['price'],fields,mode,'calculator','2021-05-01',st.session_state['_price_request_id'])
            except Exception as exc:
                st.error(f'Model unavailable: {exc}')
    saved=st.session_state.get('last_explained_price')
    with receipt:
        with st.container(key='price_receipt'):
            if saved and saved[0]==str(path):
                result=saved[1];car=saved[2]
                section('03 / Your price receipt','The numbers are in.')
                st.caption(f"{car['year']} {car['manufacturer']} {car['model']} · {car['odometer']:,} miles")
                st.metric('Estimated asking price',f"${result['price']:,.0f}")
                st.caption('Historical advertised-price estimate. Seller optimism, with a statistical chaperone.')
                from pricing.deal_detective import open_share
                st.button('📣 Share my price · image, link & group-chat fuel',on_click=open_share,key='open_share')
                st.info('Want today’s value? Open 📅 Value over time for date-adjusted Ask/Bid scenarios and the interactive timeline.')
                if result.get('range') is not None:
                    st.write(f"Indicative 80% band: ${result['range'][0]:,.0f}–${result['range'][1]:,.0f}")
                else:st.caption('Point estimate only. No validated individual prediction interval is available.')
                if result.get('predicted_text_price') is not None:
                    with st.expander('What did the description contribute?'):
                        st.metric('Description-price feature',f"${result['predicted_text_price']:,.0f}")
                        st.caption('MiniLM’s description-only scalar enters the asking model alongside vehicle facts. It is not an independent offer or a word-level explanation.')
            else:empty_receipt()
        if saved and saved[0]==str(path):
            with st.container(key='auction_receipt'):
                from marketbridge.ui import render_auction_companion
                render_auction_companion(Path(__file__).parent,saved[2])
        else:
            st.markdown('<div class="garage-science-note"><strong>Curious what moves the price?</strong><br>After estimating, explore feature contributions, mileage scenarios and all six explanation graphs below.</div>',unsafe_allow_html=True)
    if saved and saved[0]==str(path):
        st.divider()
        section('The receipts / Explainable machine learning','Pop the bonnet. Question everything.')
        st.caption('Your estimate, explained. Contributions describe the model—not causal effects or guaranteed dollar discounts.')
        from pricing.explanations import render_explanation
        render_explanation(saved[1],saved[2],saved[3],get_pipeline(str(path)))
        from pricing.explanation_ui import render_visual_explanations
        render_visual_explanations(saved[1],saved[2],saved[3],get_pipeline(str(path)))
with time_panel:
    from pricing.time_ui import render_time_studio
    time_saved=st.session_state.get('last_explained_price')
    if time_saved and time_saved[0]!=str(path):time_saved=None
    render_time_studio(Path(__file__).parent,time_saved,get_pipeline(str(path)) if time_saved else None)
with how:
    st.write('The asking default combines MiniLM description regression with CatBoost normal AFT. The auction default is tabular CatBoost quantile 0.5 on log prices, fitted after a separate quantile-residual cleaner removes the highest-residual 25% per original phase. CatBoost handles categorical labels natively and uses exact observed dollar bounds, with an internal log-price likelihood. Ask preserves its existing highest-residual quartile plus OutlierTree masks. Bid uses its new quantile-residual top-quarter cut only, without additional OutlierTree flags. Filtered test scores are optimistic selected-cohort results; original-test diagnostics are reported too. The MiniLM + CatBoost default includes the dynamic description-price feature; the earlier tabular-only CatBoost model is a comparison baseline. Auction data contain no descriptions, so auction CatBoost remains tabular. Only its cleaner can optionally use an IMR fitted from a genuine sold/unsold listing population; such data are currently unavailable. Earlier models remain available for comparison.')
    st.subheader('A pricing committee with thousands of small opinions')
    st.write('LightGBM learns lots of small decision trees, each correcting earlier mistakes. Huber loss softens the influence of large errors: one wildly ambitious seller does not get to chair the committee. The fitted tree count and exact drivers appear after an estimate.')
    st.write('Make and model enter LightGBM as ordinary categorical fields. Their transformer and separate neural price head have been removed.')
    st.write('The trained ensembles average regularized LightGBM models with age-adjusted CarBids features. New filtered training uses Cleanlab regression on vehicle facts and observed training log prices. Its internal cross-validation and uncertainty estimates identify suspected price-label errors. Historical Isolation Forest bundles keep their original names and scores. In those older benchmarks validation/test remained unfiltered. The requested CatBoost refit filters both and labels its scores accordingly. The original unfiltered ensemble remains selectable. The best-model choice uses the filtered model only if its recorded assessment passed. Benchmark results are in Evaluation.')
    st.write('The MiniLM workflow fine-tunes sentence-transformers/all-MiniLM-L6-v2 on descriptions and log prices. Three-fold predictions create predicted_text_price without using each predicted row’s target in its fitting model.')
    st.write('Cleanlab audits training labels from structured vehicle fields, excluding the text-price stacking feature to avoid indirect target leakage. It chooses a suspected-noise fraction internally; flags are not verified errors. Validation and test are unfiltered. Huber LightGBM combines retained training prices with the structured fields. The installed MiniLM assessment model runs on CPU and dynamically reads your description; GPU training is not needed to use it.')
    if minilm_record.get('outlier_filter',{}).get('algorithm')=='Cleanlab':
        st.write('The selectable MiniLM + Cleanlab baseline uses a modest Huber LightGBM with library-default statistical parameters (100 trees) and a log1p price target. MiniLM produces a dynamic dollar-price feature. Separate training-only text predictions protect the assessment; the completed full-data text head serves deployment. Make and model remain categorical fields.')
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
        elif selected_manifest.get('training_regime') in ['residual-trimmed-robust-xgb','residual-trimmed-catboost-aft','residual-trimmed-minilm-catboost-aft']:
            from pricing.residual_ui import render_assessment
            render_assessment(Path('.'))
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
        if selected_manifest.get('pipeline') in ['modest-model-suite','catboost-aft']:
            st.caption('AFT point estimate only. Its assumed log-normal distribution is not a validated individual prediction interval.')
        elif selected_manifest.get('band80_log_radius') is None and uses_text:
            st.caption('The original evaluation recorded interval coverage, but its calibration radius was not persisted. This installation serves point estimates only.')
        elif 'band80_test_coverage' in e:st.write(f"Indicative 80% band test coverage: {e['band80_test_coverage']:.1%}")
        else:st.caption('The ensemble band is calibrated on historical validation residuals; no individual or future-market coverage guarantee is claimed.')
    else:st.info('This mode has no completed evaluation. Train MiniLM to enable it; baseline scores do not apply.')

with detective:
    from pricing.deal_detective import render as render_detective
    from pricing.deal_runtime import bid_estimate
    render_detective(lambda text,fields:get_pipeline(str(path)).predict(text,fields,explain=False),bid_estimate,str(path))

from pricing.ga_events import render_events
render_events()
footer()
