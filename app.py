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
from pricing.navigation import render_navigation
_workspace=render_navigation()
if _workspace in ['🚗 Price my car','🌳 Hierarchical model'] and (Path(__file__).parent/'models/xgboost-hierarchy/release.json').exists():
    from pricing.hierarchy_sparse_studio import render
    render(Path(__file__).parent)
    from pricing.ga_events import render_events
    render_events()
    footer()
    st.stop()
if _workspace=='🧠 The science':
    from pricing.hierarchy_sparse_studio import explain
    from pricing.analytics import render_privacy
    render_privacy()
    explain()
    footer()
    st.stop()
if _workspace=='📅 Value over time':
    from pricing.current_tools import render_timeline
    from pricing.analytics import render_privacy
    render_privacy()
    render_timeline(Path(__file__).parent)
    from pricing.ga_events import render_events
    render_events()
    footer()
    st.stop()
st.caption('Historical comparison tools retain their own model names and assessment dates. They are not the current XGBoost calculator.')
st.markdown('''<div style="background:linear-gradient(110deg,#251b39,#5b43bd);color:#fff;border-radius:22px;padding:22px 26px;margin:18px 0;border:1px solid #c6baf5">
<strong style="color:#dfff92;font-size:1.2rem">🚙 Rare car? Meet the family tree.</strong><br>
An uncommon make or trim has fewer close comparisons. Explore the separate hierarchical asking and wholesale estimates, their historical ranges, and how similar cars lend evidence.
</div>''',unsafe_allow_html=True)
if st.button('Explore rare-car estimates and uncertainty ↗',type='primary',key='rare_car_entry'):
    st.query_params['hierarchy']='1'
    st.session_state['_navigate_to']='🌳 Hierarchical model'
    st.rerun()

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
robust_aft_label=('MiniLM + CatBoost AFT' if _default_root.name in ['minilm-catboost-aft','iterative-catboost'] else 'CatBoost AFT' if _default_root.name=='catboost-aft' else 'XGBoost AFT')+(' · five-stage cleaning (default)' if _default_root.name=='iterative-catboost' else ' · residual-trimmed (default)')
if robust_aft_ready(Path('.')) and 'CatBoost' not in robust_aft_label:choices.insert(0,robust_aft_label)
tabular_catboost_label='CatBoost AFT · tabular comparison'

iterative_quantile_label='MiniLM + CatBoost quantile 0.5 · five-stage comparison'

mode=st.sidebar.selectbox('Pricing model',choices,index=choices.index('Trained tabular baseline'),key='pricing_model')
locations={iterative_quantile_label:Path('models/iterative-catboost/ask-quantile'),tabular_catboost_label:Path('models/catboost-aft/ask'),robust_aft_label:model_root(Path('.'))/'ask',aft_label:aft_path,'Cleanlab trained ensemble':cleanlab_ensemble,'Best trained ensemble':best_asking_bundle(Path('.')),
           minilm_label:installed_minilm,
           'Isolation Forest trained ensemble':filtered,
           'Unfiltered trained ensemble':Path('models/robust-asking'),
           'Trained tabular baseline':Path('models/studio-baseline')}
path=Path(os.environ.get('PRICING_MODEL_DIR',str(locations[mode])))
ready=(path/'manifest.json').exists()
selected_manifest=json.loads((path/'manifest.json').read_text()) if ready else {}
uses_text=selected_manifest.get('pipeline')=='fine-tuned-minilm-l6+lightgbm' or (selected_manifest.get('pipeline') in ['modest-model-suite','catboost-aft','catboost-log-quantile'] and selected_manifest.get('text_feature',False))
with st.sidebar:
    st.caption('Historical comparison model only. Current XGBoost properties are in The science.')
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
if _workspace=='🌳 Hierarchical model':
    from pricing.hierarchical_ui import render as render_hierarchical
    render_hierarchical(Path(__file__).parent)
if _workspace=='🧹 Outlier lab':
    from pricing.outlier_lab import render as render_outlier_lab
    render_outlier_lab(Path(__file__).parent)
if _workspace=='🧪 Model lab':
    from pricing.suite_ui import render
    render(Path(__file__).parent)
if _workspace=='↔ Two-market garage':
    from marketbridge.ui import render_comparison
    render_comparison(Path(__file__).parent)
if _workspace=='🚗 Price my car':
    from pricing.price_meaning import render_market_guide,render_market_price
    render_market_guide()
    from pricing.current_value import (clock_controls,current_transport,DateAdjustedAsking,
        render_market_clock,render_adjustment,render_time_evidence)
    valuation_date,allow_newer_auction=clock_controls()
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
            st.caption('Your mileage and condition are interpreted at the selected valuation date. Historical models are adjusted for vehicle age and market movement; completed-sale values remain unverified.')
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
            except Exception as exc:
                st.error(f'Model unavailable: {exc}')
    saved=st.session_state.get('last_explained_price')
    dated_result=dated_pipeline=transport=None
    dated_receipts={};date_error=None;refresh_error=None
    st.session_state.pop('last_displayed_price',None)
    if saved and saved[0]==str(path):
        try:
            with st.spinner('Moving the price clock to your selected date…'):
                transport,refresh_error=current_transport(Path(__file__).parent)
                dated_pipeline=DateAdjustedAsking(get_pipeline(str(path)),transport,valuation_date)
                dated_result=dated_pipeline.adjust(saved[1],saved[2])
                dated_receipts['ask']=dated_result['time_adjustment']
                st.session_state['last_displayed_price']=(saved[0],dated_result,saved[2],saved[3])
            request=st.session_state.get('_price_request_id','')
            record_price('ask',dated_result['price'],saved[2],mode+' · date adjusted','calculator',str(valuation_date),request+':ask:'+str(valuation_date))
        except (ValueError,OSError,KeyError) as exc:
            date_error=str(exc)
    with receipt:
        with st.container(key='price_receipt'):
            if dated_result:
                result=dated_result;car=saved[2]
                section('03 / Your price receipt','The numbers are in.')
                st.caption(f"{car['year']} {car['manufacturer']} {car['model']} · {car['odometer']:,} miles")
                render_market_price('ask',result['price'],valuation_date,scenario=True)
                render_adjustment(dated_receipts['ask'])
                st.caption('Point estimate only. No validated individual prediction interval is available.')
            elif date_error:
                st.error('Date-adjusted asking estimate unavailable: '+date_error)
                st.caption('No historical price is substituted under today’s date. Change the date or vehicle assumptions to a supported scenario.')
            else:empty_receipt()
        if saved and saved[0]==str(path):
            with st.container(key='auction_receipt'):
                try:
                    if transport is None:raise ValueError('The date-adjustment layer is unavailable.')
                    from pricing.residual_ui import bid_folder,load
                    bid_path=bid_folder(Path(__file__).parent)
                    bid,_,_=load(str(bid_path),(bid_path/'manifest.json').stat().st_mtime_ns)
                    from marketbridge.time_engine import today
                    r=transport.estimate(bid,'',saved[2],valuation_date,'bid',allow_newer_auction,as_of=today())
                    dated_receipts['bid']=r
                    render_market_price('bid',r['price'],valuation_date,scenario=True)
                    render_adjustment(r)
                    request=st.session_state.get('_price_request_id','')
                    record_price('bid',r['price'],saved[2],'date-adjusted auction','calculator',str(valuation_date),request+':bid:'+str(valuation_date)+':'+str(allow_newer_auction))
                except (ValueError,OSError,KeyError) as exc:
                    st.warning('Date-adjusted auction estimate unavailable: '+str(exc))
                    if saved[2].get('year',0)>2015 and not allow_newer_auction:
                        st.caption('The optional newer-generation assumption is in “Valuation date & auction coverage” above. You have disabled it; enable it for an explicitly labelled newer-generation scenario.')
            st.info('Same valuation date, different markets. The gap is not dealer profit or a promised discount.')
            if dated_result:
                from pricing.deal_detective import open_share
                st.button('📣 Share my price · image, link & group-chat fuel',on_click=open_share,key='open_share')
            st.caption('Open 📅 Value over time for the interactive timeline and mileage scenarios.')
            if saved[1].get('predicted_text_price') is not None:
                with st.expander('What did the description contribute?'):
                    st.metric('Description-price feature',f"${saved[1]['predicted_text_price']:,.0f}")
                    st.caption('MiniLM’s description-only scalar enters the asking model alongside vehicle facts. It is not an independent offer or a word-level explanation.')
        else:
            st.markdown('<div class="garage-science-note"><strong>Curious what moves the price?</strong><br>After estimating, explore feature contributions, mileage scenarios and all six explanation graphs below.</div>',unsafe_allow_html=True)
    if saved and saved[0]==str(path):
        report={
            'valuation_date':str(valuation_date),
            'vehicle_facts':saved[2],
            'market_estimates':{side:{
                'estimated_price_usd':round(r['price'],2),
                'vehicle_age_years':round(r['age'],2),
                'age_multiplier':round(r['age_factor'],4),
                'market_multiplier':round(r['market_factor'],4),
                'historical_anchor_usd':round(r['anchor_price_reference_usd'],2),
                'market_index_status':r['market'],
                'pooled_depreciation':not r['model_known_to_curve'],
                'newer_generation_analogue':r['generation_analogue']
            } for side,r in dated_receipts.items()},
            'unavailable_markets':[s for s in ['ask','bid'] if s not in dated_receipts],
            'market_source':'BLS/FRED CUUR0000SETA02',
            'latest_observed_month':transport.clock.meta['latest_observation'] if transport else None,
            'scope':'Experimental date-adjusted estimates; no validated current-car prediction interval.'
        }
        from pricing.valuation_story import extend_report
        report=extend_report(report,Path(__file__).parent,saved[2],saved[1],dated_receipts,transport,path,locals().get('bid_path'))
        with st.container(key='story_receipt'):
            from pricing.bart_summary import render_story
            render_story(saved[2],saved[3],valuation=report)
        if dated_result:
            render_market_clock(transport,valuation_date,refresh_error)
            render_time_evidence(transport,dated_receipts)
            from pricing.text_attribution_ui import render_text_attribution
            render_text_attribution(dated_result,saved[2],saved[3],dated_pipeline)
        st.divider()
        section('The receipts / Explainable machine learning','Pop the bonnet. Question everything.')
        st.caption('The current-date price and phrase effects are above. The historical model diagnostics below explain the original May 2021 anchor, before time adjustment.')
        with st.expander('Historical base-model diagnostics · May 2021, before time adjustment'):
            st.metric('Original historical asking anchor · May 2021',f"${saved[1]['price']:,.0f}")
            from pricing.explanations import render_explanation
            render_explanation(saved[1],saved[2],saved[3],get_pipeline(str(path)))
            from pricing.explanation_ui import render_visual_explanations
            render_visual_explanations(saved[1],saved[2],saved[3],get_pipeline(str(path)))
if _workspace=='📅 Value over time':
    from pricing.time_ui import render_time_studio
    time_saved=st.session_state.get('last_explained_price')
    if time_saved and time_saved[0]!=str(path):time_saved=None
    render_time_studio(Path(__file__).parent,time_saved,get_pipeline(str(path)) if time_saved else None)
if _workspace=='🧠 The science':
    if _default_root.name=='iterative-catboost':
        st.success('Active default: five-stage CatBoost cleaning; MiniLM + asking AFT and wholesale log-median quantile 0.5. Full model card and all measured trade-offs are in Outlier lab. The older methods described below remain comparison experiments.')
    if _default_root.name!='iterative-catboost':st.write('The asking default combines MiniLM description regression with CatBoost normal AFT. The auction default is tabular CatBoost quantile 0.5 on log prices, fitted after a separate quantile-residual cleaner removes the highest-residual 25% per original phase. CatBoost handles categorical labels natively and uses exact observed dollar bounds, with an internal log-price likelihood. Ask preserves its existing highest-residual quartile plus OutlierTree masks. Bid uses its new quantile-residual top-quarter cut only, without additional OutlierTree flags. Filtered test scores are optimistic selected-cohort results; original-test diagnostics are reported too. The MiniLM + CatBoost default includes the dynamic description-price feature; the earlier tabular-only CatBoost model is a comparison baseline. Auction data contain no descriptions, so auction CatBoost remains tabular. Only its cleaner can optionally use an IMR fitted from a genuine sold/unsold listing population; such data are currently unavailable. Earlier models remain available for comparison.')
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
if _workspace=='📊 Evaluation':
    from pricing.hierarchy_sparse_studio import scorecard
    scorecard(Path(__file__).parent,details=True)
    st.divider()
    st.subheader('Historical experiments · different architectures and cohorts')
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
        elif selected_manifest.get('training_regime') in ['five-stage-catboost-residual','residual-trimmed-robust-xgb','residual-trimmed-catboost-aft','residual-trimmed-minilm-catboost-aft']:
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
                st.info('This is the historical MiniLM comparison model, not the current studio default. Its filter is recorded in the model manifest. The other trained models remain selectable for comparison. The recorded assessment above belongs to these installed weights; the full-data refit has not finished yet.')
                st.download_button('Download MiniLM assessment',json.dumps(e,indent=2),'minilm-assessment.json','application/json')
        if selected_manifest.get('pipeline') in ['modest-model-suite','catboost-aft','catboost-log-quantile']:
            st.caption('Point estimate only. Neither the AFT distribution nor quantile loss provides a validated current-car prediction interval here.')
        elif selected_manifest.get('band80_log_radius') is None and uses_text:
            st.caption('The original evaluation recorded interval coverage, but its calibration radius was not persisted. This installation serves point estimates only.')
        elif 'band80_test_coverage' in e:st.write(f"Indicative 80% band test coverage: {e['band80_test_coverage']:.1%}")
        else:st.caption('The ensemble band is calibrated on historical validation residuals; no individual or future-market coverage guarantee is claimed.')
    else:st.info('This mode has no completed evaluation. Train MiniLM to enable it; baseline scores do not apply.')

if _workspace=='🕵️ Deal Detective':
    from pricing.deal_detective import render as render_detective
    from pricing.deal_runtime import current_callbacks
    ask_callback,bid_callback,callback_model=current_callbacks(Path(__file__).parent)
    render_detective(ask_callback,bid_callback,callback_model)

from pricing.ga_events import render_events
render_events()
footer()
