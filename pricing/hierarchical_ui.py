"""On-demand hierarchical residual laboratory; no model promotion or raw-file storage."""
import io,json,uuid
import streamlit as st
from pricing.original_residual_results import load
import pandas as pd
from pricing.hierarchical import REQUIRED,experiment

@st.cache_data(show_spinner=False)
def installed(root,side,family='hierarchical-original'):
    return load(root,side,family)


def render(root):
    hero="""
<style>
.hierarchy-hero{background:linear-gradient(125deg,#eee8ff 0%,#f8f5eb 75%,#e6ffac 100%);border:1px solid #d4caef;border-radius:28px;padding:clamp(24px,4vw,48px);color:#272037;margin:12px 0 24px}
.hierarchy-hero .eyebrow{letter-spacing:.15em;font-size:.74rem;font-weight:800;color:#5c43b9}
.hierarchy-hero h2{font-size:clamp(2rem,4vw,3.5rem);line-height:1.08;max-width:820px;margin:15px 0;color:#272037}
.hierarchy-hero p{font-size:1.05rem;max-width:720px;line-height:1.65}
.hierarchy-status{display:inline-block;background:#292037;color:#e2ff8a;border-radius:30px;padding:8px 14px;font-size:.77rem;font-weight:750}
.hierarchy-steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:22px 0}
.hierarchy-step{background:#fffdf8;border:1px solid #e1dbef;border-radius:20px;padding:22px;color:#272037}
.hierarchy-step .number{color:#6a50cb;font-size:.78rem;font-weight:800;letter-spacing:.12em}
.hierarchy-step h3{font-size:1.18rem;margin:10px 0;color:#272037}.hierarchy-step p{font-size:.94rem;line-height:1.55;margin:0}
@media(max-width:700px){.hierarchy-steps{grid-template-columns:1fr}.hierarchy-hero{border-radius:20px}}
</style>
<section class="hierarchy-hero" aria-label="Hierarchical pricing introduction">
<div class="eyebrow">THE FAMILY TREE · ROBUST PRICES, SHARED EVIDENCE</div>
<h2>Rare trim?<br>Ask the family.</h2>
<p>One unusual car should not have to explain the entire market. This experiment lets related makes and models share evidence, while giving well-supported models their own voice.</p>
<span class="hierarchy-status">MODEL STATUS · AWAITING CALIBRATION DATA</span>
<p style="font-size:.9rem;margin-bottom:0">The laboratory is ready. A trained correction and measured accuracy gain are not available yet. Your current price calculator keeps its existing models.</p>
</section>
<div class="hierarchy-steps" aria-label="How partial pooling works">
<article class="hierarchy-step"><span class="number">01 / START WITH THE CAR</span><h3>Two models read the facts.</h3><p>Vehicle facts and the asking model’s text signal provide the starting point.</p></article>
<article class="hierarchy-step"><span class="number">02 / ASK THE FAMILY</span><h3>Related cars share evidence.</h3><p>A rare trim borrows from its make. More evidence gives its own model a stronger voice. Family advice, with fewer unsolicited opinions.</p></article>
<article class="hierarchy-step"><span class="number">03 / CHECK THE RECEIPTS</span><h3>Better is something we measure.</h3><p>Compare both versions on the same untouched cars. No difficult test cars disappear behind the garage.</p></article>
</div>
"""
    original=installed(str(root),st.session_state.get('hierarchy_side','ask'))
    if original:
        hero=hero.replace('MODEL STATUS · AWAITING CALIBRATION DATA','ORIGINAL DATA · FULL ELIGIBLE COHORTS').replace('The laboratory is ready. A trained correction and measured accuracy gain are not available yet. Your current price calculator keeps its existing models.','Real historical cars. Robust regression and a random forest. Compare each baseline with its family-tree correction on the same untouched test. The main calculator keeps its existing models.').replace('Vehicle facts and the asking model’s text signal provide the starting point.','SGD-Huber and XGBoost random forest supply genuine out-of-fold prices. MiniLM is excluded from this separate comparison.').replace('Compare both versions','Compare both versions')
    active=st.session_state.get('hierarchy_experiment')
    if active and active[0]['side']==st.session_state.get('hierarchy_side','ask'):
        hero=hero.replace('MODEL STATUS · AWAITING CALIBRATION DATA','SESSION EXPERIMENT · NOT PROMOTED').replace('The laboratory is ready. A trained correction and measured accuracy gain are not available yet. Your current price calculator keeps its existing models.','A correction is fitted in this session. Review its held-out evidence below. The main price calculator keeps its existing models.')
    st.markdown(hero,unsafe_allow_html=True)
    with st.expander('How this works · plain English and statistical recipe'):
        st.write('Start with SGD-Huber’s price. Learn log(observed price / base price) on calibration rows whose base predictions were made out of fold. Estimate global, make and model offsets, with smaller offsets for sparse groups. A regularized common age slope and supported model-age slopes explain remaining age patterns. Multiply the base estimate by exp(total correction).')
        st.write('This is sequential empirical-Bayes partial pooling with Gaussian residual assumptions. Group prior variances are estimated from calibration data only. It is not a joint MCMC fit. Correction standard errors are approximate and omit hyperparameter uncertainty; they are not individual-car prediction intervals.')
        st.warning('Asking and auction targets need separate files and models. A correction to a historical price is not a forecast of a current transaction price. The supplied base model and any text branch must exclude each validation/test vehicle and must generate genuine OOF calibration predictions. CSV declarations alone cannot prove that provenance.')
    # Streamlit executes every tab: expensive work and file handling stay on this panel.
    if st.session_state.get('studio_tabs')!='🌳 Hierarchical model':return
    st.markdown('#### Choose your market')
    side=st.radio('Hierarchical market',['ask','bid'],format_func=lambda x:'Asking · Craigslist' if x=='ask' else 'Wholesale · auction',horizontal=True,key='hierarchy_side')
    with st.expander('Developer workbench · data, fitting & provenance',expanded=False):
        from pricing.sgd_price import OBSERVATION_COLUMNS,generate
        mode=st.radio('Residual input',['Generate from vehicle observations','Use existing OOF residuals'],key='hierarchy_input_mode')
        automatic=mode.startswith('Generate')
        columns=OBSERVATION_COLUMNS if automatic else REQUIRED
        st.caption('Automatic mode fits a fresh tabular SGD-Huber assessment: Huber loss on log prices for each market. It generates group-aware OOF residuals, then fits the hierarchy. The optional upload workbench is limited to 10,000 rows; the installed comparison uses all eligible original rows. MiniLM and five-stage cleaning are not included. It does not overwrite production models.' if automatic else 'Upload verified OOF calibration and held-out base predictions. All test rows are retained.')
        st.download_button('Download residual-data template',','.join(columns)+'\n','hierarchical-input-template.csv','text/csv',key='hierarchy_template')
        source=st.file_uploader('Vehicle observations or OOF predictions · CSV',type=['csv'],key='hierarchy_data')
        confirmed=st.checkbox('Vehicle identities and partitions are correct; existing predictions, if supplied, exclude held-out vehicles.',key='hierarchy_provenance')
        if st.button('Fit and evaluate hierarchical correction',disabled=source is None or not confirmed,type='primary',key='hierarchy_fit'):
            try:
                if source.size>25*1024*1024:raise ValueError('CSV must be at most 25 MB.')
                with st.spinner('Cross-fitting SGD-Huber, then convening the family council…'):
                    data=pd.read_csv(io.BytesIO(source.getvalue()));provenance=None
                    if automatic:data,provenance=generate(data,side,max_rows=10000)
                    bundle,report,rows=experiment(data,side,include_sgd=False)
                    if provenance:report['base_assessment']=provenance
                st.session_state['hierarchy_experiment']=(bundle,report,rows)
                st.session_state['hierarchy_residual_export']=data.to_csv(index=False)
                st.session_state.pop('hierarchy_checked',None)
                st.rerun()
            except (ValueError,TypeError,KeyError,pd.errors.ParserError) as exc:st.error(str(exc))
    saved=st.session_state.get('hierarchy_experiment')
    if not saved or saved[0]['side']!=side:saved=installed(str(root),side)
    if not saved or saved[0]['side']!=side:
        st.info('No trained comparison for this market yet. The developer workbench accepts verified residual data and measures whether the correction helps before showing a price comparison.')
        return
    bundle,report,rows=saved
    forest=installed(str(root),side,'forest-original')
    if forest and 'provenance' in report:
        same_source=forest[1]['provenance']['canonical_sha256']==report['provenance']['canonical_sha256']
        same_counts=all(forest[1]['cohorts'][split]['base']['rows']==report['cohorts'][split]['base']['rows'] for split in ['validation','test'])
        if not same_source or not same_counts:
            st.warning('Forest comparison is hidden because its source or cohort counts do not match this experiment.')
            forest=None
    if 'provenance' in report:
        provenance=report['provenance']
        st.success(f"Original dataset · {provenance['eligible_rows']:,} eligible cars · 100% of the eligible cohort, no random training sample")
        st.caption(f"{provenance['source']} · {provenance['price_basis']} · {report['calibration_rows']:,} calibration cars. All validation/test cars are scored; the example browser shows a reproducible selection of at most 1,000.")
        st.link_button('Original public dataset',provenance['source_url'])
    if 'base_assessment' in report:st.caption(report['base_assessment']['warning'])
    if 'model' in report.get('base_assessment',{}):
        with st.expander('SGD-Huber baseline · simple idea, full technical recipe'):
            st.write('A robust linear model predicts natural-log price from vehicle facts. Common categories get one-hot indicators; rare levels share a bucket. Age, mileage and available auction condition are standardized. Huber limits the influence of large log-price mistakes; it does not delete any car.')
            st.write('Each OOF fold learns its own scaling and category vocabulary using only that fold’s training rows. The hierarchy then learns the remaining make/model/age patterns. This baseline uses neither trees nor transformers. The random-forest comparison below is separate. Your main calculator and description dropdown are unchanged.')
            st.json(report['base_assessment']['model'])
    st.markdown('#### 02 · Did it actually help?')
    metrics=[]
    for split,r in report['cohorts'].items():
        for method in ['base','hierarchical','sgd_huber'] if 'sgd_huber' in r else ['base','hierarchical']:metrics.append({'Cohort':split,'Model':method,**r[method]})
    if forest:
        for split,r in forest[1]['cohorts'].items():
            for method in ['base','hierarchical']:metrics.append({'Cohort':split,'Model':'forest_'+method,**r[method]})
    with st.expander('Full scorecard · MAE, RMSE, log error and bias'):
        st.dataframe(pd.DataFrame(metrics),hide_index=True)
    test=report['cohorts']['test'];lo,hi=test['mae_change_95_group_bootstrap']
    methods=['base','hierarchical']+(['sgd_huber'] if 'sgd_huber' in test else [])
    labels=['SGD-Huber baseline','Family-tree correction','Additional SGD correction']
    for column,method,label in zip(st.columns(len(methods)),methods,labels):
        if method in test:column.metric(label+' · test MAE',f"${test[method]['mae_usd']:,.0f}")
    if test['hierarchical']['rmse_usd']>test['base']['rmse_usd']:
        st.warning('A trade-off, not a victory lap: the hierarchy lowers average error only if MAE falls; here RMSE is higher, meaning extreme dollar misses receive a larger penalty. Check both before choosing a model.')
    st.caption('Lower MAE means a smaller average dollar miss on the same historical test cars. Asking advertisements and auction sales are different targets.')
    st.metric('Test MAE change · corrected minus base',f"{test['mae_change_usd']:+,.2f} USD",help='Negative means lower error. The test includes every supplied test row.')
    st.caption(f'Paired vehicle-group bootstrap, 300 resamples: 95% interval [{lo:+,.2f}, {hi:+,.2f}] USD. This interval concerns the MAE difference, not a car’s price. No automatic promotion or increased-accuracy claim.')
    if 'sgd_huber' in test:
        st.metric('SGD-Huber test MAE change · corrected minus base',f"{test['sgd_mae_change_usd']:+,.2f} USD")
        a,b=test['sgd_mae_change_95_group_bootstrap']
        st.caption(f'Paired 95% bootstrap interval: [{a:+,.2f}, {b:+,.2f}] USD. Negative is better; a second model is not automatically an improvement.')
        with st.expander('The robust linear mechanic · SGD-Huber explained'):
            st.write('SGD learns the remaining log-price error after SGD-Huber. Popular categories get one-hot indicators; levels occurring fewer than 20 times are pooled, with at most 50 encoded levels per field. Numerical facts are standardized. Everything is fitted on OOF calibration rows only.')
            st.write('Huber uses smooth squared error near zero and linear error for large misses. Its transition is 0.1 log units (roughly a 10% price ratio). Elastic-net regularization keeps the correction restrained. These coefficients describe associations, not causes.')
            st.json(bundle['sgd_huber'])
    if forest:
        rf=forest[1]['cohorts']['test']
        st.markdown('#### 🌲 Random forest · another mechanic, same cars')
        a,b=st.columns(2)
        a.metric('XGBoost RF · test MAE',f"${rf['base']['mae_usd']:,.0f}")
        b.metric('RF + family-tree correction · test MAE',f"${rf['hierarchical']['mae_usd']:,.0f}")
        st.caption('This panel displays held-out forest results; it does not replace the live calculator with a forest. Large forest weights remain offline reproduction outputs.')
        st.caption('200 parallel trees in one boosting round: a random forest implemented by XGBoost, not a sequence of boosted trees. Same original cohorts, same untouched test. No CatBoost in this experiment.')
        lo,hi=rf['mae_change_95_group_bootstrap']
        st.caption(f"Hierarchy versus RF alone: MAE change {rf['mae_change_usd']:+,.2f} USD, paired 95% interval [{lo:+,.2f}, {hi:+,.2f}]. Full RMSE and bias are in the scorecard above.")
        with st.expander('Random forest · technical receipt'):
            st.write('Log-price squared error; native categorical splits with training-only vocabularies, missing-value routing, three group-held-out calibration folds and a calibration-only final fit. No target-based test cleaning or tuning.')
            st.json(forest[1]['base_assessment'])
    with st.expander('SGD-Huber residuals · what did the base model miss?'):
        view=rows[['manufacturer','model','age','observed_price','base_price','corrected_price']].copy()
        view['base_residual_usd']=view.observed_price-view.base_price
        view['corrected_residual_usd']=view.observed_price-view.corrected_price
        st.caption('Residual = observed price minus prediction. Positive: underpriced by the model. Negative: overpriced. Both columns use the same supplied test cars; none are dropped.')
        if 'sgd_price' in rows:
            view['sgd_price']=rows.sgd_price
            view['sgd_residual_usd']=rows.observed_price-rows.sgd_price
        st.dataframe(view,hide_index=True)
    st.markdown('#### 03 · Inspect one held-out car')
    choice=st.selectbox('Held-out example',range(len(rows)),format_func=lambda i:f"Example {i+1} · {rows.iloc[i]['manufacturer']} {rows.iloc[i]['model']}",key='hierarchy_example')
    row=rows.iloc[choice]
    if st.button('Compare hierarchical estimate',key='hierarchy_compare'):
        st.session_state['hierarchy_checked']=(bundle['fingerprint'],choice)
        from pricing.hierarchical_store import record
        record(side,row,bundle,str(uuid.uuid4()))
    if st.session_state.get('hierarchy_checked')==(bundle['fingerprint'],choice):
        columns=st.columns(3 if 'sgd_price' in row else 2)
        left,right=columns[:2]
        third=columns[2] if len(columns)==3 else None
        left.metric('Base SGD-Huber · supplied historical prediction',f"${row.base_price:,.0f}")
        right.metric('Hierarchical · same price basis',f"${row.corrected_price:,.0f}",f"{100*(row.corrected_price/row.base_price-1):+.1f}%",delta_color='off')
        if 'sgd_price' in row:third.metric('SGD-Huber · same price basis',f"${row.sgd_price:,.0f}")
        if forest:
            match=forest[2][forest[2].row_id==row.row_id]
            if len(match)==1:
                ra,rb=st.columns(2);fr=match.iloc[0]
                ra.metric('XGBoost RF · same historical car',f"${fr.base_price:,.0f}")
                rb.metric('RF + hierarchy · same historical car',f"${fr.corrected_price:,.0f}")
        st.caption('This is a held-out historical check, not a live offer. Compare only prices on the supplied common basis.')
        st.bar_chart(pd.DataFrame({'Log-price correction':[row.global_effect,row.make_effect,row.model_effect,row.age_effect]},index=['Market','Make','Model','Age']))
        st.write(f"Make support: {int(row.make_n):,} calibration rows · model support: {int(row.model_n):,} · model pooling weight: {row.model_weight:.1%}.")
        st.caption(f'Approximate correction standard error: {row.correction_sd_log:.4f} log units. It excludes base-model error, hyperparameter uncertainty and market-date uncertainty.')
    if 'hierarchy_residual_export' in st.session_state:
        st.download_button('Download SGD-Huber residuals',st.session_state['hierarchy_residual_export'],'sgd-residuals.csv','text/csv',key='hierarchy_residual_download')
    else:
        st.download_button('Download inspected test examples',rows.to_csv(index=False),'original-test-examples.csv','text/csv',key='hierarchy_residual_download')
        st.caption('Full residuals, test predictions and fitted models are versioned in models/hierarchical-original in the source repository. The download above is the displayed subset, not the complete test.')
    st.download_button('Download hierarchical experiment',json.dumps({'model':bundle,'evaluation':report},indent=2),'hierarchical-experiment.json','application/json',key='hierarchy_export')
    st.caption('Uploaded rows stay in this Streamlit session; they are not added to the training repository. Only explicitly requested, consented comparison summaries are saved privately to Turso, with 90-day retention. GA receives bounded panel/action names, never the uploaded rows or model labels.')
