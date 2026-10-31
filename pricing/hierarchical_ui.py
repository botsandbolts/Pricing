"""Original-cohort forest evaluation, partial pooling and date-aware scenarios."""
import io,json,uuid
import pandas as pd
import streamlit as st
from pricing.hierarchical import REQUIRED,experiment
from pricing.original_residual_results import load


@st.cache_data(show_spinner=False)
def installed(root,side):return load(root,side)


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
<article class="hierarchy-step"><span class="number">01 / START WITH THE CAR</span><h3>The forest reads the facts.</h3><p>Vehicle facts give the forest its starting point. This separate experiment excludes text embeddings.</p></article>
<article class="hierarchy-step"><span class="number">02 / ASK THE FAMILY</span><h3>Related cars share evidence.</h3><p>A rare trim borrows from its make. More evidence gives its own model a stronger voice. Family advice, with fewer unsolicited opinions.</p></article>
<article class="hierarchy-step"><span class="number">03 / CHECK THE RECEIPTS</span><h3>Better is something we measure.</h3><p>Compare both versions on the same untouched cars. No difficult test cars disappear behind the garage.</p></article>
</div>
"""
    side=st.session_state.get('hierarchy_side','ask')
    if installed(str(root),side):
        hero=hero.replace('MODEL STATUS · AWAITING CALIBRATION DATA','XGBOOST RANDOM FOREST · ORIGINAL DATA').replace('The laboratory is ready. A trained correction and measured accuracy gain are not available yet. Your current price calculator keeps its existing models.','One random forest. One family-tree correction. Same untouched test. The main calculator and description dropdown keep their existing models.')
    st.markdown(hero,unsafe_allow_html=True)
    if st.session_state.get('studio_tabs')!='🌳 Hierarchical model':return
    side=st.radio('Hierarchical market',['ask','bid'],format_func=lambda x:'Asking · Craigslist' if x=='ask' else 'Wholesale · auction',horizontal=True,key='hierarchy_side')
    saved=st.session_state.get('hierarchy_experiment')
    if not saved or saved[0]['side']!=side:saved=installed(str(root),side)
    with st.expander('How this works · plain English and statistical recipe'):
        st.write('The forest estimates log price from vehicle facts. A family-tree correction learns the remaining out-of-fold errors: market, make and model offsets, plus supported age patterns. Sparse models borrow strength. Family advice, with fewer unsolicited opinions.')
        st.write('XGBoost RF: 200 parallel trees in one boosting round, depth 8, native categorical splits. The hierarchy is sequential empirical Bayes, not joint MCMC. Correction standard errors are not individual sale-price prediction intervals.')
        st.warning('Historical asking advertisements and wholesale sales are different targets. This experiment excludes SGD, CatBoost, MiniLM and production outlier cleaning. Your ordinary calculator and description features remain available.')
    with st.expander('Developer workbench · data, fitting & provenance'):
        from pricing.forest_residuals import OBSERVATION_COLUMNS,generate
        mode=st.radio('Residual input',['Generate from vehicle observations','Use existing OOF residuals'],key='hierarchy_input_mode')
        automatic=mode.startswith('Generate');columns=OBSERVATION_COLUMNS if automatic else REQUIRED
        st.caption('Optional uploads: maximum 10,000 rows. Installed results use all eligible original cars. Supplied base residuals must be genuinely group-held-out.')
        st.download_button('Download residual-data template',','.join(columns)+'\n','hierarchical-input-template.csv','text/csv',key='hierarchy_template')
        source=st.file_uploader('Vehicle observations or OOF predictions · CSV',type=['csv'],key='hierarchy_data')
        confirmed=st.checkbox('Vehicle identities and partitions are correct; existing predictions, if supplied, exclude held-out vehicles.',key='hierarchy_provenance')
        if st.button('Fit and evaluate hierarchical correction',disabled=source is None or not confirmed,type='primary',key='hierarchy_fit'):
            try:
                if source.size>25*1024*1024:raise ValueError('CSV must be at most 25 MB.')
                with st.spinner('Growing the forest, then convening the family council…'):
                    data=pd.read_csv(io.BytesIO(source.getvalue()));provenance=None
                    if automatic:data,provenance=generate(data,side,max_rows=10000)
                    bundle,report,rows=experiment(data,side)
                    if provenance:report['base_assessment']=provenance
                st.session_state['hierarchy_experiment']=(bundle,report,rows)
                st.session_state['hierarchy_residual_export']=data.to_csv(index=False)
                st.session_state.pop('hierarchy_checked',None);st.rerun()
            except (ValueError,TypeError,KeyError,pd.errors.ParserError) as exc:st.error(str(exc))
    if not saved:
        st.info('No installed comparison for this market yet.');return
    bundle,report,rows=saved
    provenance=report.get('provenance')
    if provenance:
        st.success(f"Original dataset · {provenance['eligible_rows']:,} eligible cars · 100% of the eligible cohort")
        st.caption(f"{provenance['source']} · {provenance['price_basis']} · {report['calibration_rows']:,} calibration cars. All test rows are scored; at most 1,000 examples are displayed.")
        st.link_button('Original public dataset',provenance['source_url'])
    if 'base_assessment' in report:st.caption(report['base_assessment']['warning'])
    st.markdown('#### 02 · Did the family advice help?')
    test=report['cohorts']['test'];a,b=st.columns(2)
    a.metric('XGBoost RF · test MAE',f"${test['base']['mae_usd']:,.0f}")
    b.metric('RF + hierarchy · test MAE',f"${test['hierarchical']['mae_usd']:,.0f}")
    st.caption(f"Same {test['base']['rows']:,} historical test cars. Lower MAE means a smaller average dollar miss. No difficult test cars disappeared behind the garage.")
    lo,hi=test['mae_change_95_group_bootstrap']
    st.metric('MAE change · hierarchy minus forest',f"{test['mae_change_usd']:+,.2f} USD")
    st.caption(f'Negative is better. Paired group bootstrap, 300 resamples: 95% interval [{lo:+,.2f}, {hi:+,.2f}] USD. This concerns the MAE difference, not a car price.')
    with st.expander('Full scorecard · MAE, RMSE, log error and bias'):
        metrics=[{'Cohort':split,'Model':'Forest' if method=='base' else 'Forest + hierarchy',**r[method]} for split,r in report['cohorts'].items() for method in ['base','hierarchical']]
        st.dataframe(pd.DataFrame(metrics),hide_index=True)
        st.caption('RMSE penalizes extreme misses. Bias is prediction minus observed price. Improving MAE does not guarantee improved bias.')
    with st.expander('Random forest · full technical receipt'):
        st.json(report.get('base_assessment',{}))
        st.caption('Large fitted forests and full residual tables are offline reproduction outputs. This panel inspects held-out cars; it does not replace the live calculator with arbitrary-input forest inference.')
    with st.expander('Residuals · what did the forest miss?'):
        view=rows[['manufacturer','model','age','observed_price','base_price','corrected_price']].copy()
        view['forest_residual_usd']=view.observed_price-view.base_price
        view['corrected_residual_usd']=view.observed_price-view.corrected_price
        st.caption('Positive residual: underestimated. Displayed rows are an inspection subset; the scorecard uses the complete test cohort.')
        st.dataframe(view,hide_index=True)
    st.markdown('#### 03 · Inspect one held-out car')
    choice=st.selectbox('Held-out example',range(len(rows)),format_func=lambda i:f"Example {i+1} · {rows.iloc[i]['manufacturer']} {rows.iloc[i]['model']}",key='hierarchy_example')
    row=rows.iloc[choice]
    if st.button('Compare hierarchical estimate',key='hierarchy_compare'):
        st.session_state['hierarchy_checked']=(bundle['fingerprint'],choice)
        from pricing.hierarchical_store import record
        record(side,row,bundle,str(uuid.uuid4()))
    if st.session_state.get('hierarchy_checked')==(bundle['fingerprint'],choice):
        a,b=st.columns(2)
        a.metric('Forest · historical prediction',f"${row.base_price:,.0f}")
        b.metric('Forest + hierarchy · same price basis',f"${row.corrected_price:,.0f}",f"{100*(row.corrected_price/row.base_price-1):+.1f}%",delta_color='off')
        st.caption(f"Observed historical target: ${row.observed_price:,.0f}. This is a held-out check, not an offer.")
        st.bar_chart(pd.DataFrame({'Log-price correction':[row.global_effect,row.make_effect,row.model_effect,row.age_effect]},index=['Market','Make','Model','Age']))
        st.write(f"Make support: {int(row.make_n):,} · model support: {int(row.model_n):,} · model pooling weight: {row.model_weight:.1%}.")
        st.caption(f'Approximate correction standard error: {row.correction_sd_log:.4f} log units; excludes base-model, hyperparameter and date uncertainty.')
        if 'year' in row and 'source_date' in row:
            from pricing.forest_time import render as render_date
            render_date(root,row,side)
    if 'hierarchy_residual_export' in st.session_state:
        st.download_button('Download forest residuals',st.session_state['hierarchy_residual_export'],'forest-residuals.csv','text/csv',key='hierarchy_residual_download')
    else:st.download_button('Download inspected test examples',rows.to_csv(index=False),'forest-test-examples.csv','text/csv',key='hierarchy_residual_download')
    st.download_button('Download hierarchical experiment',json.dumps({'model':bundle,'evaluation':report},indent=2),'hierarchical-experiment.json','application/json',key='hierarchy_export')
    st.caption('Uploads remain session-local. Only explicit, consented historical comparisons are saved privately to Turso for 90 days. Date scenarios are not logged as historical predictions. GA receives bounded action names, never uploaded rows or descriptions.')
