"""On-demand hierarchical residual laboratory; no model promotion or raw-file storage."""
import io,json,uuid
import pandas as pd
import streamlit as st
from pricing.hierarchical import REQUIRED,experiment


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
<div class="eyebrow">THE FAMILY TREE · A SECOND OPINION FOR CATBOOST</div>
<h2>Rare trim?<br>Ask the family.</h2>
<p>One unusual car should not have to explain the entire market. This experiment lets related makes and models share evidence, while giving well-supported models their own voice.</p>
<span class="hierarchy-status">MODEL STATUS · AWAITING CALIBRATION DATA</span>
<p style="font-size:.9rem;margin-bottom:0">The laboratory is ready. A trained correction and measured accuracy gain are not available yet. Your current price calculator keeps its existing models.</p>
</section>
<div class="hierarchy-steps" aria-label="How partial pooling works">
<article class="hierarchy-step"><span class="number">01 / START WITH THE CAR</span><h3>CatBoost gives its estimate.</h3><p>Vehicle facts and the asking model’s text signal provide the starting point.</p></article>
<article class="hierarchy-step"><span class="number">02 / ASK THE FAMILY</span><h3>Related cars share evidence.</h3><p>A rare trim borrows from its make. More evidence gives its own model a stronger voice. Family advice, with fewer unsolicited opinions.</p></article>
<article class="hierarchy-step"><span class="number">03 / CHECK THE RECEIPTS</span><h3>Better is something we measure.</h3><p>Compare both versions on the same untouched cars. No difficult test cars disappear behind the garage.</p></article>
</div>
"""
    active=st.session_state.get('hierarchy_experiment')
    if active and active[0]['side']==st.session_state.get('hierarchy_side','ask'):
        hero=hero.replace('MODEL STATUS · AWAITING CALIBRATION DATA','SESSION EXPERIMENT · NOT PROMOTED').replace('The laboratory is ready. A trained correction and measured accuracy gain are not available yet. Your current price calculator keeps its existing models.','A correction is fitted in this session. Review its held-out evidence below. The main price calculator keeps its existing models.')
    st.markdown(hero,unsafe_allow_html=True)
    with st.expander('How this works · plain English and statistical recipe'):
        st.write('Start with CatBoost’s price. Learn log(observed price / base price) on calibration rows whose base predictions were made out of fold. Estimate global, make and model offsets, with smaller offsets for sparse groups. A regularized common age slope and supported model-age slopes explain remaining age patterns. Multiply the base estimate by exp(total correction).')
        st.write('This is sequential empirical-Bayes partial pooling with Gaussian residual assumptions. Group prior variances are estimated from calibration data only. It is not a joint MCMC fit. Correction standard errors are approximate and omit hyperparameter uncertainty; they are not individual-car prediction intervals.')
        st.warning('Asking and auction targets need separate files and models. A correction to a historical price is not a forecast of a current transaction price. The CatBoost base model and its text branch must exclude each validation/test vehicle and must generate genuine OOF calibration predictions. CSV declarations alone cannot prove that provenance.')
    # Streamlit executes every tab: expensive work and file handling stay on this panel.
    if st.session_state.get('studio_tabs')!='🌳 Hierarchical model':return
    st.markdown('#### Choose your market')
    side=st.radio('Hierarchical market',['ask','bid'],format_func=lambda x:'Asking · Craigslist' if x=='ask' else 'Wholesale · auction',horizontal=True,key='hierarchy_side')
    with st.expander('Developer workbench · data, fitting & provenance',expanded=False):
        from pricing.catboost_residuals import OBSERVATION_COLUMNS,generate
        mode=st.radio('Residual input',['Generate from vehicle observations','Use existing OOF residuals'],key='hierarchy_input_mode')
        automatic=mode.startswith('Generate')
        columns=OBSERVATION_COLUMNS if automatic else REQUIRED
        st.caption('Automatic mode fits a fresh tabular CatBoost assessment: asking AFT or auction quantile 0.5. It generates group-aware OOF residuals, then fits the hierarchy. This is a CPU pilot (10,000 rows); MiniLM and five-stage cleaning are not included. It does not overwrite production models.' if automatic else 'Upload verified OOF calibration and held-out base predictions. All test rows are retained.')
        st.download_button('Download residual-data template',','.join(columns)+'\n','hierarchical-input-template.csv','text/csv',key='hierarchy_template')
        source=st.file_uploader('Vehicle observations or OOF predictions · CSV',type=['csv'],key='hierarchy_data')
        confirmed=st.checkbox('Vehicle identities and partitions are correct; existing predictions, if supplied, exclude held-out vehicles.',key='hierarchy_provenance')
        if st.button('Fit and evaluate hierarchical correction',disabled=source is None or not confirmed,type='primary',key='hierarchy_fit'):
            try:
                if source.size>25*1024*1024:raise ValueError('CSV must be at most 25 MB.')
                with st.spinner('Cross-fitting CatBoost, then convening the family council…'):
                    data=pd.read_csv(io.BytesIO(source.getvalue()));provenance=None
                    if automatic:data,provenance=generate(data,side)
                    bundle,report,rows=experiment(data,side)
                    if provenance:report['base_assessment']=provenance
                st.session_state['hierarchy_experiment']=(bundle,report,rows)
                st.session_state['hierarchy_residual_export']=data.to_csv(index=False)
                st.session_state.pop('hierarchy_checked',None)
                st.rerun()
            except (ValueError,TypeError,KeyError,pd.errors.ParserError) as exc:st.error(str(exc))
    saved=st.session_state.get('hierarchy_experiment')
    if not saved or saved[0]['side']!=side:
        st.info('No trained comparison for this market yet. The developer workbench accepts verified residual data and measures whether the correction helps before showing a price comparison.')
        return
    bundle,report,rows=saved
    if 'base_assessment' in report:st.caption(report['base_assessment']['warning'])
    st.markdown('#### 02 · Did it actually help?')
    metrics=[]
    for split,r in report['cohorts'].items():
        for method in ['base','hierarchical']:metrics.append({'Cohort':split,'Model':method,**r[method]})
    with st.expander('Full scorecard · MAE, RMSE, log error and bias'):
        st.dataframe(pd.DataFrame(metrics),hide_index=True)
    test=report['cohorts']['test'];lo,hi=test['mae_change_95_group_bootstrap']
    st.metric('Test MAE change · corrected minus base',f"{test['mae_change_usd']:+,.2f} USD",help='Negative means lower error. The test includes every supplied test row.')
    st.caption(f'Paired vehicle-group bootstrap, 300 resamples: 95% interval [{lo:+,.2f}, {hi:+,.2f}] USD. This interval concerns the MAE difference, not a car’s price. No automatic promotion or increased-accuracy claim.')
    with st.expander('CatBoost residuals · what did the base model miss?'):
        view=rows[['manufacturer','model','age','observed_price','base_price','corrected_price']].copy()
        view['base_residual_usd']=view.observed_price-view.base_price
        view['corrected_residual_usd']=view.observed_price-view.corrected_price
        st.caption('Residual = observed price minus prediction. Positive: underpriced by the model. Negative: overpriced. Both columns use the same supplied test cars; none are dropped.')
        st.dataframe(view,hide_index=True)
    st.markdown('#### 03 · Inspect one held-out car')
    choice=st.selectbox('Held-out example',range(len(rows)),format_func=lambda i:f"Example {i+1} · {rows.iloc[i]['manufacturer']} {rows.iloc[i]['model']}",key='hierarchy_example')
    row=rows.iloc[choice]
    if st.button('Compare hierarchical estimate',key='hierarchy_compare'):
        st.session_state['hierarchy_checked']=(bundle['fingerprint'],choice)
        from pricing.hierarchical_store import record
        record(side,row,bundle,str(uuid.uuid4()))
    if st.session_state.get('hierarchy_checked')==(bundle['fingerprint'],choice):
        left,right=st.columns(2)
        left.metric('Base CatBoost · supplied historical prediction',f"${row.base_price:,.0f}")
        right.metric('Hierarchical · same price basis',f"${row.corrected_price:,.0f}",f"{100*(row.corrected_price/row.base_price-1):+.1f}%",delta_color='off')
        st.caption('This is a held-out historical check, not a live offer. Compare only prices on the supplied common basis.')
        st.bar_chart(pd.DataFrame({'Log-price correction':[row.global_effect,row.make_effect,row.model_effect,row.age_effect]},index=['Market','Make','Model','Age']))
        st.write(f"Make support: {int(row.make_n):,} calibration rows · model support: {int(row.model_n):,} · model pooling weight: {row.model_weight:.1%}.")
        st.caption(f'Approximate correction standard error: {row.correction_sd_log:.4f} log units. It excludes base-model error, hyperparameter uncertainty and market-date uncertainty.')
    st.download_button('Download CatBoost residuals',st.session_state.get('hierarchy_residual_export',''),'catboost-residuals.csv','text/csv',key='hierarchy_residual_download')
    st.download_button('Download hierarchical experiment',json.dumps({'model':bundle,'evaluation':report},indent=2),'hierarchical-experiment.json','application/json',key='hierarchy_export')
    st.caption('Uploaded rows stay in this Streamlit session; they are not added to the training repository. Only explicitly requested, consented comparison summaries are saved privately to Turso, with 90-day retention. GA receives bounded panel/action names, never the uploaded rows or model labels.')
