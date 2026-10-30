"""On-demand hierarchical residual laboratory; no model promotion or raw-file storage."""
import io,json,uuid
import pandas as pd
import streamlit as st
from pricing.hierarchical import REQUIRED,experiment


def render(root):
    st.subheader('The family tree · a second opinion for CatBoost')
    st.markdown('### Rare trim? Ask the family. Then check the receipts.')
    st.write('A correction learns from related makes and models. Sparse groups borrow strength from the wider market; well-supported groups can speak for themselves. Even a rare Volvo gets a family meeting.')
    a,b,c=st.columns(3)
    a.metric('Layer 1','Whole market');b.metric('Layer 2','Make → model');c.metric('Layer 3','Residual age trend')
    st.info('Research panel · no hierarchical correction is installed in the landing-page models. Signed CatBoost out-of-fold residuals are not included in the deployed artifacts. Supply a properly held-out experiment below to fit and evaluate one.')
    with st.expander('How this works · plain English and statistical recipe'):
        st.write('Start with CatBoost’s price. Learn log(observed price / base price) on calibration rows whose base predictions were made out of fold. Estimate global, make and model offsets, with smaller offsets for sparse groups. A regularized common age slope and supported model-age slopes explain remaining age patterns. Multiply the base estimate by exp(total correction).')
        st.write('This is sequential empirical-Bayes partial pooling with Gaussian residual assumptions. Group prior variances are estimated from calibration data only. It is not a joint MCMC fit. Correction standard errors are approximate and omit hyperparameter uncertainty; they are not individual-car prediction intervals.')
        st.warning('Asking and auction targets need separate files and models. A correction to a historical price is not a forecast of a current transaction price. The CatBoost base model and its text branch must exclude each validation/test vehicle and must generate genuine OOF calibration predictions. CSV declarations alone cannot prove that provenance.')
    # Streamlit executes every tab: expensive work and file handling stay on this panel.
    if st.session_state.get('studio_tabs')!='🌳 Hierarchical model':return
    st.markdown('#### 01 · Bring an honest experiment')
    side=st.radio('Hierarchical market',['ask','bid'],format_func=lambda x:'Asking · Craigslist' if x=='ask' else 'Wholesale · auction',horizontal=True,key='hierarchy_side')
    st.caption('One CSV, up to 100,000 rows / 25 MB. Calibration fits the correction; validation and test remain untouched. The separate base model must already have been trained without their labels. No trimming in this experiment.')
    st.download_button('Download residual-data template',','.join(REQUIRED)+'\n','hierarchical-residual-template.csv','text/csv',key='hierarchy_template')
    source=st.file_uploader('OOF calibration and held-out predictions · CSV',type=['csv'],key='hierarchy_data')
    confirmed=st.checkbox('I verified that the base model and text predictions exclude held-out vehicles, and calibration predictions are genuinely out of fold.',key='hierarchy_provenance')
    if st.button('Fit and evaluate hierarchical correction',disabled=source is None or not confirmed,type='primary',key='hierarchy_fit'):
        try:
            if source.size>25*1024*1024:raise ValueError('CSV must be at most 25 MB.')
            with st.spinner('Convening the make-and-model family council…'):
                bundle,report,rows=experiment(pd.read_csv(io.BytesIO(source.getvalue())),side)
            st.session_state['hierarchy_experiment']=(bundle,report,rows)
            st.session_state.pop('hierarchy_checked',None)
        except (ValueError,TypeError,KeyError,pd.errors.ParserError) as exc:st.error(str(exc))
    saved=st.session_state.get('hierarchy_experiment')
    if not saved or saved[0]['side']!=side:return
    bundle,report,rows=saved
    st.markdown('#### 02 · Did it actually help?')
    metrics=[]
    for split,r in report['cohorts'].items():
        for method in ['base','hierarchical']:metrics.append({'Cohort':split,'Model':method,**r[method]})
    st.dataframe(pd.DataFrame(metrics),hide_index=True)
    test=report['cohorts']['test'];lo,hi=test['mae_change_95_group_bootstrap']
    st.metric('Test MAE change · corrected minus base',f"{test['mae_change_usd']:+,.2f} USD",help='Negative means lower error. The test includes every supplied test row.')
    st.caption(f'Paired vehicle-group bootstrap, 300 resamples: 95% interval [{lo:+,.2f}, {hi:+,.2f}] USD. This interval concerns the MAE difference, not a car’s price. No automatic promotion or increased-accuracy claim.')
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
    st.download_button('Download hierarchical experiment',json.dumps({'model':bundle,'evaluation':report},indent=2),'hierarchical-experiment.json','application/json',key='hierarchy_export')
    st.caption('Uploaded rows stay in this Streamlit session; they are not added to the training repository. Only explicitly requested, consented comparison summaries are saved privately to Turso, with 90-day retention. GA receives bounded panel/action names, never the uploaded rows or model labels.')
